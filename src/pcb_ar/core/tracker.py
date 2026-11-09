"""局部实时画面 -> 全局 PCB 图的 ORB + RANSAC 相似变换配准。

算法链路
--------
1. 全局图预先缩放、CLAHE 增强并提取 ORB 描述子（只做一次）；
2. 实时帧降采样到工作宽度后提取 ORB，KNN + Lowe ratio 筛选匹配；
3. ``estimateAffinePartial2D`` 求 4 自由度相似变换（平移、旋转、等比缩放）；
4. 用内点数、尺度跳变、旋转跳变、相似性约束做质量门控；
5. 对旋转/缩放/平移做 EMA 平滑，输出 全局图 -> 原始实时帧 的 2x3 矩阵。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pcb_ar.core.geo import (
    as_affine,
    blend_similarity,
    build_similarity,
    decompose_similarity,
    is_valid_similarity,
    matrix_rotation_deg,
    matrix_scale,
)


@dataclass
class TrackerConfig:
    global_match_width: int = 1800
    live_match_width: int = 1200
    global_features: int = 5000
    live_features: int = 1600
    ratio: float = 0.76
    min_matches: int = 18
    min_inliers: int = 12
    ransac_px: float = 4.0
    smoothing: float = 0.65
    max_lost_frames: int = 8
    max_scale_jump: float = 2.2
    max_rotation_jump_deg: float = 28.0
    min_overlap: float = 0.12
    use_clahe: bool = True


@dataclass
class TrackResult:
    matrix: np.ndarray | None
    state: str = "searching"  # searching | locked | stale | lost
    confidence: float = 0.0
    inliers: int = 0
    matches: int = 0
    latency_ms: float = 0.0
    message: str = ""
    scale: float = 0.0
    rotation_deg: float = 0.0
    timestamp: float = field(default_factory=time.time)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_pose(self) -> bool:
        return self.matrix is not None

    @property
    def is_tracking(self) -> bool:
        return self.state == "locked"


class GlobalLocalTracker:
    def __init__(self, global_image: np.ndarray, config: TrackerConfig | None = None) -> None:
        if global_image is None or global_image.size == 0:
            raise ValueError("全局图为空")
        self.config = config or TrackerConfig()
        self.global_image = global_image
        self.global_h, self.global_w = global_image.shape[:2]
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self._orb_global = cv2.ORB_create(
            nfeatures=self.config.global_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
        self._orb_live = cv2.ORB_create(
            nfeatures=self.config.live_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.global_scale = min(1.0, self.config.global_match_width / float(self.global_w))
        work_w = max(320, int(round(self.global_w * self.global_scale)))
        work_h = max(240, int(round(self.global_h * self.global_scale)))
        self.global_work_size = (work_w, work_h)
        gray = self._gray_and_enhance(global_image, self.global_scale)
        self.kp_global, self.des_global = self._orb_global.detectAndCompute(gray, None)
        if self.des_global is None or len(self.kp_global) < 30:
            raise ValueError("全局图特征点不足，无法建立跟踪基准")
        self.global_points = np.float32([kp.pt for kp in self.kp_global]).reshape(-1, 2)
        self.last_result: TrackResult | None = None
        self.last_matrix: np.ndarray | None = None
        self.last_raw_matrix: np.ndarray | None = None
        self.lost_frames = 0
        self.frame_index = 0

    def _gray_and_enhance(self, image: np.ndarray, scale: float) -> np.ndarray:
        if scale < 0.999:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.config.use_clahe:
            gray = self._clahe.apply(gray)
        return gray

    def reset(self) -> None:
        self.last_result = None
        self.last_matrix = None
        self.last_raw_matrix = None
        self.lost_frames = 0

    def track(self, frame: np.ndarray) -> TrackResult:
        started = time.perf_counter()
        self.frame_index += 1
        if frame is None or frame.size == 0:
            return self._failure("空帧", started)
        live_h, live_w = frame.shape[:2]
        live_scale = min(1.0, self.config.live_match_width / float(live_w))
        live_work_w = max(240, int(round(live_w * live_scale)))
        live_work_h = max(160, int(round(live_h * live_scale)))
        gray = self._gray_and_enhance(frame, live_scale)
        kp_live, des_live = self._orb_live.detectAndCompute(gray, None)
        if des_live is None or len(kp_live) < self.config.min_matches:
            return self._failure(f"实时帧特征不足（{len(kp_live) if kp_live else 0}）", started)

        matches = self._matcher.knnMatch(des_live, self.des_global, k=2)
        good: list[cv2.DMatch] = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio * n.distance:
                good.append(m)
        if len(good) < self.config.min_matches:
            return self._failure(f"有效匹配不足（{len(good)}）", started, matches=len(good))

        src = np.float32([self.global_points[m.trainIdx] for m in good]).reshape(-1, 2)
        dst = np.float32([kp_live[m.queryIdx].pt for m in good]).reshape(-1, 2)
        matrix_work, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_px,
            maxIters=3000,
            confidence=0.997,
            refineIters=12,
        )
        if matrix_work is None or mask is None:
            return self._failure("RANSAC 未得到有效变换", started, matches=len(good))
        inliers = int(mask.sum())
        if inliers < self.config.min_inliers:
            return self._failure(
                f"内点不足（{inliers}）",
                started,
                matches=len(good),
                inliers=inliers,
            )

        # 工作尺度矩阵还原到原始全局图 / 原始实时帧
        matrix_full = self._rescale_matrix(matrix_work, self.global_scale, live_scale)
        matrix_full = self._snap_to_similarity(matrix_full)
        if not is_valid_similarity(matrix_full):
            return self._failure(
                "变换矩阵不满足相似变换约束",
                started,
                matches=len(good),
                inliers=inliers,
            )
        if not self._plausible(matrix_full):
            return self._failure(
                "位姿跳变过大，已拒绝",
                started,
                matches=len(good),
                inliers=inliers,
            )
        if not self._overlaps_frame(matrix_full, live_w, live_h):
            return self._failure(
                "匹配区域不在当前视野内",
                started,
                matches=len(good),
                inliers=inliers,
            )

        # 重投影误差（工作坐标系，像素）
        pred = cv2.transform(src.reshape(-1, 1, 2), matrix_work).reshape(-1, 2)
        errors = np.linalg.norm(pred - dst, axis=1)
        inlier_errors = errors[mask.ravel().astype(bool)]
        median_error = float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
        if median_error > self.config.ransac_px * 1.8:
            return self._failure(
                f"重投影误差过大（{median_error:.2f}px）",
                started,
                matches=len(good),
                inliers=inliers,
            )

        alpha = self.config.smoothing
        if self.last_matrix is not None and self._is_large_change(self.last_matrix, matrix_full, live_w):
            # 快速平移/旋转或重新捕获时不做插值，避免“拖尾”几十像素
            alpha = 1.0
        smoothed = blend_similarity(
            self.last_matrix,
            matrix_full,
            alpha,
            pivot=(self.global_w / 2.0, self.global_h / 2.0),
        )
        self.last_raw_matrix = matrix_full
        self.last_matrix = smoothed
        self.lost_frames = 0
        ratio_inliers = inliers / max(len(good), 1)
        confidence = float(
            np.clip(0.55 * ratio_inliers + 0.45 * min(1.0, inliers / 60.0), 0.0, 1.0)
        )
        scale, theta, _ = decompose_similarity(matrix_full)
        result = TrackResult(
            matrix=smoothed.copy(),
            state="locked",
            confidence=confidence,
            inliers=inliers,
            matches=len(good),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            message="跟踪中",
            scale=scale,
            rotation_deg=math.degrees(theta),
            diagnostics={"median_error_px": median_error, "raw_matrix": matrix_full.copy()},
        )
        self.last_result = result
        return result

    def _failure(
        self,
        reason: str,
        started: float,
        *,
        matches: int = 0,
        inliers: int = 0,
    ) -> TrackResult:
        self.lost_frames += 1
        state = "stale" if self.last_matrix is not None and self.lost_frames <= self.config.max_lost_frames else "lost"
        matrix = self.last_matrix.copy() if state == "stale" and self.last_matrix is not None else None
        result = TrackResult(
            matrix=matrix,
            state=state,
            confidence=0.0,
            inliers=inliers,
            matches=matches,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            message=reason,
        )
        self.last_result = result
        return result

    def _rescale_matrix(self, matrix_work: np.ndarray, global_scale: float, live_scale: float) -> np.ndarray:
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = matrix_work
        g = np.diag([global_scale, global_scale, 1.0])
        inv_l = np.diag([1.0 / live_scale, 1.0 / live_scale, 1.0])
        full = inv_l @ h @ g
        return full[:2, :].astype(np.float32)

    def _snap_to_similarity(self, matrix: np.ndarray) -> np.ndarray:
        scale, theta, t = decompose_similarity(matrix)
        return build_similarity(scale, theta, float(t[0]), float(t[1]))

    def _plausible(self, matrix: np.ndarray) -> bool:
        if self.last_raw_matrix is None:
            return True
        prev_scale, prev_theta, _ = decompose_similarity(self.last_raw_matrix)
        scale, theta, _ = decompose_similarity(matrix)
        if prev_scale > 0:
            ratio = scale / prev_scale
            if ratio > self.config.max_scale_jump or ratio < 1.0 / self.config.max_scale_jump:
                return False
        delta = abs((theta - prev_theta + math.pi) % (2 * math.pi) - math.pi)
        return math.degrees(delta) <= self.config.max_rotation_jump_deg

    def _is_large_change(self, previous: np.ndarray, current: np.ndarray, live_w: int) -> bool:
        center = np.float32([[self.global_w / 2.0, self.global_h / 2.0]])
        prev_center = cv2.transform(center.reshape(-1, 1, 2), as_affine(previous)).reshape(2)
        curr_center = cv2.transform(center.reshape(-1, 1, 2), as_affine(current)).reshape(2)
        translation = float(np.linalg.norm(prev_center - curr_center))
        if translation > max(70.0, live_w * 0.06):
            return True
        prev_scale, prev_theta, _ = decompose_similarity(previous)
        curr_scale, curr_theta, _ = decompose_similarity(current)
        angle_delta = abs((curr_theta - prev_theta + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(angle_delta) > 5.0:
            return True
        if prev_scale > 0 and abs(curr_scale / prev_scale - 1.0) > 0.10:
            return True
        return False

    def _overlaps_frame(self, matrix: np.ndarray, live_w: int, live_h: int) -> bool:
        corners = np.float32(
            [[0, 0], [self.global_w, 0], [self.global_w, self.global_h], [0, self.global_h]]
        )
        mapped = cv2.transform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        frame = np.float32([[0, 0], [live_w, 0], [live_w, live_h], [0, live_h]])
        intersection, _ = cv2.intersectConvexConvex(mapped, frame)
        if intersection <= 0:
            return False
        return intersection / float(live_w * live_h) >= self.config.min_overlap

    def project_global_box(self, bbox, matrix: np.ndarray | None = None) -> np.ndarray | None:
        from pcb_ar.core.geo import as_affine

        m = matrix if matrix is not None else self.last_matrix
        if m is None:
            return None
        pts = np.float32(
            [
                [bbox.x, bbox.y],
                [bbox.x2, bbox.y],
                [bbox.x2, bbox.y2],
                [bbox.x, bbox.y2],
            ]
        )
        return cv2.transform(pts.reshape(-1, 1, 2), as_affine(m)).reshape(-1, 2)


def pose_corner_error(expected: np.ndarray, actual: np.ndarray, width: int, height: int) -> float:
    """用图像四角平均误差评估两个全局->局部矩阵的差异（像素）。"""
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    a = cv2.transform(corners, as_affine(expected)).reshape(-1, 2)
    b = cv2.transform(corners, as_affine(actual)).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))
