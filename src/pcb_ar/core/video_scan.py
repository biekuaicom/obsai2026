"""连续视频扫描建图（Video Mosaic / Mapping Mode）。

与“散图拼接”不同，扫描时相邻视频帧重叠通常高达 80%~90%。本模块只估计
当前帧 -> 上一关键帧的小幅相似变换，并累加到扫描世界坐标：

    pose_world(current) = pose_world(last_keyframe) @ T(current -> last_keyframe)

满足平移/旋转阈值时保存关键帧，扫描结束后按已累加位姿一次性渲染，避免对
几十帧重复做全局特征匹配。关键帧以 JPEG 保存在内存中，控制内存占用。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from pcb_ar.core.geo import (
    as_homography,
    decompose_similarity,
    is_valid_similarity,
)
from pcb_ar.core.stitcher import StitchResult, render_mosaic

ProgressCb = Callable[[int, int, str], None]


class ScanError(RuntimeError):
    pass


@dataclass
class ScanConfig:
    match_width: int = 1200
    features: int = 3500
    ratio: float = 0.78
    min_matches: int = 24
    min_inliers: int = 16
    ransac_px: float = 4.0
    blur_threshold: float = 14.0
    keyframe_translation_px: float = 100.0
    keyframe_rotation_deg: float = 1.4
    keyframe_interval_s: float = 1.2
    min_overlap: float = 0.18
    max_scale_change: float = 1.35
    max_keyframes: int = 120
    keyframe_max_width: int = 1280
    jpeg_quality: int = 92


@dataclass
class Keyframe:
    index: int
    jpeg: bytes
    shape: tuple[int, int]
    pose: np.ndarray  # 当前帧原图坐标 -> 扫描世界坐标 (3x3)
    timestamp: float
    center_world: tuple[float, float]
    rotation_deg: float
    inliers: int
    matches: int


@dataclass
class ScanUpdate:
    accepted: bool = False
    tracked: bool = False
    message: str = ""
    keyframe_index: int = -1
    total_keyframes: int = 0
    matches: int = 0
    inliers: int = 0
    coverage: tuple[int, int] = (0, 0)
    pose: np.ndarray | None = None


@dataclass
class VideoScanResult:
    stitch: StitchResult
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    keyframe_count: int = 0


class VideoScanSession:
    def __init__(self, config: ScanConfig | None = None) -> None:
        self.config = config or ScanConfig()
        self.keyframes: list[Keyframe] = []
        self._orb = cv2.ORB_create(
            nfeatures=self.config.features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self._last_kf_des: np.ndarray | None = None
        self._last_kf_kp: np.ndarray | None = None
        self._last_kf_scale: float = 1.0
        self._last_kf_shape: tuple[int, int] = (0, 0)
        self._last_pose: np.ndarray | None = None
        self._last_timestamp = 0.0
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._source_width = 0
        self._bounds: list[float] | None = None
        self._failure_streak = 0
        self._last_pose_guess: np.ndarray | None = None
        self._last_update: ScanUpdate | None = None

    @property
    def count(self) -> int:
        return len(self.keyframes)

    @property
    def is_active(self) -> bool:
        return bool(self.keyframes)

    @property
    def last_update(self) -> ScanUpdate | None:
        return self._last_update

    def reset(self) -> None:
        self.keyframes.clear()
        self._last_kf_des = None
        self._last_kf_kp = None
        self._last_pose = None
        self._last_timestamp = 0.0
        self._source_width = 0
        self._bounds = None
        self._failure_streak = 0
        self._last_pose_guess = None
        self._last_update = None

    def _features(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = min(1.0, self.config.match_width / float(w))
        if scale < 1.0:
            work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = image
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        kp, des = self._orb.detectAndCompute(gray, None)
        return gray, scale, kp, des

    def add_frame(self, image: np.ndarray, timestamp: float | None = None) -> ScanUpdate:
        if image is None or image.size == 0:
            return self._update_failure("空帧")
        timestamp = float(timestamp if timestamp is not None else time.time())
        h, w = image.shape[:2]
        if not self._source_width:
            self._source_width = int(w)
        gray, scale, kp, des = self._features(image)
        blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_value < self.config.blur_threshold:
            return self._update_failure(f"画面偏模糊（清晰度 {blur_value:.0f}），请放慢移动")
        if des is None or len(kp) < self.config.min_matches:
            return self._update_failure("画面特征不足，请对准有丝印/焊盘的区域")
        if not self.keyframes:
            return self._accept_first(image, kp, des, scale, (h, w), timestamp, len(kp))
        if len(self.keyframes) >= self.config.max_keyframes:
            return self._update_failure(
                f"已达到关键帧上限 {self.config.max_keyframes}，请停止并生成拼图"
            )

        assert self._last_kf_des is not None and self._last_kf_kp is not None
        knn = self._matcher.knnMatch(des, self._last_kf_des, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio * n.distance:
                good.append(m)
        if len(good) < self.config.min_matches:
            self._failure_streak += 1
            return self._update_failure(
                f"与上一关键帧重叠不足（匹配 {len(good)}），请放慢移动或回退一点",
                matches=len(good),
            )
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst = np.float32([self._last_kf_kp[m.trainIdx] for m in good]).reshape(-1, 2)
        matrix_work, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_px,
            maxIters=4000,
            confidence=0.997,
            refineIters=10,
        )
        if matrix_work is None or mask is None:
            return self._update_failure("帧间变换求解失败", matches=len(good))
        inliers = int(mask.sum())
        if inliers < self.config.min_inliers:
            return self._update_failure(
                f"帧间内点不足（{inliers}），请放慢移动",
                matches=len(good),
                inliers=inliers,
            )
        matrix_full = self._rescale_to_full(matrix_work, scale, self._last_kf_scale)
        if not is_valid_similarity(matrix_full):
            return self._update_failure("帧间变换异常", matches=len(good), inliers=inliers)
        scale_ratio = math.sqrt(abs(float(np.linalg.det(matrix_full[:2, :2]))))
        if scale_ratio > self.config.max_scale_change or scale_ratio < 1.0 / self.config.max_scale_change:
            return self._update_failure(
                f"帧间尺度变化过大（{scale_ratio:.2f}），可能失焦或移动过快",
                matches=len(good),
                inliers=inliers,
            )
        overlap = self._overlap_with_previous(matrix_full, w, h)
        if overlap < self.config.min_overlap:
            return self._update_failure(
                f"与上一关键帧重叠仅 {overlap:.0%}，请回退并放慢移动",
                matches=len(good),
                inliers=inliers,
            )
        assert self._last_pose is not None
        pose_world = as_homography(self._last_pose) @ as_homography(matrix_full)
        self._last_pose_guess = pose_world
        self._failure_streak = 0
        movement, rotation = self._movement(matrix_full, w, h)
        due = (
            movement >= self.config.keyframe_translation_px
            or abs(rotation) >= self.config.keyframe_rotation_deg
            or (
                movement >= 12.0
                and (timestamp - self._last_timestamp) >= self.config.keyframe_interval_s
            )
        )
        if not due:
            self._last_update = ScanUpdate(
                accepted=False,
                tracked=True,
                message="正在扫描…",
                total_keyframes=len(self.keyframes),
                matches=len(good),
                inliers=inliers,
                coverage=self._coverage_size(),
                pose=pose_world,
            )
            return self._last_update
        return self._accept_keyframe(
            image,
            kp,
            des,
            scale,
            (h, w),
            timestamp,
            pose_world,
            len(good),
            inliers,
            movement,
            rotation,
        )

    def _accept_first(self, image, kp, des, scale, shape, timestamp, matches) -> ScanUpdate:
        pose = np.eye(3, dtype=np.float64)
        record = self._make_record(
            image, shape, pose, timestamp, inliers=matches, matches=matches, rotation=0.0
        )
        self.keyframes.append(record)
        self._last_kf_des = des
        self._last_kf_kp = np.float32([p.pt for p in kp]).reshape(-1, 2)
        self._last_kf_scale = scale
        self._last_kf_shape = shape
        self._last_pose = pose
        self._last_timestamp = timestamp
        self._extend_bounds(shape, pose)
        self._last_update = ScanUpdate(
            accepted=True,
            tracked=True,
            message="扫描开始，缓慢移动以覆盖整块 PCB",
            keyframe_index=record.index,
            total_keyframes=1,
            matches=matches,
            inliers=matches,
            coverage=self._coverage_size(),
            pose=pose,
        )
        return self._last_update

    def _accept_keyframe(
        self,
        image,
        kp,
        des,
        scale,
        shape,
        timestamp,
        pose,
        matches,
        inliers,
        movement,
        rotation,
    ) -> ScanUpdate:
        record = self._make_record(
            image,
            shape,
            pose,
            timestamp,
            inliers=inliers,
            matches=matches,
            rotation=rotation,
        )
        self.keyframes.append(record)
        self._last_kf_des = des
        self._last_kf_kp = np.float32([p.pt for p in kp]).reshape(-1, 2)
        self._last_kf_scale = scale
        self._last_kf_shape = shape
        self._last_pose = pose
        self._last_timestamp = timestamp
        self._extend_bounds(shape, pose)
        self._last_update = ScanUpdate(
            accepted=True,
            tracked=True,
            message=f"已记录关键帧 #{record.index + 1}（位移 {movement:.0f}px）",
            keyframe_index=record.index,
            total_keyframes=len(self.keyframes),
            matches=matches,
            inliers=inliers,
            coverage=self._coverage_size(),
            pose=pose,
        )
        return self._last_update

    def _make_record(
        self,
        image: np.ndarray,
        shape: tuple[int, int],
        pose: np.ndarray,
        timestamp: float,
        inliers: int,
        matches: int,
        rotation: float,
    ) -> Keyframe:
        # 关键帧落盘前按最大宽度降采样，位姿同步缩放，显著降低内存与渲染耗时。
        h, w = image.shape[:2]
        if w > self.config.keyframe_max_width:
            factor = self.config.keyframe_max_width / float(w)
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            pose = as_homography(pose) @ np.diag([1.0 / factor, 1.0 / factor, 1.0])
        ok, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)],
        )
        if not ok:
            raise ScanError("关键帧编码失败")
        h, w = image.shape[:2]
        center = self._apply(pose, (w / 2.0, h / 2.0))
        scale, theta, _ = decompose_similarity(pose)
        return Keyframe(
            index=len(self.keyframes),
            jpeg=buffer.tobytes(),
            shape=(int(h), int(w)),
            pose=np.asarray(pose, dtype=np.float64).copy(),
            timestamp=timestamp,
            center_world=(float(center[0]), float(center[1])),
            rotation_deg=math.degrees(theta),
            inliers=inliers,
            matches=matches,
        )

    def _rescale_to_full(self, matrix_work: np.ndarray, cur_scale: float, ref_scale: float) -> np.ndarray:
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = matrix_work
        # 当前工作坐标 -> 当前原图；目标原图 -> 目标工作坐标
        inv_ref = np.diag([1.0 / ref_scale, 1.0 / ref_scale, 1.0])
        cur = np.diag([cur_scale, cur_scale, 1.0])
        full = inv_ref @ h @ cur
        return full[:2, :].astype(np.float32)

    def _overlap_with_previous(self, matrix: np.ndarray, width: int, height: int) -> float:
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        mapped = cv2.transform(corners, matrix).reshape(-1, 2)
        ref_h, ref_w = self._last_kf_shape
        if ref_w <= 0 or ref_h <= 0:
            return 0.0
        area, _ = cv2.intersectConvexConvex(
            mapped,
            np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]),
        )
        return float(area) / float(max(1, width * height))

    @staticmethod
    def _movement(matrix: np.ndarray, width: int, height: int) -> tuple[float, float]:
        scale, theta, _ = decompose_similarity(matrix)
        center = np.float32([[[width / 2.0, height / 2.0]]])
        mapped = cv2.transform(center, matrix).reshape(2)
        movement = float(np.linalg.norm(mapped - np.array([width / 2.0, height / 2.0])))
        return movement, math.degrees(theta)

    def _extend_bounds(self, shape: tuple[int, int], pose: np.ndarray) -> None:
        h, w = shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        # 3x3 齐次矩阵不能用 cv2.transform（它不会做透视除法，会返回 3 个分量）。
        mapped = cv2.transform(corners, as_homography(pose)[:2, :]).reshape(-1, 2)
        if self._bounds is None:
            self._bounds = [
                float(mapped[:, 0].min()),
                float(mapped[:, 1].min()),
                float(mapped[:, 0].max()),
                float(mapped[:, 1].max()),
            ]
        else:
            self._bounds[0] = min(self._bounds[0], float(mapped[:, 0].min()))
            self._bounds[1] = min(self._bounds[1], float(mapped[:, 1].min()))
            self._bounds[2] = max(self._bounds[2], float(mapped[:, 0].max()))
            self._bounds[3] = max(self._bounds[3], float(mapped[:, 1].max()))

    def _coverage_size(self) -> tuple[int, int]:
        if self._bounds is None:
            return (0, 0)
        return (
            int(round(self._bounds[2] - self._bounds[0])),
            int(round(self._bounds[3] - self._bounds[1])),
        )

    @property
    def coverage_size(self) -> tuple[int, int]:
        return self._coverage_size()

    @staticmethod
    def _apply(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, as_homography(matrix)).reshape(2)
        return float(out[0]), float(out[1])

    def _update_failure(self, message: str, matches: int = 0, inliers: int = 0) -> ScanUpdate:
        self._last_update = ScanUpdate(
            accepted=False,
            tracked=False,
            message=message,
            total_keyframes=len(self.keyframes),
            matches=matches,
            inliers=inliers,
            coverage=self._coverage_size(),
            pose=self._last_pose_guess,
        )
        return self._last_update

    # ------------------------------------------------------------------ 输出
    def trajectory(self) -> list[tuple[float, float]]:
        return [record.center_world for record in self.keyframes]

    def build(self, progress: ProgressCb | None = None) -> VideoScanResult:
        if len(self.keyframes) < 2:
            raise ScanError("关键帧不足，至少需要扫描 2 个不同位置")
        images: list[np.ndarray] = []
        transforms: list[np.ndarray] = []
        # 关键帧已降采样；把扫描世界坐标同步缩放到关键帧原生像素，
        # 避免生成一张“用 1280 图放大的 1600 单位”拼图，降低显存与渲染耗时。
        world_scale = 1.0
        if self._source_width > 0 and self.keyframes:
            world_scale = min(1.0, self.keyframes[0].shape[1] / float(self._source_width))
        world = np.diag([world_scale, world_scale, 1.0])
        for i, record in enumerate(self.keyframes):
            data = np.frombuffer(record.jpeg, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ScanError(f"关键帧 #{i + 1} 解码失败")
            if image.shape[:2] != record.shape:
                raise ScanError(f"关键帧 #{i + 1} 尺寸不一致")
            images.append(image)
            transforms.append(world @ record.pose)
            if progress:
                progress(i + 1, len(self.keyframes) + 1, f"解码关键帧 {i + 1}/{len(self.keyframes)}…")
        stitch = render_mosaic(
            images,
            transforms,
            progress,
            max_canvas_side=12000,
            max_canvas_pixels=90_000_000,
            feather_px=28,
        )
        return VideoScanResult(
            stitch=stitch,
            trajectory=self.trajectory(),
            keyframe_count=len(images),
        )
