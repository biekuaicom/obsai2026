"""引导式轻量捕获：稳定检查、红色覆盖、重叠校验、关键帧地图。

这是默认建图流程：不再要求用户连续扫完整板后渲染一张大图，而是让用户
“停在稳定画面→拍第一张→移动并保留一部分红色覆盖→拍下一张”。每次只保存
1200px 关键帧、ORB 特征、板坐标位姿和覆盖多边形；每拍一张立即并入低分辨率
总览，并以所有关键帧覆盖多边形的并集作为红色已拍区域。完成后只整理稀疏地图，
不把全部关键帧攒到最后重新拼接。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from pcb_ar.core.board_rectify import rectify_board
from pcb_ar.core.geo import (
    as_homography,
    invert_affine,
    is_valid_similarity,
)
from pcb_ar.core.sparse_map import (
    FeatureSet,
    SparseKeyframe,
    SparseMap,
    extract_features,
)
from pcb_ar.core.structural_map import (
    StructuralMap,
    StructuralMatch,
    extract_structural_map,
    match_structural_maps,
    transform_structural_map,
)

ProgressCb = Callable[[int, int, str], None]


class SparseCaptureError(RuntimeError):
    pass


@dataclass
class SparseCaptureConfig:
    match_width: int = 1200
    keyframe_width: int = 1200
    features: int = 2200
    ratio: float = 0.76
    min_matches: int = 20
    min_inliers: int = 14
    ransac_px: float = 4.0
    stable_frames: int = 2
    max_motion_px: float = 3.0
    min_sharpness: float = 10.0
    min_overlap: float = 0.10
    max_overlap: float = 0.82
    min_keyframes: int = 2
    overview_scale: float = 0.8
    overview_max_side: int = 3200
    overview_max_pixels: int = 8_000_000
    overview_margin_px: int = 24
    feather_px: int = 10
    jpeg_quality: int = 90
    seam_refine: bool = True
    seam_min_overlap_px: int = 2500
    seam_min_cc: float = 0.18
    seam_max_shift_px: float = 24.0
    seam_max_rotation_deg: float = 2.5
    red_area_growth_warn: float = 0.45
    red_iou_warn: float = 0.35
    red_shape_warn: float = 1.15
    red_anomaly_frames: int = 2
    rectify_board: bool = True
    rectify_max_side: int = 3200
    rectify_max_pixels: int = 8_000_000


@dataclass
class RedMetrics:
    """红区面积/外形记忆值，用于快速发现整块假红。"""

    area_px: int = 0
    area_fraction: float = 0.0
    components: int = 0
    largest_share: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: tuple[float, float] = (0.0, 0.0)
    extent: float = 0.0
    solidity: float = 0.0
    aspect: float = 0.0
    hu: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float64))

    def shape_distance(self, other: "RedMetrics | None") -> float:
        if other is None:
            return 0.0
        # log-Hu 对平移/缩放不敏感，适合做“外形值”记忆比较。
        return float(np.mean(np.abs(self.hu - other.hu)))


def compute_red_metrics(mask: np.ndarray) -> RedMetrics:
    if mask is None or mask.size == 0:
        return RedMetrics()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape[:2]
    area = int(np.count_nonzero(mask_u8))
    if area == 0:
        return RedMetrics()
    k = max(1, int(round(min(h, w) * 0.002)))
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if not np.any(clean):
        clean = mask_u8
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30]
    largest = max(comp_areas) if comp_areas else area
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RedMetrics(area_px=area, area_fraction=area / float(h * w))
    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    ys, xs = np.nonzero(clean)
    gx0, gy0 = int(xs.min()), int(ys.min())
    gw, gh = int(xs.max() - gx0 + 1), int(ys.max() - gy0 + 1)
    extent = float(area) / float(max(1, gw * gh))
    hull = cv2.convexHull(contour)
    solidity = float(cv2.contourArea(contour)) / float(max(1.0, cv2.contourArea(hull)))
    moments = cv2.moments(clean, binaryImage=True)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx, cy = gx0 + gw / 2.0, gy0 + gh / 2.0
    return RedMetrics(
        area_px=area,
        area_fraction=area / float(h * w),
        components=len(comp_areas),
        largest_share=largest / float(max(1, area)),
        bbox=(gx0, gy0, gw, gh),
        centroid=(cx, cy),
        extent=extent,
        solidity=solidity,
        aspect=float(max(gw, gh)) / float(max(1, min(gw, gh))),
        hu=hu,
    )


@dataclass
class CapturePreview:
    stable: bool = False
    message: str = "等待稳定画面…"
    pose: np.ndarray | None = None
    coverage_polygons: list[np.ndarray] = field(default_factory=list)
    verified_keyframes: int = 0
    covered_fraction: float = 0.0
    new_fraction: float = 1.0
    matches: int = 0
    inliers: int = 0
    red_metrics: RedMetrics = field(default_factory=RedMetrics)
    red_iou: float = 0.0
    red_area_growth: float = 0.0
    red_shape_distance: float = 0.0
    red_anomaly: bool = False
    red_warning: str = ""
    sharpness: float = 0.0
    motion_px: float = 999.0
    can_capture: bool = False
    reason: str = ""


@dataclass
class CapturedKeyframe:
    index: int
    shape: tuple[int, int]
    pose: np.ndarray  # stored keyframe pixels -> scan world
    coverage: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    global_descriptor: np.ndarray
    jpeg: bytes
    timestamp: float
    red_metrics: RedMetrics | None = None
    matches: int = 0
    inliers: int = 0
    pose_source: int = -1
    seam_applied: bool = False
    seam_cc: float = 0.0
    seam_shift_px: float = 0.0
    seam_rotation_deg: float = 0.0


@dataclass
class CaptureAcceptance:
    accepted: bool
    message: str
    keyframe_index: int = -1


@dataclass
class SparseCaptureResult:
    overview: np.ndarray
    sparse_map: SparseMap
    coverage_polygons: list[np.ndarray]
    keyframe_count: int
    build_seconds: float = 0.0
    rectified: bool = False
    rectify_message: str = ""
    structural_map: StructuralMap | None = None
    structural_match: StructuralMatch | None = None


class SparseCaptureSession:
    def __init__(self, config: SparseCaptureConfig | None = None) -> None:
        self.config = config or SparseCaptureConfig()
        self.keyframes: list[CapturedKeyframe] = []
        self._prev_stability_gray: np.ndarray | None = None
        self._prev_stability_size: tuple[int, int] | None = None
        self._stable_run = 0
        self._last_frame: np.ndarray | None = None
        self._preview_frame: np.ndarray | None = None
        self._last_preview: CapturePreview | None = None
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._canvas: np.ndarray | None = None
        self._coverage_mask: np.ndarray | None = None
        self._canvas_origin = np.zeros(2, dtype=np.float64)
        self._canvas_scale = 1.0
        self._overview_version = 0
        self._merge_error = ""
        self._prev_red_mask: np.ndarray | None = None
        self._prev_red_pose: np.ndarray | None = None
        self._prev_red_metrics: RedMetrics | None = None
        self._capture_red_mask: np.ndarray | None = None
        self._capture_red_pose: np.ndarray | None = None
        self._capture_red_metrics: RedMetrics | None = None
        self._red_anomaly_run = 0
        self._red_memory: list[RedMetrics] = []

    @property
    def count(self) -> int:
        return len(self.keyframes)

    @property
    def is_active(self) -> bool:
        return bool(self.keyframes)

    @property
    def last_preview(self) -> CapturePreview | None:
        return self._last_preview

    @property
    def overview_image(self) -> np.ndarray | None:
        return self._canvas

    @property
    def overview_version(self) -> int:
        return self._overview_version

    @property
    def coverage_union_mask(self) -> np.ndarray | None:
        return self._coverage_mask

    @property
    def seam_refined_count(self) -> int:
        return sum(1 for kf in self.keyframes if kf.seam_applied)

    def _clear_red_guard(self) -> None:
        self._prev_red_mask = None
        self._prev_red_pose = None
        self._prev_red_metrics = None
        self._capture_red_mask = None
        self._capture_red_pose = None
        self._capture_red_metrics = None
        self._red_anomaly_run = 0

    def reset(self) -> None:
        self.keyframes.clear()
        self._prev_stability_gray = None
        self._prev_stability_size = None
        self._stable_run = 0
        self._last_frame = None
        self._preview_frame = None
        self._last_preview = None
        self._canvas = None
        self._coverage_mask = None
        self._canvas_origin = np.zeros(2, dtype=np.float64)
        self._canvas_scale = 1.0
        self._overview_version = 0
        self._merge_error = ""
        self._prev_red_mask = None
        self._prev_red_pose = None
        self._prev_red_metrics = None
        self._capture_red_mask = None
        self._capture_red_pose = None
        self._capture_red_metrics = None
        self._red_anomaly_run = 0
        self._red_memory.clear()

    # ------------------------------------------------------------------ 稳定性
    def _stability(self, frame: np.ndarray) -> tuple[bool, float, float, float]:
        h, w = frame.shape[:2]
        target_w = 800
        scale = min(1.0, target_w / float(w))
        work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        size = gray.shape
        motion = 999.0
        if self._prev_stability_gray is not None and self._prev_stability_size == size:
            shift, _response = cv2.phaseCorrelate(
                self._prev_stability_gray.astype(np.float32),
                gray.astype(np.float32),
            )
            motion = float(math.hypot(shift[0], shift[1]) / max(scale, 1e-6))
        self._prev_stability_gray = gray
        self._prev_stability_size = size
        if sharpness < self.config.min_sharpness or motion > self.config.max_motion_px:
            self._stable_run = 0
            return False, sharpness, motion, 0.0
        self._stable_run += 1
        stable = self._stable_run >= self.config.stable_frames
        return stable, sharpness, motion, 1.0

    # ------------------------------------------------------------------ 预览
    def preview(self, frame: np.ndarray) -> CapturePreview:
        if frame is None or frame.size == 0:
            preview = CapturePreview(message="没有画面", reason="empty")
            self._last_preview = preview
            return preview
        self._last_frame = frame
        self._preview_frame = frame
        stable, sharpness, motion, _ = self._stability(frame)
        preview = CapturePreview(
            stable=stable,
            sharpness=sharpness,
            motion_px=motion,
        )
        if not self.keyframes:
            self._clear_red_guard()
            preview.coverage_polygons = []
            preview.covered_fraction = 0.0
            preview.new_fraction = 1.0
            preview.can_capture = stable
            if stable:
                preview.message = "画面稳定，可以拍摄第一张"
                preview.reason = "ready_first"
            else:
                preview.message = (
                    f"请保持稳定：清晰度 {sharpness:.0f}，移动 {motion:.1f}px"
                )
                preview.reason = "unstable_first"
            self._last_preview = preview
            return preview

        last = self.keyframes[-1]
        features = extract_features(
            frame,
            max_width=self.config.match_width,
            nfeatures=self.config.features,
        )
        if features.descriptors.size == 0:
            self._clear_red_guard()
            preview.message = "特征不足，请对准丝印/焊盘/定位孔"
            preview.reason = "no_features"
            self._last_preview = preview
            return preview
        # 先试上一张；失败时再试更早的已拍关键帧，保证回到旧区域时
        # 红色并集仍能恢复，而不是只认最后一张。
        candidates = [last]
        for kf in reversed(self.keyframes[:-1]):
            if len(candidates) >= 5:
                break
            candidates.append(kf)
        best: tuple[CapturedKeyframe, np.ndarray, int, int] | None = None
        best_good_count = 0
        for candidate in candidates:
            knn = self._matcher.knnMatch(features.descriptors, candidate.descriptors, k=2)
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.config.ratio * n.distance:
                    good.append(m)
            best_good_count = max(best_good_count, len(good))
            if len(good) < self.config.min_matches:
                continue
            src = np.float32([features.keypoints[m.queryIdx] for m in good]).reshape(-1, 2)
            dst = np.float32([candidate.keypoints[m.trainIdx] for m in good]).reshape(-1, 2)
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
                continue
            inliers = int(mask.sum())
            if inliers < self.config.min_inliers:
                continue
            h_work = np.eye(3, dtype=np.float64)
            h_work[:2, :] = matrix_work
            live_scale = features.scale
            matrix_full = h_work @ np.diag([live_scale, live_scale, 1.0])
            matrix_full = matrix_full[:2, :].astype(np.float32)
            if not is_valid_similarity(matrix_full):
                continue
            best = (candidate, matrix_full, inliers, len(good))
            break
        if best is None:
            self._clear_red_guard()
            preview.matches = best_good_count
            preview.message = (
                f"与已拍区域匹配不足（匹配 {best_good_count}），请回退一点"
            )
            preview.reason = "no_matches"
            self._last_preview = preview
            return preview
        matched_keyframe, matrix_full, inliers, good_count = best
        preview.matches = good_count
        preview.inliers = inliers
        pose_full = as_homography(matched_keyframe.pose) @ as_homography(matrix_full)
        preview.pose = pose_full
        coverage_polys = self._project_verified_coverage(
            features,
            pose_full,
            matched_keyframe.index,
            frame.shape[1],
            frame.shape[0],
        )
        preview.coverage_polygons = coverage_polys
        preview.verified_keyframes = len(coverage_polys)
        red_mask = self._rasterize_coverage(coverage_polys, frame.shape[1], frame.shape[0])
        preview.red_metrics = compute_red_metrics(red_mask)
        preview.covered_fraction = preview.red_metrics.area_fraction
        preview.new_fraction = max(0.0, 1.0 - preview.covered_fraction)
        self._evaluate_red_guard(preview, red_mask, pose_full)
        if preview.covered_fraction < self.config.min_overlap:
            preview.can_capture = False
            preview.message = (
                f"红色覆盖仅 {preview.covered_fraction:.0%}，重叠不足；"
                "请回到上一张覆盖区域内"
            )
            preview.reason = "overlap_low"
        elif preview.covered_fraction > self.config.max_overlap:
            preview.can_capture = False
            preview.message = (
                f"红色覆盖 {preview.covered_fraction:.0%}，移动太少；"
                "请移动到一个新的板区域"
            )
            preview.reason = "overlap_high"
        elif not stable:
            preview.can_capture = False
            preview.message = "位置已对好，但画面未稳定；请停稳后再拍"
            preview.reason = "unstable"
        else:
            preview.can_capture = True
            preview.message = (
                f"可拍摄 · 覆盖 {preview.covered_fraction:.0%} · "
                f"内点 {inliers} · 新区域 {preview.new_fraction:.0%}"
            )
            preview.reason = "ready"
        if preview.red_anomaly:
            preview.can_capture = False
            preview.message = (
                f"红区面积/外形突变（{preview.red_warning}），疑似误识别；"
                "请回退到上一张附近，稳定两帧再拍"
            )
            preview.reason = "red_anomaly"
        self._last_preview = preview
        return preview

    def _project_coverage(self, pose_current: np.ndarray, width: int, height: int) -> list[np.ndarray]:
        inv = invert_affine(pose_current)
        result: list[np.ndarray] = []
        for kf in self.keyframes:
            poly = np.asarray(kf.coverage, dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.transform(poly, inv).reshape(-1, 2)
            if mapped[:, 0].max() < -20 or mapped[:, 1].max() < -20:
                continue
            if mapped[:, 0].min() > width + 20 or mapped[:, 1].min() > height + 20:
                continue
            result.append(mapped)
        return result

    def _project_verified_coverage(
        self,
        features: FeatureSet,
        pose_current: np.ndarray,
        source_index: int,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """只为“当前帧能直接匹配上”的旧关键帧画红区。

        旧逻辑把全部关键帧 coverage 按当前位姿投影；只要某个旧位姿漂了，
        整块矩形就会被误画成已拍。这里要求非来源关键帧必须与当前帧有
        足够的 ORB 重叠，且匹配变换与历史位姿一致，否则不参与红区。
        """
        inv = invert_affine(pose_current)
        result: list[np.ndarray] = []
        extra = 0
        for kf in reversed(self.keyframes):
            poly = np.asarray(kf.coverage, dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.transform(poly, inv).reshape(-1, 2)
            if mapped[:, 0].max() < -20 or mapped[:, 1].max() < -20:
                continue
            if mapped[:, 0].min() > width + 20 or mapped[:, 1].min() > height + 20:
                continue
            if kf.index != source_index:
                if extra >= 3:
                    continue
                extra += 1
                if not self._coverage_match_ok(features, kf, pose_current):
                    continue
            result.append(mapped)
        return result

    def _coverage_match_ok(
        self,
        features: FeatureSet,
        keyframe: CapturedKeyframe,
        pose_current: np.ndarray,
    ) -> bool:
        if features.descriptors.size == 0 or keyframe.descriptors.size == 0:
            return False
        try:
            knn = self._matcher.knnMatch(features.descriptors, keyframe.descriptors, k=2)
        except cv2.error:
            return False
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio * n.distance:
                good.append(m)
        if len(good) < self.config.min_matches:
            return False
        src = np.float32([features.keypoints[m.queryIdx] for m in good]).reshape(-1, 2)
        dst = np.float32([keyframe.keypoints[m.trainIdx] for m in good]).reshape(-1, 2)
        matrix, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_px,
            maxIters=2000,
            confidence=0.995,
        )
        if matrix is None or mask is None:
            return False
        if int(np.count_nonzero(mask)) < self.config.min_inliers:
            return False
        # 当前帧 -> 关键帧的期望变换必须与 ORB 实测变换一致，
        # 防止“匹配到了重复纹理，但历史位姿已经漂移”的旧 coverage。
        expected_full = np.linalg.inv(as_homography(keyframe.pose)) @ as_homography(pose_current)
        s = max(features.scale, 1e-6)
        expected_work = expected_full @ np.diag([1.0 / s, 1.0 / s, 1.0])
        expected = expected_work[:2, :]
        dt = float(np.hypot(matrix[0, 2] - expected[0, 2], matrix[1, 2] - expected[1, 2]))
        angle_measured = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        angle_expected = math.atan2(float(expected[1, 0]), float(expected[0, 0]))
        da = abs(math.degrees(angle_measured - angle_expected))
        da = min(da, 360.0 - da)
        scale_measured = math.sqrt(max(1e-9, abs(float(np.linalg.det(matrix[:, :2])))))
        scale_expected = math.sqrt(max(1e-9, abs(float(np.linalg.det(expected[:, :2])))))
        ds = abs(scale_measured - scale_expected) / max(scale_expected, 1e-6)
        return dt <= 22.0 and da <= 4.0 and ds <= 0.08

    @staticmethod
    def _rasterize_coverage(polygons: list[np.ndarray], width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = np.round(poly).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def _coverage_fraction(self, polygons: list[np.ndarray], width: int, height: int) -> float:
        if not polygons:
            return 0.0
        mask = self._rasterize_coverage(polygons, width, height)
        return float(np.count_nonzero(mask)) / float(max(1, width * height))

    def _evaluate_red_guard(
        self,
        preview: CapturePreview,
        red_mask: np.ndarray,
        pose_current: np.ndarray,
    ) -> None:
        """面积 + 外形记忆的快速 guard。

        这是“体检”，不是唯一判据：面积合法变化（例如回到旧区域）可能被
        短期标记，但必须连续多帧异常才禁用拍摄，避免误杀正常移动。
        """
        metrics = preview.red_metrics
        # 优先与上一次“已接受拍摄”的红区记忆比较；这才是用户说的每拍一张记忆值。
        if (
            self._capture_red_metrics is not None
            and self._capture_red_mask is not None
            and self._capture_red_pose is not None
            and self._capture_red_metrics.area_px > 0
        ):
            prev = self._capture_red_metrics
            prev_mask = self._capture_red_mask
            prev_pose = self._capture_red_pose
        else:
            prev = self._prev_red_metrics
            prev_mask = self._prev_red_mask
            prev_pose = self._prev_red_pose
        if prev is not None and prev_mask is not None and prev_pose is not None and prev.area_px > 0:
            try:
                mapping = np.linalg.inv(as_homography(pose_current)) @ as_homography(prev_pose)
                warped = cv2.warpAffine(
                    prev_mask,
                    mapping[:2, :].astype(np.float32),
                    (red_mask.shape[1], red_mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                inter = int(np.count_nonzero((warped > 0) & (red_mask > 0)))
                union = int(np.count_nonzero((warped > 0) | (red_mask > 0)))
                preview.red_iou = float(inter) / float(max(1, union))
            except cv2.error:
                preview.red_iou = 0.0
            preview.red_area_growth = (
                float(metrics.area_px - prev.area_px) / float(max(1, prev.area_px))
            )
            preview.red_shape_distance = metrics.shape_distance(prev)
            anomaly = False
            reasons: list[str] = []
            if (
                preview.red_area_growth > self.config.red_area_growth_warn
                and preview.red_iou < self.config.red_iou_warn
            ):
                anomaly = True
                reasons.append(f"面积{preview.red_area_growth:+.0%}")
            if preview.red_area_growth > self.config.red_area_growth_warn * 1.8:
                anomaly = True
                reasons.append("面积突增")
            if (
                preview.red_shape_distance > self.config.red_shape_warn
                and preview.red_area_growth > 0.20
            ):
                anomaly = True
                reasons.append("外形突变")
            if (
                metrics.components > prev.components + 2
                and metrics.largest_share < 0.55
            ):
                anomaly = True
                reasons.append("红区碎裂")
            if anomaly:
                self._red_anomaly_run += 1
                preview.red_warning = "，".join(reasons)
            else:
                self._red_anomaly_run = 0
                preview.red_warning = ""
            preview.red_anomaly = self._red_anomaly_run >= self.config.red_anomaly_frames
        else:
            self._red_anomaly_run = 0
        self._prev_red_mask = red_mask.copy()
        self._prev_red_pose = np.asarray(pose_current, dtype=np.float64).reshape(3, 3).copy()
        self._prev_red_metrics = metrics

    # ------------------------------------------------------------------ 增量总览
    def _world_bounds(self) -> tuple[float, float, float, float] | None:
        if not self.keyframes:
            return None
        polys = [np.asarray(kf.coverage, dtype=np.float64).reshape(-1, 2) for kf in self.keyframes]
        pts = np.vstack(polys)
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    def _fit_overview_scale(self, span_x: float, span_y: float) -> float:
        span_x = max(span_x, 1.0)
        span_y = max(span_y, 1.0)
        scale = float(self.config.overview_scale)
        scale = min(scale, self.config.overview_max_side / span_x, self.config.overview_max_side / span_y)
        scale = min(scale, math.sqrt(self.config.overview_max_pixels / (span_x * span_y)))
        return max(0.08, min(1.0, scale))

    def _bounds_with_margin(self, scale: float) -> tuple[float, float, float, float] | None:
        bounds = self._world_bounds()
        if bounds is None:
            return None
        margin = self.config.overview_margin_px / max(scale, 1e-6)
        return bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin

    def _canvas_size(self, bounds: tuple[float, float, float, float], scale: float) -> tuple[int, int]:
        return (
            max(1, int(math.ceil((bounds[2] - bounds[0]) * scale))),
            max(1, int(math.ceil((bounds[3] - bounds[1]) * scale))),
        )

    def _world_to_overview(self) -> np.ndarray:
        s = self._canvas_scale
        ox, oy = float(self._canvas_origin[0]), float(self._canvas_origin[1])
        return np.array([[s, 0, -ox * s], [0, s, -oy * s], [0, 0, 1]], dtype=np.float64)

    def _allocate_canvas(
        self,
        origin: tuple[float, float],
        scale: float,
        size: tuple[int, int],
    ) -> None:
        width, height = size
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._coverage_mask = np.zeros((height, width), dtype=np.uint8)
        self._canvas_origin = np.array(origin, dtype=np.float64)
        self._canvas_scale = float(scale)

    def _expand_canvas(
        self,
        new_origin: tuple[float, float],
        new_size: tuple[int, int],
    ) -> None:
        assert self._canvas is not None and self._coverage_mask is not None
        old = self._canvas
        old_mask = self._coverage_mask
        old_origin = self._canvas_origin.copy()
        scale = self._canvas_scale
        new_w, new_h = new_size
        dx = int(round((old_origin[0] - new_origin[0]) * scale))
        dy = int(round((old_origin[1] - new_origin[1]) * scale))
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        mask = np.zeros((new_h, new_w), dtype=np.uint8)
        x0, y0 = max(0, dx), max(0, dy)
        sx0, sy0 = max(0, -dx), max(0, -dy)
        copy_w = min(old.shape[1] - sx0, new_w - x0)
        copy_h = min(old.shape[0] - sy0, new_h - y0)
        if copy_w > 0 and copy_h > 0:
            canvas[y0 : y0 + copy_h, x0 : x0 + copy_w] = old[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
            mask[y0 : y0 + copy_h, x0 : x0 + copy_w] = old_mask[sy0 : sy0 + copy_h, sx0 : sx0 + copy_w]
        self._canvas = canvas
        self._coverage_mask = mask
        self._canvas_origin = np.array(new_origin, dtype=np.float64)

    def _blend_keyframe(self, keyframe: CapturedKeyframe, image: np.ndarray) -> None:
        assert self._canvas is not None and self._coverage_mask is not None
        transform = self._world_to_overview() @ as_homography(keyframe.pose)
        h, w = image.shape[:2]
        warped = cv2.warpAffine(
            image,
            transform[:2, :],
            (self._canvas.shape[1], self._canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            np.full((h, w), 255, dtype=np.uint8),
            transform[:2, :],
            (self._canvas.shape[1], self._canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not np.any(mask):
            return
        existing = self._coverage_mask > 0
        new_area = (mask > 0) & ~existing
        overlap = (mask > 0) & existing
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip(dist / float(max(1, self.config.feather_px)), 0.0, 1.0)
        alpha[new_area] = 1.0
        if np.any(overlap):
            alpha[overlap] = np.minimum(alpha[overlap], 0.55)
        alpha = alpha[..., None]
        blended = self._canvas.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        self._canvas = np.clip(blended, 0, 255).astype(np.uint8)
        # 覆盖并集使用几何多边形，而不是图像 alpha，保证红区语义稳定。
        poly = np.round(
            cv2.transform(
                np.asarray(keyframe.coverage, dtype=np.float32).reshape(-1, 1, 2),
                self._world_to_overview()[:2, :],
            ).reshape(-1, 2)
        ).astype(np.int32)
        if len(poly) >= 3:
            cv2.fillPoly(self._coverage_mask, [poly], 255)
        self._overview_version += 1

    def _refine_keyframe_to_canvas(self, keyframe: CapturedKeyframe, image: np.ndarray) -> None:
        """用重叠区的板边/走线边缘做 ECC 局部校正，减小接缝错位。

        ECC 只作用在已有总览和新关键帧的重叠区；背景（黑边）不进入掩膜。
        """
        if not self.config.seam_refine or self._canvas is None or self._coverage_mask is None:
            return
        S = self._world_to_overview()
        B = S @ as_homography(keyframe.pose)
        warped = cv2.warpAffine(
            image,
            B[:2, :],
            (self._canvas.shape[1], self._canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        h, w = image.shape[:2]
        mask = cv2.warpAffine(
            np.full((h, w), 255, dtype=np.uint8),
            B[:2, :],
            (self._canvas.shape[1], self._canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap = ((mask > 0) & (self._coverage_mask > 0)).astype(np.uint8)
        if int(np.count_nonzero(overlap)) < self.config.seam_min_overlap_px:
            return
        x, y, rw, rh = cv2.boundingRect(overlap)
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(self._canvas.shape[1], x + rw + pad)
        y1 = min(self._canvas.shape[0], y + rh + pad)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return
        template = self._canvas[y0:y1, x0:x1]
        moving = warped[y0:y1, x0:x1]
        mask_roi = overlap[y0:y1, x0:x1]
        tg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        mg = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
        te = cv2.Canny(tg, 35, 110)
        me = cv2.Canny(mg, 35, 110)
        # 轻微膨胀，让 1px 的走线/丝印边缘也能被 ECC 捕捉。
        te = cv2.dilate(te, np.ones((3, 3), np.uint8), iterations=1)
        me = cv2.dilate(me, np.ones((3, 3), np.uint8), iterations=1)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
        try:
            cc, warp = cv2.findTransformECC(
                te.astype(np.float32),
                me.astype(np.float32),
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                mask_roi,
                5,
            )
        except cv2.error:
            return
        if not np.isfinite(cc) or cc < self.config.seam_min_cc:
            return
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        shift = math.hypot(dx, dy)
        angle = abs(math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0]))))
        if shift > self.config.seam_max_shift_px or angle > self.config.seam_max_rotation_deg:
            return
        if shift < 0.05 and angle < 0.02:
            keyframe.seam_cc = float(cc)
            return
        o = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
        o_inv = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        correction = o @ as_homography(warp) @ o_inv
        S_inv = np.linalg.inv(S)
        pose_new = S_inv @ correction @ S @ as_homography(keyframe.pose)
        if not is_valid_similarity(pose_new):
            return
        keyframe.pose = pose_new
        kh, kw = keyframe.shape
        corners = np.float32([[0, 0], [kw, 0], [kw, kh], [0, kh]]).reshape(-1, 1, 2)
        keyframe.coverage = cv2.transform(corners, pose_new[:2, :]).reshape(-1, 2)
        keyframe.seam_applied = True
        keyframe.seam_cc = float(cc)
        keyframe.seam_shift_px = shift
        keyframe.seam_rotation_deg = angle

    def _rebuild_overview(self, scale: float) -> None:
        bounds = self._bounds_with_margin(scale)
        if bounds is None:
            raise SparseCaptureError("没有可用的关键帧覆盖范围")
        size = self._canvas_size(bounds, scale)
        self._allocate_canvas((bounds[0], bounds[1]), scale, size)
        for keyframe in self.keyframes:
            image = cv2.imdecode(np.frombuffer(keyframe.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise SparseCaptureError("增量总览解码关键帧失败")
            self._blend_keyframe(keyframe, image)

    def _merge_keyframe(self, keyframe: CapturedKeyframe, image: np.ndarray) -> None:
        """把新关键帧立即并入已有低分辨率总览，并扩展覆盖并集。"""
        if self._canvas is None:
            bounds0 = self._world_bounds()
            if bounds0 is None:
                return
            span_x = bounds0[2] - bounds0[0]
            span_y = bounds0[3] - bounds0[1]
            scale = self._fit_overview_scale(span_x, span_y)
            bounds = self._bounds_with_margin(scale)
            if bounds is None:
                return
            size = self._canvas_size(bounds, scale)
            self._allocate_canvas((bounds[0], bounds[1]), scale, size)
            self._refine_keyframe_to_canvas(keyframe, image)
            self._blend_keyframe(keyframe, image)
            return

        old_origin = self._canvas_origin.copy()
        old_world_max_x = old_origin[0] + self._canvas.shape[1] / self._canvas_scale
        old_world_max_y = old_origin[1] + self._canvas.shape[0] / self._canvas_scale
        bounds = self._world_bounds()
        if bounds is None:
            return
        span_x = max(old_world_max_x, bounds[2]) - min(old_origin[0], bounds[0])
        span_y = max(old_world_max_y, bounds[3]) - min(old_origin[1], bounds[1])
        new_scale = self._fit_overview_scale(span_x, span_y)
        if new_scale < self._canvas_scale * 0.999:
            # 超出自适应上限时整体降采样重建；这是少数情况，保证总览仍可装入内存。
            self._rebuild_overview(new_scale)
            return
        margin = self.config.overview_margin_px / max(self._canvas_scale, 1e-6)
        new_origin = (min(old_origin[0], bounds[0] - margin), min(old_origin[1], bounds[1] - margin))
        new_max = (max(old_world_max_x, bounds[2] + margin), max(old_world_max_y, bounds[3] + margin))
        new_size = self._canvas_size((new_origin[0], new_origin[1], new_max[0], new_max[1]), self._canvas_scale)
        if (
            new_size[0] != self._canvas.shape[1]
            or new_size[1] != self._canvas.shape[0]
            or new_origin[0] < old_origin[0] - 1e-6
            or new_origin[1] < old_origin[1] - 1e-6
        ):
            self._expand_canvas(new_origin, new_size)
        self._refine_keyframe_to_canvas(keyframe, image)
        self._blend_keyframe(keyframe, image)

    def _estimate_keyframe_pose(
        self,
        features: FeatureSet,
        preview: CapturePreview,
        frame_width: int,
        frame_height: int,
    ) -> tuple[np.ndarray | None, int, int, int]:
        """用所有可能重叠的旧关键帧估计新关键帧位姿，降低逐张累加漂移。

        返回 (pose_new->scan_world, matches, inliers, source_keyframe_index)。
        没有足够可靠候选时返回 (None, ...)，由调用方回退到上一关键帧结果。
        """
        if preview.pose is None or features.descriptors.size == 0:
            return None, 0, 0, -1
        corners = np.float32(
            [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]]
        ).reshape(-1, 1, 2)
        view_world = cv2.transform(corners, as_homography(preview.pose)[:2, :]).reshape(-1, 2)
        candidates = [self.keyframes[-1]]
        for kf in reversed(self.keyframes[:-1]):
            if len(candidates) >= 5:
                break
            if _polygons_intersect(view_world, kf.coverage):
                candidates.append(kf)
        best_pose: np.ndarray | None = None
        best_matches = 0
        best_inliers = 0
        best_index = -1
        for kf in candidates:
            if kf.descriptors.size == 0:
                continue
            try:
                knn = self._matcher.knnMatch(features.descriptors, kf.descriptors, k=2)
            except cv2.error:
                continue
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.config.ratio * n.distance:
                    good.append(m)
            if len(good) < self.config.min_matches:
                continue
            src = np.float32([features.keypoints[m.queryIdx] for m in good]).reshape(-1, 2)
            dst = np.float32([kf.keypoints[m.trainIdx] for m in good]).reshape(-1, 2)
            matrix, mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_px,
                maxIters=2000,
                confidence=0.995,
            )
            if matrix is None or mask is None:
                continue
            inliers = int(np.count_nonzero(mask))
            if inliers < self.config.min_inliers:
                continue
            pose = as_homography(kf.pose) @ as_homography(matrix)
            if not is_valid_similarity(pose):
                continue
            if (inliers, -kf.index) > (best_inliers, -best_index if best_index >= 0 else -10**9):
                best_pose = pose
                best_matches = len(good)
                best_inliers = inliers
                best_index = kf.index
        return best_pose, best_matches, best_inliers, best_index

    # ------------------------------------------------------------------ 拍摄
    def capture(self) -> CaptureAcceptance:
        if self._last_frame is None or self._last_preview is None:
            return CaptureAcceptance(False, "还没有稳定预览，请稍候")
        if self._preview_frame is not self._last_frame:
            return CaptureAcceptance(False, "画面已更新，请重新等待稳定后拍摄")
        preview = self._last_preview
        if not preview.stable:
            return CaptureAcceptance(False, "画面未稳定，请停稳后再拍")
        if self.keyframes and not preview.can_capture:
            return CaptureAcceptance(False, preview.message)
        if not self.keyframes and not preview.can_capture:
            return CaptureAcceptance(False, "画面未稳定，请停稳后再拍")
        frame = self._last_frame
        try:
            features = extract_features(
                frame,
                max_width=self.config.keyframe_width,
                nfeatures=self.config.features,
            )
        except Exception as exc:  # noqa: BLE001
            return CaptureAcceptance(False, f"关键帧特征提取失败：{exc}")
        if features.descriptors.size < self.config.min_inliers:
            return CaptureAcceptance(False, "关键帧特征不足，请对准纹理更丰富的区域")
        pose_source = -1
        pose_matches = 0
        pose_inliers = 0
        if self.keyframes:
            if preview.pose is None:
                return CaptureAcceptance(False, "没有可用的位置关系，请回退后重试")
            # 优先与“覆盖当前视场的较老关键帧”匹配，避免只能相对上一张累加。
            pose_kf, pose_matches, pose_inliers, pose_source = self._estimate_keyframe_pose(
                features, preview, frame.shape[1], frame.shape[0]
            )
            if pose_kf is None:
                # preview.pose: full live -> scan world
                # stored keyframe image = full live * features.scale
                scale = max(features.scale, 1e-6)
                pose_kf = as_homography(preview.pose) @ np.diag([1.0 / scale, 1.0 / scale, 1.0])
                pose_matches = preview.matches
                pose_inliers = preview.inliers
        else:
            pose_kf = np.eye(3, dtype=np.float64)
        h, w = features.image.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        coverage = cv2.transform(corners, as_homography(pose_kf)[:2, :]).reshape(-1, 2)
        ok, buffer = cv2.imencode(
            ".jpg",
            features.image,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)],
        )
        if not ok:
            return CaptureAcceptance(False, "关键帧编码失败")
        record = CapturedKeyframe(
            index=len(self.keyframes),
            shape=(h, w),
            pose=pose_kf.astype(np.float64),
            coverage=coverage,
            keypoints=features.keypoints,
            descriptors=features.descriptors,
            global_descriptor=features.global_descriptor,
            jpeg=buffer.tobytes(),
            timestamp=time.time(),
            red_metrics=preview.red_metrics,
            matches=pose_matches,
            inliers=pose_inliers,
            pose_source=pose_source,
        )
        self.keyframes.append(record)
        try:
            self._merge_keyframe(record, features.image)
        except Exception as exc:  # noqa: BLE001
            self.keyframes.pop()
            return CaptureAcceptance(False, f"并入总览失败，已取消本次拍摄：{exc}")
        self._last_preview = None
        self._preview_frame = None
        self._stable_run = 0
        self._red_memory.append(preview.red_metrics)
        if preview.pose is not None and preview.red_metrics.area_px > 0:
            mask = self._rasterize_coverage(
                preview.coverage_polygons,
                self._last_frame.shape[1] if self._last_frame is not None else 0,
                self._last_frame.shape[0] if self._last_frame is not None else 0,
            )
            self._capture_red_mask = mask
            self._capture_red_pose = np.asarray(preview.pose, dtype=np.float64).reshape(3, 3).copy()
            self._capture_red_metrics = preview.red_metrics
        self._red_anomaly_run = 0
        suffix = ""
        if record.seam_applied:
            suffix = f"；板边对齐 {record.seam_shift_px:.1f}px"
        message = (
            "已拍摄第一张并立即并入总览，请移动到相邻区域并保留一部分红色覆盖"
            if len(self.keyframes) == 1
            else f"已拍摄第 {len(self.keyframes)} 张并立即并入总览{suffix}"
        )
        return CaptureAcceptance(True, message, record.index)

    # ------------------------------------------------------------------ 完成
    def finish(self, progress: ProgressCb | None = None) -> SparseCaptureResult:
        if len(self.keyframes) < self.config.min_keyframes:
            raise SparseCaptureError(
                f"至少需要 {self.config.min_keyframes} 张关键帧，当前 {len(self.keyframes)} 张"
            )
        if self._canvas is None:
            raise SparseCaptureError("增量总览尚未生成")
        started = time.perf_counter()
        if progress:
            progress(0, 4, "使用增量总览与已保存位姿…")
        overview = self._canvas.copy()
        world_to_board = self._world_to_overview()
        rectified = False
        rectify_message = ""
        if self.config.rectify_board:
            if progress:
                progress(1, 4, "检测板边并拉正外轮廓…")
            rr = rectify_board(
                overview,
                max_side=self.config.rectify_max_side,
                max_pixels=self.config.rectify_max_pixels,
            )
            rectify_message = rr.message
            if rr.applied and rr.homography is not None:
                overview = rr.image
                world_to_board = as_homography(rr.homography) @ world_to_board
                rectified = True
        sparse_keyframes: list[SparseKeyframe] = []
        coverage_polys: list[np.ndarray] = []
        for idx, kf in enumerate(self.keyframes):
            h, w = kf.shape
            matrix = world_to_board @ as_homography(kf.pose)
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            coverage = cv2.transform(corners, matrix[:2, :]).reshape(-1, 2)
            coverage_polys.append(coverage)
            sparse_keyframes.append(
                SparseKeyframe(
                    index=idx,
                    shape=(h, w),
                    pose=np.asarray(matrix, dtype=np.float64).reshape(3, 3),
                    coverage=coverage,
                    keypoints=kf.keypoints,
                    descriptors=kf.descriptors,
                    global_descriptor=kf.global_descriptor,
                    neighbors=[n for n in (idx - 1, idx + 1) if 0 <= n < len(self.keyframes)],
                    jpeg=kf.jpeg,
                    red_area_ratio=float(kf.red_metrics.area_fraction) if kf.red_metrics else 0.0,
                    red_components=int(kf.red_metrics.components) if kf.red_metrics else 0,
                    red_hu=[float(v) for v in kf.red_metrics.hu] if kf.red_metrics else [],
                )
            )
        # 根据覆盖多边形相交关系补全邻居图
        for i, poly_i in enumerate(coverage_polys):
            for j in range(i + 1, len(coverage_polys)):
                if _polygons_intersect(poly_i, coverage_polys[j]):
                    if j not in sparse_keyframes[i].neighbors:
                        sparse_keyframes[i].neighbors.append(j)
                    if i not in sparse_keyframes[j].neighbors:
                        sparse_keyframes[j].neighbors.append(i)
        sparse = SparseMap(
            keyframes=sparse_keyframes,
            board_width=int(overview.shape[1]),
            board_height=int(overview.shape[0]),
        )
        if progress:
            progress(2, 4, "提取结构特征树…")
        structural: StructuralMap | None = None
        try:
            structural = extract_structural_map(overview)
        except Exception:  # noqa: BLE001
            # 结构树是增强数据，失败不应让已经完成的建图前功尽弃。
            structural = None
        structural_match: StructuralMatch | None = None
        if structural is not None and self.keyframes:
            # B2：用一张信息量最高的关键帧做结构图自校验，验证位姿与结构树一致。
            if progress:
                progress(3, 4, "结构图匹配校验…")
            try:
                representative = max(self.keyframes, key=lambda k: k.inliers)
                data = np.frombuffer(representative.jpeg, dtype=np.uint8)
                keyframe_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if keyframe_image is not None:
                    query = extract_structural_map(keyframe_image)
                    to_board = world_to_board @ as_homography(representative.pose)
                    query_board = transform_structural_map(query, to_board)
                    structural_match = match_structural_maps(structural, query_board)
            except Exception:  # noqa: BLE001
                structural_match = None
        if progress:
            progress(4, 4, "稀疏地图与结构特征构建完成")
        return SparseCaptureResult(
            overview=overview,
            sparse_map=sparse,
            coverage_polygons=coverage_polys,
            keyframe_count=len(sparse_keyframes),
            build_seconds=time.perf_counter() - started,
            rectified=rectified,
            rectify_message=rectify_message,
            structural_map=structural,
            structural_match=structural_match,
        )


def _polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    try:
        inter, _ = cv2.intersectConvexConvex(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
        )
        return float(inter) > 1.0
    except cv2.error:
        return False


def make_sparse_localizer_keyframes(result: SparseCaptureResult) -> SparseMap:
    return result.sparse_map
