"""稀疏关键帧定位与“稀疏 + 全图”混合跟踪。

默认用关键帧 ORB 数据库恢复板坐标；若项目同时有低分辨率总览，则周期性用
全图跟踪做校验，在稀疏结果失败或明显不一致时回退到全图结果。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pcb_ar.core.geo import (
    as_homography,
    blend_similarity,
    decompose_similarity,
    invert_affine,
    is_valid_similarity,
)
from pcb_ar.core.sparse_map import FeatureSet, SparseMap, global_descriptor
from pcb_ar.core.tracker import GlobalLocalTracker, TrackResult


@dataclass
class SparseLocalizerConfig:
    match_width: int = 1200
    live_features: int = 1800
    ratio: float = 0.76
    min_matches: int = 18
    min_inliers: int = 14
    ransac_px: float = 4.0
    top_k: int = 4
    smoothing: float = 0.65
    max_lost_frames: int = 8
    max_translation_jump: float = 80.0
    max_rotation_jump_deg: float = 8.0


class SparseLocalizer:
    def __init__(
        self,
        sparse_map: SparseMap,
        config: SparseLocalizerConfig | None = None,
    ) -> None:
        if sparse_map.count == 0:
            raise ValueError("稀疏地图没有关键帧")
        self.map = sparse_map
        self.config = config or SparseLocalizerConfig()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self._orb = cv2.ORB_create(
            nfeatures=self.config.live_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
        self._global = sparse_map.global_descriptors()
        self.last_matrix: np.ndarray | None = None
        self.last_result: TrackResult | None = None
        self.lost_frames = 0
        self.last_keyframe: int = -1

    def reset(self) -> None:
        self.last_matrix = None
        self.last_result = None
        self.lost_frames = 0
        self.last_keyframe = -1

    def track(self, frame: np.ndarray) -> TrackResult:
        started = time.perf_counter()
        if frame is None or frame.size == 0:
            return self._failure("空帧", started)
        features = self._extract(frame)
        if features.descriptors.shape[0] < self.config.min_matches:
            return self._failure("实时画面特征不足", started)
        candidates = self._candidate_indices(features)
        best: tuple[float, int, np.ndarray, int, int] | None = None
        for keyframe_index in candidates:
            keyframe = self.map.sorted_keyframes()[keyframe_index]
            if keyframe.descriptors.shape[0] < self.config.min_matches:
                continue
            matches = self._match(features.descriptors, keyframe.descriptors)
            if len(matches) < self.config.min_matches:
                continue
            src = np.float32([features.keypoints[m.queryIdx] for m in matches]).reshape(-1, 2)
            dst = np.float32([keyframe.keypoints[m.trainIdx] for m in matches]).reshape(-1, 2)
            matrix_work, mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_px,
                maxIters=3500,
                confidence=0.997,
                refineIters=10,
            )
            if matrix_work is None or mask is None:
                continue
            inliers = int(mask.sum())
            if inliers < self.config.min_inliers:
                continue
            score = inliers + (60 if keyframe_index == self.last_keyframe else 0)
            if best is None or score > best[0]:
                best = (float(score), keyframe_index, matrix_work, inliers, len(matches))
            # 连续帧通常就在上一关键帧附近；内点充足时提前结束，避免每帧扫描多张关键帧。
            if keyframe_index == self.last_keyframe and inliers >= 60:
                break
            if inliers >= 100:
                break
        if best is None:
            return self._failure("未匹配到稀疏关键帧", started)
        _score, keyframe_index, matrix_work, inliers, match_count = best
        keyframe = self.map.sorted_keyframes()[keyframe_index]
        h_work = np.eye(3, dtype=np.float64)
        h_work[:2, :] = matrix_work
        matrix_full = h_work @ np.diag(
            [features.scale, features.scale, 1.0]
        )  # live full -> keyframe image
        matrix_full = matrix_full[:2, :].astype(np.float32)
        if not is_valid_similarity(matrix_full):
            return self._failure("稀疏变换异常", started, inliers, match_count)
        live_to_board = as_homography(keyframe.pose) @ as_homography(matrix_full)
        if not self._overlaps_board(live_to_board, frame.shape[1], frame.shape[0]):
            return self._failure("匹配位置超出板图范围", started, inliers, match_count)
        board_to_live = invert_affine(live_to_board).astype(np.float32)
        alpha = self.config.smoothing
        if self.last_matrix is not None and self._large_change(self.last_matrix, board_to_live, frame.shape[1]):
            alpha = 1.0
        smoothed = blend_similarity(
            self.last_matrix,
            board_to_live,
            alpha,
            pivot=(self.map.board_width / 2.0, self.map.board_height / 2.0),
        )
        self.last_matrix = smoothed
        self.last_keyframe = keyframe_index
        self.lost_frames = 0
        confidence = float(
            np.clip(0.55 * (inliers / max(match_count, 1)) + 0.45 * min(1.0, inliers / 45.0), 0.0, 1.0)
        )
        scale, theta, _ = decompose_similarity(board_to_live)
        result = TrackResult(
            matrix=smoothed.copy(),
            state="locked",
            confidence=confidence,
            inliers=inliers,
            matches=match_count,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            message="稀疏关键帧定位",
            scale=scale,
            rotation_deg=math.degrees(theta),
            diagnostics={"source": "sparse", "keyframe_index": keyframe_index},
        )
        self.last_result = result
        return result

    def _extract(self, frame: np.ndarray) -> FeatureSet:
        h, w = frame.shape[:2]
        scale = min(1.0, self.config.match_width / float(w))
        if scale < 1.0:
            work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = frame.copy()
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        kps, des = self._orb.detectAndCompute(gray, None)
        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = np.float32([[kp.pt[0], kp.pt[1]] for kp in kps]).reshape(-1, 2)
        return FeatureSet(
            image=work,
            gray=gray,
            scale=scale,
            keypoints=points,
            descriptors=des,
            global_descriptor=global_descriptor(work),
        )

    def _candidate_indices(self, features: FeatureSet) -> list[int]:
        keyframes = self.map.sorted_keyframes()
        count = len(keyframes)
        if count == 0:
            return []
        candidates: list[int] = []
        if 0 <= self.last_keyframe < count:
            candidates.append(self.last_keyframe)
        if self._global.shape[0] == count:
            distance = np.count_nonzero(self._global != features.global_descriptor.reshape(1, -1), axis=1)
            ranked = np.argsort(distance).tolist()
            for index in ranked[: self.config.top_k]:
                if index not in candidates:
                    candidates.append(int(index))
        if not candidates:
            candidates = list(range(min(count, self.config.top_k)))
        return candidates

    def _match(self, query: np.ndarray, train: np.ndarray) -> list[cv2.DMatch]:
        knn = self._matcher.knnMatch(query, train, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio * n.distance:
                good.append(m)
        return good

    def _overlaps_board(self, live_to_board: np.ndarray, width: int, height: int) -> bool:
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 2)
        warped = cv2.transform(
            corners.reshape(-1, 1, 2),
            as_homography(live_to_board)[:2, :],
        ).reshape(-1, 2)
        x1, y1 = warped[:, 0].min(), warped[:, 1].min()
        x2, y2 = warped[:, 0].max(), warped[:, 1].max()
        return x2 > -50 and y2 > -50 and x1 < self.map.board_width + 50 and y1 < self.map.board_height + 50

    def _large_change(self, previous: np.ndarray, current: np.ndarray, live_w: int) -> bool:
        center = np.float32([[self.map.board_width / 2.0, self.map.board_height / 2.0]])
        prev_center = cv2.transform(center.reshape(-1, 1, 2), previous[:2, :]).reshape(2)
        curr_center = cv2.transform(center.reshape(-1, 1, 2), current[:2, :]).reshape(2)
        if float(np.linalg.norm(prev_center - curr_center)) > max(70.0, live_w * 0.06):
            return True
        prev_scale, prev_theta, _ = decompose_similarity(previous)
        curr_scale, curr_theta, _ = decompose_similarity(current)
        delta = abs((curr_theta - prev_theta + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(delta) > 5.0:
            return True
        return prev_scale > 0 and abs(curr_scale / prev_scale - 1.0) > 0.10

    def _failure(
        self,
        message: str,
        started: float,
        inliers: int = 0,
        matches: int = 0,
    ) -> TrackResult:
        self.lost_frames += 1
        state = "stale" if self.last_matrix is not None and self.lost_frames <= self.config.max_lost_frames else "lost"
        self.last_result = TrackResult(
            matrix=self.last_matrix.copy() if state == "stale" and self.last_matrix is not None else None,
            state=state,
            confidence=0.0,
            inliers=inliers,
            matches=matches,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            message=message,
            diagnostics={"source": "sparse"},
        )
        return self.last_result


class HybridTracker:
    """稀疏定位为默认，低分辨率全图跟踪做周期校验/失败回退。"""

    def __init__(
        self,
        sparse: SparseLocalizer | None,
        full: GlobalLocalTracker | None,
        *,
        full_interval: int = 8,
        consensus_px: float = 24.0,
    ) -> None:
        if sparse is None and full is None:
            raise ValueError("至少需要稀疏或全图跟踪器之一")
        self.sparse = sparse
        self.full = full
        self.full_interval = max(1, int(full_interval))
        self.consensus_px = float(consensus_px)
        self.frame_index = 0
        self.last_result: TrackResult | None = None
        self.last_matrix: np.ndarray | None = None
        self.last_full: TrackResult | None = None
        self.last_choice = ""

    def reset(self) -> None:
        if self.sparse is not None:
            self.sparse.reset()
        if self.full is not None:
            self.full.reset()
        self.frame_index = 0
        self.last_result = None
        self.last_matrix = None
        self.last_full = None
        self.last_choice = ""

    def track(self, frame: np.ndarray) -> TrackResult:
        self.frame_index += 1
        sparse_result = self.sparse.track(frame) if self.sparse is not None else None
        need_full = self.full is not None and (
            sparse_result is None
            or not sparse_result.is_tracking
            or self.frame_index % self.full_interval == 0
        )
        full_result = self.full.track(frame) if need_full and self.full is not None else None
        if full_result is not None and full_result.is_tracking:
            self.last_full = full_result
        chosen: TrackResult | None = sparse_result
        source = "sparse"
        if sparse_result is not None and sparse_result.is_tracking:
            if full_result is not None and full_result.is_tracking:
                difference = self._matrix_difference(sparse_result.matrix, full_result.matrix)
                if difference > self.consensus_px:
                    # 全图有全局上下文；当两者明显不一致时优先全图，防止重复纹理误锁。
                    if full_result.confidence >= sparse_result.confidence * 0.70:
                        chosen = full_result
                        source = "full"
                    else:
                        chosen = sparse_result
                        source = "sparse"
                else:
                    source = "sparse"
            else:
                source = "sparse"
        elif full_result is not None:
            # 全图也可能暂时 stale；保留其矩阵，避免直接变成完全丢失。
            chosen = full_result
            source = "full"
        elif self.last_full is not None and self.last_full.is_tracking and not need_full:
            # 通常不会走到；保留一次全图结果作为临时回退
            chosen = self.last_full
            source = "full-cache"
        if chosen is not None and not chosen.is_tracking and self.last_full is not None and self.last_full.is_tracking:
            # 稀疏短暂失败时，优先沿用最近一次有效的全图结果，而不是完全丢失。
            chosen = self.last_full
            source = "full-cache"
        if chosen is None:
            chosen = sparse_result or self._empty_failure(frame)
            source = "none"
        chosen.diagnostics = dict(chosen.diagnostics or {})
        chosen.diagnostics["source"] = source
        chosen.diagnostics["full_checked"] = bool(need_full)
        chosen.diagnostics["frame_index"] = self.frame_index
        if chosen.message and source == "full":
            chosen.message = "全图增强 · " + chosen.message
        self.last_result = chosen
        self.last_matrix = chosen.matrix
        self.last_choice = source
        return chosen

    def _matrix_difference(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 1e9
        if self.sparse is not None:
            w, h = self.sparse.map.board_width, self.sparse.map.board_height
            points = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        else:
            points = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]).reshape(-1, 1, 2)
        pa = cv2.transform(points, as_homography(a)[:2, :]).reshape(-1, 2)
        pb = cv2.transform(points, as_homography(b)[:2, :]).reshape(-1, 2)
        # 用四角均方根差判断；只看中心会漏掉旋转/尺度分歧。
        return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))

    def _empty_failure(self, frame: np.ndarray) -> TrackResult:
        return TrackResult(
            matrix=None,
            state="lost",
            confidence=0.0,
            message="没有可用定位结果",
            diagnostics={"source": "none"},
        )
