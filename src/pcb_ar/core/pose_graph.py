"""关键帧位姿图与回环闭合（离线精修）。

捕获阶段仍然逐张增量预览；完成时才对关键帧做一次位姿图优化：
* 相邻关键帧 ORB 匹配给相对位姿边；
* 全局描述子找回环候选，再做 ORB 匹配；
* 用相似变换（tx,ty,θ,s）做加权最小二乘优化；
* 有回环时才重渲染总览，避免正常短扫描变慢。

这是轻量实现，不是完整 SLAM；没有 IMU、没有相机标定、没有稠密 BA。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pcb_ar.core.geo import as_homography, is_valid_similarity


@dataclass
class PoseEdge:
    i: int
    j: int
    measured: np.ndarray  # 3x3, keyframe i pixels -> keyframe j pixels
    inliers: int = 0
    loop: bool = False

    @property
    def weight(self) -> float:
        return max(0.15, min(2.0, self.inliers / 40.0))


@dataclass
class PoseGraphResult:
    poses: list[np.ndarray]
    edges: list[PoseEdge]
    loop_edges: int = 0
    rms_before: float = 0.0
    rms_after: float = 0.0
    optimized: bool = False
    message: str = ""


def _pose_to_params(pose: np.ndarray) -> np.ndarray:
    m = as_homography(pose)
    s = math.sqrt(max(1e-9, abs(float(np.linalg.det(m[:2, :2])))))
    theta = math.atan2(float(m[1, 0]), float(m[0, 0]))
    return np.array([float(m[0, 2]), float(m[1, 2]), theta, math.log(max(s, 1e-9))], dtype=np.float64)


def _params_to_pose(p: np.ndarray) -> np.ndarray:
    tx, ty, theta, log_s = (float(v) for v in p)
    s = math.exp(log_s)
    c, sn = math.cos(theta), math.sin(theta)
    return np.array(
        [[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _residuals(params: np.ndarray, edges: list[PoseEdge], fixed_pose: np.ndarray) -> np.ndarray:
    poses = [fixed_pose] + [_params_to_pose(params[i]) for i in range(len(params))]
    out = np.zeros((len(edges), 4), dtype=np.float64)
    for k, edge in enumerate(edges):
        pi = poses[edge.i]
        pj = poses[edge.j]
        rel = np.linalg.inv(pj) @ pi
        z = as_homography(edge.measured)
        dt = rel[:2, 2] - z[:2, 2]
        da = _wrap_angle(
            math.atan2(float(rel[1, 0]), float(rel[0, 0]))
            - math.atan2(float(z[1, 0]), float(z[0, 0]))
        )
        sr = math.sqrt(max(1e-9, abs(float(np.linalg.det(rel[:2, :2])))))
        sz = math.sqrt(max(1e-9, abs(float(np.linalg.det(z[:2, :2])))))
        ds = math.log(max(sr, 1e-9) / max(sz, 1e-9))
        out[k] = [dt[0] / 100.0, dt[1] / 100.0, da, ds]
    return out


def _rms(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def optimize_poses(
    initial_poses: list[np.ndarray],
    edges: list[PoseEdge],
    *,
    iterations: int = 12,
) -> PoseGraphResult:
    n = len(initial_poses)
    if n < 3 or not edges:
        return PoseGraphResult(initial_poses, edges, 0, message="关键帧或位姿边不足")
    params = np.stack([_pose_to_params(p) for p in initial_poses[1:]], axis=0)
    fixed = as_homography(initial_poses[0]).copy()
    before = _rms(_residuals(params, edges, fixed))
    current = before
    weights = np.array([math.sqrt(e.weight) for e in edges], dtype=np.float64)
    for _ in range(iterations):
        base = _residuals(params, edges, fixed)
        # 数值雅可比：参数只有 4*(N-1)，关键帧规模下足够快。
        jac = np.zeros((base.size, params.size), dtype=np.float64)
        eps = 1e-4
        for k in range(params.size):
            pert = params.copy()
            pert.reshape(-1)[k] += eps
            r = _residuals(pert, edges, fixed)
            jac[:, k] = (r - base).reshape(-1) / eps
        w = np.repeat(weights, 4)
        jw = jac * w[:, None]
        rw = base.reshape(-1) * w
        normal = jw.T @ jw + 1e-3 * np.eye(jac.shape[1])
        try:
            delta = np.linalg.solve(normal, -(jw.T @ rw))
        except np.linalg.LinAlgError:
            break
        params += delta.reshape(params.shape)
        new_rms = _rms(_residuals(params, edges, fixed))
        if abs(current - new_rms) < 1e-7:
            current = new_rms
            break
        current = new_rms
    poses = [fixed] + [_params_to_pose(p) for p in params]
    loop_edges = sum(1 for e in edges if e.loop)
    return PoseGraphResult(
        poses=poses,
        edges=edges,
        loop_edges=loop_edges,
        rms_before=before,
        rms_after=current,
        optimized=True,
        message=f"位姿图 {len(edges)} 边 / {loop_edges} 回环 / RMS {before:.3f}→{current:.3f}",
    )


def _match_pair(
    kp_i: np.ndarray,
    des_i: np.ndarray,
    kp_j: np.ndarray,
    des_j: np.ndarray,
    *,
    ratio: float = 0.76,
    min_inliers: int = 22,
    ransac_px: float = 4.0,
) -> tuple[np.ndarray | None, int]:
    if des_i.size == 0 or des_j.size == 0:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(des_i, des_j, k=2)
    except cv2.error:
        return None, 0
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < ratio * b.distance:
            good.append(a)
    if len(good) < min_inliers:
        return None, len(good)
    src = np.float32([kp_i[m.queryIdx] for m in good]).reshape(-1, 2)
    dst = np.float32([kp_j[m.trainIdx] for m in good]).reshape(-1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None or mask is None:
        return None, len(good)
    inliers = int(np.count_nonzero(mask))
    if inliers < min_inliers:
        return None, inliers
    homography = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)
    if not is_valid_similarity(homography):
        return None, inliers
    scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(matrix[:, :2])))))
    if not (0.45 <= scale <= 2.2):
        return None, inliers
    return homography, inliers


def build_pose_edges(keyframes: list[Any], *, max_loops: int = 1) -> list[PoseEdge]:
    """给关键帧列表建立相邻边 + 回环候选边。对象需具备
    keypoints/descriptors/global_descriptor/pose/coverage 属性。
    """
    n = len(keyframes)
    edges: list[PoseEdge] = []
    if n < 3:
        return edges
    # 相邻边
    for i in range(n - 1):
        matrix, inliers = _match_pair(
            keyframes[i].keypoints,
            keyframes[i].descriptors,
            keyframes[i + 1].keypoints,
            keyframes[i + 1].descriptors,
        )
        if matrix is not None:
            edges.append(PoseEdge(i, i + 1, matrix, inliers, loop=False))
    # 回环候选：全局描述子最相近的非相邻帧，再做 ORB 验证。
    used: set[tuple[int, int]] = {(min(e.i, e.j), max(e.i, e.j)) for e in edges}
    for i, kf in enumerate(keyframes):
        candidates = []
        for j in range(i + 3, n):
            if (i, j) in used:
                continue
            dist = int(np.count_nonzero(kf.global_descriptor != keyframes[j].global_descriptor))
            candidates.append((dist, j))
        candidates.sort(key=lambda item: item[0])
        for _dist, j in candidates[: max_loops]:
            matrix, inliers = _match_pair(
                kf.keypoints,
                kf.descriptors,
                keyframes[j].keypoints,
                keyframes[j].descriptors,
                min_inliers=24,
            )
            if matrix is None:
                continue
            # 回环的相对尺度/角度应接近初始位姿给出的量级；只做宽松过滤。
            scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(matrix[:2, :2])))))
            if not (0.5 <= scale <= 2.0):
                continue
            edges.append(PoseEdge(i, j, matrix, inliers, loop=True))
            used.add((i, j))
            break
    return edges
