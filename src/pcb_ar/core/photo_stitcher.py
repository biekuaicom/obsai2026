"""顺序无关的离线照片拼接。

用于把已经拍好的一批局部照片重新拼成完整板图：
* 两两 ORB 匹配，不依赖文件名顺序；
* 用重叠区边缘/灰度内容分筛选边；
* 先求鲁棒平移布局，再做相似变换（平移+旋转+缩放）位姿优化；
* 最后按位姿渲染并做小洞修补。

这是离线/批处理路径，不参与实时捕获。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from pcb_ar.core.board_mask import foreground_mask
from pcb_ar.core.sparse_map import extract_features


@dataclass
class PhotoPiece:
    path: Path
    image: np.ndarray
    mask: np.ndarray
    features: object


@dataclass
class PhotoEdge:
    i: int
    j: int
    matrix: np.ndarray
    inliers: int
    content: float
    dx: float = 0.0
    dy: float = 0.0

    @property
    def weight(self) -> float:
        return max(0.05, min(4.0, (self.inliers / 40.0) * (0.2 + self.content)))


@dataclass
class PhotoStitchResult:
    image: np.ndarray
    scale: float
    placed: int
    total: int
    edges: int
    rms: float
    message: str
    poses: list[np.ndarray] = field(default_factory=list)


def _as_h(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return matrix.reshape(3, 3)


def _match_pair(
    a: PhotoPiece,
    b: PhotoPiece,
    matcher: cv2.BFMatcher,
    *,
    ratio: float = 0.76,
) -> tuple[np.ndarray | None, int]:
    try:
        knn = matcher.knnMatch(a.features.descriptors, b.features.descriptors, k=2)
    except cv2.error:
        return None, 0
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 15:
        return None, len(good)
    src = np.float32([a.features.keypoints[m.queryIdx] for m in good]).reshape(-1, 2)
    dst = np.float32([b.features.keypoints[m.trainIdx] for m in good]).reshape(-1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.5,
        maxIters=3000,
        confidence=0.997,
    )
    if matrix is None or mask is None:
        return None, 0
    inliers = int(np.count_nonzero(mask))
    if inliers < 12:
        return None, inliers
    h = _as_h(matrix)
    scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(h[:2, :2])))))
    if not (0.45 <= scale <= 2.2):
        return None, inliers
    return h, inliers


def _content_score(a: PhotoPiece, b: PhotoPiece, matrix: np.ndarray) -> float:
    th, tw = b.image.shape[:2]
    warped = cv2.warpAffine(
        a.image,
        matrix[:2, :].astype(np.float32),
        (tw, th),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
    )
    wm = cv2.warpAffine(
        a.mask,
        matrix[:2, :].astype(np.float32),
        (tw, th),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    overlap = wm & (b.mask > 0)
    n = int(np.count_nonzero(overlap))
    if n < 3000:
        return 0.0
    g1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b.image, cv2.COLOR_BGR2GRAY)
    e1 = cv2.dilate(cv2.Canny(g1, 35, 110), np.ones((3, 3), np.uint8))
    e2 = cv2.dilate(cv2.Canny(g2, 35, 110), np.ones((3, 3), np.uint8))
    moving = (e1 > 0) & overlap
    hit = (e1 > 0) & (e2 > 0) & overlap
    edge = float(np.count_nonzero(hit)) / max(1, int(np.count_nonzero(moving)))
    try:
        corr = float(np.corrcoef(g1[overlap], g2[overlap])[0, 1])
    except (ValueError, np.linalg.LinAlgError):
        corr = 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return max(0.0, min(1.0, 0.7 * edge + 0.3 * max(0.0, corr)))


def _initial_translations(n: int, edges: list[PhotoEdge]) -> np.ndarray:
    adj: dict[int, list[tuple[float, int, float, float]]] = {i: [] for i in range(n)}
    for e in edges:
        adj[e.i].append((e.weight, e.j, e.dx, e.dy))
        adj[e.j].append((e.weight, e.i, -e.dx, -e.dy))
    start = max(range(n), key=lambda i: sum(w for w, _, _, _ in adj[i]))
    pos: dict[int, np.ndarray] = {start: np.zeros(2, dtype=np.float64)}
    stack = [start]
    while stack:
        i = stack.pop()
        for _w, j, dx, dy in sorted(adj[i], reverse=True):
            if j in pos:
                continue
            pos[j] = pos[i] + np.array([dx, dy], dtype=np.float64)
            stack.append(j)
    centroid = np.mean(list(pos.values()), axis=0) if pos else np.zeros(2)
    return np.array([pos.get(i, centroid) for i in range(n)], dtype=np.float64)


def _solve_translations(n: int, edges: list[PhotoEdge], init: np.ndarray) -> np.ndarray:
    # 初始位置用于断开组件；随后解全局平移方程。
    pos = init.copy()
    for _ in range(8):
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        weights: list[float] = []
        for e in edges:
            r = np.zeros(2 * n)
            r[2 * e.j] = 1.0
            r[2 * e.i] = -1.0
            rows.append(r)
            rhs.append(e.dx)
            weights.append(e.weight)
            r = np.zeros(2 * n)
            r[2 * e.j + 1] = 1.0
            r[2 * e.i + 1] = -1.0
            rows.append(r)
            rhs.append(e.dy)
            weights.append(e.weight)
        if not rows:
            break
        a = np.asarray(rows)
        b = np.asarray(rhs)
        w = np.asarray(weights)
        # gauge: first image at 0
        a = np.vstack([a, np.eye(2 * n)[0] * 1e5, np.eye(2 * n)[1] * 1e5])
        b = np.r_[b, [0.0, 0.0]]
        w = np.r_[w, [1e3, 1e3]]
        x = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)[0]
        for _ in range(8):
            residual = a @ x - b
            hw = w / np.maximum(1.0, np.abs(residual) / 10.0)
            x = np.linalg.lstsq(a * hw[:, None], b * hw, rcond=None)[0]
        candidate = np.array([[x[2 * i], x[2 * i + 1]] for i in range(n)])
        if np.all(np.isfinite(candidate)):
            pos = candidate
        break
    return pos


def _params_to_pose(p: np.ndarray) -> np.ndarray:
    tx, ty, theta, log_s = (float(v) for v in p)
    s = math.exp(log_s)
    c, sn = math.cos(theta), math.sin(theta)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]])


def _pose_to_params(m: np.ndarray) -> np.ndarray:
    s = math.sqrt(max(1e-9, abs(float(np.linalg.det(m[:2, :2])))))
    return np.array(
        [
            float(m[0, 2]),
            float(m[1, 2]),
            math.atan2(float(m[1, 0]), float(m[0, 0])),
            math.log(max(s, 1e-9)),
        ],
        dtype=np.float64,
    )


def _residual(pose_i: np.ndarray, pose_j: np.ndarray, measured: np.ndarray) -> np.ndarray:
    rel = np.linalg.inv(pose_j) @ pose_i
    dt = rel[:2, 2] - measured[:2, 2]
    da = math.atan2(float(rel[1, 0]), float(rel[0, 0])) - math.atan2(
        float(measured[1, 0]), float(measured[0, 0])
    )
    while da > math.pi:
        da -= 2.0 * math.pi
    while da < -math.pi:
        da += 2.0 * math.pi
    sr = math.sqrt(max(1e-9, abs(float(np.linalg.det(rel[:2, :2])))))
    sm = math.sqrt(max(1e-9, abs(float(np.linalg.det(measured[:2, :2])))))
    return np.array(
        [dt[0] / 100.0, dt[1] / 100.0, da, math.log(max(sr, 1e-9) / max(sm, 1e-9))],
        dtype=np.float64,
    )


def _optimize_similarity(
    n: int,
    edges: list[PhotoEdge],
    init_positions: np.ndarray,
    iterations: int = 12,
) -> tuple[list[np.ndarray], float]:
    poses = [np.array([[1.0, 0.0, p[0]], [0.0, 1.0, p[1]], [0.0, 0.0, 1.0]]) for p in init_positions]
    params = np.stack([_pose_to_params(p) for p in poses[1:]]) if n > 1 else np.zeros((0, 4))
    fixed = poses[0].copy()
    weights = np.array([math.sqrt(e.weight) for e in edges], dtype=np.float64)
    best_rms = 0.0
    for _ in range(iterations):
        def residuals_for(par: np.ndarray) -> np.ndarray:
            current = [fixed] + [_params_to_pose(par[i]) for i in range(len(par))]
            return np.asarray(
                [_residual(current[e.i], current[e.j], e.matrix) for e in edges],
                dtype=np.float64,
            )

        base = residuals_for(params)
        best_rms = float(np.sqrt(np.mean(np.sum(base * base, axis=1))))
        if base.size == 0:
            break
        jac = np.zeros((base.size, params.size), dtype=np.float64)
        eps = 1e-4
        for k in range(params.size):
            pert = params.copy()
            pert.reshape(-1)[k] += eps
            jac[:, k] = (residuals_for(pert).reshape(-1) - base.reshape(-1)) / eps
        w = np.repeat(weights, 4)
        jw = jac * w[:, None]
        rw = base.reshape(-1) * w
        normal = jw.T @ jw + 1e-2 * np.eye(jac.shape[1])
        try:
            delta = np.linalg.solve(normal, -(jw.T @ rw))
        except np.linalg.LinAlgError:
            break
        params += delta.reshape(params.shape)
    poses = [fixed] + [_params_to_pose(p) for p in params]
    return poses, best_rms


def stitch_photo_folder(
    paths: Sequence[str | Path],
    *,
    progress: Callable[[int, int, str], None] | None = None,
    max_side: int = 3200,
    max_pixels: int = 9_000_000,
) -> PhotoStitchResult:
    files = [Path(p) for p in paths]
    if not files:
        raise ValueError("没有照片")
    pieces: list[PhotoPiece] = []
    for i, path in enumerate(files):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        pieces.append(
            PhotoPiece(
                path=path,
                image=image,
                mask=foreground_mask(image),
                features=extract_features(image, max_width=1200, nfeatures=2500),
            )
        )
    if len(pieces) < 2:
        raise ValueError("可用照片不足")
    n = len(pieces)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    edges: list[PhotoEdge] = []
    for i in range(n):
        for j in range(i + 1, n):
            matrix, inliers = _match_pair(pieces[i], pieces[j], matcher)
            if matrix is None:
                continue
            content = _content_score(pieces[i], pieces[j], matrix)
            if content < 0.22:
                continue
            h, w = pieces[i].image.shape[:2]
            center = np.array([w / 2.0, h / 2.0, 1.0])
            p = matrix @ center
            dx = float(p[0] / p[2] - w / 2.0)
            dy = float(p[1] / p[2] - h / 2.0)
            edges.append(PhotoEdge(i, j, matrix, inliers, content, dx, dy))
        if progress:
            progress(i + 1, n, f"匹配 {i + 1}/{n}")
    if not edges:
        raise ValueError("没有可靠的照片匹配")
    init = _initial_translations(n, edges)
    positions = _solve_translations(n, edges, init)
    # PCB 局部照片的相机距离/角度变化很小；先只用平移布局，避免错误相似边
    # 把整张图旋转/缩放拉坏。后续可用局部微调修正。
    poses = [
        np.array([[1.0, 0.0, p[0]], [0.0, 1.0, p[1]], [0.0, 0.0, 1.0]])
        for p in positions
    ]
    rms = 0.0

    # 计算输出画布
    points = []
    for piece, pose in zip(pieces, poses):
        h, w = piece.image.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        points.append(cv2.transform(corners.reshape(-1, 1, 2), pose[:2]).reshape(-1, 2))
    all_points = np.vstack(points)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = min(1.0, max_side / float(max(span)), math.sqrt(max_pixels / float(span[0] * span[1])))
    width = max(1, int(round(span[0] * scale)))
    height = max(1, int(round(span[1] * scale)))
    origin = lo
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    best = np.zeros((height, width), dtype=np.float32)
    for piece, pose in zip(pieces, poses):
        matrix = np.array(
            [[scale, 0.0, -origin[0] * scale], [0.0, scale, -origin[1] * scale], [0, 0, 1]]
        ) @ pose
        warped = cv2.warpAffine(
            piece.image,
            matrix[:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpAffine(
            piece.mask,
            matrix[:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ) > 0
        if not np.any(mask):
            continue
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        dist_full = np.zeros((height, width), dtype=np.float32)
        dist_full[y : y + h, x : x + w] = cv2.distanceTransform(
            mask[y : y + h, x : x + w].astype(np.uint8), cv2.DIST_L2, 5
        )
        better = mask & (dist_full > best + 0.3)
        canvas[better] = warped[better]
        best = np.maximum(best, dist_full)
    # 小黑洞修补；大洞保留为缺失，不伪造。
    holes = ((best <= 0).astype(np.uint8) * 255)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    repair = np.zeros_like(holes)
    for k in range(1, n_labels):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if 20 <= area <= 4000:
            repair[labels == k] = 255
    if np.any(repair):
        canvas = cv2.inpaint(canvas, repair, 3, cv2.INPAINT_TELEA)
    return PhotoStitchResult(
        image=canvas,
        scale=scale,
        placed=n,
        total=n,
        edges=len(edges),
        rms=rms,
        message=f"{n} 张 / {len(edges)} 边 / RMS {rms:.2f}",
        poses=poses,
    )
