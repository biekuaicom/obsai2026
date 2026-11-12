"""C 区重叠带锚点拼接（显微镜平面网格）。

对应手绘图思路：
* 只在相邻图的 C 区重叠带找特征/锚点；
* 每对相邻图估计相似变换；
* 网格四邻域 + 2x2 闭环约束全局求解；
* 不做整图乱匹配，减少 PCB 重复纹理认错。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pcb_ar.core.board_mask import foreground_mask


@dataclass
class OverlapEdge:
    i: int
    j: int
    matrix: np.ndarray
    inliers: int
    content: float


@dataclass
class OverlapResult:
    image: np.ndarray
    message: str
    edges: int
    rms: float


def _features(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(gray, None)
    pts = np.float32([p.pt for p in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des, kp


def _similarity_from_three(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """三个非共线点求相似变换（Umeyama 最小二乘）。"""
    if len(src) != 3 or len(dst) != 3:
        return None
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    # 非共线校验
    area = abs(np.cross(src[1] - src[0], src[2] - src[0]))
    if area < 16.0:
        return None
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    s = src - mu_s
    d = dst - mu_d
    cov = (d.T @ s) / 3.0
    u, sigma, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    var = float(np.mean(np.sum(s * s, axis=1)))
    if var < 1e-6:
        return None
    scale = float(np.sum(sigma) / var)
    t = mu_d - scale * (r @ mu_s)
    m = np.eye(3)
    m[:2, :2] = scale * r
    m[:2, 2] = t
    return m


def _best_anchor_triplet(points: np.ndarray) -> np.ndarray | None:
    """在匹配点里选面积最大的三个非共线点。"""
    if len(points) < 3:
        return None
    pts = points[: min(len(points), 40)]
    best = None
    best_area = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                a = abs(float(np.cross(pts[j] - pts[i], pts[k] - pts[i])))
                if a > best_area:
                    best_area = a
                    best = np.array([i, j, k])
    return best


def _match_band(
    a: np.ndarray,
    b: np.ndarray,
    fa,
    fb,
    horizontal: bool,
    fraction: float,
) -> tuple[np.ndarray | None, int]:
    h, w = a.shape[:2]
    # 左/上图的右/下 C 区；右/下图的左/上 C 区。
    band = max(24, int((w if horizontal else h) * fraction))
    if horizontal:
        pa = np.where(fa[0][:, 0] >= w - band)[0]
        pb = np.where(fb[0][:, 0] <= band)[0]
    else:
        pa = np.where(fa[0][:, 1] >= h - band)[0]
        pb = np.where(fb[0][:, 1] <= band)[0]
    if len(pa) < 12 or len(pb) < 12:
        return None, 0
    da = fa[1][pa]
    db = fb[1][pb]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(da, db, k=2)
    except cv2.error:
        return None, 0
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 12:
        return None, len(good)
    src = np.float32([fa[0][pa[m.queryIdx]] for m in good])
    dst = np.float32([fb[0][pb[m.trainIdx]] for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000
    )
    if matrix is None or mask is None:
        return None, 0
    inliers = int(np.count_nonzero(mask))
    if inliers < 10:
        return None, inliers
    hmat = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)
    # 三点锚点优先：选面积最大、非共线的三个 RANSAC 内点。
    idx = np.where(mask.ravel() > 0)[0]
    if len(idx) >= 3:
        triplet = _best_anchor_triplet(src[idx])
        if triplet is not None:
            a_idx = idx[triplet]
            m3 = _similarity_from_three(src[a_idx], dst[a_idx])
            if m3 is not None:
                pred = cv2.transform(src.reshape(-1, 1, 2), m3[:2]).reshape(-1, 2)
                err = np.linalg.norm(pred - dst, axis=1)[idx]
                if float(np.median(err)) < 3.0 and float(np.percentile(err, 90)) < 6.0:
                    return m3, inliers
    return hmat, inliers


def _content_score(a: np.ndarray, b: np.ndarray, matrix: np.ndarray, ma, mb) -> float:
    th, tw = b.shape[:2]
    warped = cv2.warpAffine(a, matrix[:2].astype(np.float32), (tw, th), flags=cv2.INTER_AREA)
    wm = cv2.warpAffine(ma, matrix[:2].astype(np.float32), (tw, th), flags=cv2.INTER_NEAREST) > 0
    ov = wm & (mb > 0)
    n = int(np.count_nonzero(ov))
    if n < 3000:
        return 0.0
    g1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    e1 = cv2.dilate(cv2.Canny(g1, 35, 110), np.ones((3, 3), np.uint8))
    e2 = cv2.dilate(cv2.Canny(g2, 35, 110), np.ones((3, 3), np.uint8))
    hit = np.count_nonzero((e1 > 0) & (e2 > 0) & ov)
    edge = hit / max(1, np.count_nonzero((e1 > 0) & ov))
    corr = np.corrcoef(g1[ov], g2[ov])[0, 1] if n > 10 else 0
    return float(max(0.0, 0.7 * edge + 0.3 * corr))


def _pose_to_params(m):
    m = np.asarray(m, float)
    s = math.sqrt(max(1e-9, abs(float(np.linalg.det(m[:2, :2])))))
    return np.array([m[0, 2], m[1, 2], math.atan2(m[1, 0], m[0, 0]), math.log(max(s, 1e-9))])


def _params_to_pose(p):
    tx, ty, a, ls = map(float, p)
    s = math.exp(ls)
    c, sn = math.cos(a), math.sin(a)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0, 0, 1]], float)


def _residual(pi, pj, z):
    rel = np.linalg.inv(pj) @ pi
    dt = rel[:2, 2] - z[:2, 2]
    da = math.atan2(rel[1, 0], rel[0, 0]) - math.atan2(z[1, 0], z[0, 0])
    da = (da + math.pi) % (2 * math.pi) - math.pi
    sr = math.sqrt(max(1e-9, abs(float(np.linalg.det(rel[:2, :2])))))
    sz = math.sqrt(max(1e-9, abs(float(np.linalg.det(z[:2, :2])))))
    return np.array([dt[0] / 100, dt[1] / 100, da, math.log(sr / sz)])


def _optimize(n, edges, init_positions, iterations=10):
    poses = [np.array([[1, 0, p[0]], [0, 1, p[1]], [0, 0, 1]], float) for p in init_positions]
    params = np.stack([_pose_to_params(p) for p in poses[1:]]) if n > 1 else np.zeros((0, 4))
    fixed = poses[0].copy()
    weights = np.array([math.sqrt(e.inliers * (0.2 + e.content)) for e in edges])
    rms = 0.0
    for _ in range(iterations):
        def res(par):
            cur = [fixed] + [_params_to_pose(par[i]) for i in range(len(par))]
            return np.array([_residual(cur[e.i], cur[e.j], e.matrix) for e in edges])
        base = res(params)
        rms = float(np.sqrt(np.mean(np.sum(base * base, axis=1))))
        if base.size == 0:
            break
        jac = np.zeros((base.size, params.size))
        for k in range(params.size):
            pert = params.copy()
            pert.reshape(-1)[k] += 1e-4
            jac[:, k] = (res(pert).reshape(-1) - base.reshape(-1)) / 1e-4
        w = np.repeat(weights, 4)
        jw = jac * w[:, None]
        rw = base.reshape(-1) * w
        try:
            delta = np.linalg.solve(jw.T @ jw + 1e-2 * np.eye(jac.shape[1]), -(jw.T @ rw))
        except np.linalg.LinAlgError:
            break
        params += delta.reshape(params.shape)
    return [fixed] + [_params_to_pose(p) for p in params], rms


def stitch_overlap_grid(
    paths: Sequence[str | Path],
    *,
    rows: int,
    cols: int,
    serpentine: bool = True,
    overlap: float = 0.3,
    max_side: int = 2600,
) -> OverlapResult:
    images = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            images.append(im)
    if len(images) < 2:
        raise ValueError("照片不足")
    h, w = images[0].shape[:2]
    cell = {}
    order = []
    for idx in range(min(len(images), rows * cols)):
        r = idx // cols
        c = idx % cols
        if serpentine and r % 2:
            c = cols - 1 - c
        cell[(r, c)] = idx
        order.append((r, c, idx))
    features = [_features(im) for im in images]
    masks = [foreground_mask(im) for im in images]
    frac = max(0.08, min(0.45, overlap * 0.5 + 0.02))
    edges = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in cell:
                continue
            i = cell[(r, c)]
            for dr, dc, horizontal in ((0, 1, True), (1, 0, False)):
                if (r + dr, c + dc) not in cell:
                    continue
                j = cell[(r + dr, c + dc)]
                M, inliers = _match_band(images[i], images[j], features[i], features[j], horizontal, frac)
                if M is None:
                    continue
                content = _content_score(images[i], images[j], M, masks[i], masks[j])
                if content < 0.15:
                    continue
                edges.append(OverlapEdge(i, j, M, inliers, content))
    if not edges:
        raise ValueError("C 区没有可靠匹配")
    dx, dy = w * (1 - overlap), h * (1 - overlap)
    init = [np.zeros(2) for _ in images]
    for r, c, idx in order:
        init[idx] = np.array([c * dx, r * dy])
    poses, rms = _optimize(len(images), edges, init)
    # bounds/render
    polys = []
    for im, p in zip(images, poses):
        hh, ww = im.shape[:2]
        pts = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]]).reshape(-1, 2)
        polys.append(cv2.transform(pts.reshape(-1, 1, 2), p[:2]).reshape(-1, 2))
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1)
    scale = min(1.0, max_side / float(max(span)))
    W, H = int(span[0] * scale), int(span[1] * scale)
    canvas = np.zeros((H, W, 3), np.uint8)
    for im, p in zip(images, poses):
        M = np.array([[scale, 0, -lo[0] * scale], [0, scale, -lo[1] * scale], [0, 0, 1]]) @ p
        warped = cv2.warpAffine(im, M[:2].astype(np.float32), (W, H), flags=cv2.INTER_AREA)
        # 合成用完整矩形，不用板面 mask 切接缝。
        # 后放的图确定性地覆盖前一张；每个覆盖像素都写入，不做距离阈值跳像素。
        valid = np.full(im.shape[:2], 255, dtype=np.uint8)
        wm = cv2.warpAffine(
            valid, M[:2].astype(np.float32), (W, H),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ) > 0
        canvas[wm] = warped[wm]
    return OverlapResult(canvas, f"{len(edges)} C区边 / RMS {rms:.2f}", len(edges), rms)
