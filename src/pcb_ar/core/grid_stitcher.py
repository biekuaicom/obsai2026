"""坐标网格驱动的离线拼接。

冻结拼图 + 交集 C + 形状星座，只求一次相似变换：

* 第一张接受的图作为板面坐标原点，像素写入后不再改；
* 行列只用来预测粗足迹。一对邻居一旦放置成功，就用它们的中心距
  更新这一方向的步长，后面的 C 用测得步长预测；
* C = 已冻结覆盖 ∩ 这一次位姿假设的足迹（两邻居时自然是 L 或十字）。
  冻结图和新图都先按 C 清零，再抽形状、再核对；
* 只在 C 内提取孔、元件、焊盘、丝印、长走线，先按类型和尺寸配对，
  再用相对距离/角度做星座校验，估计完整相似变换（旋转、平移、轻微缩放）；
* 形状太少时，才在同一 C 内用 ORB/ECC 作为二次测量，仍然只求一个相似变换；
* 核对失败就拒绝：不改已冻结位姿，不把该图写入冻结图或覆盖；
* 通过后只把新图独占像素并入冻结图，重叠区保持先接受的参考。

``serpentine`` 仅在调用方显式打开时才把奇数行改成从右到左。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from pcb_ar.core.geo import invert_affine, matrix_rotation_deg, matrix_scale


@dataclass
class GridStitchResult:
    image: np.ndarray
    placed: int
    total: int
    step: tuple[float, float]
    message: str
    # 与成功读入的照片一一对应。未放置为 None。
    # 矩阵把该图像素映射到归一化后的全分辨率拼图画布（不含 max_side 缩放）。
    poses: list[np.ndarray | None] | None = None


@dataclass
class _Cell:
    index: int
    row: int
    col: int
    pose: np.ndarray
    accepted: bool = True
    # 预测这一格时用的步长。第一对成功后会改成测得步长。
    predicted_step: tuple[float, float] | None = None


@dataclass
class _Shape:
    kind: str
    center: np.ndarray
    major: float
    minor: float
    angle: float


@dataclass
class _ShapeMatch:
    pairs: list[tuple[int, int]]
    residual: float
    anchor_hits: int


_ANCHORS = ("hole", "component")
_SUPPORT = ("trace", "silk")
_KIND_WEIGHT = {"hole": 6.0, "component": 5.0, "trace": 3.0, "silk": 1.6, "pad": 0.2}
_MAX_ROT_DEG = 6.5
_SCALE_MIN = 0.94
_SCALE_MAX = 1.06


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64)
    return matrix[:, :2] @ p + matrix[:, 2]


def _compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """返回 first ∘ second，即先做 second 再做 first。"""
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = first[:, :2] @ second[:, :2]
    out[:, 2] = first[:, :2] @ second[:, 2] + first[:, 2]
    return out


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _angle_delta(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _similarity_ok(matrix: np.ndarray | None) -> bool:
    if matrix is None or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return False
    try:
        scale = matrix_scale(matrix)
        rotation = abs(matrix_rotation_deg(matrix))
    except (ValueError, ZeroDivisionError):
        return False
    if not _SCALE_MIN <= scale <= _SCALE_MAX or rotation > _MAX_ROT_DEG:
        return False
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    len1 = math.hypot(a, c)
    len2 = math.hypot(b, d)
    if min(len1, len2) < 1e-6:
        return False
    if abs(len1 - len2) / max(len1, len2) > 0.08:
        return False
    dot = abs(a * b + c * d) / (len1 * len2)
    return dot < 0.08 and (a * d - b * c) > 1e-6


def _sim_from_two(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray | None:
    v = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    w = np.asarray(q2, dtype=np.float64) - np.asarray(q1, dtype=np.float64)
    nv = float(np.hypot(v[0], v[1]))
    nw = float(np.hypot(w[0], w[1]))
    if nv < 18.0 or nw < 18.0:
        return None
    scale = nw / nv
    if not _SCALE_MIN <= scale <= _SCALE_MAX:
        return None
    ang = _wrap_pi(math.atan2(w[1], w[0]) - math.atan2(v[1], v[0]))
    if abs(math.degrees(ang)) > _MAX_ROT_DEG:
        return None
    cos_t, sin_t = math.cos(ang), math.sin(ang)
    linear = scale * np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    trans = np.asarray(q1, dtype=np.float64) - linear @ np.asarray(p1, dtype=np.float64)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = trans
    return matrix if _similarity_ok(matrix) else None


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """两点到多点的相似变换，src → dst。"""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 2 or len(src) != len(dst):
        return None
    if len(src) == 2:
        return _sim_from_two(src[0], src[1], dst[0], dst[1])
    n = float(len(src))
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / n
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    var = float(np.mean(np.sum(src_c * src_c, axis=1)))
    if var < 1e-8:
        return None
    scale = float(np.trace(np.diag(singular) @ sign) / var)
    trans = mu_d - scale * (rotation @ mu_s)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = trans
    return matrix if _similarity_ok(matrix) else None


def _shift_limit(width: int, height: int) -> float:
    # 只允许相似变换在当前候选足迹上做小修正。大的步长误差交给候选，
    # 不能靠把新图滑进旧图来凑重叠面积。
    return max(40.0, 0.14 * float(min(width, height)))


def _center_shift(pose: np.ndarray, rough: np.ndarray, width: int, height: int) -> float:
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    delta = _apply(pose, center) - _apply(rough, center)
    return float(np.hypot(delta[0], delta[1]))


def _major_angle(rect: tuple) -> tuple[float, float, float]:
    (_cx, _cy), (rw, rh), ang = rect
    rw, rh = float(rw), float(rh)
    if rw < rh:
        rw, rh = rh, rw
        ang = float(ang) + 90.0
    rad = math.radians(float(ang)) % math.pi
    return rw, rh, rad


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8 and int(mask.max()) > 1:
        return mask
    return (mask > 0).astype(np.uint8) * 255


def _inside_mask(center: tuple[float, float], mask: np.ndarray, *, margin: int = 1) -> bool:
    h, w = mask.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    if not (margin <= x < w - margin and margin <= y < h - margin):
        return False
    return bool(mask[y, x])


def _contour_shape(contour: np.ndarray, mask: np.ndarray, *, dark: bool) -> _Shape | None:
    area = float(cv2.contourArea(contour))
    if area < 24.0:
        return None
    mask_area = max(1, int(np.count_nonzero(mask)))
    if area > 0.40 * mask_area:
        return None
    peri = float(cv2.arcLength(contour, True))
    if peri < 8.0:
        return None
    circularity = 4.0 * math.pi * area / (peri * peri)
    rect = cv2.minAreaRect(contour)
    (cx, cy), _size, _ang = rect
    if not _inside_mask((cx, cy), mask):
        return None
    major, minor, angle = _major_angle(rect)
    if minor < 2.0 or major < 3.0:
        return None
    extent = area / max(1.0, major * minor)
    x, y, bw, bh = cv2.boundingRect(contour)
    roi = mask[y : y + bh, x : x + bw]
    if roi.size == 0 or float(np.count_nonzero(roi)) / float(roi.size) < 0.45:
        return None
    aspect = major / minor
    h, w = mask.shape[:2]
    kind: str | None = None
    if circularity >= 0.78 and aspect < 1.35 and 6.0 <= minor <= max(36.0, 0.16 * min(h, w)):
        kind = "hole" if dark else "pad"
    elif minor >= 14.0 and major >= 28.0 and aspect <= 4.2 and area >= 220.0:
        kind = "component"
    elif aspect >= 3.4 and major >= 36.0:
        kind = "trace"
    elif major <= 28.0 and minor >= 3.5 and aspect < 3.0:
        kind = "pad" if extent >= 0.55 else "silk"
    elif area <= 700.0 and aspect < 4.5:
        kind = "silk"
    if kind is None:
        return None
    return _Shape(kind, np.array([cx, cy], dtype=np.float64), major, minor, angle)


def _blobs(gray: np.ndarray, mask: np.ndarray, median: float, *, dark: bool) -> list[_Shape]:
    delta = max(14.0, 0.10 * float(median) if median > 1.0 else 14.0)
    if dark:
        binary = ((gray < median - delta) & (mask > 0)).astype(np.uint8) * 255
    else:
        binary = ((gray > median + delta) & (mask > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: list[_Shape] = []
    for contour in contours:
        shape = _contour_shape(contour, mask, dark=dark)
        if shape is not None:
            shapes.append(shape)
    return shapes


def _hough_holes(gray: np.ndarray, mask: np.ndarray, median: float) -> list[_Shape]:
    h, w = gray.shape[:2]
    min_r = 6
    max_r = max(min_r + 2, int(min(h, w) * 0.14))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(min_r * 2),
        param1=90,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    found: list[_Shape] = []
    for raw in circles[0][:24]:
        cx, cy, radius = float(raw[0]), float(raw[1]), float(raw[2])
        if not _inside_mask((cx, cy), mask, margin=2):
            continue
        x0, y0 = int(max(0, cx - 2)), int(max(0, cy - 2))
        x1, y1 = int(min(w, cx + 3)), int(min(h, cy + 3))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0 or float(patch.mean()) > median - 10.0:
            continue
        found.append(_Shape("hole", np.array([cx, cy], dtype=np.float64), radius * 2.0, radius * 2.0, 0.0))
    return found


def _traces(gray: np.ndarray, mask: np.ndarray) -> list[_Shape]:
    edges = cv2.Canny(gray, 40, 120)
    edges[mask == 0] = 0
    h, w = gray.shape[:2]
    min_len = max(36, int(0.12 * min(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=28, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return []
    # OpenCV 4.12 有的构建返回 (N, 4)，有的仍是 (N, 1, 4)。
    packed = np.asarray(lines)
    raws = packed[:, 0, :] if packed.ndim == 3 else packed.reshape(-1, 4)
    shapes: list[_Shape] = []
    for raw in raws[:40]:
        x1, y1, x2, y2 = (float(v) for v in raw)
        if not (_inside_mask((x1, y1), mask) and _inside_mask((x2, y2), mask)):
            continue
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
        shapes.append(_Shape("trace", center, length, 3.0, angle))
    return shapes


def _suppress(shapes: list[_Shape]) -> list[_Shape]:
    order = {"hole": 0, "component": 1, "trace": 2, "silk": 3, "pad": 4}
    shapes = sorted(shapes, key=lambda s: (order.get(s.kind, 9), -s.major))
    kept: list[_Shape] = []
    for shape in shapes:
        duplicate = False
        for old in kept:
            if old.kind != shape.kind:
                continue
            dist = float(np.hypot(*(shape.center - old.center)))
            if dist < 0.45 * max(shape.major, old.major, 8.0):
                duplicate = True
                break
        if not duplicate:
            kept.append(shape)
    caps = {"hole": 16, "component": 24, "trace": 30, "silk": 24, "pad": 48}
    counts = {kind: 0 for kind in caps}
    limited: list[_Shape] = []
    for shape in kept:
        if counts.get(shape.kind, 0) >= caps.get(shape.kind, 20):
            continue
        counts[shape.kind] = counts.get(shape.kind, 0) + 1
        limited.append(shape)
    return limited


def _extract_shapes(image: np.ndarray, mask: np.ndarray) -> list[_Shape]:
    """只在 C 内提取二值形状：孔、元件、焊盘、丝印、长走线。"""
    if image.size == 0 or mask.size == 0 or int(np.count_nonzero(mask)) < 80:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask_u8 = _mask_u8(mask)
    if mask_u8.shape[:2] != gray.shape[:2]:
        return []
    values = gray[mask_u8 > 0]
    if values.size < 80:
        return []
    median = float(np.median(values))
    filled = gray.copy()
    filled[mask_u8 == 0] = median
    filled = cv2.GaussianBlur(filled, (3, 3), 0)
    shapes: list[_Shape] = []
    shapes.extend(_hough_holes(filled, mask_u8, median))
    shapes.extend(_blobs(filled, mask_u8, median, dark=True))
    shapes.extend(_blobs(filled, mask_u8, median, dark=False))
    shapes.extend(_traces(filled, mask_u8))
    return _suppress(shapes)


def _size_similar(a: _Shape, b: _Shape) -> bool:
    if min(a.major, b.major) < 2.0:
        return False
    major_ratio = max(a.major, b.major) / min(a.major, b.major)
    minor_ratio = max(a.minor, b.minor) / max(1.0, min(a.minor, b.minor))
    return major_ratio <= 1.30 and minor_ratio <= 1.40


def _angle_similar(a: _Shape, b: _Shape) -> bool:
    if a.kind not in ("trace", "component"):
        return True
    if a.major < a.minor * 1.45:
        return True
    return _angle_delta(a.angle, b.angle) <= math.radians(18.0)


def _pair_candidates(src: list[_Shape], dst: list[_Shape]) -> list[tuple[int, int]]:
    buckets: dict[str, list[int]] = {}
    for index, shape in enumerate(dst):
        buckets.setdefault(shape.kind, []).append(index)
    pairs: list[tuple[int, int]] = []
    for i, shape in enumerate(src):
        pool = buckets.get(shape.kind, [])
        ranked: list[tuple[float, int]] = []
        for j in pool:
            other = dst[j]
            if not _size_similar(shape, other) or not _angle_similar(shape, other):
                continue
            cost = abs(shape.major - other.major) / max(shape.major, 1.0)
            ranked.append((cost, j))
        ranked.sort()
        cap = 8 if shape.kind in _ANCHORS else 5
        for _cost, j in ranked[:cap]:
            pairs.append((i, j))
    return pairs


def _score_pairs(
    matrix: np.ndarray,
    pairs: list[tuple[int, int]],
    src: list[_Shape],
    dst: list[_Shape],
) -> tuple[float, list[tuple[int, int]], float]:
    measured: list[tuple[float, int, int]] = []
    for i, j in pairs:
        pred = _apply(matrix, src[i].center)
        err = float(np.hypot(*(pred - dst[j].center)))
        tol = max(3.2, 0.07 * src[i].major)
        if src[i].kind == "trace":
            tol = max(tol, 5.0)
        if err <= tol:
            measured.append((err, i, j))
    measured.sort()
    used_src: set[int] = set()
    used_dst: set[int] = set()
    inliers: list[tuple[int, int]] = []
    residuals: list[float] = []
    score = 0.0
    for err, i, j in measured:
        if i in used_src or j in used_dst:
            continue
        used_src.add(i)
        used_dst.add(j)
        inliers.append((i, j))
        residuals.append(err)
        score += _KIND_WEIGHT.get(src[i].kind, 1.0)
    residual = float(np.median(residuals)) if residuals else 99.0
    return score, inliers, residual


def _kind_present(shapes: list[_Shape], kind: str) -> bool:
    return any(shape.kind == kind for shape in shapes)


def _required_kinds(src: list[_Shape], dst: list[_Shape], inliers: list[tuple[int, int]], src_shapes: list[_Shape]) -> bool:
    """两侧都有的孔/元件必须进入内点，避免周期焊盘单独决定位姿。"""
    for kind in _ANCHORS:
        if _kind_present(src, kind) and _kind_present(dst, kind):
            if not any(src_shapes[i].kind == kind for i, _j in inliers):
                return False
    return True


def _constellation_ok(src: list[_Shape], dst: list[_Shape], inliers: list[tuple[int, int]], matrix: np.ndarray) -> bool:
    pts_s: list[np.ndarray] = []
    pts_d: list[np.ndarray] = []
    for i, j in inliers:
        if src[i].kind == "pad":
            continue
        pts_s.append(src[i].center)
        pts_d.append(dst[j].center)
    if len(pts_s) < 2:
        return False
    scale = max(1e-6, matrix_scale(matrix))
    dist_err: list[float] = []
    ang_err: list[float] = []
    for a in range(len(pts_s)):
        for b in range(a + 1, len(pts_s)):
            vs = pts_s[b] - pts_s[a]
            vd = pts_d[b] - pts_d[a]
            ds = float(np.hypot(vs[0], vs[1]))
            dd = float(np.hypot(vd[0], vd[1]))
            if ds < 14.0:
                continue
            dist_err.append(abs(dd - ds * scale) / ds)
            pred = _apply(matrix, pts_s[b]) - _apply(matrix, pts_s[a])
            if dd > 14.0 and float(np.hypot(pred[0], pred[1])) > 14.0:
                delta = abs(_wrap_pi(math.atan2(vd[1], vd[0]) - math.atan2(pred[1], pred[0])))
                ang_err.append(delta)
    if dist_err and float(np.median(dist_err)) > 0.10:
        return False
    if ang_err and float(np.median(ang_err)) > math.radians(8.0):
        return False
    return True


def _support_count(src: list[_Shape], inliers: list[tuple[int, int]]) -> int:
    return sum(1 for i, _j in inliers if src[i].kind in _SUPPORT or src[i].kind in _ANCHORS)


def _anchor_count(src: list[_Shape], inliers: list[tuple[int, int]]) -> int:
    return sum(1 for i, _j in inliers if src[i].kind in _ANCHORS)


def _match_constellation(src: list[_Shape], dst: list[_Shape]) -> _ShapeMatch | None:
    """类型+尺寸配对后，用孔/元件星座求一个相似变换。焊盘不参与求解。"""
    if len(src) < 2 or len(dst) < 2:
        return None
    pairs = _pair_candidates(src, dst)
    solvers = [pair for pair in pairs if src[pair[0]].kind != "pad"]
    if len(solvers) < 2:
        return None
    # 大元件和孔优先，周期焊盘不会进入种子。
    solvers.sort(key=lambda pair: (-_KIND_WEIGHT.get(src[pair[0]].kind, 0.0), -src[pair[0]].major))
    if len(solvers) > 28:
        head = solvers[:12]
        rest = solvers[12:28]
        chosen = list(head)
        for item in rest:
            if src[item[0]].kind in _ANCHORS:
                chosen.append(item)
        solvers = chosen
    best: tuple[float, list[tuple[int, int]], float, np.ndarray] | None = None
    for a in range(len(solvers)):
        i1, j1 = solvers[a]
        for b in range(a + 1, len(solvers)):
            i2, j2 = solvers[b]
            if i1 == i2 or j1 == j2:
                continue
            # 两侧已经有孔或元件时，种子必须带上它们，避免只用走线/丝印。
            has_anchor = _kind_present(src, "hole") or _kind_present(src, "component")
            if has_anchor and src[i1].kind not in _ANCHORS and src[i2].kind not in _ANCHORS:
                continue
            matrix = _sim_from_two(src[i1].center, src[i2].center, dst[j1].center, dst[j2].center)
            if matrix is None:
                continue
            score, inliers, residual = _score_pairs(matrix, solvers, src, dst)
            if not _required_kinds(src, dst, inliers, src):
                continue
            anchors = _anchor_count(src, inliers)
            support = _support_count(src, inliers)
            if anchors < 2 and support < 3:
                continue
            if residual > 3.4:
                continue
            if best is None or score > best[0] + 0.05 or (abs(score - best[0]) <= 0.05 and residual < best[2]):
                best = (score, inliers, residual, matrix)
    if best is None:
        return None
    _score, inliers, _residual, _matrix = best
    # 最终不用焊盘，也不让碎走线改写孔/元件已经确定的相似变换。
    structural = [(i, j) for i, j in inliers if src[i].kind != "pad"]
    anchors = [(i, j) for i, j in structural if src[i].kind in _ANCHORS]
    fit_on = anchors if len(anchors) >= 2 else structural
    if len(fit_on) < 2:
        return None
    if _anchor_count(src, fit_on) < 2 and _support_count(src, fit_on) < 3:
        return None
    refit = _fit_similarity(
        np.vstack([src[i].center for i, _j in fit_on]),
        np.vstack([dst[j].center for _i, j in fit_on]),
    )
    if refit is None or not _constellation_ok(src, dst, fit_on, refit):
        return None
    _score, kept, residual = _score_pairs(refit, fit_on, src, dst)
    if residual > 3.4 or len(kept) < 2:
        return None
    if not _required_kinds(src, dst, kept, src):
        return None
    return _ShapeMatch(kept, residual, _anchor_count(src, kept))


def _resize_for_analysis(
    frozen: np.ndarray,
    new: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height, width = mask.shape[:2]
    scale = min(1.0, 960.0 / float(max(width, height, 1)))
    if scale >= 0.999:
        return frozen, new, mask, 1.0
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    frozen_s = cv2.resize(frozen, size, interpolation=cv2.INTER_AREA)
    new_s = cv2.resize(new, size, interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(_mask_u8(mask), size, interpolation=cv2.INTER_NEAREST)
    return frozen_s, new_s, mask_s > 0, scale


def _scale_shapes(shapes: list[_Shape], scale: float) -> list[_Shape]:
    if abs(scale - 1.0) < 1e-6:
        return shapes
    inv = 1.0 / scale
    return [
        _Shape(s.kind, s.center * inv, s.major * inv, s.minor * inv, s.angle)
        for s in shapes
    ]


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0
    if int(np.count_nonzero(sel)) < 200:
        return 0.0
    va = a[sel].astype(np.float64)
    vb = b[sel].astype(np.float64)
    va -= va.mean()
    vb -= vb.mean()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-6:
        return 0.0
    return float(va @ vb / denom)


def _edge_iou(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ea = cv2.Canny(a, 45, 130) > 0
    eb = cv2.Canny(b, 45, 130) > 0
    sel = mask > 0
    left = ea[sel]
    right = eb[sel]
    union = int(np.count_nonzero(left | right))
    if union < 40:
        return 1.0
    return float(np.count_nonzero(left & right)) / float(union)


def _orb_match(frozen_gray: np.ndarray, new_gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask_u8 = cv2.erode(_mask_u8(mask), np.ones((5, 5), np.uint8))
    if int(np.count_nonzero(mask_u8)) < 240:
        return None
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    key_a, desc_a = orb.detectAndCompute(frozen_gray, mask_u8)
    key_b, desc_b = orb.detectAndCompute(new_gray, mask_u8)
    if desc_a is None or desc_b is None or len(key_a) < 12 or len(key_b) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return None
    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([key_a[m.queryIdx].pt for m in good])
    dst = np.float32([key_b[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=2500,
        confidence=0.995,
    )
    if matrix is None or inliers is None or int(np.count_nonzero(inliers)) < 10:
        return None
    matrix = matrix.astype(np.float64)
    if not _similarity_ok(matrix):
        return None
    keep = inliers.ravel().astype(bool)
    pred = (matrix[:, :2] @ src[keep].T).T + matrix[:, 2]
    err = np.linalg.norm(pred - dst[keep], axis=1)
    if float(np.median(err)) > 3.5:
        return None
    if _periodic_ambiguous(src[keep], dst[keep]):
        return None
    return matrix, src[keep], dst[keep]


def _periodic_ambiguous(src: np.ndarray, dst: np.ndarray) -> bool:
    """平移内点若有两个分开的峰，说明周期结构自己就能解释多种位姿。"""
    if len(src) < 16:
        return False
    delta = dst - src
    bins: dict[tuple[int, int], int] = {}
    for dx, dy in delta:
        key = (int(round(float(dx) / 8.0)), int(round(float(dy) / 8.0)))
        bins[key] = bins.get(key, 0) + 1
    ranked = sorted(bins.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return False
    (k1, c1), (k2, c2) = ranked[0], ranked[1]
    if c2 < 0.65 * c1:
        return False
    sep = math.hypot((k1[0] - k2[0]) * 8.0, (k1[1] - k2[1]) * 8.0)
    return sep >= 12.0


def _anchors_agree(src: list[_Shape], dst: list[_Shape], matrix: np.ndarray) -> bool:
    src_anchors = [s for s in src if s.kind in _ANCHORS]
    dst_anchors = [s for s in dst if s.kind in _ANCHORS]
    if not src_anchors or not dst_anchors:
        return True
    hits = 0
    for shape in src_anchors:
        pred = _apply(matrix, shape.center)
        for other in dst_anchors:
            if other.kind != shape.kind or not _size_similar(shape, other):
                continue
            if float(np.hypot(*(pred - other.center))) <= max(4.0, 0.08 * shape.major):
                hits += 1
                break
    need = max(1, min(2, len(src_anchors), len(dst_anchors)))
    return hits >= need


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _sample_tile(image: np.ndarray, pose: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    local = pose.copy()
    local[0, 2] -= float(origin[0])
    local[1, 2] -= float(origin[1])
    return cv2.warpAffine(
        image,
        local.astype(np.float32),
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _ecc_refine(
    frozen_gray: np.ndarray,
    warped_gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    sel = int(np.count_nonzero(mask))
    if sel < 400:
        return None
    template = frozen_gray.astype(np.float32)
    moving = warped_gray.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    try:
        score, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            _mask_u8(mask),
            5,
        )
    except cv2.error:
        return None
    warp64 = warp.astype(np.float64)
    if float(score) < 0.45 or not np.all(np.isfinite(warp64)):
        return None
    shift = float(np.hypot(warp64[0, 2], warp64[1, 2]))
    rotation = abs(matrix_rotation_deg(warp64))
    if shift > 4.0 or rotation > 0.8:
        return None
    return warp64


def _pose_from_roi_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    inv_local: np.ndarray,
    origin: tuple[int, int],
) -> np.ndarray | None:
    """ROI 内的 (冻结点, 新图点) → 新图像素到画布的相似变换。"""
    tiles: list[np.ndarray] = []
    canvas: list[np.ndarray] = []
    ox, oy = float(origin[0]), float(origin[1])
    for frozen_pt, new_pt in pairs:
        tiles.append(_apply(inv_local, new_pt))
        canvas.append(np.array([ox, oy], dtype=np.float64) + np.asarray(frozen_pt, dtype=np.float64))
    return _fit_similarity(np.vstack(tiles), np.vstack(canvas))


def _pair_residual(pose: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray]], inv_local: np.ndarray, origin: tuple[int, int]) -> float:
    if not pairs:
        return 99.0
    ox, oy = float(origin[0]), float(origin[1])
    errs: list[float] = []
    for frozen_pt, new_pt in pairs:
        canvas = _apply(pose, _apply(inv_local, new_pt))
        pred = canvas - np.array([ox, oy], dtype=np.float64)
        errs.append(float(np.hypot(*(pred - np.asarray(frozen_pt, dtype=np.float64)))))
    return float(np.median(errs))


def _sizes_agree(src: list[_Shape], dst: list[_Shape], pairs: list[tuple[int, int]]) -> bool:
    ratios: list[float] = []
    for i, j in pairs:
        if src[i].kind not in _ANCHORS and src[i].kind != "trace":
            continue
        ratios.append(max(src[i].major, dst[j].major) / max(1.0, min(src[i].major, dst[j].major)))
    if not ratios:
        return True
    return float(np.median(ratios)) <= 1.22


def _verify_content(
    frozen_roi: np.ndarray,
    image: np.ndarray,
    pose: np.ndarray,
    origin: tuple[int, int],
    mask: np.ndarray,
    residual: float,
    *,
    anchor_hits: int,
    sizes_ok: bool,
) -> bool:
    if not sizes_ok or residual > 4.0:
        return False
    if not _similarity_ok(pose):
        return False
    warped = _sample_tile(image, pose, origin, (mask.shape[1], mask.shape[0]))
    # 仍在原来的 C 内，但只保留求解后足迹还盖住的像素，避免用错位条带把正确位姿判失败。
    local = pose.copy()
    local[0, 2] -= float(origin[0])
    local[1, 2] -= float(origin[1])
    foot = cv2.warpAffine(
        np.full(image.shape[:2], 255, np.uint8),
        local.astype(np.float32),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    refined = (mask > 0) & (foot > 0)
    if int(np.count_nonzero(refined)) < 300:
        return False
    ncc = _ncc(_gray(frozen_roi), _gray(warped), refined)
    edges = _edge_iou(_gray(frozen_roi), _gray(warped), refined)
    if anchor_hits >= 2 and residual <= 2.6 and ncc >= 0.45:
        return True
    if ncc >= 0.55 and residual <= 3.6:
        return True
    if ncc >= 0.72:
        return True
    if ncc >= 0.40 and edges >= 0.16 and residual <= 3.0 and anchor_hits >= 2:
        return True
    return False


def _rect_cover_count(cover: np.ndarray, x: float, y: float, width: int, height: int) -> int:
    canvas_h, canvas_w = cover.shape[:2]
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(canvas_w, int(math.ceil(x + width)))
    y1 = min(canvas_h, int(math.ceil(y + height)))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(cover[y0:y1, x0:x1]))


def _overlap_geometry(
    cover: np.ndarray,
    tx: float,
    ty: float,
    width: int,
    height: int,
) -> dict[str, float] | None:
    """轴对齐足迹和已冻结覆盖的交集。面积最大不等于正确的重叠带。"""
    canvas_h, canvas_w = cover.shape[:2]
    x0 = int(math.floor(tx))
    y0 = int(math.floor(ty))
    x1 = int(math.ceil(tx + width))
    y1 = int(math.ceil(ty + height))
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(canvas_w, x1), min(canvas_h, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    region = cover[iy0:iy1, ix0:ix1] > 0
    area = int(np.count_nonzero(region))
    if area < 1:
        return None
    ys, xs = np.where(region)
    return {
        "area": float(area),
        "frac": area / float(width * height),
        "ow": float(int(xs.max()) - int(xs.min()) + 1),
        "oh": float(int(ys.max()) - int(ys.min()) + 1),
        "mean_local_x": float(xs.mean() + ix0 - tx),
        "mean_local_y": float(ys.mean() + iy0 - ty),
    }


def _strip_cost(
    geom: dict[str, float],
    expected_overlap: tuple[float, float],
    axis: str,
    shift_x: float,
    shift_y: float,
) -> float | None:
    """越接近期望重叠带越好。重叠面积更大不能加分，滑进旧图直接丢掉。"""
    if geom["frac"] < 0.05 or geom["frac"] > 0.62:
        return None
    expect_w, expect_h = expected_overlap
    if axis == "y":
        cost = abs(geom["oh"] - expect_h) + 0.25 * abs(geom["ow"] - expect_w)
    elif axis == "xy":
        cost = abs(geom["ow"] - expect_w) + abs(geom["oh"] - expect_h)
    else:
        cost = abs(geom["ow"] - expect_w) + 0.25 * abs(geom["oh"] - expect_h)
    # 只在分数相近时偏向更小的位移，不偏向更大的面积。
    return float(cost + 0.015 * math.hypot(shift_x, shift_y))


def _search_rough(
    cover: np.ndarray,
    rough_x: float,
    rough_y: float,
    width: int,
    height: int,
    *,
    expected_overlap: tuple[float, float],
    axis: str = "x",
) -> tuple[float, float] | None:
    """粗搜索：目标是邻居方向上的期望重叠带，不是覆盖面积最大的位置。"""
    ys, xs = np.where(cover > 0)
    if xs.size == 0:
        return None
    stride_x = max(4.0, width * 0.03)
    stride_y = max(4.0, height * 0.03)
    if axis == "y":
        y_vals = np.arange(float(ys.min()) - height + 4.0, float(ys.max()) - 4.0, stride_y)
        x_vals = np.arange(rough_x - 12.0, rough_x + 12.1, 6.0)
    else:
        x_vals = np.arange(float(xs.min()) - width + 4.0, float(xs.max()) - 4.0, stride_x)
        y_vals = np.arange(rough_y - 12.0, rough_y + 12.1, 6.0)
    best: tuple[float, float] | None = None
    best_cost = 1e18
    for dy in y_vals:
        for dx in x_vals:
            x = float(dx)
            y = float(dy)
            geom = _overlap_geometry(cover, x, y, width, height)
            if geom is None:
                continue
            cost = _strip_cost(geom, expected_overlap, axis, x - rough_x, y - rough_y)
            if cost is None or cost >= best_cost:
                continue
            best_cost = cost
            best = (x, y)
    return best


def _roi_intersection(
    frozen: np.ndarray,
    cover: np.ndarray,
    image: np.ndarray,
    rough: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray] | None:
    """返回冻结块、按粗位姿采样的新图、C 掩膜、ROI 原点、ROI 内的粗变换。"""
    height, width = image.shape[:2]
    canvas_h, canvas_w = cover.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    mapped = np.vstack([_apply(rough, p) for p in corners])
    x0 = max(0, int(math.floor(float(mapped[:, 0].min()))) - 2)
    y0 = max(0, int(math.floor(float(mapped[:, 1].min()))) - 2)
    x1 = min(canvas_w, int(math.ceil(float(mapped[:, 0].max()))) + 2)
    y1 = min(canvas_h, int(math.ceil(float(mapped[:, 1].max()))) + 2)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    local = rough.copy()
    local[0, 2] -= float(x0)
    local[1, 2] -= float(y0)
    roi_size = (x1 - x0, y1 - y0)
    foot = cv2.warpAffine(
        np.full((height, width), 255, np.uint8),
        local.astype(np.float32),
        roi_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cmask = (cover[y0:y1, x0:x1] > 0) & (foot > 0)
    if int(np.count_nonzero(cmask)) < 400:
        return None
    # 工作区收成 C 的外接框，并把 C 以外清零。不能在整块预测矩形上抽形状或做 NCC。
    frozen_roi = frozen[y0:y1, x0:x1].copy()
    new_roi = _sample_tile(image, rough, (x0, y0), roi_size)
    frozen_roi[~cmask] = 0
    new_roi[~cmask] = 0
    ys, xs = np.where(cmask)
    # 收到 C 的外接框，并留几像素空边，避免形状中心贴在裁切边上被丢掉。
    # 空边仍在掩膜外，是 0，不把整块预测矩形交进去匹配。
    pad = 6
    bx0 = max(0, int(xs.min()) - pad)
    by0 = max(0, int(ys.min()) - pad)
    bx1 = min(cmask.shape[1], int(xs.max()) + 1 + pad)
    by1 = min(cmask.shape[0], int(ys.max()) + 1 + pad)
    local = local.copy()
    local[0, 2] -= float(bx0)
    local[1, 2] -= float(by0)
    return (
        frozen_roi[by0:by1, bx0:bx1].copy(),
        new_roi[by0:by1, bx0:bx1].copy(),
        cmask[by0:by1, bx0:bx1].copy(),
        (x0 + bx0, y0 + by0),
        local,
    )


def _tighten_vertical(
    frozen_roi: np.ndarray,
    image: np.ndarray,
    pose: np.ndarray,
    origin: tuple[int, int],
    mask: np.ndarray,
    point_pairs: list[tuple[np.ndarray, np.ndarray]],
    inv_local: np.ndarray,
) -> np.ndarray:
    """在同一块 C 里只微调竖直平移，收紧 A/B 接缝的高度残差。旋转和缩放保持不变。"""
    pose = pose.copy()

    def _vertical_error(candidate: np.ndarray) -> float:
        if len(point_pairs) < 2:
            return 0.0
        errors = []
        oy = float(origin[1])
        for frozen_pt, new_pt in point_pairs:
            canvas = _apply(candidate, _apply(inv_local, new_pt))
            errors.append(float(canvas[1] - (oy + float(frozen_pt[1]))))
        return abs(float(np.median(errors)))

    # 点够多时才用中心残差改 ty。两点拟合已经把这两个中心对齐，再挪会把旋转角带偏。
    if len(point_pairs) >= 4:
        signed = []
        oy = float(origin[1])
        for frozen_pt, new_pt in point_pairs:
            canvas = _apply(pose, _apply(inv_local, new_pt))
            signed.append(float(canvas[1] - (oy + float(frozen_pt[1]))))
        shift = float(np.median(signed))
        if 0.6 <= abs(shift) <= 4.0:
            pose[1, 2] -= shift
    base = _sample_tile(image, pose, origin, (mask.shape[1], mask.shape[0]))
    best_ncc = _ncc(_gray(frozen_roi), _gray(base), mask)
    anchor_limit = max(1.0, _vertical_error(pose) + 0.4)
    best = pose
    for dy in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
        trial = pose.copy()
        trial[1, 2] += dy
        if _vertical_error(trial) > anchor_limit:
            continue
        warped = _sample_tile(image, trial, origin, (mask.shape[1], mask.shape[0]))
        score = _ncc(_gray(frozen_roi), _gray(warped), mask)
        if score > best_ncc + 0.02:
            best_ncc = score
            best = trial
    return best


def _align_tile(
    frozen: np.ndarray,
    cover: np.ndarray,
    image: np.ndarray,
    rough: np.ndarray,
) -> np.ndarray | None:
    packed = _roi_intersection(frozen, cover, image, rough)
    if packed is None:
        return None
    frozen_roi, new_roi, cmask, origin, local = packed
    inv_local = invert_affine(local).astype(np.float64)
    frozen_s, new_s, mask_s, analysis = _resize_for_analysis(frozen_roi, new_roi, cmask)
    src_shapes = _scale_shapes(_extract_shapes(frozen_s, mask_s), analysis)
    dst_shapes = _scale_shapes(_extract_shapes(new_s, mask_s), analysis)
    point_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    anchor_hits = 0
    sizes_ok = True
    shape_locked = False
    matched = _match_constellation(src_shapes, dst_shapes)
    if matched is not None and _sizes_agree(src_shapes, dst_shapes, matched.pairs):
        point_pairs = [(src_shapes[i].center.copy(), dst_shapes[j].center.copy()) for i, j in matched.pairs]
        anchor_hits = matched.anchor_hits
        shape_locked = True
    else:
        orb = _orb_match(_gray(frozen_s), _gray(new_s), mask_s)
        if orb is None:
            return None
        matrix, src_pts, dst_pts = orb
        src_pts = src_pts.astype(np.float64)
        dst_pts = dst_pts.astype(np.float64)
        if analysis != 1.0:
            src_pts = src_pts / analysis
            dst_pts = dst_pts / analysis
            matrix = matrix.copy()
            matrix[:, 2] = matrix[:, 2] / analysis
        if not _anchors_agree(src_shapes, dst_shapes, matrix):
            return None
        keep = min(40, len(src_pts))
        point_pairs = [(src_pts[i], dst_pts[i]) for i in range(keep)]
        anchor_hits = 2 if _anchors_agree(src_shapes, dst_shapes, matrix) and src_shapes else 0
    pose = _pose_from_roi_pairs(point_pairs, inv_local, origin)
    if pose is None:
        return None
    pose = _tighten_vertical(frozen_roi, image, pose, origin, cmask, point_pairs, inv_local)
    height, width = image.shape[:2]
    if _center_shift(pose, rough, width, height) > _shift_limit(width, height):
        return None
    # 形状星座已经锁定时不再做 ECC，避免周期焊盘把位姿吸走。
    # 只有 ORB 二次测量才允许在同一 C 内做一次小幅 ECC。
    if not shape_locked:
        warped = _sample_tile(image, pose, origin, (cmask.shape[1], cmask.shape[0]))
        correction = _ecc_refine(_gray(frozen_roi), _gray(warped), cmask)
        if correction is not None:
            local_pose = pose.copy()
            local_pose[0, 2] -= float(origin[0])
            local_pose[1, 2] -= float(origin[1])
            refined_local = _compose(correction, local_pose)
            refined = refined_local.copy()
            refined[0, 2] += float(origin[0])
            refined[1, 2] += float(origin[1])
            if _similarity_ok(refined) and _center_shift(refined, rough, width, height) <= _shift_limit(width, height):
                pose = refined
    residual = _pair_residual(pose, point_pairs, inv_local, origin)
    if not _verify_content(
        frozen_roi,
        image,
        pose,
        origin,
        cmask,
        residual,
        anchor_hits=anchor_hits,
        sizes_ok=sizes_ok,
    ):
        return None
    return pose


def _paste_exclusive(frozen: np.ndarray, cover: np.ndarray, image: np.ndarray, pose: np.ndarray, *, erode: bool) -> None:
    height, width = image.shape[:2]
    canvas_h, canvas_w = cover.shape[:2]
    corners = np.vstack([
        _apply(pose, np.array([0.0, 0.0])),
        _apply(pose, np.array([width - 1.0, 0.0])),
        _apply(pose, np.array([width - 1.0, height - 1.0])),
        _apply(pose, np.array([0.0, height - 1.0])),
    ])
    x0 = max(0, int(math.floor(float(corners[:, 0].min()))) - 1)
    y0 = max(0, int(math.floor(float(corners[:, 1].min()))) - 1)
    x1 = min(canvas_w, int(math.ceil(float(corners[:, 0].max()))) + 2)
    y1 = min(canvas_h, int(math.ceil(float(corners[:, 1].max()))) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    warped = _sample_tile(image, pose, (x0, y0), (x1 - x0, y1 - y0))
    local = pose.copy()
    local[0, 2] -= float(x0)
    local[1, 2] -= float(y0)
    foot = cv2.warpAffine(
        np.full((height, width), 255, np.uint8),
        local.astype(np.float32),
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if erode:
        foot = cv2.erode(foot, np.ones((3, 3), np.uint8), iterations=1)
    region = frozen[y0:y1, x0:x1]
    cover_roi = cover[y0:y1, x0:x1]
    exclusive = (foot > 0) & (cover_roi == 0)
    if not np.any(exclusive):
        return
    region[exclusive] = warped[exclusive]
    cover_roi[exclusive] = 255


def _assign_cells(
    count: int,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    serpentine: bool,
) -> list[_Cell]:
    cells: list[_Cell] = []
    index = 0
    for row in range(rows):
        order = range(cols)
        if serpentine and row % 2 == 1:
            order = range(cols - 1, -1, -1)
        for col in order:
            if index >= count:
                break
            pose = np.array([[1.0, 0.0, col * dx], [0.0, 1.0, row * dy]], dtype=np.float64)
            cells.append(_Cell(index, row, col, pose, False))
            index += 1
        if index >= count:
            break
    return cells


def _tile_center(cell: _Cell, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return _apply(cell.pose, np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64))


def _measured_steps(
    accepted: list[_Cell],
    images: list[np.ndarray],
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """相邻已接受格子的中心距。有观测之后不再沿用 overlap 参数的猜测。"""
    dxs: list[float] = []
    dys: list[float] = []
    for i, left in enumerate(accepted):
        for right in accepted[i + 1 :]:
            if left.row == right.row and abs(left.col - right.col) == 1:
                a = _tile_center(left, images[left.index])
                b = _tile_center(right, images[right.index])
                dxs.append(float(b[0] - a[0]) / float(right.col - left.col))
            if left.col == right.col and abs(left.row - right.row) == 1:
                a = _tile_center(left, images[left.index])
                b = _tile_center(right, images[right.index])
                dys.append(float(b[1] - a[1]) / float(right.row - left.row))
    if dxs:
        dx = float(np.median(dxs))
    if dys:
        dy = float(np.median(dys))
    return dx, dy


def _nearest_neighbor(cell: _Cell, accepted: list[_Cell], axis: str) -> _Cell | None:
    best: _Cell | None = None
    best_d = 10**9
    for other in accepted:
        if not other.accepted:
            continue
        if axis == "x" and other.row == cell.row and other.col != cell.col:
            dist = abs(other.col - cell.col)
        elif axis == "y" and other.col == cell.col and other.row != cell.row:
            dist = abs(other.row - cell.row)
        else:
            continue
        if dist < best_d:
            best = other
            best_d = dist
    return best


def _predict_top_left(
    cell: _Cell,
    accepted: list[_Cell],
    images: list[np.ndarray],
    dx: float,
    dy: float,
    pad_x: float,
    pad_y: float,
    width: int,
    height: int,
) -> tuple[float, float, _Cell | None, _Cell | None]:
    horiz = _nearest_neighbor(cell, accepted, "x")
    vert = _nearest_neighbor(cell, accepted, "y")
    centers_x: list[float] = []
    centers_y: list[float] = []
    if horiz is not None:
        center = _tile_center(horiz, images[horiz.index])
        centers_x.append(float(center[0] + (cell.col - horiz.col) * dx))
        centers_y.append(float(center[1] + (cell.row - horiz.row) * dy))
    if vert is not None:
        center = _tile_center(vert, images[vert.index])
        centers_x.append(float(center[0] + (cell.col - vert.col) * dx))
        centers_y.append(float(center[1] + (cell.row - vert.row) * dy))
    if not centers_x:
        centers_x.append(pad_x + cell.col * dx + (width - 1) / 2.0)
        centers_y.append(pad_y + cell.row * dy + (height - 1) / 2.0)
    # 两个邻居时取平均，仍然只产生一个位姿假设，C 由这个足迹和覆盖相交得到。
    cx = float(np.mean(centers_x))
    cy = float(np.mean(centers_y))
    return cx - (width - 1) / 2.0, cy - (height - 1) / 2.0, horiz, vert


def _plausible_steps(length: int, preferred: float) -> list[float]:
    """重叠大约 8%–45% 的步长，按离期望步长的距离排序。不包含滑进旧图的更短步长。"""
    lo = float(length) * (1.0 - 0.45)
    hi = float(length) * (1.0 - 0.08)
    stride = max(12.0, float(length) * 0.03)
    values = [float(v) for v in np.arange(lo, hi + stride * 0.5, stride)]
    values.append(float(preferred))
    unique: list[float] = []
    for value in values:
        if all(abs(value - old) > 1.0 for old in unique):
            unique.append(value)
    unique.sort(key=lambda item: (abs(item - preferred), item))
    return unique


def _coarse_candidates(
    cell: _Cell,
    images: list[np.ndarray],
    cover: np.ndarray,
    pred_x: float,
    pred_y: float,
    dx: float,
    dy: float,
    width: int,
    height: int,
    horiz: _Cell | None,
    vert: _Cell | None,
) -> list[tuple[float, float]]:
    """按期望重叠带排序的候选足迹。重叠超过半幅的位置直接丢掉。"""
    raw: list[tuple[float, float, float]] = []
    # 单邻居是一条重叠带，不能超过半幅；两个邻居的 L/十字面积可以更大，
    # 但仍然丢掉几乎整幅盖住旧图的位置。
    frac_cap = 0.80 if horiz is not None and vert is not None else 0.62

    def consider(tx: float, ty: float, step_cost: float) -> None:
        geom = _overlap_geometry(cover, tx, ty, width, height)
        if geom is None or geom["frac"] < 0.05 or geom["frac"] > frac_cap:
            return
        raw.append((step_cost, tx, ty))

    if horiz is not None:
        dcol = cell.col - horiz.col
        anchor = _tile_center(horiz, images[horiz.index])
        for step in _plausible_steps(width, abs(dx) if abs(dx) > 1.0 else width * 0.7):
            cx = float(anchor[0] + dcol * step)
            cy = float(anchor[1])
            tx = cx - (width - 1) / 2.0
            ty = cy - (height - 1) / 2.0
            consider(tx, ty, abs(step - abs(dx)))
            if abs(step - abs(dx)) < 1.5:
                for jitter in (-10.0, 10.0):
                    consider(tx, ty + jitter, abs(step - abs(dx)) + 0.5)
    elif vert is not None:
        drow = cell.row - vert.row
        anchor = _tile_center(vert, images[vert.index])
        for step in _plausible_steps(height, abs(dy) if abs(dy) > 1.0 else height * 0.7):
            cx = float(anchor[0])
            cy = float(anchor[1] + drow * step)
            tx = cx - (width - 1) / 2.0
            ty = cy - (height - 1) / 2.0
            consider(tx, ty, abs(step - abs(dy)))
            if abs(step - abs(dy)) < 1.5:
                for jitter in (-10.0, 10.0):
                    consider(tx + jitter, ty, abs(step - abs(dy)) + 0.5)
    else:
        consider(pred_x, pred_y, 0.0)
    if horiz is not None and vert is not None:
        # 十字/L：横竖步长各自取靠近测得值的几档，不在二维上把图滑进已冻结区域。
        dcol = cell.col - horiz.col
        drow = cell.row - vert.row
        hcenter = _tile_center(horiz, images[horiz.index])
        vcenter = _tile_center(vert, images[vert.index])
        x_steps = _plausible_steps(width, abs(dx) if abs(dx) > 1.0 else width * 0.7)[:6]
        y_steps = _plausible_steps(height, abs(dy) if abs(dy) > 1.0 else height * 0.7)[:4]
        for sx in x_steps:
            for sy in y_steps:
                cx = float(hcenter[0] + dcol * sx)
                cy = float(vcenter[1] + drow * sy)
                consider(
                    cx - (width - 1) / 2.0,
                    cy - (height - 1) / 2.0,
                    abs(sx - abs(dx)) + abs(sy - abs(dy)),
                )
    consider(pred_x, pred_y, -0.1)
    raw.sort(key=lambda item: item[0])
    ordered: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for _cost, tx, ty in raw:
        key = (int(round(tx)), int(round(ty)))
        if key in seen:
            continue
        seen.add(key)
        ordered.append((tx, ty))
        if len(ordered) >= 16:
            break
    return ordered


def _calibrate_grid(
    cells: list[_Cell],
    images: list[np.ndarray],
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按文件夹顺序冻结。失败的图不写入 frozen/cover，也不改已接受位姿。"""
    if not cells:
        return np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8)
    max_w = max(int(image.shape[1]) for image in images)
    max_h = max(int(image.shape[0]) for image in images)
    pad_x = int(0.22 * max_w + 12)
    pad_y = int(0.22 * max_h + 12)
    extra = int(0.28 * min(max_w, max_h) + 24)
    # 画布按可能的大步长留空。测得步长比 overlap 猜测更大时不能裁掉后面的图。
    pitch_x = max(abs(dx), 0.92 * max_w)
    pitch_y = max(abs(dy), 0.92 * max_h)
    width = int(math.ceil(pad_x + max(c.col * pitch_x + images[c.index].shape[1] for c in cells) + extra))
    height = int(math.ceil(pad_y + max(c.row * pitch_y + images[c.index].shape[0] for c in cells) + extra))
    frozen = np.zeros((height, width, 3), np.uint8)
    cover = np.zeros((height, width), np.uint8)

    first = cells[0]
    fh, fw = images[first.index].shape[:2]
    frozen[pad_y : pad_y + fh, pad_x : pad_x + fw] = images[first.index]
    cover[pad_y : pad_y + fh, pad_x : pad_x + fw] = 255
    first.pose = np.array([[1.0, 0.0, float(pad_x)], [0.0, 1.0, float(pad_y)]], dtype=np.float64)
    first.accepted = True
    first.predicted_step = (float(dx), float(dy))
    dx_hat, dy_hat = float(dx), float(dy)

    for cell in cells[1:]:
        image = images[cell.index]
        ih, iw = image.shape[:2]
        accepted = [item for item in cells if item.accepted]
        dx_hat, dy_hat = _measured_steps(accepted, images, dx_hat, dy_hat)
        cell.predicted_step = (dx_hat, dy_hat)
        pred_x, pred_y, horiz, vert = _predict_top_left(
            cell, accepted, images, dx_hat, dy_hat, float(pad_x), float(pad_y), iw, ih
        )
        candidates = _coarse_candidates(
            cell, images, cover, pred_x, pred_y, dx_hat, dy_hat, iw, ih, horiz, vert
        )
        placed = False
        for tx, ty in candidates:
            rough = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
            pose = _align_tile(frozen, cover, image, rough)
            if pose is None:
                continue
            if _center_shift(pose, rough, iw, ih) > _shift_limit(iw, ih):
                continue
            cell.pose = pose
            cell.accepted = True
            _paste_exclusive(frozen, cover, image, pose, erode=True)
            placed = True
            break
        if not placed:
            cell.accepted = False

    accepted = [cell for cell in cells if cell.accepted]
    if accepted:
        points = []
        for cell in accepted:
            ih, iw = images[cell.index].shape[:2]
            points.append(_corners(cell.pose, iw, ih))
        stacked = np.vstack(points)
        min_x = float(stacked[:, 0].min())
        min_y = float(stacked[:, 1].min())
        for cell in accepted:
            cell.pose[0, 2] -= min_x
            cell.pose[1, 2] -= min_y
    return frozen, cover


def _corners(pose: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    return np.vstack([_apply(pose, p) for p in pts])


def _aabb(pose: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    pts = _corners(pose, width, height)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _apply_manual_offsets(
    cells: list[_Cell],
    row_offsets: dict[int, tuple[float, float]] | None,
    col_offsets: dict[int, tuple[float, float]] | None,
    cell_offsets: dict[tuple[int, int], tuple[float, float]] | None,
) -> None:
    if not row_offsets and not col_offsets and not cell_offsets:
        return
    for cell in cells:
        if not cell.accepted:
            continue
        if row_offsets and cell.row in row_offsets:
            cell.pose[0, 2] += float(row_offsets[cell.row][0])
            cell.pose[1, 2] += float(row_offsets[cell.row][1])
        if col_offsets and cell.col in col_offsets:
            cell.pose[0, 2] += float(col_offsets[cell.col][0])
            cell.pose[1, 2] += float(col_offsets[cell.col][1])
        if cell_offsets and (cell.row, cell.col) in cell_offsets:
            ox, oy = cell_offsets[(cell.row, cell.col)]
            cell.pose[0, 2] += max(-10.0, min(10.0, float(ox)))
            cell.pose[1, 2] += max(-10.0, min(10.0, float(oy)))


def _render_matrix(pose: np.ndarray, min_x: float, min_y: float, scale: float) -> np.ndarray:
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = pose[:, :2] * scale
    matrix[0, 2] = (pose[0, 2] - min_x) * scale
    matrix[1, 2] = (pose[1, 2] - min_y) * scale
    return matrix


def _warp_tile(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    interp = cv2.INTER_LINEAR
    if abs(float(matrix[0, 1])) < 1e-6 and abs(float(matrix[1, 0])) < 1e-6 and abs(float(matrix[0, 0])) < 0.999:
        interp = cv2.INTER_AREA
    warped = cv2.warpAffine(
        image,
        matrix.astype(np.float32),
        size,
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    foot = cv2.warpAffine(
        np.full(image.shape[:2], 255, np.uint8),
        matrix.astype(np.float32),
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, foot


def _render(cells: list[_Cell], images: list[np.ndarray], *, max_side: int, render_mode: str) -> np.ndarray:
    accepted = [cell for cell in cells if cell.accepted]
    if not accepted:
        raise ValueError("没有可放置的照片")
    points = []
    for cell in accepted:
        ih, iw = images[cell.index].shape[:2]
        points.append(_corners(cell.pose, iw, ih))
    stacked = np.vstack(points)
    min_x = float(stacked[:, 0].min())
    min_y = float(stacked[:, 1].min())
    max_x = float(stacked[:, 0].max())
    max_y = float(stacked[:, 1].max())
    scale = min(1.0, max_side / float(max(max_x - min_x, max_y - min_y, 1.0)))
    out_w = max(1, int(round((max_x - min_x) * scale)))
    out_h = max(1, int(round((max_y - min_y) * scale)))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if render_mode == "top":
        for cell in accepted:
            image = images[cell.index]
            matrix = _render_matrix(cell.pose, min_x, min_y, scale)
            warped, foot = _warp_tile(image, matrix, (out_w, out_h))
            sel = foot > 0
            canvas[sel] = warped[sel]
        return canvas

    # 直线 / L 型 cell 接缝。只使用已接受位姿，不再回改。
    x_bounds: dict[int, list[float]] = {}
    row_top: dict[int, float] = {}
    row_bottom: dict[int, float] = {}
    rows = sorted({cell.row for cell in accepted})
    for row in rows:
        row_cells = [cell for cell in accepted if cell.row == row]
        row_cells.sort(key=lambda cell: _aabb(cell.pose, images[cell.index].shape[1], images[cell.index].shape[0])[0])
        boxes = [_aabb(cell.pose, images[cell.index].shape[1], images[cell.index].shape[0]) for cell in row_cells]
        bounds = [boxes[0][0]]
        for prev, nxt in zip(boxes, boxes[1:]):
            if nxt[0] < prev[2]:
                bounds.append((nxt[0] + prev[2]) / 2.0)
            else:
                bounds.append((prev[2] + nxt[0]) / 2.0)
        bounds.append(boxes[-1][2])
        x_bounds[row] = bounds
        row_top[row] = float(np.mean([box[1] for box in boxes]))
        row_bottom[row] = float(np.mean([box[3] for box in boxes]))
    y_bounds = [row_top[rows[0]]]
    for above, below in zip(rows, rows[1:]):
        if row_top[below] < row_bottom[above]:
            y_bounds.append((row_top[below] + row_bottom[above]) / 2.0)
        else:
            y_bounds.append((row_bottom[above] + row_top[below]) / 2.0)
    y_bounds.append(row_bottom[rows[-1]])

    for row_i, row in enumerate(rows):
        row_cells = [cell for cell in accepted if cell.row == row]
        row_cells.sort(key=lambda cell: _aabb(cell.pose, images[cell.index].shape[1], images[cell.index].shape[0])[0])
        for col_i, cell in enumerate(row_cells):
            image = images[cell.index]
            matrix = _render_matrix(cell.pose, min_x, min_y, scale)
            warped, foot = _warp_tile(image, matrix, (out_w, out_h))
            pad = 2
            x0 = max(0, int(round((x_bounds[row][col_i] - min_x) * scale)) - pad)
            x1 = min(out_w, int(round((x_bounds[row][col_i + 1] - min_x) * scale)) + pad)
            y0 = max(0, int(round((y_bounds[row_i] - min_y) * scale)) - pad)
            y1 = min(out_h, int(round((y_bounds[row_i + 1] - min_y) * scale)) + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            sel = np.zeros((out_h, out_w), dtype=bool)
            sel[y0:y1, x0:x1] = True
            sel &= foot > 0
            canvas[sel] = warped[sel]
    return canvas


def stitch_grid_folder(
    paths,
    *,
    rows: int,
    cols: int,
    serpentine: bool = False,
    overlap: float = 0.3,
    step: tuple[float, float] | None = None,
    max_side: int = 3200,
    row_offsets: dict[int, tuple[float, float]] | None = None,
    col_offsets: dict[int, tuple[float, float]] | None = None,
    cell_offsets: dict[tuple[int, int], tuple[float, float]] | None = None,
    render_mode: str = "cell",
) -> GridStitchResult:
    from pathlib import Path

    files = [Path(p) for p in paths]
    images: list[np.ndarray] = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError("没有可读照片")
    height, width = images[0].shape[:2]
    if step is None:
        step = (width * (1.0 - overlap), height * (1.0 - overlap))
    dx, dy = float(step[0]), float(step[1])
    cells = _assign_cells(len(images), rows, cols, dx, dy, serpentine)
    if not cells:
        raise ValueError("网格没有放置任何照片")
    _calibrate_grid(cells, images, dx, dy)
    _apply_manual_offsets(cells, row_offsets, col_offsets, cell_offsets)
    canvas = _render(cells, images, max_side=max_side, render_mode=render_mode)
    poses: list[np.ndarray | None] = [None] * len(images)
    placed = 0
    for cell in cells:
        if cell.accepted:
            poses[cell.index] = cell.pose.astype(np.float64).copy()
            placed += 1
    rejected = len(cells) - placed
    seam = "上层覆盖" if render_mode == "top" else "直线接缝"
    message = f"{placed}/{len(images)} 张 / 步长 {dx:.1f},{dy:.1f} / {seam} / 交集星座相似变换"
    if rejected:
        message += f" / 拒绝 {rejected} 张"
    return GridStitchResult(
        image=canvas,
        placed=placed,
        total=len(images),
        step=(dx, dy),
        message=message,
        poses=poses,
    )
