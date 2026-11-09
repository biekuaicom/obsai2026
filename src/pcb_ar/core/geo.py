"""二维几何、仿射矩阵与标注框工具。

坐标约定
--------
* 全局图坐标 / 实时帧坐标均为图像像素坐标，x 向右、y 向下。
* ``M`` 为 2x3 仿射矩阵，方向为 **全局图坐标 -> 实时帧坐标**。
* 旋转角按图像坐标系（y 向下）使用 ``atan2`` 计算，角度正方向为视觉顺时针。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w < 0:
            object.__setattr__(self, "w", 0.0)
        if self.h < 0:
            object.__setattr__(self, "h", 0.0)

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def is_empty(self) -> bool:
        return self.w <= 0.5 or self.h <= 0.5

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]

    def as_int_list(self) -> list[int]:
        return [int(round(v)) for v in self.as_list()]

    def corners(self) -> np.ndarray:
        return np.array(
            [
                [self.x, self.y],
                [self.x2, self.y],
                [self.x2, self.y2],
                [self.x, self.y2],
            ],
            dtype=np.float32,
        )

    def expanded(self, margin: float) -> "BBox":
        return BBox(self.x - margin, self.y - margin, self.w + 2 * margin, self.h + 2 * margin)

    def clipped(self, width: int | float, height: int | float) -> "BBox":
        x1 = max(0.0, min(float(width), self.x))
        y1 = max(0.0, min(float(height), self.y))
        x2 = max(x1, min(float(width), self.x2))
        y2 = max(y1, min(float(height), self.y2))
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x2 and self.y <= y <= self.y2

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x2 < other.x
            or other.x2 < self.x
            or self.y2 < other.y
            or other.y2 < self.y
        )

    @classmethod
    def from_sequence(cls, values) -> "BBox":
        if isinstance(values, BBox):
            return values
        if hasattr(values, "x") and hasattr(values, "y") and hasattr(values, "w"):
            return cls(float(values.x), float(values.y), float(values.w), float(values.h))
        seq = list(values)
        if len(seq) < 4:
            raise ValueError(f"BBox 需要 4 个数，收到 {seq!r}")
        return cls(*(float(v) for v in seq[:4]))


def bbox_corners(box: BBox) -> np.ndarray:
    return box.corners()


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """对 Nx2 点应用 2x3 / 3x3 仿射矩阵，返回 Nx2 float32。"""
    m = as_affine(matrix)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts, ones])
    out = homo @ m.T
    return out.astype(np.float32)


def transform_bbox(matrix: np.ndarray, box: BBox) -> BBox:
    """返回变换后的水平外接框（用于筛选/定位，不用于绘制旋转框）。"""
    pts = apply_affine(matrix, bbox_corners(box))
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return BBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    out = apply_affine(matrix, np.array([point], dtype=np.float32))[0]
    return float(out[0]), float(out[1])


def as_affine(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape == (3, 3):
        m = m[:2, :]
    if m.shape != (2, 3):
        raise ValueError(f"仿射矩阵形状应为 2x3 或 3x3，收到 {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("仿射矩阵包含非有限值")
    return m


def as_homography(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = m
        return h
    if m.shape == (3, 3):
        return m.copy()
    raise ValueError(f"矩阵形状应为 2x3 或 3x3，收到 {m.shape}")


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    inv = cv2.invertAffineTransform(as_affine(matrix))
    return inv.astype(np.float32)


def matrix_scale(matrix: np.ndarray) -> float:
    m = as_affine(matrix)
    return float(math.sqrt(max(0.0, abs(float(np.linalg.det(m[:, :2]))))))


def matrix_rotation_deg(matrix: np.ndarray) -> float:
    m = as_affine(matrix)
    return float(math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0]))))


def matrix_translation(matrix: np.ndarray) -> tuple[float, float]:
    m = as_affine(matrix)
    return float(m[0, 2]), float(m[1, 2])


def is_valid_similarity(matrix: np.ndarray, min_scale: float = 1e-4, max_scale: float = 50.0) -> bool:
    try:
        m = as_affine(matrix)
    except ValueError:
        return False
    a, b = float(m[0, 0]), float(m[0, 1])
    c, d = float(m[1, 0]), float(m[1, 1])
    det = a * d - b * c
    if det <= 1e-8:
        return False
    # 近似等比例：两个基向量长度以及正交性
    len1 = math.hypot(a, c)
    len2 = math.hypot(b, d)
    if len1 < min_scale or len1 > max_scale or len2 < min_scale or len2 > max_scale:
        return False
    if abs(len1 - len2) / max(len1, len2) > 0.12:
        return False
    dot = abs(a * b + c * d) / max(len1 * len2, EPS)
    return dot < 0.12


def decompose_similarity(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    """将近似相似变换分解为 (scale, theta_rad, translation)，使用 SVD 抗噪。"""
    m = as_affine(matrix).astype(np.float64)
    a = m[:, :2]
    u, s, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        # 修正反射，保证旋转矩阵行列式为 +1
        d = np.diag([1.0, -1.0])
        r = u @ d @ vt
        s = np.array([s[0], -s[1]])
    scale = float(np.mean(np.abs(s)))
    theta = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return scale, theta, m[:, 2].astype(np.float64)


def build_similarity(scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t, scale * cos_t, ty],
        ],
        dtype=np.float32,
    )


def blend_similarity(
    previous: np.ndarray | None,
    current: np.ndarray,
    alpha: float,
    pivot: tuple[float, float] | None = None,
) -> np.ndarray:
    """对旋转/缩放/平移做指数平滑。

    平移不直接对矩阵的 ``t`` 插值，而是在 ``pivot``（通常是全局图中心）处
    对映射后的位置插值。这样旋转/缩放变化不会造成整幅标注的明显拖尾。
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    if previous is None or alpha >= 0.999:
        return as_affine(current).copy()
    if alpha <= 0.001:
        return as_affine(previous).copy()
    ps, pt, ptl = decompose_similarity(previous)
    cs, ct, ctl = decompose_similarity(current)
    delta = (ct - pt + math.pi) % (2 * math.pi) - math.pi
    theta = pt + alpha * delta
    scale = math.exp((1 - alpha) * math.log(max(ps, 1e-6)) + alpha * math.log(max(cs, 1e-6)))
    if pivot is None:
        return build_similarity(scale, theta, float(ptl[0]), float(ptl[1]))
    px, py = float(pivot[0]), float(pivot[1])
    prev_a = as_affine(previous)[:, :2]
    curr_a = as_affine(current)[:, :2]
    prev_p = prev_a @ np.array([px, py]) + ptl
    curr_p = curr_a @ np.array([px, py]) + ctl
    smooth_p = (1 - alpha) * prev_p + alpha * curr_p
    base = build_similarity(scale, theta, 0.0, 0.0)
    smooth_a = base[:, :2]
    smooth_t = smooth_p - smooth_a @ np.array([px, py])
    base[:, 2] = smooth_t
    return base.astype(np.float32)


def bbox_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area + b.area - inter
    return float(inter / union) if union > EPS else 0.0


def bbox_contains_ratio(inner: BBox, outer: BBox) -> float:
    if inner.area <= EPS:
        return 0.0
    x1, y1 = max(inner.x, outer.x), max(inner.y, outer.y)
    x2, y2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1) / inner.area)


def non_max_suppression(boxes: list[BBox], iou_threshold: float = 0.35) -> list[int]:
    """按面积从大到小做简单 NMS，返回保留索引。"""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].area, reverse=True)
    keep: list[int] = []
    for idx in order:
        if all(bbox_iou(boxes[idx], boxes[j]) <= iou_threshold for j in keep):
            keep.append(idx)
    return keep


def rotation_matrix_about(center: tuple[float, float], angle_deg: float, scale: float = 1.0) -> np.ndarray:
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_deg), float(scale))
    return m.astype(np.float32)


def map_points_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, as_homography(matrix))
    return out.reshape(-1, 2)


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
