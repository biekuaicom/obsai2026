"""Qt 与 NumPy/OpenCV 的轻量转换工具。"""
from __future__ import annotations

import re

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QImage, QTransform


def qimage_from_bgr(frame: np.ndarray) -> QImage:
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    # 拼图裁剪/ROI 结果经常是非连续视图；PyQt6 的 QImage 构造函数不接受
    # 这类 buffer 视图，必须先转成 C-contiguous 的 uint8 缓冲。
    if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    stride = int(frame.strides[0])
    image = QImage(frame.data, w, h, stride, QImage.Format.Format_BGR888)
    return image.copy()


def qtransform_from_affine(matrix: np.ndarray) -> QTransform:
    m = np.asarray(matrix, dtype=float)
    if m.shape == (3, 3):
        m = m[:2, :]
    if m.shape != (2, 3):
        raise ValueError(f"期望 2x3 矩阵，收到 {m.shape}")
    a, b, tx = m[0]
    c, d, ty = m[1]
    # Qt: x' = m11*x + m21*y + m31 ; y' = m12*x + m22*y + m32
    return QTransform(float(a), float(c), float(tx), float(b), float(d), float(ty), 0.0, 0.0, 1.0)


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    color = QColor(hex_color)
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


def scene_rect_from_points(points: np.ndarray) -> QRectF:
    pts = np.asarray(points)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return QRectF(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text or "")]


def qpoint(value: tuple[float, float]) -> QPointF:
    return QPointF(float(value[0]), float(value[1]))
