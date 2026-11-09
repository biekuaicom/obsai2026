"""基于 PCB 外轮廓的板边校正。

轻量捕获的最后一步用绿色/彩色板体掩膜找最大外轮廓，拟合最小外接矩形，
再用单应变换把板子拉正。它只处理板体，背景不参与校正。

这不是 CAD 级几何修复：它修正整板的旋转/透视和边缘弯曲，不能凭空恢复
焊盘中心、走线宽度等绝对尺寸；局部拼接缝仍需要重叠区校验/回环闭合。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class RectifyError(RuntimeError):
    pass


@dataclass
class RectifyResult:
    image: np.ndarray
    applied: bool
    message: str
    homography: np.ndarray | None = None
    quad: np.ndarray | None = None
    source_size: tuple[int, int] = (0, 0)


def order_quad(points: np.ndarray) -> np.ndarray:
    """按“上边两点、下边两点”排序，适合人工摆放常见的 ±45° 内旋转。

    不用简单的 x+y/x-y：板子旋转几度后，这两种排序会把左右边误当成上下边，
    导致校正后的总览横竖颠倒。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def detect_board_quad(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """返回板体外接矩形四角（原图坐标）和说明。"""
    if image is None or image.size == 0:
        return None, "没有图像"
    h, w = image.shape[:2]
    if min(h, w) < 80:
        return None, "图像太小"
    scale = min(1.0, 900.0 / max(h, w))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 0.999 else image
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    b, g, r = cv2.split(work.astype(np.int16))
    green_dom = g - np.maximum(r, b)
    # 绿色阻焊是最常见的板体；同时接受高饱和的蓝/红板。用较大核闭运算
    # 跨过元件、丝印和焊盘，得到整块板体，而不是单个元件。
    colored = ((sat > 38) & (val > 22)).astype(np.uint8) * 255
    green = ((green_dom > 8) & (val > 20)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(colored, green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "未找到板体轮廓"
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 0.10 * float(work.shape[0] * work.shape[1]):
        return None, "板体区域太小，无法可靠校正"
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if min(rw, rh) < 60:
        return None, "板体轮廓过窄"
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_quad(box)
    if scale < 0.999:
        box = box / scale
    # 贴边通常意味着板子没有完整进入总览；只允许一侧贴边（常见于板子
    # 刚好顶到画面边缘），两侧以上则强行拉正会把板外背景也拉进来。
    touches = int(box[:, 0].min() < 2) + int(box[:, 1].min() < 2)
    touches += int(box[:, 0].max() > w - 3) + int(box[:, 1].max() > h - 3)
    if touches >= 2:
        return None, "板边未完整进入总览，跳过校正"
    return box, f"板体 {rw * scale:.0f}×{rh * scale:.0f}px"


def rectify_board(
    image: np.ndarray,
    *,
    max_side: int = 3200,
    max_pixels: int = 8_000_000,
    min_side: int = 80,
) -> RectifyResult:
    quad, message = detect_board_quad(image)
    if quad is None:
        return RectifyResult(image=image, applied=False, message=message, source_size=image.shape[:2])
    tl, tr, br, bl = quad
    width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    if width < min_side or height < min_side:
        return RectifyResult(image=image, applied=False, message="板体尺寸过小", quad=quad, source_size=image.shape[:2])
    scale = min(1.0, max_side / width, max_side / height, (max_pixels / max(1.0, width * height)) ** 0.5)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    homography = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        image,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return RectifyResult(
        image=warped,
        applied=True,
        message=f"已按板边校正为 {out_w}×{out_h}px",
        homography=homography,
        quad=quad,
        source_size=image.shape[:2],
    )
