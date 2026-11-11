"""坐标网格驱动的离线拼接。

适用于显微镜平台/手动网格拍照：照片按行列拍摄，重叠区只做小范围
相位相关微调，不做全局 ORB 盲拼。这样避免 PCB 重复焊盘/引脚把全局
位姿拉错。

如果拍摄顺序乱了，可把 `serpentine` 设为 False/True 或传入每张图的
(row, col) 坐标；没有坐标的照片不应强行塞入。
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
class GridStitchResult:
    image: np.ndarray
    placed: int
    total: int
    step: tuple[float, float]
    message: str


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """返回 b 相对 a 的平移和响应；输入大小相同。"""
    if a.shape != b.shape or a.size == 0:
        return 0.0, 0.0, 0.0
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ga * win, gb * win)
    except cv2.error:
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(response)


def _refine_pair(
    previous: np.ndarray,
    current: np.ndarray,
    expected: tuple[float, float],
    previous_pos: tuple[float, float],
) -> tuple[float, float] | None:
    """已知上一张位置和预期步长，在重叠区求小修正。"""
    h, w = current.shape[:2]
    dx, dy = expected
    # 重叠区域（按预期偏移裁切）。
    if abs(dx) >= w or abs(dy) >= h:
        return None
    if dx >= 0:
        a = previous[:, w - int(dx) :]
    else:
        a = previous[:, : int(-dx)]
    if dy >= 0:
        a = a[: h - int(dy), :]
    else:
        a = a[int(-dy) :, :]
    # current 对应区域
    cx0 = 0 if dx >= 0 else int(-dx)
    cy0 = 0 if dy >= 0 else int(-dy)
    ov_w = int(w - abs(dx))
    ov_h = int(h - abs(dy))
    b = current[cy0 : cy0 + ov_h, cx0 : cx0 + ov_w]
    a = a[:ov_h, :ov_w]
    if a.size == 0 or b.size == 0:
        return None
    shift_x, shift_y, response = _phase_shift(a, b)
    if not np.isfinite(shift_x) or not np.isfinite(shift_y) or response < 0.02:
        return None
    if abs(shift_x) > 40 or abs(shift_y) > 40:
        return None
    return previous_pos[0] + dx + shift_x, previous_pos[1] + dy + shift_y


def stitch_grid_folder(
    paths: Sequence[str | Path],
    *,
    rows: int,
    cols: int,
    serpentine: bool = True,
    overlap: float = 0.3,
    step: tuple[float, float] | None = None,
    max_side: int = 3200,
) -> GridStitchResult:
    files = [Path(p) for p in paths]
    images = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError("没有可读照片")
    h, w = images[0].shape[:2]
    if step is None:
        step = (w * (1.0 - overlap), h * (1.0 - overlap))
    dx, dy = float(step[0]), float(step[1])
    positions: list[np.ndarray] = []
    index = 0
    for r in range(rows):
        order = range(cols)
        if serpentine and r % 2 == 1:
            order = range(cols - 1, -1, -1)
        for c in order:
            if index >= len(images):
                break
            x = c * dx
            y = r * dy
            # 左侧/上方已放置邻居做局部相位相关，不做全局优化。
            if c > 0:
                prev_idx = index - 1
                if 0 <= prev_idx < len(positions):
                    refined = _refine_pair(
                        images[prev_idx],
                        images[index],
                        (dx, dy),
                        (float(positions[prev_idx][0]), float(positions[prev_idx][1])),
                    )
                    if refined is not None:
                        x, y = refined
            positions.append(np.array([x, y], dtype=np.float64))
            index += 1
        if index >= len(images):
            break
    if not positions:
        raise ValueError("网格没有放置任何照片")

    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_x = max(p[0] for p in positions) + w
    max_y = max(p[1] for p in positions) + h
    scale = min(1.0, max_side / float(max(max_x - min_x, max_y - min_y)))
    width = max(1, int(round((max_x - min_x) * scale)))
    height = max(1, int(round((max_y - min_y) * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    best = np.zeros((height, width), dtype=np.float32)
    for image, pos in zip(images, positions):
        matrix = np.array(
            [
                [scale, 0.0, (pos[0] - min_x) * scale],
                [0.0, scale, (pos[1] - min_y) * scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        warped = cv2.warpAffine(
            image,
            matrix[:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpAffine(
            foreground_mask(image),
            matrix[:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ) > 0
        if not np.any(mask):
            continue
        x, y, ww, hh = cv2.boundingRect(mask.astype(np.uint8))
        dist = np.zeros((height, width), dtype=np.float32)
        dist[y : y + hh, x : x + ww] = cv2.distanceTransform(
            mask[y : y + hh, x : x + ww].astype(np.uint8), cv2.DIST_L2, 5
        )
        better = mask & (dist > best + 0.3)
        canvas[better] = warped[better]
        best = np.maximum(best, dist)
    return GridStitchResult(
        image=canvas,
        placed=min(index, len(images)),
        total=len(images),
        step=(dx, dy),
        message=f"{min(index, len(images))}/{len(images)} 张 / 步长 {dx:.1f},{dy:.1f}",
    )
