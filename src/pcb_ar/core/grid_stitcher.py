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


def _calibrate_grid(cells, images, dx: float, dy: float) -> None:
    """用 C 区相位相关求“整行/整列”的偏移量，补偿红色覆盖与实际步长的容差。

    模型：
      X(r,c) = c*dx + ColX[c] + RowX[r]
      Y(r,c) = r*dy + RowY[r] + ColY[c]
    只修正整行/整列，不把每张图自由乱拖。
    """
    by_row = {}
    by_col = {}
    for t in cells:
        by_row.setdefault(t[1], []).append(t)
        by_col.setdefault(t[2], []).append(t)
    # 行内相邻：测列方向偏移
    colx: dict[int, list[float]] = {}
    coly: dict[int, list[float]] = {}
    for r, items in by_row.items():
        items.sort(key=lambda t: t[3][0])
        for left, right in zip(items, items[1:]):
            expected = (float(dx), 0.0)
            refined = _refine_pair(
                images[left[0]], images[right[0]], expected,
                (float(left[3][0]), float(left[3][1])),
            )
            if refined is None:
                continue
            meas_dx = refined[0] - float(left[3][0])
            meas_dy = refined[1] - float(left[3][1])
            c0, c1 = left[2], right[2]
            colx.setdefault(c0, []).append(0.0)
            colx.setdefault(c1, []).append(meas_dx - dx)
            coly.setdefault(c0, []).append(0.0)
            coly.setdefault(c1, []).append(meas_dy)
    # 列内相邻：测行方向偏移
    rowx: dict[int, list[float]] = {}
    rowy: dict[int, list[float]] = {}
    for c, items in by_col.items():
        items.sort(key=lambda t: t[3][1])
        for top, bottom in zip(items, items[1:]):
            expected = (0.0, float(dy))
            refined = _refine_pair(
                images[top[0]], images[bottom[0]], expected,
                (float(top[3][0]), float(top[3][1])),
            )
            if refined is None:
                continue
            meas_dx = refined[0] - float(top[3][0])
            meas_dy = refined[1] - float(top[3][1])
            r0, r1 = top[1], bottom[1]
            rowx.setdefault(r0, []).append(0.0)
            rowx.setdefault(r1, []).append(meas_dx)
            rowy.setdefault(r0, []).append(0.0)
            rowy.setdefault(r1, []).append(meas_dy - dy)

    def cumulative(values: dict[int, list[float]], scale: float = 1.0) -> dict[int, float]:
        out = {0: 0.0}
        for idx in sorted(values):
            if idx == 0:
                continue
            prev = idx - 1
            res = float(np.median(values.get(idx, [0.0]))) * scale
            out[idx] = out.get(prev, 0.0) + res
        return out

    col_x = cumulative(colx)
    col_y = cumulative(coly)
    row_x = cumulative(rowx)
    row_y = cumulative(rowy)
    for t in cells:
        idx, r, c, pos = t
        pos[0] = c * dx + col_x.get(c, 0.0) + row_x.get(r, 0.0)
        pos[1] = r * dy + row_y.get(r, 0.0) + col_y.get(c, 0.0)


def stitch_grid_folder(
    paths: Sequence[str | Path],
    *,
    rows: int,
    cols: int,
    serpentine: bool = True,
    overlap: float = 0.3,
    step: tuple[float, float] | None = None,
    max_side: int = 3200,
    row_offsets: dict[int, tuple[float, float]] | None = None,
    col_offsets: dict[int, tuple[float, float]] | None = None,
    cell_offsets: dict[tuple[int, int], tuple[float, float]] | None = None,
    render_mode: str = "top",
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
    cells: list[tuple[int, int, int, np.ndarray]] = []
    index = 0
    for r in range(rows):
        order = range(cols)
        if serpentine and r % 2 == 1:
            order = range(cols - 1, -1, -1)
        for c in order:
            if index >= len(images):
                break
            cells.append((index, r, c, np.array([c * dx, r * dy], dtype=np.float64)))
            index += 1
        if index >= len(images):
            break
    if not cells:
        raise ValueError("网格没有放置任何照片")
    _calibrate_grid(cells, images, dx, dy)
    # 人工容差补偿：整行/整列平移，不破坏网格结构。
    if row_offsets or col_offsets:
        for idx, r, c, pos in cells:
            if row_offsets and r in row_offsets:
                pos[0] += float(row_offsets[r][0])
                pos[1] += float(row_offsets[r][1])
            if col_offsets and c in col_offsets:
                pos[0] += float(col_offsets[c][0])
                pos[1] += float(col_offsets[c][1])
            if cell_offsets and (r, c) in cell_offsets:
                # 单图局部微调只允许 ±10px，不能重排整格。
                ox = max(-10.0, min(10.0, float(cell_offsets[(r, c)][0])))
                oy = max(-10.0, min(10.0, float(cell_offsets[(r, c)][1])))
                pos[0] += ox
                pos[1] += oy

    if render_mode == "top":
        # 层级合成：后放的照片在上层，整张照片参与，不按 cell 裁掉。
        # 这样 B 图里完整圆不会被接缝多切掉；接缝落在照片矩形边缘（直线）。
        min_x = min(float(t[3][0]) for t in cells)
        min_y = min(float(t[3][1]) for t in cells)
        max_x = max(float(t[3][0]) + w for t in cells)
        max_y = max(float(t[3][1]) + h for t in cells)
        scale = min(1.0, max_side / float(max(max_x - min_x, max_y - min_y)))
        width = max(1, int(round((max_x - min_x) * scale)))
        height = max(1, int(round((max_y - min_y) * scale)))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, r, c, pos in cells:
            matrix = np.array(
                [[scale, 0, (pos[0] - min_x) * scale], [0, scale, (pos[1] - min_y) * scale], [0, 0, 1]],
                dtype=np.float64,
            )
            warped = cv2.warpAffine(
                images[idx], matrix[:2].astype(np.float32), (width, height),
                flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
            )
            # 只修剪物理外边缘 2px（坏边），不切重叠区内部。
            footprint = np.zeros(image.shape[:2], dtype=np.uint8)
            b = 2
            footprint[b : image.shape[0] - b, b : image.shape[1] - b] = 255
            fmask = cv2.warpAffine(
                footprint, matrix[:2].astype(np.float32), (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0
            canvas[fmask] = warped[fmask]
        return GridStitchResult(
            image=canvas,
            placed=len(cells),
            total=len(images),
            step=(dx, dy),
            message=f"{len(cells)}/{len(images)} 张 / 步长 {dx:.1f},{dy:.1f} / 上层覆盖",
        )

    # 每一行/列用中点作为接缝：严格垂直线/水平线/L 线，不做距离场曲线。
    x_bounds: dict[int, list[float]] = {}
    row_y: dict[int, float] = {}
    for r in range(rows):
        row_cells = sorted([x for x in cells if x[1] == r], key=lambda t: t[3][0])
        if not row_cells:
            continue
        xs = [float(t[3][0]) for t in row_cells]
        # 外侧接缝在首/末照片边缘；内部接缝在重叠区中点（不是图心中点）。
        bounds = [xs[0]]
        for a, b in zip(xs, xs[1:]):
            if b < a + w:
                bounds.append((b + a + w) / 2.0)
            else:
                bounds.append((a + b) / 2.0)
        bounds.append(xs[-1] + w)
        x_bounds[r] = bounds
        row_y[r] = float(np.mean([t[3][1] for t in row_cells]))
    row_ids = sorted(row_y)
    y_bounds: list[float] = []
    if row_ids:
        y_bounds = [row_y[row_ids[0]]]
        for a, b in zip(row_ids, row_ids[1:]):
            ya, yb = row_y[a], row_y[b]
            if yb < ya + h:
                y_bounds.append((yb + ya + h) / 2.0)
            else:
                y_bounds.append((ya + yb) / 2.0)
        y_bounds.append(row_y[row_ids[-1]] + h)

    min_x = min(x_bounds[r][0] for r in x_bounds)
    max_x = max(x_bounds[r][-1] for r in x_bounds)
    min_y = y_bounds[0]
    max_y = y_bounds[-1]
    scale = min(1.0, max_side / float(max(max_x - min_x, max_y - min_y)))
    width = max(1, int(round((max_x - min_x) * scale)))
    height = max(1, int(round((max_y - min_y) * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, r, c, pos in cells:
        image = images[idx]
        row_cells = sorted([t for t in cells if t[1] == r], key=lambda t: t[3][0])
        ci = next((k for k, t in enumerate(row_cells) if t[0] == idx), None)
        if ci is None:
            continue
        x0w = x_bounds[r][ci]
        x1w = x_bounds[r][ci + 1]
        ri = row_ids.index(r)
        y0w = y_bounds[ri]
        y1w = y_bounds[ri + 1]
        # warp 整图，只拷贝本 cell，cell 外不参与。
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
        x0 = max(0, int(round((x0w - min_x) * scale)))
        x1 = min(width, int(round((x1w - min_x) * scale)))
        y0 = max(0, int(round((y0w - min_y) * scale)))
        y1 = min(height, int(round((y1w - min_y) * scale)))
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = warped[y0:y1, x0:x1]
    return GridStitchResult(
        image=canvas,
        placed=len(cells),
        total=len(images),
        step=(dx, dy),
        message=f"{len(cells)}/{len(images)} 张 / 步长 {dx:.1f},{dy:.1f} / 直线接缝",
    )
