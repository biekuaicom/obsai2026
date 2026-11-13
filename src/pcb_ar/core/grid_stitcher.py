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


def _calibrate_grid(cells, images, dx: float, dy: float, overlap: float) -> None:
    """冻结拼图 + 实际交集 + 单次相似变换。"""
    if not cells:
        return
    h, w = images[0].shape[:2]
    W = int((len(images) + 2) * w)
    H = int((len(images) + 2) * h)
    frozen = np.zeros((H, W, 3), np.uint8)
    cover = np.zeros((H, W), np.uint8)
    first = cells[0]
    first[3][0] = float(w)
    first[3][1] = float(h)
    frozen[h:h+h, w:w+w] = images[0]
    cover[h:h+h, w:w+w] = 255

    for t in cells[1:]:
        r, c = t[1], t[2]
        rough_x = float(w + c * dx)
        rough_y = float(h + r * dy)
        x0 = max(0, int(rough_x))
        y0 = max(0, int(rough_y))
        x1 = min(W, int(rough_x) + w)
        y1 = min(H, int(rough_y) + h)
        if x1 - x0 < 50 or y1 - y0 < 50:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        cm = cover[y0:y1, x0:x1]
        if int(np.count_nonzero(cm)) < 500:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        frozen_crop = frozen[y0:y1, x0:x1].copy()
        frozen_crop[cm == 0] = 0
        new_crop = images[t[0]][0:y1-y0, 0:x1-x0]
        ga = cv2.cvtColor(frozen_crop, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=2500)
        ka, da = orb.detectAndCompute(ga, None)
        kb, db = orb.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 10 or len(kb) < 10:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            knn = bf.knnMatch(da, db, k=2)
        except cv2.error:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if len(good) < 10:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        pa = np.float32([ka[m.queryIdx].pt for m in good])
        pb = np.float32([kb[m.trainIdx].pt for m in good])
        M, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000)
        if M is None or mask is None or int(np.count_nonzero(mask)) < 10:
            t[3][0], t[3][1] = rough_x, rough_y
            continue
        # M: frozen crop -> new crop. 新图原点 = C 起点 - M 平移。
        px = x0 - float(M[0, 2])
        py = y0 - float(M[1, 2])
        if abs(px - rough_x) > max(60.0, 0.6 * w) or abs(py - rough_y) > max(60.0, 0.6 * h):
            t[3][0], t[3][1] = rough_x, rough_y
        else:
            t[3][0], t[3][1] = px, py
        xx = int(round(t[3][0])); yy = int(round(t[3][1]))
        if 0 <= xx < W - w and 0 <= yy < H - h:
            frozen[yy:yy+h, xx:xx+w] = images[t[0]]
            cover[yy:yy+h, xx:xx+w] = 255

    min_x = min(float(t[3][0]) for t in cells)
    min_y = min(float(t[3][1]) for t in cells)
    for t in cells:
        t[3][0] -= min_x
        t[3][1] -= min_y


def stitch_grid_folder(
    paths: Sequence[str | Path],
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
    _calibrate_grid(cells, images, dx, dy, overlap)
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
            # 合成用完整矩形：不在每张图上修边，避免 A 修右边、B 修左边
            # 导致接缝中间少一列。物理外边缘留给最终输出统一处理。
            footprint = np.full(image.shape[:2], 255, dtype=np.uint8)
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
        # 只向外补 2px，消掉接缝少列；不移动整段接缝。
        pad = 2
        x0 = max(0, int(round((x0w - min_x) * scale)) - pad)
        x1 = min(width, int(round((x1w - min_x) * scale)) + pad)
        y0 = max(0, int(round((y0w - min_y) * scale)) - pad)
        y1 = min(height, int(round((y1w - min_y) * scale)) + pad)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = warped[y0:y1, x0:x1]
    return GridStitchResult(
        image=canvas,
        placed=len(cells),
        total=len(images),
        step=(dx, dy),
        message=f"{len(cells)}/{len(images)} 张 / 步长 {dx:.1f},{dy:.1f} / 直线接缝",
    )
