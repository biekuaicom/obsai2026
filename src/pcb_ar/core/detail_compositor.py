"""基于 OpenCV detail 的接缝/曝光/多频段合成。

几何仍由我们自己的关键帧位姿提供；这里只负责：
* ExposureCompensator 均衡亮度；
* DpSeamFinder 选内容最优接缝；
* MultiBandBlender 做多频段融合。

不采用 cv2.Stitcher 的自动位姿估计，因为 PCB 重复引脚会让它认错。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetailPiece:
    image: np.ndarray
    mask: np.ndarray
    matrix: np.ndarray  # image pixels -> output canvas
    name: str = ""


def composite_detail(
    pieces: list[DetailPiece],
    width: int,
    height: int,
    *,
    num_bands: int = 5,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not pieces:
        raise ValueError("没有可合成的局部图")
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    corners: list[tuple[int, int]] = []
    for piece in pieces:
        matrix = np.asarray(piece.matrix, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        warped = cv2.warpAffine(
            piece.image,
            matrix[:2, :].astype(np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            piece.mask,
            matrix[:2, :].astype(np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = ((mask > 0).astype(np.uint8) * 255)
        x, y, w, h = cv2.boundingRect(mask)
        if w < 8 or h < 8:
            continue
        roi = warped[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        images.append(np.ascontiguousarray(roi))
        masks.append(np.ascontiguousarray(roi_mask))
        corners.append((x, y))
    if not images:
        raise ValueError("所有局部图都为空")

    try:
        compensator = cv2.detail.ExposureCompensator_createDefault(
            cv2.detail.ExposureCompensator_GAIN_BLOCKS
        )
        compensator.feed(corners, images, masks)
        for i, (corner, mask) in enumerate(zip(corners, masks)):
            compensator.apply(i, corner, images[i], mask)
        exposure_msg = "曝光均衡"
    except cv2.error:
        exposure_msg = "曝光均衡跳过"

    try:
        seam = cv2.detail.DpSeamFinder("COLOR_GRAD")
        seam.find(images, corners, masks)
        seam_msg = "DP 接缝"
    except cv2.error:
        seam_msg = "接缝跳过"

    try:
        blender = cv2.detail.MultiBandBlender()
        blender.setNumBands(int(num_bands))
    except cv2.error:
        blender = cv2.detail.MultiBandBlender()
    blender.prepare((0, 0, width, height))
    for image, mask, corner in zip(images, masks, corners):
        blender.feed(image, mask, corner)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result_mask = np.zeros((height, width), dtype=np.uint8)
    try:
        result, result_mask = blender.blend(result, result_mask)
    except cv2.error as exc:
        raise RuntimeError(f"多频段融合失败：{exc}") from exc
    return result, result_mask, f"{exposure_msg} + {seam_msg} + 多频段"
