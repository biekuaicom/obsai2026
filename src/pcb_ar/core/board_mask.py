"""从关键帧里分离 PCB 板面，排除白纸/桌面/背景。

这一步很重要：只按“非黑”取有效像素时，白色纸张也会被算进覆盖和拼接，
总览边缘就会出现白色锯齿条，红区也会盖到板外。板面 mask 让覆盖、红区和
混合只发生在 PCB 上。
"""
from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """返回 uint8 0/255 的板面 mask；失败时退回非黑区域。"""
    if image is None or image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    h, w = image.shape[:2]
    # 不确定时整张图都算板面；宁可保留纸边，也不能切掉板内白色区域。
    fallback = np.full((h, w), 255, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # 绿色板/彩色丝印/黑色元件；白纸是低饱和高亮，被排除。
    cand = (((s > 45) & (v > 18)) | (v < 95)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    try:
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(cand, 8)
        if n <= 1:
            return fallback
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        if area < int(0.05 * h * w):
            return fallback
        mask = (labels == largest).astype(np.uint8) * 255
    except cv2.error:
        return fallback
    # 只保留板的外轮廓；外轮廓内部全部有效。
    # 白色丝印、白色连接器、金属亮面都属于板内，不能按颜色删除。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    if int(np.count_nonzero(filled)) < int(0.03 * h * w):
        return fallback
    return filled
