"""重叠区元素分割与对齐。

这不是深度学习分割模型，而是一个可替换的“经典分割基线”：
* 元件本体/焊盘/孔/走线复用结构基元检测；
* 文本用 MSER 找小连通区域；
* 大色块用 HSV 连通域补充；
* 每个元素保存 bbox、中心、尺寸、方向、颜色和局部灰度描述子。

后续可以把 `extract_elements` 替换成真实分割模型（U-Net/YOLO/SAM），
匹配层和拼接层不需要改。元素级匹配用于给 ORB 位姿做二次校正。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pcb_ar.core.structural_map import extract_structural_map


ElementKind = str
KINDS = ("component", "pad", "hole", "line", "text", "color")


@dataclass
class Element:
    id: int
    kind: ElementKind
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    angle: float = 0.0
    area: int = 0
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    descriptor: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def size(self) -> float:
        return float(math.hypot(self.width, self.height))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "bbox": [int(v) for v in self.bbox],
            "center": [round(float(v), 2) for v in self.center],
            "angle": round(float(self.angle), 2),
            "area": int(self.area),
            "color": [round(float(v), 1) for v in self.color],
        }


@dataclass
class ElementSet:
    width: int
    height: int
    elements: list[Element] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {kind: 0 for kind in KINDS}
        for e in self.elements:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def describe(self) -> str:
        c = self.counts()
        return (
            f"元件 {c['component']} · 焊盘 {c['pad']} · 孔 {c['hole']} · "
            f"文本 {c['text']} · 线 {c['line']} · 色块 {c['color']}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "counts": self.counts(),
            "elements": [e.to_dict() for e in self.elements],
        }


@dataclass
class ElementMatch:
    ok: bool
    matrix: np.ndarray | None
    inliers: int
    confidence: float
    kinds: int = 0
    message: str = ""


def _patch_descriptor(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = bbox
    if w < 3 or h < 3:
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = cv2.resize(gray[y0:y1, x0:x1], (12, 12), interpolation=cv2.INTER_AREA)
    patch = patch.astype(np.float32)
    patch -= patch.mean()
    std = float(patch.std())
    if std > 1e-3:
        patch /= std
    return patch.reshape(-1)


def _make_element(
    idx: int,
    kind: str,
    bbox: tuple[int, int, int, int],
    image: np.ndarray,
    gray: np.ndarray,
    angle: float = 0.0,
) -> Element:
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    color = image[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) if x1 > x0 and y1 > y0 else (0, 0, 0)
    return Element(
        id=idx,
        kind=kind,
        bbox=(int(x), int(y), int(w), int(h)),
        center=(float(x + w / 2), float(y + h / 2)),
        angle=float(angle),
        area=int(max(1, w * h)),
        color=(float(color[0]), float(color[1]), float(color[2])),
        descriptor=_patch_descriptor(gray, bbox),
    )


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(max(1, min(aw * ah, bw * bh)))


def extract_elements(image: np.ndarray, *, max_text: int = 80) -> ElementSet:
    """经典分割基线：结构基元 + MSER 文本 + HSV 色块。"""
    if image is None or image.size == 0:
        return ElementSet(0, 0, [])
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elements: list[Element] = []
    # 结构基元：元件/焊盘/孔/线
    try:
        structural = extract_structural_map(image)
    except Exception:  # noqa: BLE001
        structural = None
    if structural is not None:
        for n in structural.nodes:
            if n.kind in ("component", "pad", "hole", "line"):
                elements.append(
                    _make_element(len(elements), n.kind, n.bbox, image, gray, n.angle)
                )
    # 文本：MSER 小连通区域，排除与元件大面积重叠的。
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        text_boxes: list[tuple[int, int, int, int]] = []
        for pts in regions:
            x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
            area = bw * bh
            if not (6 <= bw <= 70 and 5 <= bh <= 45 and 12 <= area <= 1800):
                continue
            aspect = bw / float(max(1, bh))
            if not (0.18 <= aspect <= 6.0):
                continue
            if any(_overlap_ratio((x, y, bw, bh), e.bbox) > 0.65 for e in elements if e.kind == "component"):
                continue
            text_boxes.append((x, y, bw, bh))
            if len(text_boxes) >= max_text:
                break
        # 去掉几乎重复的 MSER 框
        kept: list[tuple[int, int, int, int]] = []
        for box in text_boxes:
            if any(_overlap_ratio(box, old) > 0.75 for old in kept):
                continue
            kept.append(box)
            elements.append(_make_element(len(elements), "text", box, image, gray))
    except cv2.error:
        pass
    # 色块：HSV 中大面积、非板绿的区域，限制数量。
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 非绿色、饱和度较高的大块（黄色/蓝色/红色/白色丝印块）
        mask = cv2.inRange(hsv, (0, 40, 50), (179, 255, 255))
        green = cv2.inRange(hsv, (35, 35, 25), (95, 255, 255))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_count = 0
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(cnt))
            if area < 450:
                break
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 12 or bh < 8:
                continue
            if any(_overlap_ratio((x, y, bw, bh), e.bbox) > 0.55 for e in elements if e.kind == "component"):
                continue
            elements.append(_make_element(len(elements), "color", (x, y, bw, bh), image, gray))
            color_count += 1
            if color_count >= 20:
                break
    except cv2.error:
        pass
    for i, e in enumerate(elements):
        e.id = i
    return ElementSet(w, h, elements)


def _candidate_pairs(
    reference: ElementSet,
    query: ElementSet,
    initial: np.ndarray,
    tolerance: float,
) -> list[tuple[float, int, int]]:
    m = np.asarray(initial, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]])
    m = m.reshape(3, 3)
    pairs: list[tuple[float, int, int]] = []
    for qi, q in enumerate(query.elements):
        qp = m @ np.array([q.center[0], q.center[1], 1.0])
        if abs(qp[2]) < 1e-9:
            continue
        qx, qy = qp[0] / qp[2], qp[1] / qp[2]
        for ri, r in enumerate(reference.elements):
            if r.kind != q.kind:
                continue
            ratio = (q.width * q.height) / float(max(1, r.width * r.height))
            if not (0.3 <= ratio <= 3.0):
                continue
            d = math.hypot(qx - r.center[0], qy - r.center[1])
            if d <= tolerance:
                pairs.append((d, qi, ri))
    pairs.sort(key=lambda item: item[0])
    return pairs


def match_elements(
    reference: ElementSet,
    query: ElementSet,
    initial: np.ndarray,
    *,
    min_inliers: int = 6,
) -> ElementMatch:
    if not reference.elements or not query.elements:
        return ElementMatch(False, None, 0, 0.0, message="无元素")
    diag = math.hypot(reference.width, reference.height)
    tolerance = max(12.0, diag * 0.055)
    pairs = _candidate_pairs(reference, query, initial, tolerance)
    if len(pairs) < min_inliers:
        return ElementMatch(False, None, len(pairs), 0.0, message="元素对应不足")
    used_q: set[int] = set()
    used_r: set[int] = set()
    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    kinds: set[str] = set()
    for _d, qi, ri in pairs:
        if qi in used_q or ri in used_r:
            continue
        used_q.add(qi)
        used_r.add(ri)
        src.append(query.elements[qi].center)
        dst.append(reference.elements[ri].center)
        kinds.add(query.elements[qi].kind)
        if len(src) >= 80:
            break
    if len(src) < min_inliers:
        return ElementMatch(False, None, len(src), 0.0, message="元素对应不足")
    src_arr = np.float32(src).reshape(-1, 1, 2)
    dst_arr = np.float32(dst).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        src_arr,
        dst_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=4000,
        confidence=0.997,
    )
    if matrix is None or mask is None:
        return ElementMatch(False, None, 0, 0.0, message="元素 RANSAC 失败")
    inliers = int(np.count_nonzero(mask))
    if inliers < min_inliers:
        return ElementMatch(False, None, inliers, 0.0, message="元素内点不足")
    h = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)
    confidence = min(1.0, inliers / max(6.0, min(len(src), 14) * 0.75))
    return ElementMatch(
        True,
        h,
        inliers,
        float(confidence),
        len(kinds),
        f"元素对齐 {inliers} 个 / {len(kinds)} 类",
    )
