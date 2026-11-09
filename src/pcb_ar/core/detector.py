"""元件候选检测。

传统视觉检测器用于“冷启动”：
1. 利用阻焊层绿色与元件/焊盘的低绿通道优势，构造非板材掩膜；
2. 去掉纯白丝印（避免把标题识别成元件）；
3. 形态学开闭运算聚合元件本体与焊盘；
4. 连通域 + 尺寸/填充率/边缘过滤，输出待人工确认的候选框。

这不是语义识别，不会自动给出准确位号/参数。生产环境建议训练 YOLO 模型，
通过 ``YoloDetector`` 适配器替换本检测器；无论哪种方式，都必须保留人工核对流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pcb_ar.core.geo import BBox, non_max_suppression


class DetectorUnavailable(RuntimeError):
    pass


@dataclass
class Detection:
    bbox: BBox
    score: float = 0.0
    label: str = "候选元件"
    category: str = "候选元件"
    confidence: float = 0.0
    source: str = "classical"


@dataclass
class DetectorConfig:
    max_work_width: int = 2000
    min_area_ratio: float = 0.00014
    max_area_ratio: float = 0.075
    min_width: int = 24
    min_height: int = 14
    max_aspect: float = 9.0
    min_extent: float = 0.28
    edge_margin_ratio: float = 0.03
    open_kernel: int = 5
    close_kernel: int = 7
    max_candidates: int = 90
    green_dominance: int = 12
    dark_value: int = 40
    white_value: int = 208


def _working_mask(image: np.ndarray, config: DetectorConfig) -> np.ndarray:
    b, g, r = cv2.split(image)
    green_dom = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = ((green_dom < config.green_dominance) | (gray < config.dark_value)).astype(np.uint8) * 255
    # 纯白丝印/文字不是元件；保留银色焊盘（通常 < 200）
    white = (b >= config.white_value) & (g >= config.white_value) & (r >= config.white_value)
    mask[white] = 0
    if config.open_kernel > 0:
        k = np.ones((config.open_kernel, config.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if config.close_kernel > 0:
        k = np.ones((config.close_kernel, config.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_component_candidates(
    image: np.ndarray,
    config: DetectorConfig | None = None,
) -> list[Detection]:
    if image is None or image.size == 0:
        return []
    config = config or DetectorConfig()
    original_h, original_w = image.shape[:2]
    scale = min(1.0, config.max_work_width / float(original_w))
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image
    h, w = work.shape[:2]
    mask = _working_mask(work, config)
    # 深色板框会把整幅图连成一个外轮廓，因此必须取 RETR_LIST 才能看到内部元件。
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.min_area_ratio * original_w * original_h * scale * scale
    max_area = config.max_area_ratio * original_w * original_h * scale * scale
    margin_x = int(w * config.edge_margin_ratio)
    margin_y = int(h * config.edge_margin_ratio)
    candidates: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < config.min_width * scale or bh < config.min_height * scale:
            continue
        if bw > w * 0.45 or bh > h * 0.45:
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > config.max_aspect:
            continue
        extent = area / float(max(1, bw * bh))
        if extent < config.min_extent:
            continue
        if x < margin_x or y < margin_y or x + bw > w - margin_x or y + bh > h - margin_y:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity > 0.92 and 0.75 < bw / max(bh, 1) < 1.33:
            # 定位点/安装孔通常是接近完美的圆
            continue
        x0, y0 = int(round(x / scale)), int(round(y / scale))
        x1, y1 = int(round((x + bw) / scale)), int(round((y + bh) / scale))
        bbox = BBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        if bbox.x < 0 or bbox.y < 0 or bbox.x2 > original_w or bbox.y2 > original_h:
            continue
        extent_score = min(1.0, extent / 0.85)
        size_score = min(1.0, area / (min_area * 4.0))
        score = 0.45 * extent_score + 0.35 * size_score + 0.20 * min(1.0, 2.2 / aspect)
        candidates.append(Detection(bbox, float(score), confidence=float(score)))

    candidates.sort(key=lambda d: d.score, reverse=True)
    if len(candidates) > config.max_candidates:
        candidates = candidates[: config.max_candidates]
    keep = non_max_suppression([c.bbox for c in candidates], iou_threshold=0.35)
    candidates = [candidates[i] for i in keep]
    candidates.sort(key=lambda d: (round(d.bbox.y / 40), d.bbox.x))
    return candidates


class YoloDetector:
    """可选的 YOLO 适配器。

    使用前需在工作环境额外安装 ``ultralytics`` 并提供已训练的 PCB 权重。
    该适配器不随安装包捆绑，避免把 PyTorch 带入 1GB 级依赖。
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str] | None = None,
        confidence: float = 0.25,
        device: str = "",
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(
                "未安装 ultralytics。请仅在需要 YOLO 的开发环境中安装，"
                "或使用默认的传统候选检测器。"
            ) from exc
        path = Path(model_path)
        if not path.exists():
            raise DetectorUnavailable(f"YOLO 权重不存在：{path}")
        self.model = YOLO(str(path))
        self.class_names = class_names or []
        self.confidence = float(confidence)
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        kwargs: dict[str, Any] = {"conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(image, **kwargs)
        detections: list[Detection] = []
        for result in results:
            names = result.names or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                conf = float(box.conf[0].detach().cpu().item())
                cls_id = int(box.cls[0].detach().cpu().item())
                label = str(names.get(cls_id, cls_id))
                if self.class_names and label.isdigit() and int(label) < len(self.class_names):
                    label = self.class_names[int(label)]
                x0, y0, x1, y1 = xyxy
                detections.append(
                    Detection(
                        BBox(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                        score=conf,
                        label=label,
                        category=label,
                        confidence=conf,
                        source="yolo",
                    )
                )
        return detections
