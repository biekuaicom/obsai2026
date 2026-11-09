"""合成 PCB 演示板与模拟显微镜画面。

演示板不依赖任何外部数据集或摄像头，用于：
* 首次启动时展示完整 AR 跟随效果；
* 自动化测试中的全局图/局部图配准与拼图；
* 打包后验证 OpenCV / Qt 资源是否完整。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcb_ar.core.geo import BBox
from pcb_ar.core.models import (
    STATUS_ISSUE,
    STATUS_PENDING,
    STATUS_VERIFIED,
    Component,
    Fiducial,
    new_component,
)
from pcb_ar.core.project import ProjectError, ProjectStore

DEMO_VERSION = 3
BOARD_WIDTH = 3200
BOARD_HEIGHT = 2200

# BGR 调色板
BOARD_GREEN = (58, 102, 35)
BOARD_DARK = (39, 77, 23)
TRACE_DARK = (45, 88, 42)
TRACE_LIGHT = (76, 128, 70)
COPPER = (56, 112, 152)
GOLD = (54, 146, 205)
SILVER = (166, 178, 182)
SILK = (228, 236, 235)
BODY_DARK = (31, 34, 39)
BODY_GRAY = (52, 57, 64)
BLUE_PLASTIC = (104, 77, 47)
TEXT_DIM = (176, 194, 184)


@dataclass(frozen=True)
class Placement:
    refdes: str
    category: str
    value: str
    package: str
    x: int
    y: int
    w: int
    h: int
    note: str = ""
    status: str = STATUS_PENDING


PLACEMENTS: tuple[Placement, ...] = (
    Placement("J1", "连接器", "12V-5.08mm", "5.08-2P", 180, 620, 170, 300, "电源输入"),
    Placement("J2", "连接器", "USB-C 5V", "USB-C-16P", 180, 1180, 180, 220, "USB 供电/数据"),
    Placement("J3", "连接器", "SWD 2x5", "2.54-10P", 430, 1810, 250, 110, "调试接口"),
    Placement("U1", "芯片", "STM32F407VGT6", "LQFP100", 850, 500, 380, 380, "主控 MCU", STATUS_VERIFIED),
    Placement("U2", "芯片", "ESP32-S3-WROOM", "MODULE", 1950, 520, 360, 360, "无线模组", STATUS_VERIFIED),
    Placement("U3", "芯片", "AMS1117-3.3", "SOT-223", 1470, 1090, 230, 160, "3.3V LDO", STATUS_VERIFIED),
    Placement("U4", "芯片", "MAX3232ESE", "SOIC-16", 2380, 1080, 250, 190, "RS-232 收发"),
    Placement("U5", "芯片", "LM2596S-5.0", "TO-263", 2060, 1560, 260, 180, "5V 降压"),
    Placement("Y1", "晶振", "8MHz", "HC-49S", 900, 1070, 170, 90, "主时钟", STATUS_VERIFIED),
    Placement("R1", "电阻", "10k", "0603", 560, 430, 110, 55, "BOOT 上拉"),
    Placement("R2", "电阻", "22R", "0603", 710, 430, 110, 55, "USB 串阻"),
    Placement("C1", "电容", "100nF", "0603", 560, 560, 110, 55, "去耦"),
    Placement("C2", "电容", "10uF", "0805", 710, 560, 120, 60, "去耦"),
    Placement("R3", "电阻", "4.7k", "0603", 1320, 500, 110, 55, "I2C 上拉", STATUS_VERIFIED),
    Placement("C3", "电容", "1uF", "0603", 1320, 620, 110, 55, "去耦"),
    Placement("R4", "电阻", "100R", "0603", 1320, 750, 110, 55, "LED 限流", STATUS_ISSUE),
    Placement("C4", "电容", "22pF", "0402", 860, 980, 100, 50, "晶振负载"),
    Placement("C5", "电容", "22pF", "0402", 1040, 980, 100, 50, "晶振负载"),
    Placement("C6", "电容", "100nF", "0603", 2700, 1080, 110, 55, "去耦"),
    Placement("R5", "电阻", "1k", "0603", 2400, 1370, 110, 55, "RX 上拉"),
    Placement("D1", "二极管", "SS34", "SMA", 2610, 1460, 130, 80, "防反接"),
    Placement("R6", "电阻", "10k", "0603", 1640, 1540, 110, 55, "反馈分压"),
    Placement("C7", "电容", "470uF", "CP-8x12", 1810, 1490, 150, 150, "输入储能", STATUS_ISSUE),
    Placement("L1", "电感", "33uH", "CD54", 2380, 1620, 150, 100, "Buck 电感"),
    Placement("J4", "连接器", "SENSOR 2x8", "2.54-16P", 2790, 560, 210, 300, "传感器接口"),
    Placement("J5", "连接器", "MOTOR 4P", "KF128-4P", 2790, 1470, 210, 280, "电机接口"),
    Placement("D2", "二极管", "LED-G", "0805", 1430, 1900, 120, 70, "运行指示"),
    Placement("SW1", "开关", "RESET", "TACT-6x6", 1640, 1900, 150, 90, "复位按键"),
)


def _text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = SILK,
    thickness: int = 1,
    center: bool = False,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _draw_via(image: np.ndarray, x: int, y: int, radius: int = 7) -> None:
    cv2.circle(image, (x, y), radius + 3, (42, 78, 50), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, GOLD, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), max(2, radius - 4), (20, 30, 33), -1, cv2.LINE_AA)


def _draw_fiducial(image: np.ndarray, x: int, y: int, radius: int = 24) -> None:
    cv2.circle(image, (x, y), radius + 12, (24, 49, 31), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 7, GOLD, 3, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, (231, 233, 220), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), max(4, radius // 3), (26, 35, 37), -1, cv2.LINE_AA)
    cv2.line(image, (x - radius - 15, y), (x + radius + 15, y), SILK, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - radius - 15), (x, y + radius + 15), SILK, 1, cv2.LINE_AA)


def _draw_mounting_hole(image: np.ndarray, x: int, y: int, radius: int = 34) -> None:
    cv2.circle(image, (x, y), radius + 12, (184, 190, 178), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 12, (98, 106, 100), 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, (16, 24, 26), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius - 4, (4, 7, 9), -1, cv2.LINE_AA)


def _trace_path(
    image: np.ndarray,
    rng: np.random.Generator,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
) -> None:
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])
    mode = int(rng.integers(0, 3))
    if mode == 0:
        mid_x = (x1 + x2) // 2
        pts = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
    elif mode == 1:
        mid_y = (y1 + y2) // 2
        pts = [(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
    else:
        dx = x2 - x1
        dy = y2 - y1
        pts = [(x1, y1), (x1 + dx // 3, y1 + dy // 2), (x1 + 2 * dx // 3, y1 + dy // 2), (x2, y2)]
    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(image, [arr], False, TRACE_DARK, width + 4, cv2.LINE_AA)
    cv2.polylines(image, [arr], False, TRACE_LIGHT, max(1, width - 1), cv2.LINE_AA)
    for point in pts[1:-1]:
        cv2.circle(image, point, max(2, width // 2), TRACE_LIGHT, -1, cv2.LINE_AA)
    for point in (pts[0], pts[-1]):
        _draw_via(image, int(point[0]), int(point[1]), radius=6)


def _bbox_anchors(bbox: BBox) -> list[tuple[int, int]]:
    cx, cy = bbox.center
    return [
        (int(bbox.x - 12), int(cy)),
        (int(bbox.x2 + 12), int(cy)),
        (int(cx), int(bbox.y - 12)),
        (int(cx), int(bbox.y2 + 12)),
    ]


def _draw_passive(image: np.ndarray, item: Placement, rng: np.random.Generator) -> None:
    x, y, w, h = item.x, item.y, item.w, item.h
    pad_w = max(10, int(w * 0.24))
    pad_h = max(8, int(h * 0.86))
    pad_y = y + (h - pad_h) // 2
    cv2.rectangle(image, (x, pad_y), (x + pad_w, pad_y + pad_h), SILVER, -1)
    cv2.rectangle(image, (x + w - pad_w, pad_y), (x + w, pad_y + pad_h), SILVER, -1)
    body_x = x + pad_w - 3
    body_x2 = x + w - pad_w + 3
    body_y = y + int(h * 0.14)
    body_y2 = y + int(h * 0.86)
    if item.category == "电容":
        body = (60, 86, 102) if item.package.startswith("CP") else (84, 104, 109)
    elif item.category == "电感":
        body = (55, 60, 64)
    elif item.category == "二极管":
        body = (42, 40, 46)
    elif item.category == "晶振":
        body = (142, 150, 152)
    else:
        body = (35, 37, 43)
    cv2.rectangle(image, (body_x, body_y), (body_x2, body_y2), body, -1)
    cv2.rectangle(image, (body_x, body_y), (body_x2, body_y2), (18, 20, 24), 1)
    if item.category == "二极管":
        mid = body_x + (body_x2 - body_x) // 2
        cv2.line(image, (mid, body_y + 2), (mid, body_y2 - 2), (205, 210, 205), 3)
        cv2.line(image, (mid + 8, body_y + 5), (mid + 8, body_y2 - 5), (180, 190, 185), 1)
    if item.category == "电容" and item.package.startswith("CP"):
        cv2.circle(image, (body_x + (body_x2 - body_x) // 2, body_y + 8), 5, (210, 218, 220), -1)
    if item.category == "晶振":
        cv2.ellipse(
            image,
            (body_x + (body_x2 - body_x) // 2, body_y + (body_y2 - body_y) // 2),
            (max(2, (body_x2 - body_x) // 2 - 6), max(2, (body_y2 - body_y) // 2 - 6)),
            0,
            0,
            360,
            (93, 101, 103),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (x - 4, y - 4), (x + w + 4, y + h + 4), (182, 197, 193), 1)
    _text(image, item.refdes, (x, max(20, y - 9)), 0.44, SILK, 1)
    if item.w >= 105:
        (tw, _), _ = cv2.getTextSize(item.value, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(
            image,
            item.value,
            (x + w - tw, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            TEXT_DIM,
            1,
            cv2.LINE_AA,
        )
    if rng.random() < 0.35:
        cv2.line(image, (x + w + 8, y + h // 2), (x + w + 28, y + h // 2), (160, 180, 170), 1)


def _draw_ic(image: np.ndarray, item: Placement) -> None:
    x, y, w, h = item.x, item.y, item.w, item.h
    if item.refdes == "Y1":
        _draw_passive(image, item, np.random.default_rng(0))
        return
    if w >= 330:
        per_side = 12
        pin_len = 22
    elif w >= 240:
        per_side = 8
        pin_len = 18
    else:
        per_side = 5
        pin_len = 15
    body_x1, body_y1 = x + pin_len, y + pin_len
    body_x2, body_y2 = x + w - pin_len, y + h - pin_len
    pin_w = max(4, min(9, w // 45))
    for side in range(4):
        if side in (0, 1):
            count = max(3, int(per_side * (h / max(w, 1))))
            span = body_y2 - body_y1
            for i in range(count):
                py = body_y1 + int((i + 0.5) * span / count)
                if side == 0:
                    cv2.rectangle(image, (x + 4, py - pin_w // 2), (body_x1 + 2, py + pin_w // 2), SILVER, -1)
                else:
                    cv2.rectangle(image, (body_x2 - 2, py - pin_w // 2), (x + w - 4, py + pin_w // 2), SILVER, -1)
        else:
            count = per_side
            span = body_x2 - body_x1
            for i in range(count):
                px = body_x1 + int((i + 0.5) * span / count)
                if side == 2:
                    cv2.rectangle(image, (px - pin_w // 2, y + 4), (px + pin_w // 2, body_y1 + 2), SILVER, -1)
                else:
                    cv2.rectangle(image, (px - pin_w // 2, body_y2 - 2), (px + pin_w // 2, y + h - 4), SILVER, -1)
    cv2.rectangle(image, (body_x1, body_y1), (body_x2, body_y2), BODY_DARK, -1)
    cv2.rectangle(image, (body_x1 + 3, body_y1 + 3), (body_x2 - 3, body_y2 - 3), (43, 46, 52), 1)
    cv2.rectangle(image, (x - 3, y - 3), (x + w + 3, y + h + 3), (184, 198, 193), 1)
    cx, cy = (body_x1 + body_x2) // 2, (body_y1 + body_y2) // 2
    cv2.circle(image, (body_x1 + 18, body_y1 + 18), 7, (80, 84, 86), -1, cv2.LINE_AA)
    _text(image, item.refdes, (cx, cy - 8), 0.82, SILK, 2, center=True)
    short_value = item.value.split("-")[0][:18]
    _text(image, short_value, (cx, cy + 32), 0.42, (169, 185, 178), 1, center=True)
    _text(image, item.package, (cx, cy + 56), 0.34, (131, 148, 142), 1, center=True)
    # 左下角方向标记
    cv2.circle(image, (body_x1 + 14, body_y2 - 14), 3, (145, 158, 150), -1)


def _draw_connector(image: np.ndarray, item: Placement) -> None:
    x, y, w, h = item.x, item.y, item.w, item.h
    body_x1, body_y1 = x + 10, y + 10
    body_x2, body_y2 = x + w - 10, y + h - 10
    cv2.rectangle(image, (body_x1, body_y1), (body_x2, body_y2), (48, 45, 43), -1)
    cv2.rectangle(image, (body_x1 + 4, body_y1 + 4), (body_x2 - 4, body_y2 - 4), BLUE_PLASTIC, -1)
    if item.refdes in ("J1", "J3", "J4", "J5"):
        cols = 2 if item.refdes != "J5" else 2
        rows = 7 if item.refdes == "J4" else 4
        if item.refdes in ("J1", "J3"):
            cols, rows = 1, 2 if item.refdes == "J1" else 5
        for row in range(rows):
            for col in range(cols):
                px = body_x1 + int((col + 0.5) * (body_x2 - body_x1) / cols)
                py = body_y1 + int((row + 0.5) * (body_y2 - body_y1) / rows)
                cv2.circle(image, (px, py), 9, (36, 39, 41), -1, cv2.LINE_AA)
                cv2.circle(image, (px, py), 6, GOLD, -1, cv2.LINE_AA)
    else:
        for col in range(8):
            px = body_x1 + int((col + 0.5) * (body_x2 - body_x1) / 8)
            cv2.rectangle(image, (px - 5, body_y1 + 18), (px + 5, body_y2 - 18), (39, 42, 44), -1)
            cv2.rectangle(image, (px - 3, body_y1 + 24), (px + 3, body_y2 - 24), GOLD, -1)
    cv2.rectangle(image, (x - 4, y - 4), (x + w + 4, y + h + 4), (190, 201, 197), 1)
    _text(image, item.refdes, (x, max(20, y - 10)), 0.5, SILK, 1)
    _text(image, item.value, (x + 4, y + h + 23), 0.4, TEXT_DIM, 1)


def _draw_switch(image: np.ndarray, item: Placement) -> None:
    x, y, w, h = item.x, item.y, item.w, item.h
    cv2.rectangle(image, (x, y), (x + w, y + h), (61, 63, 66), -1)
    cv2.rectangle(image, (x + 8, y + 8), (x + w - 8, y + h - 8), (108, 111, 112), -1)
    cv2.circle(image, (x + w // 2, y + h // 2), min(w, h) // 5, (47, 48, 51), -1, cv2.LINE_AA)
    cv2.rectangle(image, (x - 4, y - 4), (x + w + 4, y + h + 4), (190, 201, 197), 1)
    _text(image, item.refdes, (x, y - 10), 0.46, SILK, 1)
    _text(image, "RESET", (x + w // 2, y + h + 22), 0.36, TEXT_DIM, 1, center=True)


def _draw_component(image: np.ndarray, item: Placement, rng: np.random.Generator) -> None:
    if item.category == "芯片":
        _draw_ic(image, item)
    elif item.category == "连接器":
        _draw_connector(image, item)
    elif item.category == "开关":
        _draw_switch(image, item)
    else:
        _draw_passive(image, item, rng)


def make_synthetic_board(seed: int = 20260412) -> tuple[np.ndarray, list[Component], list[Fiducial]]:
    """生成带丝印、焊盘、走线、定位点的合成 PCB 及对应标注数据。"""
    rng = np.random.default_rng(seed)
    width, height = BOARD_WIDTH, BOARD_HEIGHT
    board = np.zeros((height, width, 3), dtype=np.uint8)
    board[:] = BOARD_GREEN

    # 大面积阻焊层色差
    for _ in range(70):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        axes = (int(rng.integers(80, 420)), int(rng.integers(40, 260)))
        angle = float(rng.uniform(0, 180))
        color = BOARD_DARK if rng.random() < 0.55 else (64, 111, 43)
        overlay = board.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1)
        board = cv2.addWeighted(overlay, 0.08, board, 0.92, 0)

    noise = rng.normal(0, 4.0, board.shape).astype(np.int16)
    board = np.clip(board.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 走线始终在元件下方
    trace_layer = board.copy()
    anchors: list[tuple[int, int]] = []
    for item in PLACEMENTS:
        anchors.extend(_bbox_anchors(BBox(item.x, item.y, item.w, item.h)))
    for _ in range(64):
        start = anchors[int(rng.integers(0, len(anchors)))]
        candidates = [a for a in anchors if 120 < abs(a[0] - start[0]) + abs(a[1] - start[1]) < 1050]
        if not candidates:
            continue
        end = candidates[int(rng.integers(0, len(candidates)))]
        _trace_path(trace_layer, rng, start, end, int(rng.integers(2, 7)))

    # 明确的并行总线，增加局部纹理
    for i in range(8):
        y = 700 + i * 13
        cv2.line(trace_layer, (1250, y), (1940, y + 45), TRACE_DARK, 3, cv2.LINE_AA)
        cv2.line(trace_layer, (1250, y), (1940, y + 45), TRACE_LIGHT, 1, cv2.LINE_AA)
        _draw_via(trace_layer, 1250, y, 5)
        _draw_via(trace_layer, 1940, y + 45, 5)
    board = cv2.addWeighted(trace_layer, 0.9, board, 0.1, 0)

    # 丝印：板名、版本、序列号、测试点
    _text(board, "PCB AR  DEMO  BOARD", (150, 250), 1.0, SILK, 2)
    _text(board, "REV A  2026-04", (150, 292), 0.55, TEXT_DIM, 1)
    _text(board, "SN DEMO-2026-001", (2400, 250), 0.55, SILK, 1)
    _text(board, "FOR VISION CALIBRATION ONLY", (150, 2040), 0.45, TEXT_DIM, 1)
    _text(board, "MADE IN DEMO / 80mm x 55mm", (150, 2080), 0.42, TEXT_DIM, 1)
    for i in range(18):
        px = int(rng.integers(450, 2500))
        py = int(rng.integers(1000, 2050))
        if any(
            item.x - 30 < px < item.x + item.w + 30 and item.y - 30 < py < item.y + item.h + 30
            for item in PLACEMENTS
        ):
            continue
        cv2.circle(board, (px, py), 8, GOLD, -1, cv2.LINE_AA)
        cv2.circle(board, (px, py), 4, (18, 28, 29), -1, cv2.LINE_AA)
        _text(board, f"TP{i + 1:02d}", (px + 12, py + 5), 0.3, TEXT_DIM, 1)

    for item in PLACEMENTS:
        _draw_component(board, item, rng)

    # 定位点 / 安装孔 / 板框
    fiducials = [
        Fiducial("FID1", 330, 330, "fiducial", 24),
        Fiducial("FID2", 2870, 330, "fiducial", 24),
        Fiducial("FID3", 330, 1870, "fiducial", 24),
        Fiducial("FID4", 2870, 1870, "fiducial", 24),
    ]
    for fid in fiducials:
        _draw_fiducial(board, int(fid.x), int(fid.y), int(fid.radius))
    for x, y in ((105, 105), (3095, 105), (105, 2095), (3095, 2095)):
        _draw_mounting_hole(board, x, y)
    cv2.rectangle(board, (60, 60), (width - 60, height - 60), (177, 190, 176), 3, cv2.LINE_AA)
    cv2.rectangle(board, (70, 70), (width - 70, height - 70), (23, 47, 28), 1, cv2.LINE_AA)

    # 边框外的深色背景，保证 warp 后没有异常亮边
    board[:52, :] = (12, 18, 19)
    board[-52:, :] = (12, 18, 19)
    board[:, :52] = (12, 18, 19)
    board[:, -52:] = (12, 18, 19)

    components: list[Component] = []
    for index, item in enumerate(PLACEMENTS):
        comp = new_component(
            item.refdes,
            BBox(item.x, item.y, item.w, item.h),
            category=item.category,
            value=item.value,
            package=item.package,
            note=item.note,
            source="demo",
        )
        comp.status = item.status
        if index % 3 == 0 and comp.status == STATUS_PENDING:
            comp.status = STATUS_VERIFIED
        components.append(comp)
    return board, components, fiducials


def create_demo_project(base_dir: str | Path, force: bool = False) -> ProjectStore:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "project.pcbproj.json"
    if not force and path.exists():
        try:
            store = ProjectStore.load(path)
            if int(store.project.extra.get("demo_version", 0)) == DEMO_VERSION:
                return store
        except ProjectError:
            pass
    board, components, fiducials = make_synthetic_board()
    store = ProjectStore.create(
        path,
        "PCB AR 演示板 REV A",
        board,
        mm_per_px=0.025,
        demo=True,
    )
    store.project.components = components
    store.project.fiducials = fiducials
    store.project.extra["demo_version"] = DEMO_VERSION
    store.project.notes = (
        "这是自动生成的演示工程。移动视图可观察标注随电路板平移/旋转；"
        "双击元件气泡可快速修改最多 3 个核心参数，右侧栏可核对全部字段。"
    )
    store.save()
    return store


def synthetic_view(
    board: np.ndarray,
    center: tuple[float, float],
    scale: float = 0.8,
    angle_deg: float = 0.0,
    size: tuple[int, int] = (1600, 1000),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """从全局板图生成一个带旋转/缩放的局部视野，返回 (frame, M_global->frame)。"""
    out_w, out_h = int(size[0]), int(size[1])
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_deg), float(scale))
    matrix[0, 2] += out_w / 2.0 - float(center[0])
    matrix[1, 2] += out_h / 2.0 - float(center[1])
    frame = cv2.warpAffine(
        board,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(10, 16, 18),
    )
    if seed:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 3.5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        gain = float(rng.uniform(0.96, 1.04))
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return frame, matrix.astype(np.float32)


def demoTrackPose(t: float) -> tuple[tuple[float, float], float, float]:
    """演示轨迹：缓慢环绕 + 轻微旋转/缩放，模拟显微镜手动移动。"""
    cx = 1600 + 620 * math.sin(0.105 * t) + 90 * math.sin(0.31 * t + 0.7)
    cy = 1100 + 430 * math.sin(0.072 * t + 1.1) + 30 * math.sin(0.47 * t)
    angle = 2.8 * math.sin(0.14 * t + 0.2) + 0.35 * math.sin(0.9 * t)
    scale = 0.78 + 0.045 * math.sin(0.083 * t + 2.0)
    return (cx, cy), angle, scale
