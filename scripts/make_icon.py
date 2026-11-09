"""生成应用图标（PNG + 多尺寸 PNG-ICO），仅构建时使用。"""
from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"))

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packaging"
OUT.mkdir(parents=True, exist_ok=True)


def draw_icon(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = size / 256.0
    p.scale(scale, scale)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#08121C"))
    p.drawRoundedRect(QRectF(4, 4, 248, 248), 46, 46)
    p.setPen(QPen(QColor("#2DD4A7"), 5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(12, 12, 232, 232), 39, 39)

    # PCB
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#0B6B4A"))
    p.drawRoundedRect(QRectF(34, 34, 188, 188), 24, 24)
    p.setPen(QPen(QColor("#3EBD91"), 3))
    for a, b, c, d in [
        (48, 68, 98, 68),
        (98, 68, 98, 104),
        (158, 78, 206, 78),
        (206, 78, 206, 120),
        (48, 184, 92, 184),
        (92, 184, 92, 150),
        (158, 186, 208, 186),
    ]:
        p.drawLine(QPointF(a, b), QPointF(c, d))
    p.setPen(QPen(QColor("#D9B15F"), 5))
    for x, y in ((48, 68), (206, 120), (48, 184), (208, 186)):
        p.drawPoint(QPointF(x, y))

    # Chip
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1C2430"))
    p.drawRoundedRect(QRectF(78, 82, 100, 84), 9, 9)
    p.setBrush(QColor("#AAB7C0"))
    for i in range(5):
        x = 88 + i * 18
        p.drawRect(QRectF(x, 72, 8, 12))
        p.drawRect(QRectF(x, 164, 8, 12))
    for i in range(3):
        y = 96 + i * 22
        p.drawRect(QRectF(68, y, 11, 8))
        p.drawRect(QRectF(177, y, 11, 8))
    p.setPen(QColor("#74F5CD"))
    p.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
    p.drawText(QRectF(78, 82, 100, 84), Qt.AlignmentFlag.AlignCenter, "AR")

    # Corner brackets
    pen = QPen(QColor("#EAF6F1"), 6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    for sx, sy, dx, dy in (
        (31, 31, 1, 1),
        (225, 31, -1, 1),
        (31, 225, 1, -1),
        (225, 225, -1, -1),
    ):
        p.drawLine(QPointF(sx, sy), QPointF(sx + 26 * dx, sy))
        p.drawLine(QPointF(sx, sy), QPointF(sx, sy + 26 * dy))
    p.end()
    return image


def ico_bytes(images: list[tuple[int, bytes]]) -> bytes:
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    payload = b""
    for size, data in images:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        payload += data
    return header + entries + payload


def main() -> None:
    app = QApplication.instance() or QApplication([])
    pngs: list[tuple[int, bytes]] = []
    for size in (16, 32, 48, 64, 128, 256):
        image = draw_icon(size)
        path = OUT / f"app_{size}.png"
        image.save(str(path), "PNG")
        pngs.append((size, path.read_bytes()))
    (OUT / "app.png").write_bytes(pngs[-1][1])
    (OUT / "app.ico").write_bytes(ico_bytes(pngs))
    print("icon written:", OUT / "app.ico")


if __name__ == "__main__":
    main()
