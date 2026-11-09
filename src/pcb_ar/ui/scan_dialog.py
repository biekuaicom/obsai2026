"""视频扫描建图工作台。"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pcb_ar.core.imageio import read_image, write_image
from pcb_ar.core.stitcher import StitchError, StitchResult, stitch_images
from pcb_ar.core.video_scan import ScanError, VideoScanSession
from pcb_ar.ui.qt_utils import qimage_from_bgr

FrameProvider = Callable[[], tuple[np.ndarray | None, float]]


class TrajectoryWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: list[tuple[float, float]] = []
        self.setMinimumHeight(150)
        self.setMaximumHeight(190)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = list(points)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0A111B"))
        painter.setPen(QPen(QColor("#1D2C3E"), 1))
        for x in range(20, self.width(), 32):
            painter.drawLine(x, 0, x, self.height())
        for y in range(20, self.height(), 32):
            painter.drawLine(0, y, self.width(), y)
        if not self.points:
            painter.setPen(QColor("#526274"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待扫描轨迹…")
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        pad = 16
        avail_w = max(10, self.width() - 2 * pad)
        avail_h = max(10, self.height() - 2 * pad)
        scale = min(avail_w / span_x, avail_h / span_y)
        offset_x = pad + (avail_w - span_x * scale) / 2
        offset_y = pad + (avail_h - span_y * scale) / 2

        def to_screen(point: tuple[float, float]) -> QPointF:
            return QPointF(
                offset_x + (point[0] - min_x) * scale,
                offset_y + (point[1] - min_y) * scale,
            )

        path = [to_screen(p) for p in self.points]
        painter.setPen(QPen(QColor(45, 212, 167, 150), 1.5))
        for a, b in zip(path, path[1:]):
            painter.drawLine(a, b)
        for index, point in enumerate(path):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2DD4A7") if index == len(path) - 1 else QColor("#41B8E8"))
            radius = 4.0 if index == len(path) - 1 else 2.5
            painter.drawEllipse(point, radius, radius)
        painter.setPen(QColor("#6D8298"))
        painter.drawText(
            QRectF(8, self.height() - 20, self.width() - 16, 16),
            Qt.AlignmentFlag.AlignRight,
            f"{len(self.points)} 个关键帧 · 轨迹仅作覆盖参考",
        )


class MosaicBuildThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, mode: str, session: VideoScanSession | None, images: list[np.ndarray] | None, parent=None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.session = session
        self.images = images

    def run(self) -> None:
        try:
            if self.mode == "video":
                assert self.session is not None
                result = self.session.build(progress=self.progress.emit)
                self.finished_ok.emit(result.stitch)
            else:
                assert self.images is not None
                result = stitch_images(self.images, progress=self.progress.emit)
                self.finished_ok.emit(result)
        except (ScanError, StitchError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"生成拼图失败：{exc}\n{traceback.format_exc(limit=2)}")


class VideoScanDialog(QDialog):
    def __init__(self, frame_provider: FrameProvider, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("视频扫描建图 · Mapping")
        self.setModal(True)
        self.resize(1120, 720)
        self.frame_provider = frame_provider
        self.session = VideoScanSession()
        self.result_image: np.ndarray | None = None
        self._scanning = False
        self._last_frame_ref: np.ndarray | None = None
        self._last_capture_time = 0.0
        self._build_thread: MosaicBuildThread | None = None
        self._build_started = 0.0
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(120)
        self._live_timer.timeout.connect(self._refresh_preview)
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(240)
        self._capture_timer.timeout.connect(self._capture_tick)
        self._build_ui()
        self._live_timer.start()
        self._refresh_preview()
        self._update_counters()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(8)
        title = QLabel("连续视频扫描")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #F1F7FD;")
        left.addWidget(title)
        subtitle = QLabel(
            "缓慢、连续地移动显微镜覆盖整块 PCB。相邻画面保持 80% 以上重叠，"
            "软件会自动记录关键帧并累加位置。"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Muted")
        left.addWidget(subtitle)
        self.preview = QLabel("等待实时画面…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(620, 390)
        self.preview.setStyleSheet(
            "background: #05080D; border: 1px solid #223044; border-radius: 8px; color: #526274;"
        )
        left.addWidget(self.preview, 1)
        hint = QLabel(
            "建议路径：从左到右逐行扫描（弓字形），每行之间也保留重叠；不要快速甩动或改变焦距。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #F5A524;")
        left.addWidget(hint)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)
        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(7)
        card_title = QLabel("扫描状态")
        card_title.setObjectName("SectionTitle")
        card_layout.addWidget(card_title)
        self.status_label = QLabel("未开始")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #DDE9F4;")
        card_layout.addWidget(self.status_label)
        self.counter_label = QLabel("关键帧 0 · 覆盖 0 × 0 px")
        self.counter_label.setObjectName("Muted")
        card_layout.addWidget(self.counter_label)
        self.trajectory = TrajectoryWidget()
        card_layout.addWidget(self.trajectory)
        right.addWidget(card)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.start_button = QPushButton("开始扫描")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_scan)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_scan)
        self.reset_button = QPushButton("清空")
        self.reset_button.setObjectName("Ghost")
        self.reset_button.clicked.connect(self.reset_scan)
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.stop_button, 1)
        buttons.addWidget(self.reset_button, 1)
        right.addLayout(buttons)

        self.build_button = QPushButton("生成全局拼图")
        self.build_button.setObjectName("Primary")
        self.build_button.setEnabled(False)
        self.build_button.clicked.connect(self.build_mosaic)
        right.addWidget(self.build_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        right.addWidget(self.progress)
        self.build_status = QLabel("扫描至少 2 个位置后生成拼图")
        self.build_status.setObjectName("Muted")
        self.build_status.setWordWrap(True)
        right.addWidget(self.build_status)

        self.result_bar = QFrame()
        self.result_bar.setObjectName("Card")
        result_layout = QHBoxLayout(self.result_bar)
        result_layout.setContentsMargins(8, 6, 8, 6)
        result_layout.setSpacing(6)
        save = QPushButton("保存 PNG…")
        save.clicked.connect(self.save_result)
        apply = QPushButton("作为新工程导入")
        apply.setObjectName("Primary")
        apply.clicked.connect(self.accept_result)
        result_layout.addWidget(save)
        result_layout.addWidget(apply)
        self.result_bar.hide()
        right.addWidget(self.result_bar)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #223044;")
        right.addWidget(divider)
        fallback = QLabel("散图备用")
        fallback.setObjectName("SectionTitle")
        right.addWidget(fallback)
        import_button = QPushButton("导入多张重叠照片拼接…")
        import_button.setObjectName("Ghost")
        import_button.clicked.connect(self.import_images)
        right.addWidget(import_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        right.addWidget(close_button)
        right.addStretch(1)
        root.addLayout(right, 2)

    # ------------------------------------------------------------------ 扫描
    def start_scan(self) -> None:
        self.session.reset()
        self.result_image = None
        self.result_bar.hide()
        self.preview.clear()
        self.preview.setText("等待扫描画面…")
        self._scanning = True
        self._last_frame_ref = None
        self._last_capture_time = 0.0
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.build_button.setEnabled(False)
        self.import_button_set_enabled(False)
        self.status_label.setText("正在扫描：请缓慢移动，覆盖完整 PCB…")
        self.build_status.setText("扫描至少 2 个位置后生成拼图")
        self.trajectory.set_points([])
        self._update_counters()
        self._capture_timer.start()

    def stop_scan(self) -> None:
        self._scanning = False
        self._capture_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.import_button_set_enabled(True)
        self.build_button.setEnabled(self.session.count >= 2)
        if self.session.count < 2:
            self.status_label.setText("关键帧不足，请重新扫描并移动更远一些")
            self.build_status.setText("关键帧不足，至少需要 2 个不同位置")
        else:
            self.status_label.setText("扫描已停止，可以生成全局拼图")
            self.build_status.setText(
                f"已记录 {self.session.count} 个关键帧，点击「生成全局拼图」"
            )
        self._update_counters()

    def reset_scan(self) -> None:
        self._scanning = False
        self._capture_timer.stop()
        self.session.reset()
        self.result_image = None
        self.result_bar.hide()
        self.preview.clear()
        self.preview.setText("等待实时画面…")
        self.status_label.setText("未开始")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.build_button.setEnabled(False)
        self.import_button_set_enabled(True)
        self.progress.setValue(0)
        self.build_status.setText("扫描至少 2 个位置后生成拼图")
        self.trajectory.set_points([])
        self._update_counters()

    def _capture_tick(self) -> None:
        if not self._scanning:
            return
        frame, timestamp = self.frame_provider()
        if frame is None:
            self.status_label.setText("没有实时画面，请先在主界面启动摄像头/演示")
            return
        if frame is self._last_frame_ref:
            return
        self._last_frame_ref = frame
        update = self.session.add_frame(frame, timestamp=timestamp or time.time())
        self.status_label.setText(update.message)
        self._update_counters()
        self.trajectory.set_points(self.session.trajectory())
        if self.session.count >= 120:
            self.stop_scan()
        elif self.session.count >= 2:
            self.build_button.setEnabled(True)

    def _update_counters(self) -> None:
        width, height = self.session.coverage_size if self.session.count else (0, 0)
        self.counter_label.setText(
            f"关键帧 {self.session.count} · 覆盖范围约 {width} × {height} px"
        )

    def import_button_set_enabled(self, enabled: bool) -> None:
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.build_button.setEnabled(enabled and self.session.count >= 2)

    # ------------------------------------------------------------------ 预览
    def _refresh_preview(self) -> None:
        if self.result_image is None:
            frame, _ = self.frame_provider()
            if frame is not None:
                self._set_preview(frame)

    def _set_preview(self, image: np.ndarray) -> None:
        qimage = qimage_from_bgr(image)
        pixmap = QPixmap.fromImage(qimage)
        target = self.preview.size()
        if target.width() > 20 and target.height() > 20:
            pixmap = pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview.setPixmap(pixmap)

    # ------------------------------------------------------------------ 拼接
    def build_mosaic(self) -> None:
        if self._build_thread is not None and self._build_thread.isRunning():
            return
        if self.session.count < 2:
            QMessageBox.information(self, "关键帧不足", "请先扫描至少两个不同位置。")
            return
        self.stop_scan()
        self._start_build("video", session=self.session)

    def import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择重叠的局部图片（按拍摄顺序）",
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not files:
            return
        images = []
        for path in files:
            try:
                images.append(read_image(path))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "读取失败", f"{path}\n{exc}")
                return
        if len(images) < 2:
            QMessageBox.information(self, "图片不足", "至少选择 2 张有重叠的图片。")
            return
        self._scanning = False
        self._capture_timer.stop()
        self.status_label.setText(f"已导入 {len(images)} 张图片，将按选择顺序配准")
        self._start_build("images", images=images)

    def _start_build(self, mode: str, session: VideoScanSession | None = None, images: list[np.ndarray] | None = None) -> None:
        self._build_started = time.time()
        self.progress.setValue(0)
        self.build_status.setText("正在配准与合成，请稍候…")
        self.build_button.setEnabled(False)
        self.import_button_set_enabled(False)
        self._build_thread = MosaicBuildThread(mode, session, images, self)
        self._build_thread.progress.connect(self._on_build_progress)
        self._build_thread.finished_ok.connect(self._on_build_ok)
        self._build_thread.failed.connect(self._on_build_failed)
        self._build_thread.finished.connect(self._on_build_done)
        self._build_thread.start()

    def _on_build_progress(self, current: int, total: int, message: str) -> None:
        total = max(1, total)
        self.progress.setRange(0, total)
        self.progress.setValue(min(current, total))
        self.build_status.setText(message)

    def _on_build_ok(self, result: StitchResult) -> None:
        self.result_image = result.image
        self._set_preview(result.image)
        elapsed = time.time() - self._build_started
        self.build_status.setText(
            f"拼图完成：{result.image.shape[1]} × {result.image.shape[0]} px，"
            f"缩放 {result.output_scale:.2f}，用时 {elapsed:.1f}s"
        )
        self.status_label.setText("拼图已生成，可以保存或作为新工程继续标注")
        self._show_result_actions()

    def _on_build_failed(self, message: str) -> None:
        self.build_status.setText("拼图失败")
        self.result_bar.hide()
        QMessageBox.warning(self, "拼图失败", message)
        self.import_button_set_enabled(True)
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def _on_build_done(self) -> None:
        thread = self._build_thread
        if thread is not None:
            thread.deleteLater()
        self._build_thread = None
        self.progress.setValue(self.progress.maximum())

    def _show_result_actions(self) -> None:
        self.result_bar.show()

    def save_result(self) -> None:
        if self.result_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存全局拼图",
            str(Path.home() / "Desktop" / "pcb_global_map.png"),
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg)",
        )
        if not path:
            return
        try:
            write_image(path, self.result_image)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        QMessageBox.information(self, "已保存", f"全局拼图已保存到：\n{path}")

    def accept_result(self) -> None:
        if self.result_image is None:
            return
        self.accept()

    def closeEvent(self, event) -> None:
        if self._build_thread is not None and self._build_thread.isRunning():
            QMessageBox.information(self, "正在生成", "拼图正在生成，请等待完成后再关闭。")
            event.ignore()
            return
        self._live_timer.stop()
        self._capture_timer.stop()
        super().closeEvent(event)
