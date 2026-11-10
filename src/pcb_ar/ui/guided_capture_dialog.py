"""默认轻量捕获工作台：稳定检查、红色覆盖、重叠校验、关键帧地图。"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox,
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

from pcb_ar.core.imageio import write_image
from pcb_ar.core.sparse_capture import (
    CapturePreview,
    SparseCaptureError,
    SparseCaptureResult,
    SparseCaptureSession,
)
from pcb_ar.ui.qt_utils import qimage_from_bgr

FrameProvider = Callable[[], tuple[np.ndarray | None, float]]


class CapturePreviewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._polygons: list[np.ndarray] = []
        self._status = "等待实时画面"
        self._can_capture = False
        self._show_legend = True
        self.setMinimumSize(760, 480)
        self.setStyleSheet("background: #05080D; border-radius: 8px;")

    def set_data(
        self,
        frame: np.ndarray | None,
        polygons: list[np.ndarray],
        status: str,
        can_capture: bool = False,
        show_legend: bool = True,
    ) -> None:
        self._frame = frame
        self._polygons = polygons
        self._status = status
        self._can_capture = can_capture
        self._show_legend = show_legend
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#05080D"))
        if self._frame is None:
            painter.setPen(QColor("#526274"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待实时画面…")
            return
        frame = self._frame
        h, w = frame.shape[:2]
        area = self.rect().adjusted(6, 6, -6, -6)
        scale = min(area.width() / max(w, 1), area.height() / max(h, 1))
        draw_w, draw_h = int(w * scale), int(h * scale)
        x0 = area.left() + (area.width() - draw_w) // 2
        y0 = area.top() + (area.height() - draw_h) // 2
        image = qimage_from_bgr(frame)
        painter.drawImage(QRectF(x0, y0, draw_w, draw_h), image)

        # 已覆盖区域：全部关键帧覆盖多边形的并集，整体只填充一次。
        # 这样重叠处不会因为 alpha 叠加而变深，视觉上就是“全部已拍并集”。
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        has_polygon = False
        for poly in self._polygons:
            if len(poly) < 3:
                continue
            qpoly = QPolygonF(
                [QPointF(x0 + float(p[0]) * scale, y0 + float(p[1]) * scale) for p in poly]
            )
            one = QPainterPath()
            one.addPolygon(qpoly)
            one.closeSubpath()
            # 真正做布尔并集，只画外轮廓；否则重叠关键帧的内部边会画成
            # 一道道红竖线，让人误以为是对齐缝。
            path = path.united(one) if has_polygon else one
            has_polygon = True
        if has_polygon:
            painter.save()
            # 只允许红色出现在图像区域，不能画到上下黑边/工具栏上。
            painter.setClipRect(QRectF(x0, y0, draw_w, draw_h))
            painter.fillPath(path, QColor(255, 70, 90, 66))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 82, 104, 220), 2.0))
            painter.drawPath(path)
            painter.restore()

        border = QColor("#2DD4A7") if self._can_capture else QColor("#F5A524")
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border, 2))
        painter.drawRect(QRectF(x0, y0, draw_w, draw_h))

        # 底部状态条
        bar = QRectF(x0 + 8, y0 + draw_h - 38, min(520, draw_w - 16), 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(7, 13, 21, 218))
        painter.drawRoundedRect(bar, 6, 6)
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.setPen(border)
        painter.drawText(bar.adjusted(9, 0, -9, 0), Qt.AlignmentFlag.AlignVCenter, self._status)

        # 左下角红色图例（仅在捕获模式显示）
        if self._show_legend:
            legend = QRectF(x0 + 8, y0 + 8, 176, 25)
            painter.setBrush(QColor(7, 13, 21, 210))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(legend, 5, 5)
            painter.setBrush(QColor(255, 70, 90, 75))
            painter.setPen(QPen(QColor(255, 82, 104, 220), 1.5))
            painter.drawRect(QRectF(legend.left() + 8, legend.top() + 6, 13, 13))
            painter.setPen(QColor("#DCE8F4"))
            painter.drawText(
                legend.adjusted(28, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter,
                "红色 = 已拍摄覆盖区",
            )


class SparseBuildThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, session: SparseCaptureSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session

    def run(self) -> None:
        try:
            result = self.session.finish(progress=self.progress.emit)
            self.finished_ok.emit(result)
        except SparseCaptureError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"生成稀疏地图失败：{exc}\n{traceback.format_exc(limit=2)}")


class GuidedCaptureDialog(QDialog):
    def __init__(self, frame_provider: FrameProvider, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("轻量扫描建图 · Guided Capture")
        self.setModal(True)
        self.resize(1240, 850)
        self.frame_provider = frame_provider
        self.session = SparseCaptureSession()
        self.result: SparseCaptureResult | None = None
        self._running = False
        self._first_auto_done = False
        self._build_thread: SparseBuildThread | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(110)
        self._timer.timeout.connect(self._tick)
        self._thumb_version = -1
        self._build_ui()
        self._update_buttons()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(8)
        title = QLabel("轻量捕获 · 不生成大拼图")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #F1F7FD;")
        left.addWidget(title)
        subtitle = QLabel(
            "停在清晰稳定画面拍第一张；移动后确保红色覆盖还在，再拍下一张。"
            "红色是全部已拍关键帧的覆盖并集，每拍一张立即并入总览，不攒到最后再拼。"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Muted")
        left.addWidget(subtitle)
        self.preview = CapturePreviewWidget()
        left.addWidget(self.preview, 1)
        hint = QLabel(
            "红色区域是全部已拍关键帧的覆盖并集。规则：红色不能完全消失（重叠不足会拒绝），"
            "也不能几乎铺满（移动太少会拒绝）。建议每次移动约 20%~50% 视场。"
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
        card_layout.addWidget(QLabel("捕获状态"))
        self.status_label = QLabel("点击「开始捕获」")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #DDE9F4;")
        card_layout.addWidget(self.status_label)
        self.metrics_label = QLabel("关键帧 0 · 清晰度 — · 移动 —")
        self.metrics_label.setObjectName("Muted")
        self.metrics_label.setWordWrap(True)
        card_layout.addWidget(self.metrics_label)
        self.overlap_label = QLabel("已覆盖 —")
        self.overlap_label.setObjectName("Muted")
        card_layout.addWidget(self.overlap_label)
        self.red_guard_label = QLabel("红区面积/外形：—")
        self.red_guard_label.setObjectName("Muted")
        self.red_guard_label.setWordWrap(True)
        card_layout.addWidget(self.red_guard_label)
        self.quality_label = QLabel("稳定性：等待")
        self.quality_label.setObjectName("Muted")
        card_layout.addWidget(self.quality_label)
        self.auto_first = QCheckBox("稳定后自动拍第一张")
        self.auto_first.setChecked(True)
        card_layout.addWidget(self.auto_first)
        self.rectify_check = QCheckBox("完成后按板边拉正（推荐）")
        self.rectify_check.setChecked(True)
        card_layout.addWidget(self.rectify_check)
        right.addWidget(card)

        overview_title = QLabel("增量总览（逐张合并）")
        overview_title.setObjectName("Muted")
        right.addWidget(overview_title)
        self.overview_label = QLabel("每拍一张会立即并入这里")
        self.overview_label.setObjectName("Card")
        self.overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overview_label.setMinimumHeight(118)
        self.overview_label.setMaximumHeight(138)
        self.overview_label.setWordWrap(True)
        self.overview_label.setStyleSheet("color: #7C8FA5; font-size: 11px;")
        right.addWidget(self.overview_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.start_button = QPushButton("开始捕获")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_capture)
        self.capture_button = QPushButton("拍摄当前区域")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_now)
        self.finish_button = QPushButton("完成并建立地图")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finish_capture)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.capture_button, 2)
        right.addLayout(buttons)
        right.addWidget(self.finish_button)
        right.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        right.addWidget(self.progress)
        self.build_status = QLabel("至少拍摄 2 张关键帧后才能建立地图")
        self.build_status.setObjectName("Muted")
        self.build_status.setWordWrap(True)
        right.addWidget(self.build_status)

        self.result_bar = QFrame()
        self.result_bar.setObjectName("Card")
        result_layout = QVBoxLayout(self.result_bar)
        result_layout.setContentsMargins(8, 7, 8, 7)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_buttons = QHBoxLayout()
        apply_button = QPushButton("作为新工程导入")
        apply_button.setObjectName("Primary")
        apply_button.clicked.connect(self.accept)
        self.export_button = QPushButton("导出整张图…")
        self.export_button.clicked.connect(self.export_overview)
        result_buttons.addWidget(apply_button, 2)
        result_buttons.addWidget(self.export_button, 1)
        result_layout.addLayout(result_buttons)
        self.result_bar.hide()
        right.addWidget(self.result_bar)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #223044;")
        right.addWidget(divider)
        note = QLabel(
            "说明：每拍一张都会立即并入上方低分辨率总览；完成后用于人工查看/自动标注，"
            "实时定位默认使用稀疏关键帧数据库。"
        )
        note.setWordWrap(True)
        note.setObjectName("Muted")
        right.addWidget(note)
        right.addStretch(1)
        root.addLayout(right, 2)

    # ------------------------------------------------------------------ 控制
    def start_capture(self) -> None:
        self.session.config.rectify_board = self.rectify_check.isChecked()
        self.session.reset()
        self.result = None
        self._first_auto_done = False
        self._running = True
        self.result_bar.hide()
        self.build_status.setText("等待稳定画面…")
        self.progress.setValue(0)
        self.start_button.setText("重新开始")
        self.preview.set_data(None, [], "等待实时画面…")
        self._thumb_version = -1
        self.overview_label.setPixmap(QPixmap())
        self.overview_label.setText("每拍一张会立即并入这里")
        self.red_guard_label.setText("红区面积/外形：—")
        self.red_guard_label.setStyleSheet("color: #7C8FA5; font-size: 11px;")
        self._timer.start()
        self._update_buttons()

    def _tick(self) -> None:
        frame, _timestamp = self.frame_provider()
        if frame is None:
            self.status_label.setText("没有实时画面，请先在主界面启动摄像头/演示")
            return
        preview = self.session.preview(frame)
        self.preview.set_data(frame, preview.coverage_polygons, preview.message, preview.can_capture)
        self._update_metrics(preview)
        self._refresh_overview_thumb()
        if (
            self.auto_first.isChecked()
            and self.session.count == 0
            and preview.stable
            and not self._first_auto_done
        ):
            acceptance = self.session.capture()
            if acceptance.accepted:
                self._first_auto_done = True
                self.status_label.setText(acceptance.message)
                self.build_status.setText("第一张已保存并并入总览；移动电路板/视野并保留红色覆盖，再拍下一张。")
                self._refresh_overview_thumb()
        self._update_buttons()

    def _update_metrics(self, preview: CapturePreview) -> None:
        self.status_label.setText(preview.message)
        self.quality_label.setText(
            f"稳定性：{'稳定' if preview.stable else '未稳定'} · "
            f"清晰度 {preview.sharpness:.0f}"
        )
        self.overlap_label.setText(
            f"红色覆盖 {preview.covered_fraction:.0%} · 新区域 {preview.new_fraction:.0%} · "
            f"内点 {preview.inliers} · 已复核 {preview.verified_keyframes} 张"
        )
        red = preview.red_metrics
        if preview.red_anomaly:
            state = "异常"
            color = "#FF5F73"
        elif preview.red_warning:
            state = "波动"
            color = "#F5A524"
        else:
            state = "稳定"
            color = "#7C8FA5"
        text = (
            f"红区面积 {red.area_fraction:.0%} · 连通 {red.components} · "
            f"外形 {state}"
        )
        if preview.red_warning:
            text += f"（{preview.red_warning}）"
        self.red_guard_label.setText(text)
        self.red_guard_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.metrics_label.setText(
            f"关键帧 {self.session.count} · 移动 {preview.motion_px:.1f}px · "
            f"匹配 {preview.matches}"
        )

    def _refresh_overview_thumb(self) -> None:
        image = self.session.overview_image
        version = self.session.overview_version
        if image is None:
            return
        if version == self._thumb_version:
            return
        self._thumb_version = version
        self._show_overview(image)

    def _show_overview(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            return
        qimage = qimage_from_bgr(image)
        target_w = max(220, self.overview_label.width() - 12)
        target_h = max(100, self.overview_label.height() - 10)
        pixmap = QPixmap.fromImage(qimage).scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.overview_label.setText("")
        self.overview_label.setPixmap(pixmap)

    def capture_now(self) -> None:
        acceptance = self.session.capture()
        self.status_label.setText(acceptance.message)
        if self.session.count >= 2:
            self.build_status.setText(f"已拍摄 {self.session.count} 张，可以完成建图。")
        self._refresh_overview_thumb()
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = self._build_thread is not None and self._build_thread.isRunning()
        self.capture_button.setEnabled(
            self._running
            and not busy
            and self.session.last_preview is not None
            and self.session.last_preview.can_capture
        )
        self.finish_button.setEnabled(self.session.count >= 2 and not busy and self.result is None)
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(not busy)

    def finish_capture(self) -> None:
        if self.session.count < 2:
            QMessageBox.information(self, "关键帧不足", "至少拍摄 2 张关键帧。")
            return
        self._running = False
        self._timer.stop()
        self.start_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.finish_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.build_status.setText("正在保存增量总览和稀疏地图…")
        self._build_thread = SparseBuildThread(self.session, self)
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

    def _on_build_ok(self, result: SparseCaptureResult) -> None:
        self.result = result
        self.preview.set_data(
            result.overview,
            [],
            "增量总览已生成",
            True,
            show_legend=False,
        )
        self._thumb_version = self.session.overview_version
        self._show_overview(result.overview)
        self.status_label.setText("稀疏关键帧地图与增量总览已生成")
        self.quality_label.setText("定位模式：稀疏关键帧默认 + 全图周期校验")
        self.red_guard_label.setText(
            f"红区记忆：{self.session.count} 张；面积/外形参数已写入稀疏地图"
        )
        self.red_guard_label.setStyleSheet("color: #7C8FA5; font-size: 11px;")
        self.overlap_label.setText(
            f"边缘对齐 {self.session.seam_refined_count} 张 · "
            f"{result.rectify_message or '板边未校正'}"
        )
        self.metrics_label.setText(
            f"关键帧 {result.keyframe_count} 张 · 已逐张合并 · "
            f"{'板边已拉正' if result.rectified else '保留原板边'}"
        )
        structure = (
            f"\n结构：{result.structural_map.describe()}"
            if result.structural_map is not None
            else "\n结构：未生成"
        )
        if result.structural_match is not None:
            structure += (
                f"\n结构复核：{result.structural_match.inliers} 内点 / "
                f"置信度 {result.structural_match.confidence:.0%}"
            )
        self.result_label.setText(
            f"关键帧 {result.keyframe_count} 张 · 增量总览 "
            f"{result.overview.shape[1]}×{result.overview.shape[0]} px · "
            f"收尾耗时 {result.build_seconds:.2f}s{structure}"
        )
        self.result_bar.show()
        self.build_status.setText(
            "可以导入为新工程，开始自动检测和人工核对；结构特征已一并保存。"
        )
        self.cancel_button.setEnabled(True)

    def _on_build_failed(self, message: str) -> None:
        self.build_status.setText("建立地图失败")
        QMessageBox.warning(self, "建立地图失败", message)
        self._running = True
        self._timer.start()
        self.start_button.setEnabled(True)
        self.finish_button.setEnabled(self.session.count >= 2)
        self.cancel_button.setEnabled(True)

    def _on_build_done(self) -> None:
        thread = self._build_thread
        if thread is not None:
            thread.deleteLater()
        self._build_thread = None
        self._update_buttons()

    def export_overview(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "尚未完成", "请先完成并建立地图。")
            return
        default_name = time.strftime("PCB_AR_整板_%Y%m%d-%H%M%S.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出整张板图",
            default_name,
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        try:
            write_image(target, self.result.overview)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(
            self,
            "整张图已导出",
            f"已导出：\n{target}\n\n尺寸 {self.result.overview.shape[1]}×"
            f"{self.result.overview.shape[0]} px",
        )

    def accept(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "尚未完成", "请先完成并建立地图。")
            return
        self._timer.stop()
        super().accept()

    def reject(self) -> None:
        if self._build_thread is not None and self._build_thread.isRunning():
            QMessageBox.information(self, "正在生成", "请等待地图生成完成后再关闭。")
            return
        self._timer.stop()
        super().reject()

    def closeEvent(self, event) -> None:
        if self._build_thread is not None and self._build_thread.isRunning():
            QMessageBox.information(self, "正在生成", "请等待地图生成完成后再关闭。")
            event.ignore()
            return
        self._timer.stop()
        super().closeEvent(event)
