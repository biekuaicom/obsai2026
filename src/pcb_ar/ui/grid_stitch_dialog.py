"""离线照片文件夹 → 网格/自动拼接工作台。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pcb_ar.core.grid_stitcher import GridStitchResult, stitch_grid_folder
from pcb_ar.core.imageio import write_image
from pcb_ar.core.photo_stitcher import PhotoStitchResult, stitch_photo_folder
from pcb_ar.ui.qt_utils import qimage_from_bgr


class _StitchThread(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, files: list[Path], method: str, rows: int, cols: int, overlap: float, serpentine: bool):
        super().__init__()
        self.files = files
        self.method = method
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.serpentine = serpentine

    def run(self) -> None:
        try:
            if self.method == "grid":
                result = stitch_grid_folder(
                    self.files,
                    rows=self.rows,
                    cols=self.cols,
                    serpentine=self.serpentine,
                    overlap=self.overlap,
                    max_side=2600,
                )
            else:
                result = stitch_photo_folder(self.files, max_side=2600)
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class GridStitchDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("照片文件夹拼接 · Grid Stitch")
        self.setModal(True)
        self.resize(1120, 820)
        self._thread: _StitchThread | None = None
        self._result: GridStitchResult | PhotoStitchResult | None = None
        self._build_ui()
        self._update_buttons()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(8)
        path_row = QHBoxLayout()
        self.path_label = QLabel("照片文件夹：未选择")
        self.path_label.setWordWrap(True)
        browse = QPushButton("选择文件夹…")
        browse.clicked.connect(self.choose_folder)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(browse)
        root.addLayout(path_row)

        options = QHBoxLayout()
        options.addWidget(QLabel("方式"))
        self.method = QComboBox()
        self.method.addItem("坐标网格（推荐）", "grid")
        self.method.addItem("自动鲁棒（实验）", "auto")
        options.addWidget(self.method)
        options.addWidget(QLabel("行"))
        self.rows = QSpinBox()
        self.rows.setRange(1, 40)
        self.rows.setValue(11)
        options.addWidget(self.rows)
        options.addWidget(QLabel("列"))
        self.cols = QSpinBox()
        self.cols.setRange(1, 40)
        self.cols.setValue(5)
        options.addWidget(self.cols)
        options.addWidget(QLabel("重叠"))
        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0.05, 0.9)
        self.overlap.setSingleStep(0.05)
        self.overlap.setValue(0.33)
        self.overlap.setDecimals(2)
        options.addWidget(self.overlap)
        self.serpentine = QCheckBox("蛇形顺序")
        self.serpentine.setChecked(True)
        options.addWidget(self.serpentine)
        options.addStretch(1)
        root.addLayout(options)

        self.preview = QLabel("选择照片文件夹后开始拼接")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(600, 520)
        self.preview.setStyleSheet("background:#0d1117;color:#8b949e;")
        root.addWidget(self.preview, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("开始拼接")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start)
        self.save_button = QPushButton("保存完整图…")
        self.save_button.clicked.connect(self.save)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.save_button, 2)
        buttons.addWidget(close, 1)
        root.addLayout(buttons)
        self._files: list[Path] = []

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择局部照片文件夹")
        if not path:
            return
        files = sorted(
            p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self._files = files
        self.path_label.setText(f"照片文件夹：{path}（{len(files)} 张）")

    def start(self) -> None:
        if not self._files:
            QMessageBox.information(self, "没有照片", "请先选择照片文件夹。")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = _StitchThread(
            list(self._files),
            str(self.method.currentData()),
            self.rows.value(),
            self.cols.value(),
            float(self.overlap.value()),
            self.serpentine.isChecked(),
        )
        self._thread.finished_ok.connect(self._on_ok)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()
        self.status.setText("拼接中… 大图可能需要几十秒")
        self._update_buttons()

    def _on_ok(self, result) -> None:
        self._result = result
        image = result.image
        qimage = qimage_from_bgr(image)
        target_w = max(320, self.preview.width() - 16)
        target_h = max(240, self.preview.height() - 16)
        pix = QPixmap.fromImage(qimage).scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)
        self.status.setText(
            f"{result.message} · 输出 {image.shape[1]}×{image.shape[0]} px"
        )
        self._update_buttons()

    def _on_failed(self, message: str) -> None:
        self.status.setText("拼接失败")
        QMessageBox.warning(self, "拼接失败", message)
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = self._thread is not None and self._thread.isRunning()
        self.start_button.setEnabled(bool(self._files) and not busy)
        self.save_button.setEnabled(self._result is not None and not busy)

    def save(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存完整图",
            time.strftime("PCB_AR_网格拼接_%Y%m%d-%H%M%S.png"),
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        try:
            write_image(target, self._result.image)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        QMessageBox.information(self, "已保存", str(target))
