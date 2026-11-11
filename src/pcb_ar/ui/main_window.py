"""主窗口：工程、实时画面、AR 标注、核对与扫描建图入口。"""
from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pcb_ar import APP_NAME_CN, __version__
from pcb_ar.config import AppConfig
from pcb_ar.core.bom import BomError, apply_bom, export_bom, load_bom
from pcb_ar.core.detector import detect_component_candidates
from pcb_ar.core.geo import BBox, invert_affine, transform_bbox
from pcb_ar.core.imageio import read_image, write_image
from pcb_ar.core.models import (
    STATUS_ISSUE,
    STATUS_PENDING,
    STATUS_VERIFIED,
    Component,
    Project,
    new_component,
)
from pcb_ar.core.project import ProjectError, ProjectStore
from pcb_ar.core.sparse_localizer import HybridTracker, SparseLocalizer
from pcb_ar.core.sparse_map import SparseMap, SparseMapError
from pcb_ar.core.synthetic import create_demo_project
from pcb_ar.core.tracker import GlobalLocalTracker, TrackerConfig, TrackResult, pose_corner_error
from pcb_ar.core.video import CameraSource, FrameData, SyntheticMicroscopeSource, VideoFileSource
from pcb_ar.paths import default_project_dir, demo_dir
from pcb_ar.ui.grid_stitch_dialog import GridStitchDialog
from pcb_ar.ui.guided_capture_dialog import GuidedCaptureDialog
from pcb_ar.ui.inline_editor import InlineParamEditor
from pcb_ar.ui.live import LiveWorker
from pcb_ar.ui.overlay import AnnotationOverlay
from pcb_ar.ui.qt_utils import natural_key
from pcb_ar.ui.scan_dialog import VideoScanDialog
from pcb_ar.ui.sidebar import SidebarPanel
from pcb_ar.ui.view import CanvasView

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        config_path: Path | None = None,
        demo_base: Path | None = None,
        initial_project: Path | None = None,
        auto_bootstrap: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config or AppConfig.load()
        self.config_path = config_path
        self.demo_base = Path(demo_base) if demo_base else demo_dir()
        self.initial_project = Path(initial_project) if initial_project else None
        self.store: ProjectStore | None = None
        self.project: Project | None = None
        self.global_image: np.ndarray | None = None
        self.tracker: HybridTracker | GlobalLocalTracker | None = None
        self.sparse_map: SparseMap | None = None
        self.sparse_localizer: SparseLocalizer | None = None
        self.worker: LiveWorker | None = None
        self.source = None
        self.last_frame: np.ndarray | None = None
        self.last_frame_timestamp: float = 0.0
        self.last_result: TrackResult | None = None
        self._static_matrix: np.ndarray | None = None
        self._dirty = False
        self._syncing_selection = False
        self._syncing_draw = False
        self._inline_open = False
        self._inline_component_id = ""
        self._fps = 0.0
        self._last_error_message = ""
        self._current_source_kind = ""
        self._closing = False
        self._build_ui()
        self._build_actions()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()
        self._restore_window_state()
        self._update_actions()
        if auto_bootstrap:
            QTimer.singleShot(30, self._bootstrap)

    # ================================================================== UI
    def _build_ui(self) -> None:
        self.setWindowTitle(APP_NAME_CN)
        self.resize(1420, 900)
        self.setMinimumSize(1100, 720)
        host = QWidget()
        host.setObjectName("CentralHost")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.canvas = CanvasView()
        layout.addWidget(self.canvas, 1)
        self.setCentralWidget(host)

        self.sidebar = SidebarPanel()
        self.sidebar_dock = QDockWidget("元件核对", self)
        self.sidebar_dock.setObjectName("SidebarDock")
        self.sidebar_dock.setWidget(self.sidebar)
        self.sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.sidebar_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.sidebar_dock.setMinimumWidth(365)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sidebar_dock)
        self.resizeDocks([self.sidebar_dock], [390], Qt.Orientation.Horizontal)

        self.overlay = AnnotationOverlay(self.canvas.scene(), self.canvas, self)
        self.inline_editor = InlineParamEditor(self.canvas.viewport())
        self.inline_editor.hide()

        self.source_status = QLabel("● 未连接")
        self.source_status.setStyleSheet("color: #8294A8; padding-left: 4px;")
        self.track_status = QLabel("等待图像输入")
        self.track_status.setStyleSheet("color: #8294A8;")
        self.fps_status = QLabel("— FPS")
        self.fps_status.setStyleSheet("color: #8294A8; padding: 0 8px;")
        self.zoom_status = QLabel("—")
        self.zoom_status.setStyleSheet("color: #8294A8; padding: 0 8px;")
        self.count_status = QLabel("—")
        self.count_status.setStyleSheet("color: #B8C8D8; padding: 0 8px;")
        self.statusBar().addWidget(self.source_status)
        self.statusBar().addWidget(self.track_status, 1)
        self.statusBar().addPermanentWidget(self.fps_status)
        self.statusBar().addPermanentWidget(self.zoom_status)
        self.statusBar().addPermanentWidget(self.count_status)

    def _action(
        self,
        text: str,
        slot=None,
        shortcut: str | QKeySequence | None = None,
        *,
        checkable: bool = False,
        tip: str = "",
    ) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.setCheckable(checkable)
        if slot is not None:
            action.triggered.connect(slot)
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        return action

    def _build_actions(self) -> None:
        self.action_new = self._action("新建工程", self.new_project, QKeySequence.StandardKey.New, tip="从一张完整板图创建工程")
        self.action_open = self._action("打开工程", self.open_project_dialog, QKeySequence.StandardKey.Open, tip="打开 .pcbproj.json 工程")
        self.action_save = self._action("保存工程", self.save_project, QKeySequence.StandardKey.Save)
        self.action_save_as = self._action("另存为…", self.save_project_as, "Ctrl+Shift+S")
        self.action_import_bom = self._action("导入 BOM / CSV…", self.import_bom)
        self.action_export_board = self._action(
            "导出整张板图…",
            self.export_board_image,
            tip="把当前工程的总览图导出为 PNG/JPEG",
        )
        self.action_open_photos = self._action(
            "打开拍照文件夹",
            self.open_photos_folder,
            tip="在资源管理器中打开工程 assets/photos",
        )
        self.action_export_photos = self._action(
            "导出拍照文件夹…",
            self.export_photos_folder,
            tip="复制工程 assets/photos 下的全部局部照片",
        )
        self.action_export_bom = self._action("导出 BOM / CSV…", self.export_bom)
        self.action_exit = self._action("退出", self.close, "Alt+F4")

        self.action_demo = self._action("演示", self.start_demo, "Ctrl+D", tip="启动合成显微镜演示，不需要硬件")
        self.action_camera = self._action("摄像头…", self.start_camera, "Ctrl+K", tip="连接 USB/HDMI 采集摄像头")
        self.action_video = self._action("视频文件…", self.open_video_file, tip="使用录制好的显微镜视频")
        self.action_pause = self._action("暂停", self.toggle_pause, "F5", checkable=True, tip="暂停/继续实时采集")

        self.action_grid_stitch = self._action(
            "照片文件夹网格拼接…",
            self.open_grid_stitch,
            tip="把已拍好的局部照片按行列/重叠重新拼成完整图",
        )
        self.action_guided_scan = self._action(
            "轻量扫描建图…",
            self.open_guided_capture,
            "Ctrl+M",
            tip="稳定拍照 + 红色重叠校验 + 稀疏关键帧地图（默认）",
        )
        self.action_scan = self._action(
            "全图扫描（独立备用）…",
            self.open_scan_dialog,
            "Ctrl+Shift+M",
            tip="生成独立全图工程；轻量工程的全图增强会自动使用低分辨率总览并周期校验",
        )
        self.action_detect = self._action("自动检测候选", self.auto_detect, "Ctrl+G")
        self.action_draw = self._action("框选新增", self._toggle_draw_action, "Ctrl+Shift+A", checkable=True)
        self.action_verify = self._action("标记已核对", self.verify_selected, "Ctrl+Return")
        self.action_issue = self._action("标记异常", self.mark_issue_selected, "Ctrl+Shift+I")
        self.action_delete = self._action("删除元件", self.delete_selected, "Ctrl+Shift+Delete")
        self.action_edit = self._action("快速编辑 3 参数", self.edit_selected, "Ctrl+E")
        self.action_next = self._action("下一待核对", self.next_pending, "Ctrl+Shift+N")

        self.action_fit = self._action("适应窗口", self.canvas.fit_to_window, "Ctrl+0")
        self.action_100 = self._action("100%", self.canvas.zoom_100, "Ctrl+1")
        self.action_zoom_in = self._action("放大", lambda: self.canvas.zoom_by(1.2), "Ctrl+=")
        self.action_zoom_out = self._action("缩小", lambda: self.canvas.zoom_by(1 / 1.2), "Ctrl+-")
        self.action_labels = self._action("显示标签", self._toggle_labels, "F3", checkable=True)
        self.action_labels.setChecked(True)
        self.action_expand_labels = self._action("全部展开参数", self._toggle_expand_labels, "F4", checkable=True)
        self.action_overlay = self._action("显示覆盖层", self._toggle_overlay, "F2", checkable=True)
        self.action_overlay.setChecked(True)
        self.action_project_info = self._action("工程信息 / 比例标定…", self.edit_project_info)

        self.action_shortcuts = self._action("快捷键与流程", self.show_shortcuts, "F1")
        self.action_about = self._action("关于", self.show_about)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_grid_stitch)
        file_menu.addAction(self.action_import_bom)
        file_menu.addAction(self.action_export_board)
        file_menu.addAction(self.action_open_photos)
        file_menu.addAction(self.action_export_photos)
        file_menu.addAction(self.action_export_bom)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        input_menu = menu.addMenu("输入")
        input_menu.addAction(self.action_demo)
        input_menu.addAction(self.action_camera)
        input_menu.addAction(self.action_video)
        input_menu.addAction(self.action_pause)
        input_menu.addSeparator()
        input_menu.addAction(self.action_guided_scan)
        input_menu.addAction(self.action_scan)

        mark_menu = menu.addMenu("标注与核对")
        mark_menu.addAction(self.action_detect)
        mark_menu.addAction(self.action_draw)
        mark_menu.addSeparator()
        mark_menu.addAction(self.action_edit)
        mark_menu.addAction(self.action_verify)
        mark_menu.addAction(self.action_issue)
        mark_menu.addAction(self.action_delete)
        mark_menu.addSeparator()
        mark_menu.addAction(self.action_next)

        view_menu = menu.addMenu("视图")
        view_menu.addAction(self.action_fit)
        view_menu.addAction(self.action_100)
        view_menu.addAction(self.action_zoom_in)
        view_menu.addAction(self.action_zoom_out)
        view_menu.addSeparator()
        view_menu.addAction(self.action_overlay)
        view_menu.addAction(self.action_labels)
        view_menu.addAction(self.action_expand_labels)
        view_menu.addSeparator()
        view_menu.addAction(self.sidebar_dock.toggleViewAction())

        project_menu = menu.addMenu("工程")
        project_menu.addAction(self.action_project_info)

        help_menu = menu.addMenu("帮助")
        help_menu.addAction(self.action_shortcuts)
        help_menu.addAction(self.action_about)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        brand = QLabel('<span style="color:#2DD4A7;font-weight:800;">PCB AR</span>'
                       '<span style="color:#61758B;font-size:11px;">  显微标注</span>')
        brand.setContentsMargins(5, 0, 14, 0)
        toolbar.addWidget(brand)
        for action in (self.action_open, self.action_save):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in (self.action_demo, self.action_camera, self.action_video, self.action_pause):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.action_guided_scan)
        toolbar.addSeparator()
        toolbar.addAction(self.action_detect)
        toolbar.addAction(self.action_draw)
        toolbar.addAction(self.action_verify)
        toolbar.addSeparator()
        toolbar.addAction(self.action_fit)
        toolbar.addAction(self.action_labels)
        self.toolbar = toolbar

    def _connect_signals(self) -> None:
        self.canvas.empty_clicked.connect(self._clear_selection)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.draw_mode_changed.connect(self._on_draw_mode_changed)
        self.overlay.selection_changed.connect(self._on_overlay_selection)
        self.overlay.edit_requested.connect(self.open_inline_editor)
        self.sidebar.selection_requested.connect(self._on_sidebar_selection)
        self.sidebar.component_applied.connect(self._on_component_applied)
        self.sidebar.verify_requested.connect(lambda cid: self._set_status(cid, STATUS_VERIFIED))
        self.sidebar.issue_requested.connect(lambda cid: self._set_status(cid, STATUS_ISSUE))
        self.sidebar.delete_requested.connect(self.delete_component)
        self.sidebar.next_pending_requested.connect(self.next_pending)
        self.inline_editor.saved.connect(self._on_inline_saved)
        self.inline_editor.dismissed.connect(self._on_inline_dismissed)

    # ================================================================== 启动
    def _bootstrap(self) -> None:
        if self.initial_project is not None:
            self._open_path(self.initial_project, autoplay=True)
            return
        if self.config.last_project:
            path = Path(self.config.last_project)
            if path.exists():
                if self._open_path(path, autoplay=True):
                    return
        self.load_demo(auto_start=True)

    def load_demo(self, auto_start: bool = True) -> bool:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.statusBar().showMessage("正在生成/载入演示工程…")
            QApplication.processEvents()
            store = create_demo_project(self.demo_base)
            return self._open_store(store, autoplay=auto_start, start_demo_source=auto_start)
        except Exception as exc:  # noqa: BLE001
            logger.exception("载入演示失败")
            QMessageBox.critical(self, "演示工程失败", f"无法创建演示工程：\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _open_path(self, path: Path, autoplay: bool = False) -> bool:
        try:
            store = ProjectStore.load(path)
        except ProjectError as exc:
            QMessageBox.warning(self, "打开工程失败", str(exc))
            return False
        is_demo = store.project.demo
        return self._open_store(
            store,
            autoplay=autoplay and is_demo,
            start_demo_source=autoplay and is_demo,
        )

    def _open_store(
        self,
        store: ProjectStore,
        *,
        autoplay: bool = False,
        start_demo_source: bool = False,
    ) -> bool:
        try:
            image = store.load_image()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "全局图读取失败", str(exc))
            return False
        self.stop_live(clear=True)
        self.store = store
        self.project = store.project
        self.global_image = image
        self.tracker = None
        self.sparse_map = None
        self.sparse_localizer = None
        sparse_path = store.sparse_map_path
        if sparse_path is not None:
            try:
                self.sparse_map = SparseMap.load(sparse_path)
                image_h, image_w = image.shape[:2]
                if (self.sparse_map.board_width, self.sparse_map.board_height) != (image_w, image_h):
                    raise SparseMapError(
                        "稀疏地图板尺寸与总览图不一致："
                        f"稀疏 {self.sparse_map.board_width}×{self.sparse_map.board_height}，"
                        f"总览 {image_w}×{image_h}"
                    )
                self.project.map_mode = "hybrid"
            except SparseMapError as exc:
                logger.exception("稀疏地图加载失败")
                self.sparse_map = None
                self.project.map_mode = "full"
                QMessageBox.warning(
                    self,
                    "稀疏地图加载失败",
                    f"工程仍可打开，但将回退到全图跟踪：\n{exc}",
                )
        self.last_frame = None
        self.last_result = None
        self._dirty = False
        self._static_matrix = None
        self._inline_component_id = ""
        self.inline_editor.dismiss()
        self.overlay.set_project(self.project)
        self.sidebar.set_project(self.project)
        self._show_static_preview()
        self._update_title()
        self._update_counts()
        self._update_actions()
        self._save_config(last_project=str(store.path))
        self.statusBar().showMessage(
            f"工程已载入：{store.project.name} · {self.project.count} 个元件", 5000
        )
        if start_demo_source:
            QTimer.singleShot(50, self.start_demo)
        return True

    def _show_static_preview(self) -> None:
        if self.global_image is None:
            self.canvas.clear_frame()
            return
        h, w = self.global_image.shape[:2]
        scale = min(1.0, 2200.0 / max(w, 1))
        if scale < 1.0:
            preview = cv2.resize(
                self.global_image,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = self.global_image
        matrix = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
        self._static_matrix = matrix
        self.canvas.set_frame(preview)
        self.overlay.update_pose(matrix, "static")
        self.canvas.fit_to_window()
        self._set_source_status("静态全局图", "#4DA8FF")
        self._set_track_status("等待实时画面 · 可先核对静态标注", "#8294A8")
        self.last_frame = None
        self.last_frame_timestamp = 0.0
        self._current_source_kind = "static"

    # ================================================================== 输入源
    def _ensure_tracker(self) -> bool:
        if self.global_image is None:
            return False
        if self.tracker is not None:
            return True
        try:
            cfg = TrackerConfig(
                global_match_width=self.config.global_match_width,
                global_features=self.config.tracker_global_features,
                live_features=self.config.tracker_live_features,
                ratio=self.config.tracker_ratio,
                min_matches=self.config.tracker_min_matches,
                min_inliers=self.config.tracker_min_inliers,
                ransac_px=self.config.tracker_ransac_px,
                smoothing=self.config.tracker_smoothing,
                max_lost_frames=self.config.tracker_max_lost_frames,
            )
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            sparse_localizer: SparseLocalizer | None = None
            if self.sparse_map is not None and self.project is not None and self.project.map_mode != "full":
                try:
                    sparse_localizer = SparseLocalizer(self.sparse_map)
                    self.sparse_localizer = sparse_localizer
                except Exception:  # noqa: BLE001
                    logger.exception("稀疏定位器初始化失败，将使用全图")
                    self.sparse_localizer = None
            full_tracker: GlobalLocalTracker | None = None
            need_full = sparse_localizer is None or (
                self.project is not None and self.project.map_mode in ("hybrid", "full")
            )
            if need_full:
                try:
                    full_tracker = GlobalLocalTracker(self.global_image, cfg)
                except Exception:  # noqa: BLE001
                    logger.exception("全图增强跟踪器初始化失败")
                    if sparse_localizer is None:
                        raise
            if sparse_localizer is not None and full_tracker is not None:
                self.tracker = HybridTracker(
                    sparse_localizer,
                    full_tracker,
                    full_interval=10,
                    consensus_px=24.0,
                )
                logger.info("使用稀疏定位 + 全图增强跟踪")
            elif sparse_localizer is not None:
                self.tracker = sparse_localizer
                logger.info("使用纯稀疏关键帧定位")
            elif full_tracker is not None:
                self.tracker = full_tracker
                logger.info("使用全图跟踪")
            else:
                raise RuntimeError("无法建立任何跟踪器")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("初始化跟踪器失败")
            QMessageBox.warning(self, "跟踪器初始化失败", str(exc))
            self.tracker = None
            self.sparse_localizer = None
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _start_worker(self, source, kind: str) -> bool:
        self.stop_live(clear=False)
        tracker = None
        if self.project is not None and self.global_image is not None:
            if not self._ensure_tracker():
                tracker = None
            else:
                tracker = self.tracker
        self.source = source
        worker = LiveWorker(
            source,
            tracker=tracker,
            max_width=self.config.max_display_width,
            parent=self,
        )
        worker.frame_ready.connect(self._on_frame)
        worker.fatal_error.connect(self._on_live_error)
        worker.tracker_error.connect(self._on_tracker_error)
        worker.fps_updated.connect(self._on_fps)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self.worker = worker
        self._current_source_kind = kind
        worker.start()
        self.action_pause.setEnabled(True)
        self.action_pause.setChecked(False)
        self.action_pause.setText("暂停")
        self._update_actions()
        self._set_source_status(f"● {worker.source_description}", "#2DD4A7")
        self._set_track_status(
            "正在初始化跟踪…" if tracker is not None else "实时画面（未载入工程，无标注）",
            "#F5A524",
        )
        self.statusBar().showMessage("实时输入已启动", 2500)
        return True

    def start_demo(self) -> None:
        if self.project is None or self.global_image is None:
            if not self.load_demo(auto_start=False):
                return
        try:
            source = SyntheticMicroscopeSource(
                self.global_image,
                size=(1600, 1000),
                fps=20.0,
                seed=11,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "演示启动失败", str(exc))
            return
        self._start_worker(source, "demo")
        self._set_source_status("● 模拟显微镜演示", "#2DD4A7")
        self.statusBar().showMessage(
            "演示模式：画面模拟显微镜平移/旋转，标注应同步吸附在元件上。", 6000
        )

    def start_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "打开摄像头", "摄像头索引：", 0, 0, 16, 1)
        if not ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            source = CameraSource(index=index)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "摄像头打开失败",
                f"{exc}\n\n请确认摄像头未被其他程序占用；必要时在设备管理器检查驱动。",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._start_worker(source, "camera")

    def open_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开显微镜视频",
            "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)",
        )
        if not path:
            return
        try:
            source = VideoFileSource(path, loop=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "视频打开失败", str(exc))
            return
        self._start_worker(source, "video")

    def stop_live(self, clear: bool = False) -> None:
        worker = self.worker
        self.worker = None
        self.source = None
        if worker is not None:
            try:
                worker.stop()
                if not worker.wait(3500):
                    logger.warning("采集线程未在超时内退出")
                worker.deleteLater()
            except RuntimeError:
                pass
        self.action_pause.setEnabled(False)
        self.action_pause.setChecked(False)
        self.action_pause.setText("暂停")
        self.last_result = None
        self._update_actions()
        if clear and self.project is not None:
            self._show_static_preview()
        elif clear:
            self.canvas.clear_frame()
            self.overlay.update_pose(None, "lost")
            self._set_source_status("● 未连接", "#8294A8")
            self._set_track_status("等待图像输入", "#8294A8")

    def toggle_pause(self) -> None:
        if self.worker is None:
            self.action_pause.setChecked(False)
            return
        paused = not self.worker.is_paused()
        self.worker.set_paused(paused)
        self.action_pause.setChecked(paused)
        self.action_pause.setText("继续" if paused else "暂停")
        self._set_track_status("已暂停采集" if paused else "继续采集…", "#F5A524" if paused else "#2DD4A7")

    # ================================================================== 帧处理
    def _on_frame(self, data: FrameData, result: TrackResult | None) -> None:
        if not isinstance(data, FrameData):
            return
        self.last_frame = data.image
        self.last_frame_timestamp = data.timestamp
        self.canvas.set_frame(data.image)
        if self.project is None:
            self.overlay.update_pose(None, "lost")
            self._set_track_status("实时画面（未载入工程）", "#8294A8")
            return
        if result is None:
            self.overlay.update_pose(None, "lost")
            self._set_track_status("未启用跟踪", "#F5A524")
            return
        self.last_result = result
        self.overlay.update_pose(result.matrix, result.state)
        source = str((result.diagnostics or {}).get("source", ""))
        source_label = {
            "sparse": "稀疏定位",
            "full": "全图增强",
            "full-cache": "全图缓存",
            "none": "无定位",
        }.get(source, "定位")
        if result.state == "locked":
            text = (
                f"{source_label} · {result.inliers} 内点 / {result.matches} 匹配 · "
                f"置信度 {result.confidence:.0%} · {result.latency_ms:.0f} ms"
            )
            if source == "sparse" and (result.diagnostics or {}).get("full_checked"):
                text += " · 全图校验通过"
            color = "#2DD4A7"
            if data.truth_matrix is not None and result.matrix is not None:
                h, w = data.image.shape[:2]
                err = pose_corner_error(data.truth_matrix, result.matrix, w, h)
                text += f" · 演示误差 {err:.1f}px"
                self.track_status.setToolTip(f"与合成真值四角平均误差：{err:.2f} 像素")
            self._set_track_status(text, color)
        elif result.state == "stale":
            self._set_track_status(
                f"{source_label}暂失：{result.message} · 保持上一位置", "#F5A524"
            )
        else:
            self._set_track_status(f"{source_label}未锁定：{result.message}", "#FF5C7A")
        if self._inline_open:
            self._position_inline_editor()

    def _on_fps(self, fps: float) -> None:
        self._fps = fps
        self.fps_status.setText(f"{fps:.1f} FPS")

    def _on_live_error(self, message: str) -> None:
        logger.error("实时输入错误：%s", message)
        if message == self._last_error_message:
            return
        self._last_error_message = message
        self._set_track_status("实时输入中断", "#FF5C7A")
        self.statusBar().showMessage(message.splitlines()[0], 8000)
        if not self._closing:
            QMessageBox.warning(self, "实时输入中断", message.split("\n\n")[0])

    def _on_tracker_error(self, message: str) -> None:
        logger.error("跟踪错误：%s", message)
        self.last_result = None
        self.overlay.update_pose(None, "lost")
        self._set_track_status(message.splitlines()[0], "#FF5C7A")

    def _on_worker_finished(self, worker: LiveWorker) -> None:
        if worker is self.worker:
            self.worker = None
            self.source = None
            self.action_pause.setEnabled(False)
            self.action_pause.setChecked(False)
            self.action_pause.setText("暂停")
            self._set_source_status("● 输入已停止", "#8294A8")
            self._update_actions()

    # ================================================================== 标注交互
    def _select_component(self, component_id: str, *, from_overlay: bool = False) -> None:
        if self.project is None:
            self.overlay.clear_selection(emit=False)
            self.sidebar.clear_selection()
            return
        comp = self.project.find_by_id(component_id) if component_id else None
        if comp is None:
            self.overlay.clear_selection(emit=False)
            self.sidebar.clear_selection()
            self.inline_editor.dismiss()
            self._update_actions()
            return
        if self._inline_open and self._inline_component_id != comp.id:
            self.inline_editor.dismiss()
            self._inline_open = False
            self._inline_component_id = ""
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.overlay.select(comp.id, emit=False)
            self.sidebar.select_component(comp.id, emit=False)
        finally:
            self._syncing_selection = False
        if not from_overlay:
            self._ensure_selected_visible()
        self._update_actions()
        self._position_inline_editor()

    def _clear_selection(self) -> None:
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.overlay.clear_selection(emit=False)
            self.sidebar.clear_selection()
        finally:
            self._syncing_selection = False
        self.inline_editor.dismiss()
        self._update_actions()

    def _on_overlay_selection(self, component_id: str) -> None:
        if self._syncing_selection:
            return
        if self._inline_open and component_id != self._inline_component_id:
            self.inline_editor.dismiss()
            self._inline_open = False
            self._inline_component_id = ""
        self._syncing_selection = True
        try:
            if component_id:
                self.sidebar.select_component(component_id, emit=False)
            else:
                self.sidebar.clear_selection()
        finally:
            self._syncing_selection = False
        if not component_id:
            self.inline_editor.dismiss()
        self._update_actions()

    def _on_sidebar_selection(self, component_id: str) -> None:
        if self._syncing_selection:
            return
        self._select_component(component_id)
        self._ensure_selected_visible()

    def _ensure_selected_visible(self) -> None:
        if self.project is None or not self.overlay.selected_id:
            return
        if self._current_source_kind in ("camera", "video", "demo"):
            matrix = self.last_result.matrix if self.last_result is not None else None
        else:
            matrix = self._static_matrix
        if matrix is None:
            return
        comp = self.project.find_by_id(self.overlay.selected_id)
        if comp is None:
            return
        projected = transform_bbox(matrix, comp.bbox)
        rect = QRectF(
            self.canvas.mapToScene(self.canvas.viewport().rect().topLeft()),
            self.canvas.mapToScene(self.canvas.viewport().rect().bottomRight()),
        )
        if not rect.contains(QPointF(projected.x, projected.y)) or not rect.contains(
            QPointF(projected.x2, projected.y2)
        ):
            self.canvas.ensure_scene_rect_visible(
                QRectF(projected.x, projected.y, projected.w, projected.h)
            )

    def _update_actions(self) -> None:
        has_project = self.project is not None
        has_selection = bool(self.overlay.selected_id) if hasattr(self, "overlay") else False
        has_live = self.worker is not None and self.worker.isRunning()
        for action in (
            self.action_save,
            self.action_save_as,
            self.action_import_bom,
            self.action_export_board,
            self.action_open_photos,
            self.action_export_photos,
            self.action_export_bom,
            self.action_detect,
            self.action_draw,
            self.action_next,
            self.action_project_info,
        ):
            action.setEnabled(has_project)
        self.action_scan.setEnabled(has_live)
        self.action_guided_scan.setEnabled(has_live)
        for action in (self.action_edit, self.action_verify, self.action_issue, self.action_delete):
            action.setEnabled(has_project and has_selection)
        self.action_pause.setEnabled(has_live)
        self.action_labels.setEnabled(has_project)
        self.action_expand_labels.setEnabled(has_project)
        self.action_overlay.setEnabled(has_project)
        self.action_draw.setEnabled(has_project)

    # ================================================================== 快速编辑
    def edit_selected(self) -> None:
        if self.overlay.selected_id:
            self.open_inline_editor(self.overlay.selected_id)

    def open_inline_editor(self, component_id: str) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        if comp is None:
            return
        if self.overlay.selected_id != component_id:
            self._select_component(component_id)
        self.inline_editor.open_for(comp)
        self._inline_component_id = component_id
        self._inline_open = True
        self._position_inline_editor()
        self.inline_editor.raise_()

    def _position_inline_editor(self) -> None:
        if not self._inline_open or not self.inline_editor.isVisible():
            return
        cid = self.overlay.selected_id
        if not cid:
            return
        anchor = self.overlay.bubble_screen_rect(cid)
        if anchor.isNull():
            anchor = self.overlay.selected_box_screen_rect()
        if anchor.isNull():
            return
        self.inline_editor.position_near(anchor, self.canvas.viewport().rect())

    def _on_inline_saved(self, values: list, status: str) -> None:
        if self.project is None:
            self.inline_editor.hide()
            self._inline_open = False
            return
        cid = self._inline_component_id or self.overlay.selected_id
        comp = self.project.find_by_id(cid)
        if comp is None:
            self.inline_editor.hide()
            self._inline_open = False
            self._inline_component_id = ""
            return
        comp.set_core_values(values)
        comp.set_status(status)
        comp.touch()
        self.project.touch()
        self._mark_dirty()
        self.overlay.refresh_component(cid)
        self.sidebar.refresh_component(cid)
        self.inline_editor.hide()
        self._inline_open = False
        self._inline_component_id = ""
        self._update_counts()
        self.statusBar().showMessage(f"{comp.refdes} 已更新", 2500)

    def _on_inline_dismissed(self) -> None:
        self._inline_open = False
        self._inline_component_id = ""

    # ================================================================== 核对操作
    def _set_status(self, component_id: str, status: str) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        if comp is None:
            return
        comp.set_status(status)
        self.project.touch()
        self._mark_dirty()
        self.overlay.refresh_component(component_id)
        self.sidebar.refresh_component(component_id)
        self._update_counts()
        label = "已核对" if status == STATUS_VERIFIED else "已标记异常"
        self.statusBar().showMessage(f"{comp.refdes} {label}", 2000)

    def verify_selected(self) -> None:
        if self.overlay.selected_id:
            self._set_status(self.overlay.selected_id, STATUS_VERIFIED)

    def mark_issue_selected(self) -> None:
        if self.overlay.selected_id:
            self._set_status(self.overlay.selected_id, STATUS_ISSUE)

    def delete_selected(self) -> None:
        if self.overlay.selected_id:
            self.delete_component(self.overlay.selected_id)

    def delete_component(self, component_id: str) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        if comp is None:
            return
        answer = QMessageBox.question(
            self,
            "删除元件",
            f"确定删除 {comp.refdes} 及其全部参数吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.inline_editor.dismiss()
        self._inline_open = False
        self.project.remove_component(component_id)
        self._mark_dirty()
        self.overlay.rebuild(self.project)
        self.sidebar.refresh_project()
        self._clear_selection()
        self._update_counts()
        self.statusBar().showMessage(f"已删除 {comp.refdes}", 2500)

    def next_pending(self) -> None:
        if self.project is None:
            return
        ordered = sorted(self.project.components, key=lambda c: natural_key(c.refdes))
        pending = [c for c in ordered if c.status == STATUS_PENDING]
        if not pending:
            QMessageBox.information(self, "核对完成", "当前工程没有待核对元件。")
            return
        current = self.overlay.selected_id
        index = next((i for i, c in enumerate(pending) if c.id == current), -1)
        target = pending[(index + 1) % len(pending)]
        self._select_component(target.id)
        self._ensure_selected_visible()

    def _on_component_applied(self, component_id: str) -> None:
        if self.project is None:
            return
        self.project.touch()
        self._mark_dirty()
        self.overlay.refresh_component(component_id)
        self._update_counts()
        self.statusBar().showMessage("元件参数已应用", 2500)

    # ================================================================== 新增与检测
    def _toggle_draw_action(self, checked: bool) -> None:
        if self._syncing_draw:
            return
        if checked:
            if self.project is None:
                self.action_draw.setChecked(False)
                return
            if self.last_result is None and self._static_matrix is None:
                QMessageBox.information(self, "暂无可绘制画面", "请先打开工程或启动实时输入。")
                self.action_draw.setChecked(False)
                return
            self.inline_editor.dismiss()
            self.overlay.set_interactive(False)
            self.canvas.set_draw_mode(True)
            self.statusBar().showMessage("在画面上拖拽绘制元件框，松开后自动创建待核对元件。", 0)
        else:
            self.overlay.set_interactive(True)
            self.canvas.set_draw_mode(False)
            self.statusBar().clearMessage()

    def _on_draw_mode_changed(self, enabled: bool) -> None:
        self._syncing_draw = True
        self.action_draw.setChecked(enabled)
        self._syncing_draw = False
        if not enabled:
            self.overlay.set_interactive(True)
            self.statusBar().clearMessage()

    def _on_box_drawn(self, rect: QRectF) -> None:
        if self.project is None:
            return
        if self._current_source_kind in ("camera", "video", "demo"):
            matrix = self.last_result.matrix if self.last_result is not None else None
            if matrix is None:
                QMessageBox.information(
                    self,
                    "跟踪未锁定",
                    "实时画面当前未与全局图配准，暂时无法把框转换到全局坐标。\n"
                    "请让视野包含更多丝印/焊盘，等待跟踪锁定后再画框。",
                )
                return
        else:
            matrix = self._static_matrix
        if matrix is None:
            QMessageBox.information(self, "无法创建", "当前没有可用的画面坐标变换。")
            return
        inverse = invert_affine(matrix)
        pts = np.float32(
            [
                [rect.left(), rect.top()],
                [rect.right(), rect.top()],
                [rect.right(), rect.bottom()],
                [rect.left(), rect.bottom()],
            ]
        ).reshape(-1, 1, 2)
        mapped = cv2.transform(pts, inverse).reshape(-1, 2)
        x1, y1 = float(mapped[:, 0].min()), float(mapped[:, 1].min())
        x2, y2 = float(mapped[:, 0].max()), float(mapped[:, 1].max())
        bbox = BBox(x1, y1, x2 - x1, y2 - y1).clipped(
            self.project.width or self.global_image.shape[1],
            self.project.height or self.global_image.shape[0],
        )
        if bbox.w < 2 or bbox.h < 2:
            QMessageBox.information(self, "框太小", "请在画面上拖出更大的区域。")
            return
        refdes = self.project.next_refdes("X")
        comp = new_component(
            refdes,
            bbox,
            category="候选元件",
            value="",
            package="",
            note="人工框选",
            source="manual",
        )
        self.project.add_component(comp)
        self._mark_dirty()
        self.overlay.rebuild(self.project)
        self.sidebar.refresh_project()
        self._select_component(comp.id)
        self._update_counts()
        self.statusBar().showMessage(
            f"已创建 {refdes}，请在右侧栏填写位号/参数并核对。", 5000
        )

    def auto_detect(self) -> None:
        if self.project is None or self.global_image is None:
            return
        answer = QMessageBox.question(
            self,
            "自动检测候选",
            "将在全局图上检测元件候选框，并追加到现有标注中。\n"
            "自动结果仅作为起点，必须人工核对/修正。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("正在检测候选元件…")
        QApplication.processEvents()
        try:
            detections = detect_component_candidates(self.global_image)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "检测失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not detections:
            QMessageBox.information(
                self,
                "未检测到候选",
                "当前全局图未检测到符合尺寸的元件候选。\n"
                "可尝试使用框选新增，或训练 YOLO 模型替换传统检测器。",
            )
            return
        created = 0
        for detection in detections:
            refdes = self.project.next_refdes("X")
            comp = new_component(
                refdes,
                detection.bbox,
                category="候选元件",
                value="",
                package="",
                note=f"自动候选 · 置信度 {detection.confidence:.2f}",
                source="auto",
            )
            self.project.add_component(comp)
            created += 1
        self.project.touch()
        self._mark_dirty()
        self.overlay.rebuild(self.project)
        self.sidebar.refresh_project()
        self._update_counts()
        self.statusBar().showMessage(f"已追加 {created} 个候选元件，请逐一核对。", 6000)
        QMessageBox.information(
            self,
            "检测完成",
            f"已创建 {created} 个候选元件。\n请在右侧栏修正位号、参数与框位置后标记核对。",
        )

    # ================================================================== 工程操作
    def new_project(self) -> None:
        if not self._maybe_save():
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择完整电路板图（建议正视、无畸变）",
            "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not image_path:
            return
        try:
            image = read_image(image_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取图像失败", str(exc))
            return
        default_name = Path(image_path).stem
        name, ok = QInputDialog.getText(self, "工程名称", "名称：", text=default_name)
        if not ok or not name.strip():
            return
        default_path = default_project_dir() / f"{name.strip()}.pcbproj.json"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存工程",
            str(default_path),
            "PCB AR 工程 (*.pcbproj.json)",
        )
        if not save_path:
            return
        try:
            store = ProjectStore.create(save_path, name.strip(), image, mm_per_px=0.0)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "创建工程失败", str(exc))
            return
        self._open_store(store)
        self.statusBar().showMessage("工程已创建，可启动摄像头或自动检测候选。", 5000)

    def open_project_dialog(self) -> None:
        if not self._maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开工程",
            str(default_project_dir()),
            "PCB AR 工程 (*.pcbproj.json *.json)",
        )
        if path:
            self._open_path(Path(path), autoplay=False)

    def save_project(self) -> bool:
        if self.store is None:
            return self.save_project_as()
        try:
            self.store.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return False
        self._set_dirty(False)
        self._save_config(last_project=str(self.store.path))
        self.statusBar().showMessage(f"工程已保存：{self.store.path}", 4000)
        return True

    def save_project_as(self) -> bool:
        if self.project is None or self.store is None:
            return False
        stem = self.project.name.replace("/", "_").replace("\\", "_") or "pcb_project"
        default_path = default_project_dir() / f"{stem}.pcbproj.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "工程另存为",
            str(default_path),
            "PCB AR 工程 (*.pcbproj.json)",
        )
        if not path:
            return False
        try:
            self.store.save_as(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "另存失败", str(exc))
            return False
        self._set_dirty(False)
        self._save_config(last_project=str(self.store.path))
        self._update_title()
        self.statusBar().showMessage(f"工程已另存：{self.store.path}", 4000)
        return True

    def _maybe_save(self) -> bool:
        if not self._dirty or self.project is None:
            return True
        answer = QMessageBox.question(
            self,
            "保存修改",
            "当前工程有未保存的修改，是否先保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _mark_dirty(self, dirty: bool = True) -> None:
        self._set_dirty(dirty)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._update_title()

    def _update_title(self) -> None:
        name = self.project.name if self.project else "无工程"
        marker = " *" if self._dirty else ""
        mode = " · 演示" if self.project is not None and self.project.demo else ""
        if self.project is not None and self.project.map_mode in ("hybrid", "sparse"):
            mode += " · 轻量/增强"
        self.setWindowTitle(f"{APP_NAME_CN} — {name}{marker}{mode}")

    def _update_counts(self) -> None:
        if self.project is None:
            self.count_status.setText("—")
            return
        counts = self.project.status_counts()
        self.count_status.setText(
            f"已核对 {counts.get(STATUS_VERIFIED, 0)}/{self.project.count} · "
            f"待核对 {counts.get(STATUS_PENDING, 0)}"
        )

    def edit_project_info(self) -> None:
        if self.project is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("工程信息 / 比例标定")
        dialog.setMinimumWidth(440)
        form = QFormLayout(dialog)
        name_edit = QLineEdit(self.project.name)
        form.addRow("工程名称", name_edit)
        mm_spin = QDoubleSpinBox()
        mm_spin.setRange(0.0, 10.0)
        mm_spin.setDecimals(6)
        mm_spin.setSingleStep(0.001)
        mm_spin.setSuffix(" mm/px")
        mm_spin.setValue(float(self.project.mm_per_px or 0.0))
        form.addRow("物理比例", mm_spin)
        notes = QPlainTextEdit(self.project.notes)
        notes.setMaximumHeight(120)
        form.addRow("工程备注", notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = name_edit.text().strip()
        if new_name:
            self.project.name = new_name
        self.project.mm_per_px = float(mm_spin.value())
        self.project.notes = notes.toPlainText().strip()
        self.project.touch()
        self._mark_dirty()
        self.sidebar.set_project(self.project)
        self.overlay.refresh_all_content()
        self._update_title()
        self.statusBar().showMessage("工程信息已更新", 3000)

    # ================================================================== BOM
    def import_bom(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 BOM / 元件清单",
            "",
            "CSV 文件 (*.csv *.txt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            rows = load_bom(path)
            report = apply_bom(self.project, rows)
        except (BomError, Exception) as exc:  # noqa: BLE001
            QMessageBox.warning(self, "BOM 导入失败", str(exc))
            return
        if report.updated:
            self._mark_dirty()
            self.overlay.refresh_all_content()
            self.sidebar.refresh_project()
            self._update_counts()
        message = report.summary
        if report.unmatched:
            message += "\n\n未匹配位号：\n" + "、".join(report.unmatched[:25])
        if report.missing:
            message += "\n\nBOM 缺少的已标注元件：\n" + "、".join(report.missing[:25])
        QMessageBox.information(self, "BOM 导入结果", message)

    def export_board_image(self) -> None:
        if self.project is None or self.global_image is None:
            QMessageBox.information(self, "没有整张图", "请先打开或生成一个工程。")
            return
        default_path = default_project_dir() / f"{self.project.name}_整板.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出整张板图",
            str(default_path),
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        try:
            self._write_board_image(target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))

    def _write_board_image(self, target: str | Path) -> Path:
        if self.global_image is None:
            raise ValueError("没有可导出的整张图")
        target = Path(target)
        write_image(target, self.global_image)
        self.statusBar().showMessage(f"整张板图已导出：{target}", 5000)
        return target

    def open_grid_stitch(self) -> None:
        GridStitchDialog(self).exec()

    def open_photos_folder(self) -> None:
        if self.store is None:
            return
        folder = self.store.assets_dir / "photos"
        if not folder.exists():
            QMessageBox.information(self, "没有照片", "当前工程还没有保存局部照片。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def export_photos_folder(self) -> None:
        if self.project is None or self.store is None:
            return
        source = self.store.assets_dir / "photos"
        if not source.exists() or not any(source.iterdir()):
            QMessageBox.information(self, "没有照片", "当前工程没有保存局部照片。")
            return
        parent = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not parent:
            return
        destination = Path(parent) / f"{self.project.name}_photos"
        if destination.exists():
            destination = Path(parent) / (
                f"{self.project.name}_photos_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        try:
            shutil.copytree(source, destination)
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "拍照文件夹已导出", f"已复制到：\n{destination}")

    def export_bom(self) -> None:
        if self.project is None:
            return
        default_path = default_project_dir() / f"{self.project.name}_BOM.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 BOM",
            str(default_path),
            "CSV 文件 (*.csv)",
        )
        if not path:
            return
        try:
            export_bom(self.project, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"BOM 已导出：{path}", 4000)

    # ================================================================== 扫描建图
    def _require_live(self, purpose: str) -> bool:
        if self.worker is None or not self.worker.isRunning():
            QMessageBox.information(
                self,
                "需要实时画面",
                f"{purpose}需要连续画面。\n请先点击「演示」「摄像头」或「视频文件」。",
            )
            return False
        if self.worker.is_paused():
            self.worker.set_paused(False)
            self.action_pause.setChecked(False)
            self.action_pause.setText("暂停")
        return True

    def open_guided_capture(self) -> None:
        if not self._require_live("轻量扫描建图"):
            return
        if not self._maybe_save():
            return
        dialog = GuidedCaptureDialog(self._scan_frame_provider, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result is not None:
            self._create_project_from_guided_capture(dialog.result)

    def _create_project_from_guided_capture(self, result) -> None:
        if not self._maybe_save():
            return
        default_name = "轻量工程 " + datetime.now().strftime("%Y%m%d-%H%M")
        name, ok = QInputDialog.getText(self, "轻量扫描新工程", "工程名称：", text=default_name)
        if not ok or not name.strip():
            return
        default_path = default_project_dir() / f"{name.strip()}.pcbproj.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存轻量工程",
            str(default_path),
            "PCB AR 工程 (*.pcbproj.json)",
        )
        if not path:
            return
        try:
            store = ProjectStore.create(path, name.strip(), result.overview, mm_per_px=0.0)
            manifest = result.sparse_map.save(store.assets_dir)
            try:
                relative = manifest.relative_to(store.path.parent).as_posix()
            except ValueError:
                relative = str(manifest)
            store.project.sparse_map = relative
            if result.structural_map is not None:
                structural_path = result.structural_map.save(store.assets_dir / "structural_map.json")
                try:
                    structural_rel = structural_path.relative_to(store.path.parent).as_posix()
                except ValueError:
                    structural_rel = str(structural_path)
                store.project.structural_map = structural_rel
            store.project.map_mode = "hybrid"
            structure_text = (
                f"；结构基元：{result.structural_map.describe()}"
                if result.structural_map is not None
                else ""
            )
            if result.structural_match is not None:
                structure_text += (
                    f"；结构复核：{result.structural_match.inliers} 内点 / "
                    f"置信度 {result.structural_match.confidence:.0%}"
                )
            if result.pose_graph_optimized:
                structure_text += (
                    f"；位姿图：{result.pose_graph_edges} 边 / "
                    f"{result.loop_closures} 回环 / "
                    f"RMS {result.pose_rms_before:.3f}→{result.pose_rms_after:.3f}"
                )
            photos_dir = store.assets_dir / "photos"
            photos_dir.mkdir(parents=True, exist_ok=True)
            for i, jpeg in enumerate(result.photos):
                (photos_dir / f"kf_{i:03d}.jpg").write_bytes(jpeg)
            element_total = 0
            if result.element_sets:
                manifest_path = store.assets_dir / "elements_manifest.json"
                data = {
                    "keyframes": [
                        {"index": i, **es.to_dict()} for i, es in enumerate(result.element_sets)
                    ],
                    "final": result.final_elements.to_dict()
                    if result.final_elements is not None
                    else None,
                }
                manifest_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                element_total = sum(len(es.elements) for es in result.element_sets)
                structure_text += f"；元素签名：{element_total} 个"
            structure_text += f"；局部照片：{len(result.photos)} 张"
            store.project.notes = (
                f"由轻量引导捕获生成：{result.keyframe_count} 个关键帧{structure_text}。"
                "稀疏关键帧定位为默认，低分辨率总览用于全图增强/人工查看。"
                "建议先标定 mm/px 并人工核对。"
            )
            store.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "创建轻量工程失败", str(exc))
            return
        self._open_store(store)
        QMessageBox.information(
            self,
            "轻量工程已创建",
            f"已保存 {result.keyframe_count} 个关键帧。\n"
            "实时模式默认使用稀疏关键帧定位，周期性用低分辨率全图校验/回退。\n"
            "下一步可运行自动候选检测并人工核对。",
        )
        answer = QMessageBox.question(
            self,
            "自动检测",
            "是否立即对总览图运行自动候选检测？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.auto_detect()

    def open_scan_dialog(self) -> None:
        if not self._require_live("全图扫描（独立备用）"):
            return
        dialog = VideoScanDialog(self._scan_frame_provider, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_image is not None:
            self._create_project_from_scan(dialog.result_image)

    def _scan_frame_provider(self) -> tuple[np.ndarray | None, float]:
        return self.last_frame, self.last_frame_timestamp

    def _create_project_from_scan(self, image: np.ndarray) -> None:
        if not self._maybe_save():
            return
        default_name = "扫描工程 " + datetime.now().strftime("%Y%m%d-%H%M")
        name, ok = QInputDialog.getText(self, "新扫描工程", "工程名称：", text=default_name)
        if not ok or not name.strip():
            return
        default_path = default_project_dir() / f"{name.strip()}.pcbproj.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存扫描工程",
            str(default_path),
            "PCB AR 工程 (*.pcbproj.json)",
        )
        if not path:
            return
        try:
            store = ProjectStore.create(path, name.strip(), image, mm_per_px=0.0)
            store.project.map_mode = "full"
            store.project.notes = "由全图视频扫描生成。建议先标定 mm/px 并人工核对。"
            store.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "创建扫描工程失败", str(exc))
            return
        self._open_store(store)
        answer = QMessageBox.question(
            self,
            "扫描工程已创建",
            "是否立即对全局图运行自动候选检测，开始人工核对？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.auto_detect()

    # ================================================================== 显示
    def _toggle_labels(self, checked: bool) -> None:
        self.config.labels_visible = bool(checked)
        self.overlay.set_labels_visible(checked)
        if not checked:
            self.inline_editor.dismiss()
            self._inline_open = False

    def _toggle_expand_labels(self, checked: bool) -> None:
        self.config.expanded_labels = bool(checked)
        self.overlay.set_expanded_all(checked)
        if self._inline_open:
            self._position_inline_editor()

    def _toggle_overlay(self, checked: bool) -> None:
        self.overlay.set_overlay_visible(checked)
        if not checked:
            self.inline_editor.dismiss()
            self._inline_open = False

    def _on_view_changed(self, scale: float) -> None:
        self.zoom_status.setText(f"{scale * 100:.0f}%")
        # 气泡是屏幕固定尺寸的顶层图元，缩放/平移后必须重新计算锚点，
        # 否则暂停或静态图状态下标签会与元件框脱离。
        self.overlay.refresh_geometry()
        if self._inline_open:
            self._position_inline_editor()

    def _set_source_status(self, text: str, color: str) -> None:
        self.source_status.setText(text)
        self.source_status.setStyleSheet(f"color: {color}; padding-left: 4px;")

    def _set_track_status(self, text: str, color: str) -> None:
        self.track_status.setText(text)
        self.track_status.setStyleSheet(f"color: {color};")

    # ================================================================== 帮助
    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "快捷键与推荐流程",
            "<b>推荐流程</b><br>"
            "1. 输入 → 演示/摄像头，观察标注是否吸附在元件上；<br>"
            "2. 默认用「轻量扫描建图」：稳定拍照 + 红色覆盖 + 重叠校验；<br>"
            "   需要时再用「全图扫描（独立备用）」生成整图；<br>"
            "3. 自动检测候选 → 框选补漏 → 点击气泡/右侧栏核对参数；<br>"
            "4. 保存工程，导出 BOM。<br><br>"
            "<b>快捷键</b><br>"
            "Ctrl+O 打开 · Ctrl+S 保存 · Ctrl+D 演示 · Ctrl+M 轻量建图<br>"
            "Ctrl+Shift+M 全图扫描（独立备用）<br>"
            "Ctrl+G 自动检测 · Ctrl+Shift+A 框选新增 · Ctrl+E 快速编辑<br>"
            "F3 标签 · F4 全部展开 · Ctrl+Enter 标记已核对 · Ctrl+Shift+N 下一待核对<br>"
            "Ctrl+Shift+Delete 删除 · F5 暂停<br>"
            "滚轮缩放 · 空白处左键拖动平移 · 双击气泡修改 3 个核心参数",
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            f"<b>{APP_NAME_CN}</b><br>版本 {__version__}<br><br>"
            "面向显微镜/正上方相机的 PCB 元件 AR 标注与人工核对工具。<br>"
            "默认：轻量关键帧捕获 + 稀疏定位；增强：低分辨率全图周期校验。<br>"
            "技术栈：PyQt6 · OpenCV · NumPy<br><br>"
            "所有工程数据保存在本地；扫描建图与跟踪均在本机完成。",
        )

    # ================================================================== 配置/关闭
    def _save_config(self, **updates) -> None:
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        try:
            self.config.save(self.config_path)
        except Exception:  # noqa: BLE001
            logger.exception("保存应用配置失败")

    def _restore_window_state(self) -> None:
        if self.config.window_geometry:
            try:
                self.restoreGeometry(QByteArray.fromBase64(self.config.window_geometry.encode("ascii")))
            except Exception:  # noqa: BLE001
                pass
        if self.config.window_state:
            try:
                self.restoreState(QByteArray.fromBase64(self.config.window_state.encode("ascii")))
            except Exception:  # noqa: BLE001
                pass

    def closeEvent(self, event) -> None:
        if not self._maybe_save():
            event.ignore()
            return
        self._closing = True
        self._save_config(
            window_geometry=bytes(self.saveGeometry().toBase64()).decode("ascii"),
            window_state=bytes(self.saveState().toBase64()).decode("ascii"),
            labels_visible=self.action_labels.isChecked(),
            expanded_labels=self.action_expand_labels.isChecked(),
        )
        self.stop_live(clear=False)
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._inline_open:
            self._position_inline_editor()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            self.action_labels.setChecked(self.config.labels_visible)
            self.action_expand_labels.setChecked(self.config.expanded_labels)
            self.overlay.set_labels_visible(self.config.labels_visible)
            self.overlay.set_expanded_all(self.config.expanded_labels)
