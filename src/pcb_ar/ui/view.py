"""视频/图像画布视图：缩放、平移、框选新增。"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)

from pcb_ar.ui.qt_utils import qimage_from_bgr


class CanvasView(QGraphicsView):
    view_changed = pyqtSignal(float)
    empty_clicked = pyqtSignal()
    box_drawn = pyqtSignal(object)
    draw_mode_changed = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setBackgroundBrush(QColor("#05080D"))
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._background = QGraphicsPixmapItem()
        self._background.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._background.setZValue(-100)
        self._scene.addItem(self._background)
        self._has_frame = False
        self._auto_fit = True
        self._draw_mode = False
        self._drawing = False
        self._draw_start = QPointF()
        self._rubber: QGraphicsRectItem | None = None
        self._panning = False
        self._pan_last = QPoint()
        self._pan_moved = False
        self._ready = False

        self._placeholder = QLabel(self.viewport())
        self._placeholder.setObjectName("CanvasPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "QLabel#CanvasPlaceholder {"
            " color: #5D7186; background: rgba(10, 17, 27, 150);"
            " border: 1px dashed #26384D; border-radius: 10px;"
            " padding: 24px; font-size: 14px;"
            "}"
        )
        self._placeholder.setText(
            "等待图像输入\n\n"
            "点击工具栏「演示」快速体验，或打开工程 / 摄像头 / 视频文件\n"
            "滚轮缩放 · 左键空白拖动平移 · 双击元件气泡快速修改"
        )
        self._placeholder.show()
        self._ready = True

    # ------------------------------------------------------------------ 图像
    @property
    def has_frame(self) -> bool:
        return self._has_frame

    def set_frame(self, frame) -> None:
        if frame is None:
            return
        image: QImage = qimage_from_bgr(frame)
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        old_size = self._scene.sceneRect().size()
        self._background.setPixmap(pixmap)
        self._background.setPos(0, 0)
        new_size = QSize(frame.shape[1], frame.shape[0])
        size_changed = not self._has_frame or old_size != new_size
        self._has_frame = True
        self._placeholder.hide()
        if size_changed:
            self._scene.setSceneRect(QRectF(0, 0, new_size.width(), new_size.height()))
            if self._auto_fit or not self._has_frame:
                QTimer.singleShot(0, self.fit_to_window)
        self._scene.invalidate(QRectF(0, 0, new_size.width(), new_size.height()))
        self.viewport().update()

    def clear_frame(self) -> None:
        self._background.setPixmap(QPixmap())
        self._has_frame = False
        self._placeholder.show()
        self._placeholder.raise_()

    def set_placeholder(self, text: str) -> None:
        if not self._has_frame:
            self._placeholder.setText(text)
            self._placeholder.show()
            self._placeholder.raise_()

    # ------------------------------------------------------------------ 缩放
    @property
    def current_scale(self) -> float:
        return float(self.transform().m11())

    def fit_to_window(self) -> None:
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self._auto_fit = True
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_view_changed()

    def zoom_100(self) -> None:
        if not self._has_frame:
            return
        self._auto_fit = False
        self.resetTransform()
        self._emit_view_changed()

    def zoom_by(self, factor: float, anchor_pos: QPoint | None = None) -> None:
        if not self._has_frame:
            return
        new_scale = self.current_scale * factor
        if new_scale < 0.02 or new_scale > 12.0:
            return
        self._auto_fit = False
        if anchor_pos is not None:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.scale(factor, factor)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.scale(factor, factor)
        self._emit_view_changed()

    def ensure_scene_rect_visible(self, rect: QRectF) -> None:
        if rect.isEmpty():
            return
        self.ensureVisible(rect, 80, 80)

    # ------------------------------------------------------------------ 框选
    @property
    def draw_mode(self) -> bool:
        return self._draw_mode

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if not enabled and self._rubber is not None:
            self._scene.removeItem(self._rubber)
            self._rubber = None
        self.draw_mode_changed.emit(self._draw_mode)

    def _scene_pos(self, pos: QPoint) -> QPointF:
        point = self.mapToScene(pos)
        rect = self._scene.sceneRect()
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)

    # ------------------------------------------------------------------ 鼠标
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._draw_mode and self._has_frame:
            self._drawing = True
            self._draw_start = self._scene_pos(event.position().toPoint())
            self._rubber = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
            pen = QPen(QColor("#2DD4A7"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber.setPen(pen)
            self._rubber.setBrush(QColor(45, 212, 167, 28))
            self._rubber.setZValue(900)
            self._scene.addItem(self._rubber)
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event.position().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is None or not hasattr(item, "component_id"):
                self.empty_clicked.emit()
                self._start_pan(event.position().toPoint())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing and self._rubber is not None:
            current = self._scene_pos(event.position().toPoint())
            self._rubber.setRect(QRectF(self._draw_start, current).normalized())
            event.accept()
            return
        if self._panning:
            delta = event.position().toPoint() - self._pan_last
            if abs(delta.x()) + abs(delta.y()) > 2:
                self._pan_moved = True
            self._pan_last = event.position().toPoint()
            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            self._emit_view_changed()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            rect = self._rubber.rect() if self._rubber is not None else QRectF()
            if self._rubber is not None:
                self._scene.removeItem(self._rubber)
                self._rubber = None
            min_size = max(4.0, 8.0 / max(self.current_scale, 1e-6))
            if rect.width() >= min_size and rect.height() >= min_size:
                self.box_drawn.emit(rect)
            self.set_draw_mode(False)
            event.accept()
            return
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self.setCursor(
                Qt.CursorShape.CrossCursor if self._draw_mode else Qt.CursorShape.ArrowCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        if not self._has_frame:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.18 if delta > 0 else 1 / 1.18
        self.zoom_by(factor, event.position().toPoint())
        event.accept()

    def _start_pan(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_moved = False
        self._pan_last = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._emit_view_changed()

    # ------------------------------------------------------------------ 布局
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_placeholder()
        if self._auto_fit and self._has_frame:
            QTimer.singleShot(0, self._fit_if_needed)

    def _fit_if_needed(self) -> None:
        if self._auto_fit and self._has_frame:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._emit_view_changed()

    def _position_placeholder(self) -> None:
        width = max(240, min(640, self.viewport().width() - 100))
        height = max(120, min(220, self.viewport().height() - 140))
        self._placeholder.setGeometry(
            (self.viewport().width() - width) // 2,
            (self.viewport().height() - height) // 2,
            width,
            height,
        )

    def _emit_view_changed(self) -> None:
        if self._ready:
            self.view_changed.emit(self.current_scale)
