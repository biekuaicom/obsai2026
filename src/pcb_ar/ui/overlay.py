"""AR 标注覆盖层：旋转矩形框、引线、常驻像素大小的参数气泡。"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PyQt6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from pcb_ar.core.geo import BBox, transform_bbox
from pcb_ar.core.models import Component, Project
from pcb_ar.ui.qt_utils import qcolor, qtransform_from_affine


class ComponentBoxItem(QGraphicsRectItem):
    def __init__(
        self,
        component: Component,
        on_click: Callable[[str], None],
        on_edit: Callable[[str], None],
    ) -> None:
        super().__init__(QRectF(*component.bbox.as_list()))
        self.component_id = component.id
        self._color = QColor(component.status_color)
        self._selected = False
        self._hover = False
        self._on_click = on_click
        self._on_edit = on_edit
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(10)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self.rect().adjusted(-5, -5, 5, 5)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        color = QColor(self._color)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._selected:
            glow = QColor(color)
            glow.setAlpha(70)
            pen = QPen(glow, 5.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect())
        pen = QPen(color, 2.7 if self._selected else 1.7)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(44 if (self._hover or self._selected) else 20)
        painter.setBrush(fill)
        painter.drawRect(self.rect())
        if self._selected:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            size = 2.6
            for x, y in (
                (self.rect().left(), self.rect().top()),
                (self.rect().right(), self.rect().top()),
                (self.rect().right(), self.rect().bottom()),
                (self.rect().left(), self.rect().bottom()),
            ):
                painter.drawEllipse(QPointF(x, y), size, size)

    def set_status_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def set_selected(self, selected: bool) -> None:
        if self._selected != selected:
            self._selected = selected
            self.update()

    def set_stale(self, stale: bool) -> None:
        self.setOpacity(0.55 if stale else 1.0)

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self.component_id)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_edit(self.component_id)
        event.accept()


class LabelBubble(QGraphicsObject):
    """固定屏幕尺寸的气泡。通过 ItemIgnoresTransformations 保证缩放后仍可读。"""

    def __init__(
        self,
        component: Component,
        on_click: Callable[[str], None],
        on_edit: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.component_id = component.id
        self._component = component
        self._on_click = on_click
        self._on_edit = on_edit
        self._expanded = False
        self._selected = False
        self._hover = False
        self._stale = False
        self._compact_size = (170, 30)
        self._expanded_size = (238, 106)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(30)
        self._font_header = QFont("Microsoft YaHei UI", 9)
        self._font_header.setBold(True)
        self._font_body = QFont("Microsoft YaHei UI", 9)
        self._font_tiny = QFont("Microsoft YaHei UI", 8)
        self._measure()
        self._refresh_tooltip()

    def _measure(self) -> None:
        fm_h = QFontMetricsF(self._font_header)
        fm_b = QFontMetricsF(self._font_body)
        text = self._component.refdes
        value = self._component.primary_value
        width = 24 + fm_h.horizontalAdvance(text) + 7 + min(120.0, fm_b.horizontalAdvance(value)) + 18
        width = max(116.0, min(250.0, width))
        self._compact_size = (int(round(width)), 30)
        # 展开态：标题 + 最多 3 行参数 + 内边距
        max_line = 0.0
        for name, val in self._component.inline_lines(limit=3):
            max_line = max(max_line, fm_b.horizontalAdvance(f"{name}  {val}"))
        width = max(228.0, min(294.0, 18 + max_line + 18))
        self._expanded_size = (int(round(width)), 104)
        self.prepareGeometryChange()
        self.update()

    def size_for(self, expanded: bool) -> tuple[int, int]:
        return self._expanded_size if expanded else self._compact_size

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self.prepareGeometryChange()
            self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def set_stale(self, stale: bool) -> None:
        self._stale = stale
        self.update()

    def set_component(self, component: Component) -> None:
        self._component = component
        self._measure()
        self._refresh_tooltip()
        self.update()

    def _refresh_tooltip(self) -> None:
        comp = self._component
        lines = [f"{comp.refdes}  [{comp.status_label}]", f"类别：{comp.category}"]
        for name, value in comp.inline_lines(limit=3):
            lines.append(f"{name}：{value or '—'}")
        lines.append("单击选中，再次单击或双击快速修改 3 个核心参数")
        self.setToolTip("\n".join(lines))

    def boundingRect(self) -> QRectF:  # noqa: N802
        width, height = self.size_for(self._expanded)
        return QRectF(-1, -1, width + 4, height + 6)

    def shape(self) -> QPainterPath:
        width, height = self.size_for(self._expanded)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, width, height), 7, 7)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        comp = self._component
        width, height = self.size_for(self._expanded)
        status = QColor(comp.status_color)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # 阴影
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 95))
        painter.drawRoundedRect(QRectF(1.5, 2.5, width, height), 8, 8)
        # 主体
        bg = QColor(11, 20, 32, 244)
        painter.setBrush(bg)
        border = QColor(status)
        border.setAlpha(230 if (self._selected or self._hover) else 150)
        pen = QPen(border, 2.0 if self._selected else 1.1)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(0, 0, width, height), 7, 7)
        # 左侧状态条
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status)
        painter.drawRoundedRect(QRectF(0.5, 7, 2.6, height - 14), 1.3, 1.3)
        # 状态点
        painter.setBrush(status)
        painter.drawEllipse(QPointF(13, 15), 3.6, 3.6)

        header_font = QFont(self._font_header)
        body_font = QFont(self._font_body)
        tiny_font = QFont(self._font_tiny)
        fm_h = QFontMetricsF(header_font)
        fm_b = QFontMetricsF(body_font)
        painter.setFont(header_font)
        painter.setPen(QColor("#F1F7FD"))
        header_width = fm_h.horizontalAdvance(comp.refdes)
        painter.drawText(QPointF(23, 19), comp.refdes)

        if not self._expanded:
            painter.setFont(body_font)
            painter.setPen(QColor("#9FB2C5"))
            x = 23 + header_width + 7
            available = width - x - 10
            value = fm_b.elidedText(comp.primary_value or comp.category, Qt.TextElideMode.ElideRight, available)
            painter.drawText(QPointF(x, 19), value)
            if self._stale:
                painter.setFont(tiny_font)
                painter.setPen(QColor("#F5A524"))
                painter.drawText(QPointF(width - 28, 19), "跟丢")
            return

        # 状态胶囊
        pill_w, pill_h = 46.0, 17.0
        pill_rect = QRectF(width - pill_w - 8, 5, pill_w, pill_h)
        pill_bg = QColor(status)
        pill_bg.setAlpha(35)
        painter.setBrush(pill_bg)
        painter.setPen(QPen(QColor(status.red(), status.green(), status.blue(), 120), 0.8))
        painter.drawRoundedRect(pill_rect, 8, 8)
        painter.setFont(tiny_font)
        painter.setPen(status)
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, comp.status_label)

        # 分隔线
        painter.setPen(QPen(QColor(40, 58, 78, 180), 0.8))
        painter.drawLine(QPointF(11, 27.5), QPointF(width - 10, 27.5))

        lines = comp.inline_lines(limit=3)
        while len(lines) < 3:
            lines.append((f"参数{len(lines) + 1}", ""))
        painter.setFont(body_font)
        for index, (name, value) in enumerate(lines[:3]):
            y = 43.5 + index * 18.3
            painter.setPen(QColor("#8294A8"))
            label = fm_b.elidedText(name or f"参数{index + 1}", Qt.TextElideMode.ElideRight, 52)
            painter.drawText(QPointF(13, y), label)
            painter.setPen(QColor("#DDE9F4") if value else QColor("#526274"))
            available = width - 75
            text = fm_b.elidedText(value or "—", Qt.TextElideMode.ElideRight, available)
            painter.drawText(QPointF(70, y), text)
        painter.setFont(tiny_font)
        painter.setPen(QColor("#526B84"))
        hint = "双击编辑 · 右侧栏查看全部字段"
        painter.drawText(
            QPointF(13, height - 8),
            fm_b.elidedText(hint, Qt.TextElideMode.ElideRight, width - 26),
        )

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self.component_id)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_edit(self.component_id)
        event.accept()


class AnnotationOverlay(QObject):
    selection_changed = pyqtSignal(str)
    edit_requested = pyqtSignal(str)

    def __init__(self, scene: QGraphicsScene, view: QGraphicsView, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.view = view
        self.project: Project | None = None
        self.matrix: np.ndarray | None = None
        self.state = "searching"
        self._items: dict[str, dict[str, QGraphicsItem]] = {}
        self._selected_id = ""
        self._labels_visible = True
        self._expanded_all = False
        self._overlay_visible = True
        self._interactive = True

    # ------------------------------------------------------------------ 构建
    def set_project(self, project: Project | None) -> None:
        self._clear_items()
        self.project = project
        self._selected_id = ""
        if project is not None:
            self._build_items(project)
        self.emit_selection()
        self.refresh_geometry()

    def rebuild(self, project: Project | None = None) -> None:
        if project is not None:
            self.project = project
        keep = self._selected_id
        self.set_project(self.project)
        if keep and self.project is not None and self.project.find_by_id(keep):
            self.select(keep, emit=True)

    def _clear_items(self) -> None:
        for entry in self._items.values():
            for item in entry.values():
                if item.scene() is self.scene:
                    self.scene.removeItem(item)
        self._items.clear()

    def _build_items(self, project: Project) -> None:
        for comp in project.components:
            box = ComponentBoxItem(comp, self._on_box_clicked, self._request_edit)
            box.setToolTip(
                f"{comp.refdes} · {comp.status_label}\n{comp.category}  {comp.primary_value}\n"
                "单击选中，双击编辑"
            )
            bubble = LabelBubble(comp, self._on_bubble_clicked, self._request_edit)
            leader = QGraphicsLineItem()
            pen = QPen(QColor(comp.status_color), 1.1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            pen.setDashPattern([4, 4])
            leader.setPen(pen)
            leader.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            leader.setZValue(20)
            self.scene.addItem(box)
            self.scene.addItem(leader)
            self.scene.addItem(bubble)
            self._items[comp.id] = {"box": box, "bubble": bubble, "leader": leader}
        self._apply_visibility()

    # ------------------------------------------------------------------ 交互
    @property
    def selected_id(self) -> str:
        return self._selected_id

    def _on_box_clicked(self, component_id: str) -> None:
        self.select(component_id)

    def _on_bubble_clicked(self, component_id: str) -> None:
        if self._selected_id == component_id:
            self.edit_requested.emit(component_id)
        else:
            self.select(component_id)

    def _request_edit(self, component_id: str) -> None:
        self.select(component_id)
        self.edit_requested.emit(component_id)

    def select(self, component_id: str, emit: bool = True) -> None:
        if component_id and component_id not in self._items:
            component_id = ""
        if component_id == self._selected_id:
            if emit:
                self.selection_changed.emit(component_id)
            return
        if self._selected_id in self._items:
            self._items[self._selected_id]["box"].set_selected(False)
            self._items[self._selected_id]["bubble"].set_selected(False)
        self._selected_id = component_id
        if component_id in self._items:
            self._items[component_id]["box"].set_selected(True)
            self._items[component_id]["bubble"].set_selected(True)
            self.refresh_geometry()
        else:
            self.refresh_geometry()
        if emit:
            self.selection_changed.emit(component_id)

    def clear_selection(self, emit: bool = True) -> None:
        self.select("", emit=emit)

    def emit_selection(self) -> None:
        self.selection_changed.emit(self._selected_id)

    def set_interactive(self, interactive: bool) -> None:
        self._interactive = bool(interactive)
        buttons = Qt.MouseButton.LeftButton if interactive else Qt.MouseButton.NoButton
        for entry in self._items.values():
            entry["box"].setAcceptedMouseButtons(buttons)
            entry["bubble"].setAcceptedMouseButtons(buttons)

    def set_labels_visible(self, visible: bool) -> None:
        self._labels_visible = bool(visible)
        self.refresh_geometry()

    def set_expanded_all(self, expanded: bool) -> None:
        self._expanded_all = bool(expanded)
        self.refresh_geometry()

    def set_overlay_visible(self, visible: bool) -> None:
        self._overlay_visible = bool(visible)
        self._apply_visibility()
        self.refresh_geometry()

    # ------------------------------------------------------------------ 位姿
    def update_pose(self, matrix: np.ndarray | None, state: str = "locked") -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float32)
        self.state = state
        self._apply_visibility()
        if self.matrix is None or self.project is None:
            self._hide_annotations()
            return
        try:
            qtransform = qtransform_from_affine(self.matrix)
        except ValueError:
            self._hide_annotations()
            return
        stale = state in ("stale", "lost")
        for comp in self.project.components:
            entry = self._items.get(comp.id)
            if entry is None:
                continue
            entry["box"].setTransform(qtransform)
            entry["box"].set_stale(stale)
            entry["bubble"].set_stale(state == "stale")
        self.refresh_geometry()

    def refresh_geometry(self) -> None:
        if self.project is None:
            return
        if not self._overlay_visible:
            self._hide_annotations()
            return
        if self.matrix is None:
            self._hide_annotations()
            return
        scene_rect = self.scene.sceneRect()
        if scene_rect.isEmpty():
            self._hide_annotations()
            return
        stale = self.state in ("stale", "lost")
        for comp in self.project.components:
            entry = self._items.get(comp.id)
            if entry is None:
                continue
            projected = transform_bbox(self.matrix, comp.bbox)
            intersects = projected.intersects(
                BBox(
                    scene_rect.left() - 20,
                    scene_rect.top() - 20,
                    scene_rect.width() + 40,
                    scene_rect.height() + 40,
                )
            )
            entry["box"].setVisible(intersects)
            if not intersects:
                entry["bubble"].setVisible(False)
                entry["leader"].setVisible(False)
        self._layout_labels()

    def _layout_labels(self) -> None:
        if self.project is None or self.matrix is None:
            return
        if not self._labels_visible:
            for entry in self._items.values():
                entry["bubble"].setVisible(False)
                entry["leader"].setVisible(False)
            return
        viewport = self.view.viewport().rect()
        if viewport.width() < 60 or viewport.height() < 60:
            return
        scene_rect = self.scene.sceneRect()
        visible_items: list[tuple[float, float, Component, dict, QRect]] = []
        for comp in self.project.components:
            entry = self._items.get(comp.id)
            if entry is None or not entry["box"].isVisible():
                continue
            projected = transform_bbox(self.matrix, comp.bbox)
            if not projected.intersects(
                BBox(
                    scene_rect.left() - 5,
                    scene_rect.top() - 5,
                    scene_rect.width() + 10,
                    scene_rect.height() + 10,
                )
            ):
                continue
            top_left = self.view.mapFromScene(QPointF(projected.x, projected.y))
            bottom_right = self.view.mapFromScene(QPointF(projected.x2, projected.y2))
            box_screen = QRect(top_left, bottom_right).normalized()
            if not box_screen.intersects(viewport.adjusted(-30, -30, 30, 30)):
                continue
            # 框大部分移出视口时不再把气泡钉在边缘，避免形成误导性的“幽灵标签”。
            if not viewport.adjusted(-8, -8, 8, 8).contains(box_screen.center()):
                continue
            visible_items.append((float(box_screen.y()), float(box_screen.x()), comp, entry, box_screen))

        visible_items.sort(
            key=lambda item: (0 if item[2].id == self._selected_id else 1, item[0], item[1])
        )
        occupied: list[QRect] = []
        for _, _, comp, entry, box_screen in visible_items:
            bubble: LabelBubble = entry["bubble"]  # type: ignore[assignment]
            leader: QGraphicsLineItem = entry["leader"]  # type: ignore[assignment]
            expanded = self._expanded_all or comp.id == self._selected_id
            bubble.set_expanded(expanded)
            width, height = bubble.size_for(expanded)
            candidates = self._candidate_positions(box_screen, width, height, viewport)
            chosen = candidates[0]
            best_score = float("inf")
            for rect, distance in candidates:
                overlap = sum(self._intersection_area(rect, other) for other in occupied)
                score = overlap * 10.0 + distance * 0.02
                if score < best_score:
                    best_score = score
                    chosen = rect
            occupied.append(chosen.adjusted(-2, -2, 2, 2))
            bubble.setPos(self.view.mapToScene(chosen.topLeft()))
            bubble.setVisible(True)
            # 引线：从框上边中点到气泡靠近框的一侧
            start_screen = QPointF(box_screen.center().x(), box_screen.top())
            if chosen.center().y() <= box_screen.center().y():
                end_screen = QPointF(chosen.center().x(), chosen.bottom())
            else:
                end_screen = QPointF(chosen.center().x(), chosen.top())
            start = self.view.mapToScene(start_screen.toPoint())
            end = self.view.mapToScene(end_screen.toPoint())
            leader.setLine(start.x(), start.y(), end.x(), end.y())
            color = QColor(comp.status_color)
            color.setAlpha(160)
            pen = QPen(color, 1.1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            pen.setDashPattern([4, 4])
            leader.setPen(pen)
            leader.setVisible(True)
            leader.setOpacity(0.55 if self.state == "stale" else 1.0)

    def _candidate_positions(
        self, box: QRect, width: int, height: int, viewport: QRect
    ) -> list[tuple[QRect, float]]:
        gap = 8
        raw = [
            QRect(box.left(), box.top() - height - gap, width, height),
            QRect(box.left(), box.bottom() + gap, width, height),
            QRect(box.right() - width, box.top() - height - gap, width, height),
            QRect(box.right() - width, box.bottom() + gap, width, height),
            QRect(box.right() + gap, box.center().y() - height // 2, width, height),
            QRect(box.left() - width - gap, box.center().y() - height // 2, width, height),
        ]
        results: list[tuple[QRect, float]] = []
        for rect in raw:
            clamped = self._clamp_rect(rect, viewport)
            center = clamped.center()
            dx = max(0, box.left() - center.x(), center.x() - box.right())
            dy = max(0, box.top() - center.y(), center.y() - box.bottom())
            distance = math.hypot(dx, dy)
            results.append((clamped, distance))
        return results

    @staticmethod
    def _clamp_rect(rect: QRect, viewport: QRect) -> QRect:
        x = min(max(rect.x(), viewport.left() + 5), viewport.right() - rect.width() - 5)
        y = min(max(rect.y(), viewport.top() + 5), viewport.bottom() - rect.height() - 5)
        return QRect(x, y, rect.width(), rect.height())

    @staticmethod
    def _intersection_area(a: QRect, b: QRect) -> int:
        inter = a.intersected(b)
        if inter.isEmpty():
            return 0
        return inter.width() * inter.height()

    def _apply_visibility(self) -> None:
        for entry in self._items.values():
            entry["box"].setVisible(self._overlay_visible)
            if not self._overlay_visible:
                entry["bubble"].setVisible(False)
                entry["leader"].setVisible(False)

    def _hide_annotations(self) -> None:
        for entry in self._items.values():
            entry["box"].setVisible(False)
            entry["bubble"].setVisible(False)
            entry["leader"].setVisible(False)

    # ------------------------------------------------------------------ 内容刷新
    def refresh_component(self, component_id: str) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        entry = self._items.get(component_id)
        if comp is None or entry is None:
            return
        box: ComponentBoxItem = entry["box"]  # type: ignore[assignment]
        bubble: LabelBubble = entry["bubble"]  # type: ignore[assignment]
        box.setRect(QRectF(*comp.bbox.as_list()))
        box.set_status_color(comp.status_color)
        box.setToolTip(
            f"{comp.refdes} · {comp.status_label}\n{comp.category}  {comp.primary_value}\n"
            "单击选中，双击编辑"
        )
        bubble.set_component(comp)
        if comp.id == self._selected_id:
            box.set_selected(True)
            bubble.set_selected(True)
        self.refresh_geometry()

    def refresh_all_content(self) -> None:
        if self.project is None:
            return
        for comp in self.project.components:
            entry = self._items.get(comp.id)
            if entry is not None:
                entry["box"].setToolTip(
                    f"{comp.refdes} · {comp.status_label}\n{comp.category}  {comp.primary_value}\n"
                    "单击选中，双击编辑"
                )
                entry["bubble"].set_component(comp)
        self.refresh_geometry()

    # ------------------------------------------------------------------ 查询
    def bubble_screen_rect(self, component_id: str) -> QRect:
        entry = self._items.get(component_id)
        if entry is None or not entry["bubble"].isVisible():
            return QRect()
        bubble: LabelBubble = entry["bubble"]  # type: ignore[assignment]
        width, height = bubble.size_for(bubble.expanded)
        top_left = self.view.mapFromScene(bubble.pos())
        return QRect(top_left, QPoint(top_left.x() + width, top_left.y() + height))

    def selected_box_screen_rect(self) -> QRect:
        if not self._selected_id or self.project is None or self.matrix is None:
            return QRect()
        comp = self.project.find_by_id(self._selected_id)
        if comp is None:
            return QRect()
        projected = transform_bbox(self.matrix, comp.bbox)
        tl = self.view.mapFromScene(QPointF(projected.x, projected.y))
        br = self.view.mapFromScene(QPointF(projected.x2, projected.y2))
        return QRect(tl, br).normalized()
