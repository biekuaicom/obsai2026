"""右侧工程栏：元件列表、核心参数、扩展参数与核对操作。"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pcb_ar.core.geo import BBox
from pcb_ar.core.models import (
    CORE_SLOT_COUNT,
    STATUS_ISSUE,
    STATUS_LABELS,
    STATUS_PENDING,
    STATUS_VERIFIED,
    STATUSES,
    Component,
    Project,
)
from pcb_ar.ui.qt_utils import natural_key


def _status_icon(color: str, size: int = 10) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)


class SidebarPanel(QWidget):
    selection_requested = pyqtSignal(str)
    component_applied = pyqtSignal(str)
    verify_requested = pyqtSignal(str)
    issue_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    next_pending_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.project: Project | None = None
        self._current_id = ""
        self._refreshing = False
        self._loading_detail = False
        self._build_ui()
        self._set_detail_enabled(False)
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------ 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        header = QFrame()
        header.setObjectName("Card")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(11, 9, 11, 9)
        header_layout.setSpacing(5)
        self.project_name = QLabel("未打开工程")
        self.project_name.setObjectName("ProjectName")
        header_layout.addWidget(self.project_name)
        self.project_meta = QLabel("等待载入…")
        self.project_meta.setObjectName("Muted")
        header_layout.addWidget(self.project_meta)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        header_layout.addWidget(self.progress)
        self.stats_label = QLabel("—")
        self.stats_label.setObjectName("Muted")
        header_layout.addWidget(self.stats_label)
        root.addWidget(header)

        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索位号 / 参数 / 备注…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.refresh_list)
        search_row.addWidget(self.search_edit, 1)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部", "all")
        self.filter_combo.addItem("待核对", STATUS_PENDING)
        self.filter_combo.addItem("已核对", STATUS_VERIFIED)
        self.filter_combo.addItem("异常", STATUS_ISSUE)
        self.filter_combo.currentIndexChanged.connect(self.refresh_list)
        search_row.addWidget(self.filter_combo)
        root.addLayout(search_row)

        list_header = QHBoxLayout()
        list_header.setSpacing(6)
        self.list_count = QLabel("0 个元件")
        self.list_count.setObjectName("SectionTitle")
        list_header.addWidget(self.list_count)
        list_header.addStretch(1)
        self.next_button = QPushButton("下一待核对 →")
        self.next_button.setObjectName("Ghost")
        self.next_button.setToolTip("跳转到下一个待核对元件（N）")
        self.next_button.clicked.connect(self.next_pending_requested.emit)
        list_header.addWidget(self.next_button)
        root.addLayout(list_header)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setAlternatingRowColors(False)
        self.list_widget.setMinimumHeight(170)
        self.list_widget.setMaximumHeight(290)
        self.list_widget.currentItemChanged.connect(self._on_list_current_changed)
        root.addWidget(self.list_widget, 2)

        self._stack = QStackedWidget()
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(16, 24, 16, 24)
        empty_label = QLabel("从列表或画布中选择一个元件\n查看并核对全部参数")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setObjectName("Muted")
        empty_layout.addStretch(1)
        empty_layout.addWidget(empty_label)
        empty_layout.addStretch(1)
        self._stack.addWidget(empty)
        self._stack.addWidget(self._build_detail_scroll())
        root.addWidget(self._stack, 5)

    def _build_detail_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        content = QWidget()
        content.setStyleSheet("QWidget { background: transparent; }")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 2, 4)
        layout.setSpacing(7)

        identity = QGroupBox("元件属性")
        identity_form = QFormLayout(identity)
        identity_form.setContentsMargins(9, 6, 9, 8)
        identity_form.setHorizontalSpacing(8)
        identity_form.setVerticalSpacing(5)
        self.refdes_edit = QLineEdit()
        self.refdes_edit.setMaxLength(32)
        identity_form.addRow("位号", self.refdes_edit)
        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        for category in ("电阻", "电容", "芯片", "连接器", "二极管", "电感", "晶振", "开关", "候选元件", "未分类"):
            self.category_combo.addItem(category)
        identity_form.addRow("类别", self.category_combo)
        self.status_combo = QComboBox()
        for key in STATUSES:
            self.status_combo.addItem(STATUS_LABELS[key], key)
        identity_form.addRow("状态", self.status_combo)
        layout.addWidget(identity)

        core_group = QGroupBox("核心参数（气泡显示）")
        core_layout = QVBoxLayout(core_group)
        core_layout.setContentsMargins(9, 6, 9, 8)
        core_layout.setSpacing(4)
        hint = QLabel("最多 3 个；在画布气泡中双击可快速修改这些值。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        core_layout.addWidget(hint)
        self.core_name_edits: list[QLineEdit] = []
        self.core_value_edits: list[QLineEdit] = []
        for index in range(CORE_SLOT_COUNT):
            row = QHBoxLayout()
            row.setSpacing(5)
            name = QLineEdit()
            name.setPlaceholderText(f"参数{index + 1}")
            name.setFixedWidth(72)
            value = QLineEdit()
            value.setPlaceholderText("值")
            row.addWidget(name)
            row.addWidget(value, 1)
            core_layout.addLayout(row)
            self.core_name_edits.append(name)
            self.core_value_edits.append(value)
        layout.addWidget(core_group)

        ext_group = QGroupBox("扩展参数")
        ext_layout = QVBoxLayout(ext_group)
        ext_layout.setContentsMargins(9, 6, 9, 8)
        ext_layout.setSpacing(5)
        self.ext_table = QTableWidget(0, 2)
        self.ext_table.setHorizontalHeaderLabels(["参数名", "值"])
        self.ext_table.verticalHeader().setVisible(False)
        self.ext_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.ext_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ext_table.setMinimumHeight(105)
        self.ext_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.ext_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        ext_layout.addWidget(self.ext_table)
        ext_buttons = QHBoxLayout()
        ext_buttons.setSpacing(5)
        add_ext = QPushButton("+ 添加字段")
        add_ext.setObjectName("Ghost")
        add_ext.clicked.connect(self._add_ext_row)
        del_ext = QPushButton("删除选中")
        del_ext.setObjectName("Ghost")
        del_ext.clicked.connect(self._remove_ext_rows)
        ext_buttons.addWidget(add_ext)
        ext_buttons.addWidget(del_ext)
        ext_buttons.addStretch(1)
        ext_layout.addLayout(ext_buttons)
        layout.addWidget(ext_group)

        geometry = QGroupBox("位置与尺寸（全局像素）")
        geo_layout = QGridLayout(geometry)
        geo_layout.setContentsMargins(9, 7, 9, 8)
        geo_layout.setHorizontalSpacing(6)
        geo_layout.setVerticalSpacing(5)
        self.x_spin = self._coord_spin()
        self.y_spin = self._coord_spin()
        self.w_spin = self._coord_spin(minimum=0.5)
        self.h_spin = self._coord_spin(minimum=0.5)
        geo_layout.addWidget(QLabel("X"), 0, 0)
        geo_layout.addWidget(self.x_spin, 0, 1)
        geo_layout.addWidget(QLabel("Y"), 0, 2)
        geo_layout.addWidget(self.y_spin, 0, 3)
        geo_layout.addWidget(QLabel("宽"), 1, 0)
        geo_layout.addWidget(self.w_spin, 1, 1)
        geo_layout.addWidget(QLabel("高"), 1, 2)
        geo_layout.addWidget(self.h_spin, 1, 3)
        self.mm_label = QLabel("未标定 mm/px")
        self.mm_label.setObjectName("Muted")
        self.mm_label.setWordWrap(True)
        geo_layout.addWidget(self.mm_label, 2, 0, 1, 4)
        for spin in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
            spin.valueChanged.connect(lambda _value: self._update_mm_label())
        layout.addWidget(geometry)

        notes_group = QGroupBox("备注 / 工艺说明")
        notes_layout = QVBoxLayout(notes_group)
        notes_layout.setContentsMargins(9, 6, 9, 8)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("记录供应商、批次、工艺注意事项等…")
        self.notes_edit.setMaximumHeight(92)
        notes_layout.addWidget(self.notes_edit)
        layout.addWidget(notes_group)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.apply_button = QPushButton("应用修改")
        self.apply_button.setObjectName("Primary")
        self.apply_button.clicked.connect(self._apply_changes)
        self.verify_button = QPushButton("标记已核对")
        self.verify_button.clicked.connect(lambda: self._verify_with_apply(STATUS_VERIFIED))
        self.issue_button = QPushButton("标记异常")
        self.issue_button.setObjectName("Danger")
        self.issue_button.clicked.connect(lambda: self._verify_with_apply(STATUS_ISSUE))
        actions.addWidget(self.apply_button, 2)
        actions.addWidget(self.verify_button, 1)
        actions.addWidget(self.issue_button, 1)
        layout.addLayout(actions)

        danger_row = QHBoxLayout()
        danger_row.addStretch(1)
        self.delete_button = QPushButton("删除此元件")
        self.delete_button.setObjectName("Ghost")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self._current_id))
        danger_row.addWidget(self.delete_button)
        layout.addLayout(danger_row)
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _coord_spin(minimum: float = 0.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, 1_000_000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" px")
        spin.setKeyboardTracking(False)
        return spin

    # ------------------------------------------------------------------ 工程
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self._current_id = ""
        if project is None:
            self.project_name.setText("未打开工程")
            self.project_meta.setText("等待载入…")
            self.stats_label.setText("—")
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.list_widget.clear()
            self._stack.setCurrentIndex(0)
            self._set_detail_enabled(False)
            return
        self.project_name.setText(project.name)
        mode = "轻量+全图增强" if project.map_mode == "hybrid" else ("纯稀疏" if project.map_mode == "sparse" else "全图")
        self.project_meta.setText(
            f"{project.width} × {project.height} px · "
            + (f"{project.mm_per_px:.4f} mm/px" if project.mm_per_px > 0 else "未标定物理比例")
            + f" · {mode}"
        )
        self.refresh_project()
        self._stack.setCurrentIndex(0)
        self._set_detail_enabled(False)

    def refresh_project(self) -> None:
        if self.project is None:
            return
        counts = self.project.status_counts()
        total = max(1, self.project.count)
        verified = counts.get(STATUS_VERIFIED, 0)
        pending = counts.get(STATUS_PENDING, 0)
        issue = counts.get(STATUS_ISSUE, 0)
        self.progress.setRange(0, total)
        self.progress.setValue(verified)
        self.stats_label.setText(
            f'<span style="color:#2DD4A7">已核对 {verified}</span> / {self.project.count} · '
            f'<span style="color:#F5A524">待核对 {pending}</span> · '
            f'<span style="color:#FF5C7A">异常 {issue}</span>'
        )
        self.refresh_list()

    def refresh_list(self, *_args) -> None:
        if self.project is None:
            return
        query = self.search_edit.text().strip().lower()
        filter_key = self.filter_combo.currentData()
        selected = self._current_id
        self._refreshing = True
        self.list_widget.clear()
        filtered = 0
        for comp in sorted(self.project.components, key=lambda c: natural_key(c.refdes)):
            haystack = " ".join(
                [
                    comp.refdes,
                    comp.category,
                    comp.primary_value,
                    comp.notes,
                    *[f"{p.name} {p.value}" for p in comp.core],
                    *[f"{k} {v}" for k, v in comp.extended.items()],
                ]
            ).lower()
            if query and query not in haystack:
                continue
            if filter_key != "all" and comp.status != filter_key:
                continue
            filtered += 1
            item = QListWidgetItem(_status_icon(comp.status_color), self._list_text(comp))
            item.setData(Qt.ItemDataRole.UserRole, comp.id)
            item.setToolTip(
                f"{comp.refdes} · {comp.category}\n{comp.primary_value}\n状态：{comp.status_label}"
                f"\n位置：({comp.bbox.x:.0f}, {comp.bbox.y:.0f})  {comp.bbox.w:.0f}×{comp.bbox.h:.0f} px"
            )
            item.setForeground(QBrush(QColor("#DCE8F4")))
            item.setSizeHint(QSize(0, 29))
            self.list_widget.addItem(item)
            if comp.id == selected:
                self.list_widget.setCurrentItem(item)
        self._refreshing = False
        self.list_count.setText(
            f"{filtered} 个元件" if filtered == self.project.count else f"{filtered} / {self.project.count} 个元件"
        )
        if filtered == 0:
            self._stack.setCurrentIndex(0)
            self._set_detail_enabled(False)

    @staticmethod
    def _list_text(comp: Component) -> str:
        value = comp.primary_value or comp.category
        return f"{comp.refdes}    {value}    · {comp.status_label}"

    def select_component(self, component_id: str, emit: bool = False) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        if comp is None:
            return
        self._refreshing = True
        target_item = None
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == component_id:
                target_item = item
                break
        if target_item is None:
            # 当前筛选/search 隐藏了目标，回到全部列表
            self.search_edit.blockSignals(True)
            self.filter_combo.blockSignals(True)
            self.search_edit.clear()
            self.filter_combo.setCurrentIndex(0)
            self.search_edit.blockSignals(False)
            self.filter_combo.blockSignals(False)
            self.refresh_list()
            self._refreshing = True
            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == component_id:
                    target_item = item
                    break
        if target_item is not None:
            self.list_widget.setCurrentItem(target_item)
            self.list_widget.scrollToItem(target_item)
        self._refreshing = False
        self._current_id = component_id
        self.load_component(comp)
        if emit:
            self.selection_requested.emit(component_id)

    def refresh_component(self, component_id: str) -> None:
        if self.project is None:
            return
        comp = self.project.find_by_id(component_id)
        if comp is None:
            self.refresh_project()
            return
        self._refreshing = True
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == component_id:
                item.setText(self._list_text(comp))
                item.setIcon(_status_icon(comp.status_color))
                item.setToolTip(
                    f"{comp.refdes} · {comp.category}\n{comp.primary_value}\n状态：{comp.status_label}"
                )
                break
        self._refreshing = False
        self.refresh_project()
        if component_id == self._current_id:
            self.load_component(comp)

    def _on_list_current_changed(self, current: QListWidgetItem | None, previous=None) -> None:
        if self._refreshing or current is None:
            return
        component_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        if not component_id or component_id == self._current_id:
            return
        self._current_id = component_id
        if self.project is not None:
            comp = self.project.find_by_id(component_id)
            if comp is not None:
                self.load_component(comp)
        self.selection_requested.emit(component_id)

    def clear_selection(self) -> None:
        self._current_id = ""
        self._refreshing = True
        self.list_widget.clearSelection()
        self.list_widget.setCurrentItem(None)
        self._refreshing = False
        self._stack.setCurrentIndex(0)
        self._set_detail_enabled(False)

    # ------------------------------------------------------------------ 详情
    def load_component(self, comp: Component) -> None:
        self._loading_detail = True
        self._stack.setCurrentIndex(1)
        self._set_detail_enabled(True)
        self.refdes_edit.setText(comp.refdes)
        self.category_combo.setCurrentText(comp.category)
        index = self.status_combo.findData(comp.status)
        self.status_combo.setCurrentIndex(max(0, index))
        for i in range(CORE_SLOT_COUNT):
            self.core_name_edits[i].setText(comp.core[i].name if i < len(comp.core) else "")
            self.core_value_edits[i].setText(comp.core[i].value if i < len(comp.core) else "")
        self.ext_table.setRowCount(0)
        for key, value in comp.extended.items():
            row = self.ext_table.rowCount()
            self.ext_table.insertRow(row)
            self.ext_table.setItem(row, 0, QTableWidgetItem(key))
            self.ext_table.setItem(row, 1, QTableWidgetItem(value))
        self.x_spin.setValue(comp.bbox.x)
        self.y_spin.setValue(comp.bbox.y)
        self.w_spin.setValue(comp.bbox.w)
        self.h_spin.setValue(comp.bbox.h)
        self.notes_edit.setPlainText(comp.notes)
        self._update_mm_label()
        self._loading_detail = False

    def _set_detail_enabled(self, enabled: bool) -> None:
        for widget in (
            self.refdes_edit,
            self.category_combo,
            self.status_combo,
            self.ext_table,
            self.x_spin,
            self.y_spin,
            self.w_spin,
            self.h_spin,
            self.notes_edit,
            self.apply_button,
            self.verify_button,
            self.issue_button,
            self.delete_button,
        ):
            widget.setEnabled(enabled)
        for widget in self.core_name_edits + self.core_value_edits:
            widget.setEnabled(enabled)

    def _update_mm_label(self) -> None:
        if self.project is None:
            return
        if self.project.mm_per_px <= 0:
            self.mm_label.setText("未标定 mm/px；可在工程信息中设置物理比例")
            return
        scale = self.project.mm_per_px
        x_mm = self.x_spin.value() * scale
        y_mm = self.y_spin.value() * scale
        w_mm = self.w_spin.value() * scale
        h_mm = self.h_spin.value() * scale
        self.mm_label.setText(
            f"物理坐标约 ({x_mm:.2f}, {y_mm:.2f}) mm · 尺寸 {w_mm:.2f} × {h_mm:.2f} mm"
        )

    def _add_ext_row(self) -> None:
        row = self.ext_table.rowCount()
        self.ext_table.insertRow(row)
        self.ext_table.setItem(row, 0, QTableWidgetItem(""))
        self.ext_table.setItem(row, 1, QTableWidgetItem(""))
        self.ext_table.setCurrentCell(row, 0)
        self.ext_table.editItem(self.ext_table.item(row, 0))

    def _remove_ext_rows(self) -> None:
        rows = sorted({index.row() for index in self.ext_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.ext_table.removeRow(row)

    def _apply_changes(self) -> bool:
        if self.project is None or not self._current_id:
            return False
        comp = self.project.find_by_id(self._current_id)
        if comp is None:
            return False
        refdes = self.refdes_edit.text().strip()
        if not refdes:
            QMessageBox.warning(self, "位号不能为空", "请填写唯一的元件位号，例如 R12。")
            self.refdes_edit.setFocus()
            return False
        other = self.project.find_by_refdes(refdes)
        if other is not None and other.id != comp.id:
            QMessageBox.warning(self, "位号重复", f"位号 {refdes} 已被其他元件使用。")
            self.refdes_edit.setFocus()
            return False
        width = self.w_spin.value()
        height = self.h_spin.value()
        if width < 0.5 or height < 0.5:
            QMessageBox.warning(self, "尺寸无效", "标注框宽高必须大于 0.5 像素。")
            return False
        extended: dict[str, str] = {}
        duplicate_keys: set[str] = set()
        for row in range(self.ext_table.rowCount()):
            key_item = self.ext_table.item(row, 0)
            value_item = self.ext_table.item(row, 1)
            key = (key_item.text() if key_item else "").strip()
            value = (value_item.text() if value_item else "").strip()
            if not key:
                continue
            if key in extended:
                duplicate_keys.add(key)
            extended[key] = value
        if duplicate_keys:
            QMessageBox.warning(
                self,
                "扩展参数重名",
                "以下扩展参数名重复，请合并后再应用：\n" + "、".join(sorted(duplicate_keys)),
            )
            return False
        comp.refdes = refdes
        comp.category = self.category_combo.currentText().strip() or "未分类"
        comp.status = str(self.status_combo.currentData() or comp.status)
        for i in range(CORE_SLOT_COUNT):
            name = self.core_name_edits[i].text().strip() or f"参数{i + 1}"
            value = self.core_value_edits[i].text().strip()
            comp.core[i].name = name
            comp.core[i].value = value
        comp.extended = extended
        comp.bbox = BBox(
            self.x_spin.value(),
            self.y_spin.value(),
            width,
            height,
        )
        comp.notes = self.notes_edit.toPlainText().strip()
        comp.touch()
        self.project.touch()
        self._current_id = comp.id
        self.component_applied.emit(comp.id)
        self.refresh_component(comp.id)
        return True

    def _verify_with_apply(self, status: str) -> None:
        if not self._current_id:
            return
        # 先保存当前表单，避免“改了参数点核对”时修改被刷新掉。
        if not self._apply_changes():
            return
        if status == STATUS_VERIFIED:
            self.verify_requested.emit(self._current_id)
        else:
            self.issue_requested.emit(self._current_id)

    def _emit_status(self, status: str) -> None:
        if not self._current_id:
            return
        if status == STATUS_VERIFIED:
            self.verify_requested.emit(self._current_id)
        else:
            self.issue_requested.emit(self._current_id)

    def next_pending(self) -> None:
        if self.project is None:
            return
        ordered = sorted(self.project.components, key=lambda c: natural_key(c.refdes))
        pending = [c for c in ordered if c.status == STATUS_PENDING]
        if not pending:
            QMessageBox.information(self, "核对完成", "当前工程没有待核对元件。")
            return
        current_index = next((i for i, c in enumerate(pending) if c.id == self._current_id), -1)
        target = pending[(current_index + 1) % len(pending)]
        self.select_component(target.id, emit=True)
