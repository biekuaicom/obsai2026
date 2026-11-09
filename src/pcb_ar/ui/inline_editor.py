"""画布内联快速编辑器：只编辑气泡展示的最多 3 个核心参数。"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from pcb_ar.core.models import CORE_SLOT_COUNT, STATUS_LABELS, STATUSES, Component


class InlineParamEditor(QFrame):
    saved = pyqtSignal(list, str)
    dismissed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._component_id = ""
        self._names: list[str] = []
        self.setObjectName("InlineEditor")
        self.setStyleSheet(
            "QFrame#InlineEditor {"
            " background: #101A27; border: 1px solid #2DD4A7; border-radius: 9px;"
            "}"
            "QFrame#InlineEditor QLabel { background: transparent; }"
            "QLabel#InlineTitle { font-size: 14px; font-weight: 700; color: #F1F7FD; }"
            "QLabel#InlineHint { font-size: 10px; color: #7E92A6; }"
            "QLabel#InlineParamName { color: #91A5B9; font-size: 11px; }"
            "QLineEdit { padding: 4px 7px; }"
            "QPushButton#InlineSave {"
            " background: #16A57F; border-color: #2DD4A7; color: #03120D;"
            " font-weight: 700; padding: 5px 12px;"
            "}"
            "QPushButton#InlineClose {"
            " background: transparent; border: none; color: #7E92A6;"
            " padding: 2px 7px; font-size: 16px;"
            "}"
            "QPushButton#InlineClose:hover { color: #FFFFFF; background: #1B2A3B; }"
        )
        self.setFixedWidth(360)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 7)
        shadow.setColor(QColor(0, 0, 0, 190))
        self.setGraphicsEffect(shadow)
        self._build_ui()
        self.hide()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(13, 10, 13, 11)
        root.setSpacing(7)
        header = QHBoxLayout()
        header.setSpacing(6)
        self.title = QLabel("快速修改")
        self.title.setObjectName("InlineTitle")
        header.addWidget(self.title)
        header.addStretch(1)
        self.hint = QLabel("最多 3 个核心参数")
        self.hint.setObjectName("InlineHint")
        header.addWidget(self.hint)
        close = QPushButton("×")
        close.setObjectName("InlineClose")
        close.setToolTip("关闭（Esc）")
        close.clicked.connect(self.dismiss)
        header.addWidget(close)
        root.addLayout(header)

        self.rows: list[tuple[QLabel, QLineEdit]] = []
        for index in range(CORE_SLOT_COUNT):
            row = QHBoxLayout()
            row.setSpacing(7)
            name = QLabel(f"参数{index + 1}")
            name.setObjectName("InlineParamName")
            name.setFixedWidth(54)
            edit = QLineEdit()
            edit.setPlaceholderText("留空则不显示")
            edit.returnPressed.connect(self._save)
            row.addWidget(name)
            row.addWidget(edit, 1)
            root.addLayout(row)
            self.rows.append((name, edit))

        status_row = QHBoxLayout()
        status_row.setSpacing(7)
        status_label = QLabel("核对状态")
        status_label.setObjectName("InlineParamName")
        status_label.setFixedWidth(54)
        self.status_combo = QComboBox()
        for key in STATUSES:
            self.status_combo.addItem(STATUS_LABELS[key], key)
        status_row.addWidget(status_label)
        status_row.addWidget(self.status_combo, 1)
        root.addLayout(status_row)

        buttons = QHBoxLayout()
        buttons.setSpacing(7)
        self.key_hint = QLabel("Enter 保存 · Esc 取消")
        self.key_hint.setObjectName("InlineHint")
        buttons.addWidget(self.key_hint)
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.dismiss)
        buttons.addWidget(cancel)
        save = QPushButton("保存")
        save.setObjectName("InlineSave")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)

    def open_for(self, component: Component) -> None:
        self._component_id = component.id
        self._names = [param.name or f"参数{i + 1}" for i, param in enumerate(component.core[:CORE_SLOT_COUNT])]
        self.title.setText(f"快速修改 · {component.refdes}")
        for index, (label, edit) in enumerate(self.rows):
            name = self._names[index] if index < len(self._names) else f"参数{index + 1}"
            label.setText(name)
            label.setToolTip(name)
            value = component.core[index].value if index < len(component.core) else ""
            edit.setText(value)
            edit.setPlaceholderText(f"输入{name}")
        status_index = self.status_combo.findData(component.status)
        self.status_combo.setCurrentIndex(max(0, status_index))
        self.adjustSize()
        self.setFixedWidth(360)
        self.show()
        self.raise_()
        self.rows[0][1].setFocus(Qt.FocusReason.OtherFocusReason)
        self.rows[0][1].selectAll()

    @property
    def component_id(self) -> str:
        return self._component_id

    def values(self) -> list[str]:
        return [edit.text().strip() for _, edit in self.rows]

    def status_key(self) -> str:
        return str(self.status_combo.currentData() or STATUSES[0])

    def position_near(self, anchor: QRect, viewport: QRect) -> None:
        if anchor.isNull() or not anchor.isValid():
            return
        width = self.width()
        height = self.height()
        x = anchor.center().x() - width // 2
        y = anchor.top() - height - 10
        if y < viewport.top() + 8:
            y = anchor.bottom() + 10
        x = max(viewport.left() + 8, min(x, viewport.right() - width - 8))
        y = max(viewport.top() + 8, min(y, viewport.bottom() - height - 8))
        self.move(x, y)

    def _save(self) -> None:
        if not self._component_id:
            return
        self.saved.emit(self.values(), self.status_key())
        self._component_id = ""

    def dismiss(self) -> None:
        if not self.isVisible():
            return
        self._component_id = ""
        self.hide()
        self.dismissed.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.dismiss()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Save) or (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._save()
            event.accept()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event) -> None:
        self._component_id = ""
        super().hideEvent(event)
