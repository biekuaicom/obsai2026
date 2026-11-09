"""暗色工业风主题。"""
from __future__ import annotations

BG = "#080C13"
PANEL = "#0E1520"
PANEL_ALT = "#111B28"
BORDER = "#223044"
TEXT = "#E8F0F7"
MUTED = "#8294A8"
ACCENT = "#2DD4A7"
ACCENT_DARK = "#16A57F"
AMBER = "#F5A524"
DANGER = "#FF5C7A"
INFO = "#4DA8FF"

STYLE_SHEET = f"""
* {{
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI";
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QWidget#CentralHost {{
    background: {BG};
}}
QMenuBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
    padding: 2px 6px;
}}
QMenuBar::item {{
    padding: 6px 12px;
    background: transparent;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background: {PANEL_ALT};
}}
QMenu {{
    background: {PANEL};
    border: 1px solid {BORDER};
    padding: 6px;
}}
QMenu::item {{
    padding: 7px 26px 7px 22px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: rgba(45, 212, 167, 0.14);
    color: {ACCENT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}
QToolBar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 3px;
    padding: 5px 8px;
}}
QToolBar::separator {{
    width: 1px;
    background: {BORDER};
    margin: 5px 7px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 9px;
    color: #C9D7E4;
}}
QToolButton:hover {{
    background: {PANEL_ALT};
    border-color: {BORDER};
    color: {TEXT};
}}
QToolButton:pressed, QToolButton:checked {{
    background: rgba(45, 212, 167, 0.14);
    border-color: rgba(45, 212, 167, 0.45);
    color: {ACCENT};
}}
QToolButton:disabled {{
    color: #47576A;
}}
QStatusBar {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
    color: {MUTED};
    padding: 2px 8px;
}}
QStatusBar::item {{ border: none; }}
QDockWidget {{
    color: {TEXT};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {PANEL_ALT};
    border-bottom: 1px solid {BORDER};
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
}}
QWidget#Sidebar {{
    background: {PANEL};
}}
QFrame#Card, QGroupBox {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QGroupBox {{
    margin-top: 14px;
    padding: 12px 10px 8px 10px;
    font-weight: 600;
    color: #C8D7E6;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    top: 1px;
    padding: 0 5px;
    color: {MUTED};
    font-size: 12px;
}}
QLabel#SectionTitle {{
    color: {MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#ProjectName {{
    font-size: 16px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Accent {{ color: {ACCENT}; }}
QLabel#Warning {{ color: {AMBER}; }}
QLabel#Danger {{ color: {DANGER}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #0A111B;
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 7px;
    selection-background-color: {ACCENT_DARK};
    selection-color: #04110D;
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_DARK};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: #526274;
    background: #0C121B;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: rgba(45, 212, 167, 0.18);
    selection-color: {TEXT};
    outline: none;
}}
QPushButton {{
    background: #162231;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 12px;
    color: #D6E2ED;
}}
QPushButton:hover {{
    background: #1B2A3B;
    border-color: #334A64;
}}
QPushButton:pressed {{
    background: #0F1924;
}}
QPushButton:disabled {{
    color: #4D5D6F;
    border-color: #172333;
}}
QPushButton#Primary {{
    background: {ACCENT_DARK};
    border-color: {ACCENT};
    color: #03120D;
    font-weight: 700;
}}
QPushButton#Primary:hover {{
    background: {ACCENT};
}}
QPushButton#Danger {{
    background: rgba(255, 92, 122, 0.10);
    border-color: rgba(255, 92, 122, 0.45);
    color: {DANGER};
}}
QPushButton#Ghost {{
    background: transparent;
    border-color: transparent;
    color: {MUTED};
    padding: 4px 7px;
}}
QPushButton#Ghost:hover {{
    color: {TEXT};
    background: {PANEL_ALT};
}}
QListWidget, QTableWidget, QTreeWidget {{
    background: #0A111B;
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
    alternate-background-color: #0D1520;
}}
QListWidget::item {{
    padding: 7px 8px;
    border-bottom: 1px solid #141F2D;
    border-radius: 3px;
}}
QListWidget::item:hover {{
    background: #121E2C;
}}
QListWidget::item:selected {{
    background: rgba(45, 212, 167, 0.12);
    border-left: 2px solid {ACCENT};
    color: {TEXT};
}}
QHeaderView::section {{
    background: #111B28;
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 5px 6px;
    font-size: 11px;
}}
QTableWidget::item {{
    padding: 4px;
}}
QTableWidget::item:selected {{
    background: rgba(45, 212, 167, 0.12);
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #2A3A4E;
    min-height: 28px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background: #3A5068; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #2A3A4E;
    min-width: 28px;
    border-radius: 4px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QProgressBar {{
    background: #0A111B;
    border: 1px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    color: {MUTED};
    font-size: 10px;
    min-height: 14px;
    max-height: 14px;
}}
QProgressBar::chunk {{
    background: {ACCENT_DARK};
    border-radius: 4px;
}}
QCheckBox {{
    spacing: 6px;
    color: #B8C8D8;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid #354960;
    border-radius: 3px;
    background: #0A111B;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_DARK};
    border-color: {ACCENT};
}}
QToolTip {{
    background: #0B1420;
    color: {TEXT};
    border: 1px solid #334A64;
    padding: 5px;
}}
QSplitter::handle {{ background: {BORDER}; }}
QGraphicsView {{
    border: none;
    background: #05080D;
}}
QMessageBox {{
    background: {PANEL};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {PANEL};
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 7px 14px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #1A2838;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {ACCENT};
}}
"""
