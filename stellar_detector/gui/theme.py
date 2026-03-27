"""Dark and light theme stylesheets for the application."""

from __future__ import annotations

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected { background-color: #313244; }
QMenu {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
}
QMenu::item:selected { background-color: #45475a; }

QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 4px;
    padding: 2px;
}

QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
}

QDockWidget {
    titlebar-close-icon: none;
    color: #cdd6f4;
}
QDockWidget::title {
    background-color: #181825;
    padding: 6px;
    border-bottom: 1px solid #313244;
}

QTabWidget::pane { border: 1px solid #313244; }
QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    padding: 8px 16px;
    border: 1px solid #313244;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}

QTreeView, QTableView, QListView {
    background-color: #1e1e2e;
    alternate-background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    selection-background-color: #45475a;
    gridline-color: #313244;
}
QHeaderView::section {
    background-color: #181825;
    color: #a6adc8;
    padding: 5px;
    border: 1px solid #313244;
    font-weight: bold;
}

QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}
QPushButton:hover { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QPushButton:disabled { color: #6c7086; background-color: #313244; }
QPushButton[primary="true"] {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}
QPushButton[primary="true"]:hover { background-color: #74c7ec; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #89b4fa;
}

QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}

QGroupBox {
    border: 1px solid #313244;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    padding: 0 6px;
}

QTextEdit, QPlainTextEdit {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
    font-family: 'JetBrains Mono', 'Menlo', monospace;
    font-size: 12px;
}

QSplitter::handle {
    background-color: #313244;
    width: 2px;
    height: 2px;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background-color: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 20px;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 4px;
}

QLabel[heading="true"] {
    font-size: 16px;
    font-weight: bold;
    color: #89b4fa;
}
"""

LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #e6e9ef;
    color: #4c4f69;
    border-bottom: 1px solid #ccd0da;
}
QMenuBar::item:selected { background-color: #ccd0da; }
QMenu {
    background-color: #eff1f5;
    color: #4c4f69;
    border: 1px solid #ccd0da;
}
QMenu::item:selected { background-color: #bcc0cc; }

QToolBar {
    background-color: #e6e9ef;
    border-bottom: 1px solid #ccd0da;
    spacing: 4px;
    padding: 2px;
}

QStatusBar {
    background-color: #e6e9ef;
    color: #6c6f85;
    border-top: 1px solid #ccd0da;
}

QDockWidget::title {
    background-color: #e6e9ef;
    padding: 6px;
    border-bottom: 1px solid #ccd0da;
}

QTabBar::tab {
    background-color: #e6e9ef;
    color: #6c6f85;
    padding: 8px 16px;
    border: 1px solid #ccd0da;
    border-bottom: none;
}
QTabBar::tab:selected {
    background-color: #eff1f5;
    color: #1e66f5;
    border-bottom: 2px solid #1e66f5;
}

QTreeView, QTableView, QListView {
    background-color: #ffffff;
    alternate-background-color: #f5f5f9;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    selection-background-color: #dce0e8;
    gridline-color: #e6e9ef;
}
QHeaderView::section {
    background-color: #e6e9ef;
    color: #5c5f77;
    padding: 5px;
    border: 1px solid #ccd0da;
    font-weight: bold;
}

QPushButton {
    background-color: #dce0e8;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}
QPushButton:hover { background-color: #bcc0cc; }
QPushButton[primary="true"] {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    font-weight: bold;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 5px;
}
QLineEdit:focus { border: 1px solid #1e66f5; }

QProgressBar {
    background-color: #dce0e8;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    text-align: center;
    height: 20px;
}
QProgressBar::chunk { background-color: #1e66f5; border-radius: 3px; }

QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #1e66f5;
}

QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
}

QToolTip {
    background-color: #e6e9ef;
    color: #4c4f69;
    border: 1px solid #ccd0da;
}

QLabel[heading="true"] {
    font-size: 16px;
    font-weight: bold;
    color: #1e66f5;
}
"""

THEMES = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}
