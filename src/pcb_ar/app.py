"""程序入口与日志初始化。"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from multiprocessing import freeze_support
from pathlib import Path

from pcb_ar import APP_NAME, APP_NAME_CN, __version__


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_NAME_CN)
    parser.add_argument("--project", "-p", type=str, default="", help="启动时打开指定工程")
    parser.add_argument("--demo", action="store_true", help="强制启动合成演示")
    parser.add_argument("--self-test", action="store_true", help="离屏自检并退出（用于打包验证）")
    parser.add_argument("--report", type=str, default="", help="自检报告 JSON 输出路径")
    parser.add_argument("--screenshot", type=str, default="", help="离屏截图输出路径")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return parser.parse_args(argv)


def _setup_logging() -> Path | None:
    try:
        from pcb_ar.paths import log_dir

        directory = log_dir()
    except Exception:  # noqa: BLE001
        return None
    log_file = directory / "app.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(handler, RotatingFileHandler) for handler in root.handlers):
        handler = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)
    if sys.stderr is not None:
        stream = logging.StreamHandler()
        stream.setLevel(logging.WARNING)
        stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(stream)
    return log_file


def _apply_style(app) -> None:
    from PyQt6.QtGui import QFont, QIcon

    from pcb_ar.paths import resource_path
    from pcb_ar.ui.theme import STYLE_SHEET

    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei UI", 9)
    if not font.exactMatch():
        font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    app.setStyleSheet(STYLE_SHEET)
    icon_path = resource_path("assets/app.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))


def main(argv: list[str] | None = None) -> int:
    freeze_support()
    # PyInstaller windowed 模式没有控制台，stdout/stderr 可能为 None。
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    args = _parse_args(argv)
    if args.self_test or args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        windir = os.environ.get("WINDIR", r"C:\Windows")
        fonts = Path(windir) / "Fonts"
        if fonts.exists():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
    log_file = _setup_logging()
    logger = logging.getLogger("pcb_ar")
    logger.info("启动 %s %s（frozen=%s, argv=%s）", APP_NAME, __version__, getattr(sys, "frozen", False), argv)

    try:
        from PyQt6.QtCore import QCoreApplication, Qt
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except Exception as exc:  # noqa: BLE001
        print(f"PyQt6 加载失败：{exc}", file=sys.stderr)
        return 1

    if args.self_test or args.screenshot:
        from pcb_ar.selftest import run_self_test

        report_path = args.report or ""
        if args.self_test and not report_path:
            report_path = str(Path.cwd() / "self_test_report.json")
        return run_self_test(args.screenshot or None, report_path or None)

    app = QApplication(sys.argv[:1])
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setOrganizationName("PCB AR Project")
    QCoreApplication.setApplicationVersion(__version__)
    _apply_style(app)

    from pcb_ar.ui.main_window import MainWindow

    def excepthook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("未捕获异常：\n%s", text)
        QMessageBox.critical(
            None,
            "程序异常",
            f"发生未处理异常：\n{exc_value}\n\n详细信息已写入日志：\n{log_file or '不可用'}",
        )

    sys.excepthook = excepthook

    try:
        window = MainWindow(initial_project=Path(args.project) if args.project else None)
        window.show()
        if args.demo:
            window.load_demo(auto_start=True)
        code = app.exec()
    except Exception:  # noqa: BLE001
        logger.exception("主程序异常")
        QMessageBox.critical(None, "启动失败", traceback.format_exc(limit=5))
        return 1
    logger.info("退出，code=%s", code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
