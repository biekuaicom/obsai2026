"""应用路径与运行环境辅助。

所有可写数据均放在用户目录或工程目录，虚拟环境位于项目根目录，不触碰系统 Python。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def resource_path(relative: str) -> Path:
    """返回随程序打包的资源路径（兼容 PyInstaller）。"""
    base = Path(getattr(sys, "_MEIPASS", package_dir()))
    candidate = base / relative
    if candidate.exists():
        return candidate
    return package_dir() / relative


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        path = Path(base) / "PCB_AR_Annotator"
    else:
        path = Path.home() / ".pcb_ar_annotator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return app_data_dir() / "config.json"


def log_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def demo_dir() -> Path:
    path = app_data_dir() / "demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_project_dir() -> Path:
    docs = Path.home() / "Documents"
    base = docs if docs.exists() else Path.home()
    path = base / "PCB_AR_Projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_root() -> Path:
    """源码运行时返回仓库根目录；打包运行时返回 exe 所在目录。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
