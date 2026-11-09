# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
HIDDEN = sorted(set(collect_submodules("pcb_ar") + ["pcb_ar.selftest"]))
DATAS = [(str(SRC / "pcb_ar" / "assets" / "app.png"), "pcb_ar/assets")]

a = Analysis(
    [str(SRC / "pcb_ar" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pandas",
        "PIL",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PCB_AR_Annotator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "packaging" / "app.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PCB_AR_Annotator",
)
