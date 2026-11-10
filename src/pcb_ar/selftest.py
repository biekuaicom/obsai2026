"""离屏自检：验证 OpenCV、跟踪、UI、内联编辑、工程保存与打包资源。

打包后的 exe 可通过 ``--self-test --report path.json`` 运行，不需要显示器。
"""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# 必须在导入 PyQt6 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if _font_dir.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_font_dir))

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from pcb_ar import __version__
from pcb_ar.config import AppConfig
from pcb_ar.core.detector import detect_component_candidates
from pcb_ar.core.project import ProjectStore
from pcb_ar.core.sparse_capture import SparseCaptureConfig, SparseCaptureSession
from pcb_ar.core.sparse_localizer import SparseLocalizer
from pcb_ar.core.synthetic import make_synthetic_board, synthetic_view
from pcb_ar.core.tracker import GlobalLocalTracker, pose_corner_error
from pcb_ar.core.video import FrameData
from pcb_ar.ui.main_window import MainWindow
from pcb_ar.ui.theme import STYLE_SHEET


def _safe_print(text: str) -> None:
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001
        pass


def _check(report: dict[str, Any], name: str, ok: bool, detail: str = "") -> None:
    report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    _safe_print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def run_self_test(
    screenshot: str | Path | None = None,
    report_path: str | Path | None = None,
    *,
    keep_open_seconds: float = 0.0,
) -> int:
    started = time.time()
    report: dict[str, Any] = {
        "app": "PCB AR Annotator",
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "checks": [],
    }
    exit_code = 1
    temp_dir = tempfile.TemporaryDirectory(prefix="pcb_ar_selftest_")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    window: MainWindow | None = None
    try:
        from PyQt6.QtCore import Qt

        base = Path(temp_dir.name)
        config = AppConfig(autoplay_demo=False)
        window = MainWindow(
            config=config,
            config_path=base / "config.json",
            demo_base=base / "demo",
            auto_bootstrap=False,
        )
        window.resize(1500, 940)
        window.show()
        app.processEvents()
        _check(report, "demo_project_created", window.load_demo(auto_start=False), str(window.store.path if window.store else ""))

        board = window.global_image
        _check(report, "global_image_loaded", board is not None, str(None if board is None else board.shape))

        detections = detect_component_candidates(board) if board is not None else []
        _check(report, "candidate_detector", len(detections) >= 10, f"{len(detections)} candidates")

        if board is not None:
            sparse_session = SparseCaptureSession(SparseCaptureConfig(min_sharpness=5))
            for center in ((1100, 1000), (1650, 1000), (2200, 1000)):
                sparse_frame, _ = synthetic_view(board, center, 0.9, 1.5, (1200, 760), seed=0)
                for _ in range(3):
                    sparse_session.preview(sparse_frame)
                sparse_session.capture()
            sparse_result = sparse_session.finish()
            sparse_map = sparse_result.sparse_map
            sparse_localizer = SparseLocalizer(sparse_map)
            sparse_test, _ = synthetic_view(board, (1500, 1030), 0.9, 1.5, (1200, 760), seed=0)
            sparse_track = sparse_localizer.track(sparse_test)
            _check(
                report,
                "sparse_capture_localization",
                sparse_map.count == 3 and sparse_track.state == "locked",
                f"keyframes={sparse_map.count}, overview={sparse_result.overview.shape}, "
                f"state={sparse_track.state}, inliers={sparse_track.inliers}",
            )
            structural = sparse_result.structural_map
            _check(
                report,
                "structural_map",
                structural is not None and structural.component_count > 0,
                structural.describe() if structural is not None else "未生成",
            )
            if sparse_map.count == 3:
                sparse_dir = base / "sparse_assets"
                manifest = sparse_map.save(sparse_dir)
                _check(report, "sparse_map_persist", manifest.exists(), str(manifest))

        if board is not None:
            frame, truth = synthetic_view(board, (1600, 1100), 0.8, 2.0, (1600, 1000), seed=101)
            tracker = GlobalLocalTracker(board)
            result = tracker.track(frame)
            error = (
                pose_corner_error(truth, result.matrix, board.shape[1], board.shape[0])
                if result.matrix is not None
                else 9999.0
            )
            _check(
                report,
                "orb_tracking",
                result.state == "locked" and error < 12.0,
                f"state={result.state}, inliers={result.inliers}, corner_error={error:.2f}px",
            )
            window.tracker = tracker
            window._current_source_kind = "demo"
            data = FrameData(frame, time.time(), 1, truth, "self-test")
            window._on_frame(data, result)
            app.processEvents()
            _check(report, "canvas_rendered", window.canvas.has_frame, "")

            if window.project is not None and window.project.components:
                comp = next(
                    (c for c in window.project.components if c.refdes == "U1"),
                    window.project.components[0],
                )
                window._select_component(comp.id)
                app.processEvents()
                _check(report, "overlay_has_items", bool(window.overlay.selected_id), comp.refdes)
                window.open_inline_editor(comp.id)
                app.processEvents()
                _check(report, "inline_editor_visible", window.inline_editor.isVisible(), "")
                values = ["10kΩ ±1%", "0603", "自检修改"]
                window._on_inline_saved(values, "verified")
                app.processEvents()
                updated = window.project.find_by_id(comp.id)
                _check(
                    report,
                    "inline_edit_saved",
                    updated is not None
                    and updated.core[0].value == values[0]
                    and updated.status == "verified",
                    f"status={updated.status if updated else 'missing'}",
                )

            window._set_dirty(False)
            save_path = base / "selftest.pcbproj.json"
            if window.store is not None:
                window.store.save_as(save_path)
                window.store.save()
                reloaded = ProjectStore.load(save_path)
                _check(
                    report,
                    "project_roundtrip",
                    reloaded.project.count == window.project.count,
                    f"{reloaded.project.count} components",
                )

        window.resize(1500, 940)
        app.processEvents()
        if screenshot:
            shot = Path(screenshot)
            shot.parent.mkdir(parents=True, exist_ok=True)
            pixmap = window.grab()
            if not pixmap.isNull():
                pixmap.save(str(shot), "PNG")
            _check(report, "screenshot_saved", shot.exists(), str(shot))
        _check(report, "ui_alive", window.isVisible(), "")

        if keep_open_seconds > 0:
            end = time.time() + keep_open_seconds
            while time.time() < end:
                app.processEvents()
                time.sleep(0.03)

        exit_code = 0 if all(item["ok"] for item in report["checks"]) else 2
    except Exception as exc:  # noqa: BLE001
        import traceback

        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        _safe_print(report["traceback"])
        exit_code = 1
    finally:
        if window is not None:
            try:
                window._set_dirty(False)
                window.close()
            except Exception:  # noqa: BLE001
                pass
        report["elapsed_sec"] = round(time.time() - started, 2)
        report["success"] = exit_code == 0
        if report_path:
            target = Path(report_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _safe_print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
        temp_dir.cleanup()
        app.processEvents()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_self_test())
