from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from pcb_ar.config import AppConfig
from pcb_ar.core.synthetic import synthetic_view
from pcb_ar.core.tracker import GlobalLocalTracker
from pcb_ar.core.video import FrameData
from pcb_ar.ui.guided_capture_dialog import CapturePreviewWidget
from pcb_ar.ui.main_window import MainWindow


@pytest.mark.usefixtures("qapp")
def test_noncontiguous_capture_preview_renders(qapp):
    # 回归：拼图裁剪产生的是非连续 ndarray，QImage 直接接收会抛
    # “unexpected type 'memoryview'”并可能让 Qt 崩溃。
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    frame[20:100, 30:190] = (40, 120, 80)
    cropped = frame[10:110, 20:200]
    assert not cropped.flags.c_contiguous
    widget = CapturePreviewWidget()
    widget.resize(640, 400)
    widget.set_data(cropped, [], "test", True)
    widget.show()
    pixmap = widget.grab()
    assert not pixmap.isNull()
    assert pixmap.size().width() >= 640
    widget.close()


@pytest.mark.usefixtures("qapp")
def test_main_window_demo_inline_edit_and_screenshot(tmp_path: Path):
    window = MainWindow(
        config=AppConfig(autoplay_demo=False),
        config_path=tmp_path / "config.json",
        demo_base=tmp_path / "demo",
        auto_bootstrap=False,
    )
    window.resize(1400, 880)
    window.show()
    try:
        assert window.load_demo(auto_start=False)
        assert window.project is not None
        assert window.project.count >= 20
        assert window.canvas.has_frame
        exported = window._write_board_image(tmp_path / "board_export.png")
        assert exported.exists() and exported.stat().st_size > 1000
        assert window.action_export_board.isEnabled()

        board = window.global_image
        frame, truth = synthetic_view(board, (1600, 1050), 0.82, 2.0, (1600, 1000), seed=0)
        tracker = GlobalLocalTracker(board)
        result = tracker.track(frame)
        assert result.state == "locked"
        window.tracker = tracker
        window._current_source_kind = "demo"
        window._on_frame(FrameData(frame, time.time(), 1, truth, "test"), result)
        window.repaint()

        comp = window.project.find_by_refdes("U1")
        assert comp is not None
        window._select_component(comp.id)
        window.open_inline_editor(comp.id)
        assert window.inline_editor.isVisible()
        window._on_inline_saved(["4.7k", "0603", "UI 测试"], "verified")
        updated = window.project.find_by_id(comp.id)
        assert updated is not None
        assert updated.core[0].value == "4.7k"
        assert updated.status == "verified"
        assert not window.inline_editor.isVisible()

        shot = tmp_path / "ui.png"
        assert window.grab().save(str(shot), "PNG")
        assert shot.stat().st_size > 10_000

        # 通过侧栏修改位号，验证数据回写与列表刷新
        window.sidebar.select_component(comp.id)
        window.sidebar.refdes_edit.setText("U99")
        window.sidebar._apply_changes()  # noqa: SLF001
        assert window.project.find_by_refdes("U99") is not None
        assert window.project.find_by_id(comp.id).refdes == "U99"
    finally:
        window._set_dirty(False)
        window.close()
