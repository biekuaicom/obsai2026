from __future__ import annotations

import time
from pathlib import Path

import pytest

from pcb_ar.config import AppConfig
from pcb_ar.core.project import ProjectStore
from pcb_ar.core.sparse_capture import SparseCaptureConfig, SparseCaptureSession
from pcb_ar.core.synthetic import synthetic_view
from pcb_ar.core.video import FrameData
from pcb_ar.ui.main_window import MainWindow


@pytest.mark.usefixtures("qapp")
def test_sparse_project_opens_with_hybrid_tracker(tmp_path: Path, board):
    session = SparseCaptureSession(SparseCaptureConfig(min_sharpness=5))
    frames = []
    for cx, cy in [(1000, 1000), (1550, 1000), (2100, 1000)]:
        frame, matrix = synthetic_view(board, (cx, cy), 0.9, 1.5, (1600, 1000), seed=0)
        frames.append((frame, matrix))
        for _ in range(3):
            session.preview(frame)
        assert session.capture().accepted
    result = session.finish()
    store = ProjectStore.create(tmp_path / "sparse.pcbproj.json", "轻量测试", result.overview, mm_per_px=0.02)
    manifest = result.sparse_map.save(store.assets_dir)
    store.project.sparse_map = manifest.relative_to(store.path.parent).as_posix()
    store.project.map_mode = "hybrid"
    store.save()

    window = MainWindow(
        config=AppConfig(autoplay_demo=False),
        config_path=tmp_path / "config.json",
        demo_base=tmp_path / "demo",
        auto_bootstrap=False,
    )
    window.resize(1400, 880)
    window.show()
    try:
        assert window._open_store(ProjectStore.load(store.path))
        assert window.sparse_map is not None
        assert window.project is not None and window.project.map_mode == "hybrid"
        assert window._ensure_tracker()
        assert window.tracker is not None
        assert type(window.tracker).__name__ == "HybridTracker"

        frame, truth = frames[1]
        track = window.tracker.track(frame)
        assert track.state == "locked"
        window._current_source_kind = "demo"
        window._on_frame(FrameData(frame, time.time(), 1, truth, "test"), track)
        window.repaint()
        assert window.canvas.has_frame
        assert window.overlay.project is not None
        assert window.overlay.project.sparse_map
    finally:
        window._set_dirty(False)
        window.close()
