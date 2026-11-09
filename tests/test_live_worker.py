from __future__ import annotations

import pytest
from PyQt6.QtCore import QEventLoop, QTimer

from pcb_ar.core.video import SyntheticMicroscopeSource
from pcb_ar.ui.live import LiveWorker


@pytest.mark.usefixtures("qapp")
def test_live_worker_emits_scaled_frames(board):
    source = SyntheticMicroscopeSource(board, size=(800, 500), fps=15, seed=3)
    worker = LiveWorker(source, tracker=None, max_width=640)
    frames = []

    def on_frame(data, result):
        frames.append(data.image)

    worker.frame_ready.connect(on_frame)
    worker.start()
    loop = QEventLoop()
    QTimer.singleShot(1400, loop.quit)
    loop.exec()
    worker.stop()
    assert worker.wait(3000)
    assert len(frames) >= 2
    assert frames[-1].shape[1] == 640
    assert frames[-1].shape[0] == 400
