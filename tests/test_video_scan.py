from __future__ import annotations

import numpy as np

from pcb_ar.core.synthetic import synthetic_view
from pcb_ar.core.video_scan import ScanConfig, VideoScanSession


def test_video_scan_keyframes_and_build(board):
    session = VideoScanSession(
        ScanConfig(
            keyframe_translation_px=60,
            keyframe_rotation_deg=1.0,
            keyframe_max_width=640,
            blur_threshold=4,
            min_matches=16,
            min_inliers=12,
        )
    )
    frames = []
    for index in range(14):
        center = (1000 + index * 70, 900 + (index % 3) * 20)
        frame, truth = synthetic_view(board, center, 0.9, 1.2, (1000, 650), seed=0)
        frames.append(frame)
        update = session.add_frame(frame, timestamp=index * 0.2)
        assert update.total_keyframes >= 1
    assert session.count >= 4
    result = session.build()
    assert result.stitch.image.size > 0
    assert result.stitch.image.shape[1] > 900
    assert result.stitch.image.shape[0] > 400
    assert result.keyframe_count == session.count


def test_video_scan_rejects_blank_frame(board):
    session = VideoScanSession(ScanConfig(blur_threshold=10))
    blank = np.zeros((500, 800, 3), dtype=np.uint8)
    update = session.add_frame(blank)
    assert update.accepted is False
    assert update.total_keyframes == 0
