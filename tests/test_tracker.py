from __future__ import annotations

import numpy as np

from pcb_ar.core.synthetic import synthetic_view
from pcb_ar.core.tracker import GlobalLocalTracker, TrackerConfig, pose_corner_error


def test_tracker_locks_and_follows_rotation(board):
    tracker = GlobalLocalTracker(board, TrackerConfig())
    frame, truth = synthetic_view(board, (1500, 1050), 0.82, 2.5, (1600, 1000), seed=0)
    result = tracker.track(frame)
    assert result.state == "locked"
    assert result.matrix is not None
    assert result.inliers >= 50
    assert pose_corner_error(truth, result.matrix, board.shape[1], board.shape[0]) < 10.0

    frame2, truth2 = synthetic_view(board, (1750, 1180), 0.84, -1.6, (1600, 1000), seed=0)
    result2 = tracker.track(frame2)
    assert result2.state == "locked"
    assert result2.matrix is not None
    assert pose_corner_error(truth2, result2.matrix, board.shape[1], board.shape[0]) < 12.0


def test_tracker_stale_then_lost_then_reacquire(board):
    tracker = GlobalLocalTracker(board, TrackerConfig(max_lost_frames=3))
    frame, _ = synthetic_view(board, (1600, 1100), 0.8, 1.0, (1600, 1000), seed=0)
    assert tracker.track(frame).state == "locked"
    blank = np.zeros((1000, 1600, 3), dtype=np.uint8)
    stale = tracker.track(blank)
    assert stale.state == "stale"
    assert stale.matrix is not None
    for _ in range(4):
        lost = tracker.track(blank)
    assert lost.state == "lost"
    assert lost.matrix is None

    frame2, truth2 = synthetic_view(board, (1600, 1100), 0.8, 1.0, (1600, 1000), seed=0)
    reacquired = tracker.track(frame2)
    assert reacquired.state == "locked"
    assert reacquired.matrix is not None
    assert pose_corner_error(truth2, reacquired.matrix, board.shape[1], board.shape[0]) < 10.0
