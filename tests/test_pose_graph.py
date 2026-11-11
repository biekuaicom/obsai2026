from __future__ import annotations

import numpy as np

from pcb_ar.core.pose_graph import PoseEdge, optimize_poses


def _pose(tx: float, ty: float, angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [scale * c, -scale * s, tx],
            [scale * s, scale * c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_pose_graph_loop_reduces_error():
    truth = [_pose(0, 0), _pose(80, 0), _pose(160, 0), _pose(80, 70)]
    # 所有相对测量来自真值，模拟回环中的长距离约束。
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            measured = np.linalg.inv(truth[j]) @ truth[i]
            edges.append(PoseEdge(i, j, measured, inliers=80, loop=(j > i + 2)))
    initial = [p.copy() for p in truth]
    initial[2] = _pose(172, 8)
    initial[3] = _pose(95, 82)
    result = optimize_poses(initial, edges)
    assert result.optimized
    assert result.rms_after < result.rms_before
    # 最后一帧应明显靠近真值
    error_before = np.linalg.norm(initial[3][:2, 2] - truth[3][:2, 2])
    error_after = np.linalg.norm(result.poses[3][:2, 2] - truth[3][:2, 2])
    assert error_after < error_before
