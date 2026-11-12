import numpy as np

from pcb_ar.core.overlap_stitcher import _similarity_from_three


def test_three_point_similarity_recovers_transform():
    src = np.float32([[10, 20], [110, 20], [60, 100]])
    theta = np.deg2rad(7.0)
    s = 1.03
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([12.0, -5.0])
    dst = (s * (r @ src.T)).T + t
    m = _similarity_from_three(src, dst)
    assert m is not None
    pred = (m[:2, :2] @ src.T).T + m[:2, 2]
    assert np.allclose(pred, dst, atol=1e-4)


def test_three_point_similarity_rejects_collinear():
    src = np.float32([[0, 0], [10, 10], [20, 20]])
    dst = src + 5
    assert _similarity_from_three(src, dst) is None
