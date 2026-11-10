from __future__ import annotations

import numpy as np

from pcb_ar.core.sparse_capture import compute_red_metrics


def test_red_area_and_shape_metrics():
    mask = np.zeros((300, 500), dtype=np.uint8)
    mask[30:150, 40:240] = 255
    mask[180:270, 300:460] = 255
    metrics = compute_red_metrics(mask)
    assert metrics.area_px == 120 * 200 + 90 * 160
    assert metrics.area_fraction > 0.2
    assert metrics.components == 2
    assert 0.5 < metrics.largest_share < 0.8
    assert metrics.bbox == (40, 30, 420, 240)
    assert metrics.extent > 0.3
    assert metrics.hu.shape == (7,)
    assert np.isfinite(metrics.hu).all()


def test_empty_red_mask_metrics():
    metrics = compute_red_metrics(np.zeros((100, 100), dtype=np.uint8))
    assert metrics.area_px == 0
    assert metrics.area_fraction == 0.0
    assert metrics.components == 0
