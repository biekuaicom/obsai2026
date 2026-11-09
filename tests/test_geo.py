from __future__ import annotations

import math

import numpy as np
import pytest

from pcb_ar.core.geo import (
    BBox,
    apply_affine,
    as_affine,
    blend_similarity,
    build_similarity,
    decompose_similarity,
    invert_affine,
    is_valid_similarity,
    transform_bbox,
)


def test_bbox_operations():
    box = BBox(10, 20, 30, 40)
    assert box.center == (25, 40)
    assert box.x2 == 40
    assert box.y2 == 60
    clipped = BBox(-10, -5, 30, 40).clipped(100, 100)
    assert clipped == BBox(0, 0, 20, 35)
    assert BBox(0, 0, 10, 10).intersects(BBox(9, 9, 5, 5))
    assert not BBox(0, 0, 10, 10).intersects(BBox(11, 0, 5, 5))


def test_affine_roundtrip():
    matrix = build_similarity(1.5, math.radians(12), 40, -15)
    points = np.float32([[0, 0], [100, 0], [100, 60], [0, 60]])
    mapped = apply_affine(matrix, points)
    restored = apply_affine(invert_affine(matrix), mapped)
    assert np.allclose(restored, points, atol=0.01)
    transformed = transform_bbox(matrix, BBox(0, 0, 100, 60))
    assert transformed.w > 100
    assert transformed.h > 60


def test_similarity_decompose_reconstruct():
    for scale, angle, tx, ty in [
        (0.8, 0.15, -300, 120),
        (1.7, -1.2, 55, -10),
        (1.0, 0.0, 0, 0),
    ]:
        matrix = build_similarity(scale, angle, tx, ty)
        s, theta, t = decompose_similarity(matrix)
        rebuilt = build_similarity(s, theta, float(t[0]), float(t[1]))
        assert np.allclose(rebuilt, matrix, atol=1e-5)
        assert s == pytest.approx(scale, rel=1e-5)
        assert ((theta - angle + math.pi) % (2 * math.pi) - math.pi) == pytest.approx(0.0, abs=1e-5)


def test_similarity_validation():
    good = build_similarity(1.2, 0.3, 10, 20)
    assert is_valid_similarity(good)
    reflected = np.array([[-1, 0, 0], [0, 1, 0]], dtype=np.float32)
    assert not is_valid_similarity(reflected)
    sheared = np.array([[1, 0.7, 0], [0, 1, 0]], dtype=np.float32)
    assert not is_valid_similarity(sheared)


def test_blend_similarity_preserves_pivot_progress():
    pivot = (800.0, 500.0)
    a = build_similarity(1.0, 0.0, 10, 20)
    b = build_similarity(1.2, 0.2, 80, 60)
    blended = blend_similarity(a, b, 0.5, pivot=pivot)
    pa = apply_affine(a, np.array([pivot]))[0]
    pb = apply_affine(b, np.array([pivot]))[0]
    pm = apply_affine(blended, np.array([pivot]))[0]
    assert np.allclose(pm, (pa + pb) / 2.0, atol=0.02)


def test_as_affine_shapes():
    m = np.eye(3, dtype=np.float32)
    assert as_affine(m).shape == (2, 3)
    with pytest.raises(ValueError):
        as_affine(np.eye(4, dtype=np.float32))
