from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pcb_ar.core.structural_map import (
    StructuralMap,
    extract_structural_map,
    match_structural_maps,
    transform_structural_map,
)


def test_structural_primitives_and_roundtrip(board, tmp_path: Path):
    structural = extract_structural_map(board)
    assert structural.width == board.shape[1]
    assert structural.height == board.shape[0]
    assert structural.component_count >= 10
    assert structural.counts.get("component", 0) >= 10
    assert structural.counts.get("hole", 0) >= 1
    assert structural.counts.get("line", 0) >= 5
    assert len(structural.nodes) > structural.component_count
    assert len(structural.edges) > 0
    assert "元件" in structural.describe()

    path = structural.save(tmp_path / "structural_map.json")
    loaded = StructuralMap.load(path)
    assert loaded.component_count == structural.component_count
    assert len(loaded.nodes) == len(structural.nodes)
    assert len(loaded.edges) == len(structural.edges)

    # 已知相似变换下的结构图应能被重新匹配。
    s = 0.8
    a = math.radians(5.0)
    transform = np.array(
        [
            [s * math.cos(a), -s * math.sin(a), 300.0],
            [s * math.sin(a), s * math.cos(a), 180.0],
            [0.0, 0.0, 1.0],
        ]
    )
    query = transform_structural_map(structural, transform)
    match = match_structural_maps(structural, query)
    assert match.ok, match.message
    assert match.inliers >= 5
    assert match.matrix is not None
