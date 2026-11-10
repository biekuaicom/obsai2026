from __future__ import annotations

from pathlib import Path

from pcb_ar.core.structural_map import StructuralMap, extract_structural_map


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
