from __future__ import annotations

import cv2

from pcb_ar.core.detector import detect_component_candidates
from pcb_ar.core.geo import bbox_iou
from pcb_ar.core.stitcher import stitch_images


def test_discrete_stitch_of_ordered_tiles(board):
    tiles = []
    for y, x in ((0, 0), (0, 450), (0, 900), (500, 0), (500, 450), (500, 900)):
        tiles.append(board[y : y + 650, x : x + 650].copy())
    result = stitch_images(tiles, feather_px=24)
    assert result.image.shape[0] > 900
    assert result.image.shape[1] > 1200
    assert len(result.pairs) == 5
    # 不应出现完全黑图
    assert float(result.image.mean()) > 10


def test_candidate_detector_finds_most_demo_components(board, sample_components):
    detections = detect_component_candidates(board)
    assert len(detections) >= 20
    matched = set()
    for det in detections:
        for comp in sample_components:
            if bbox_iou(det.bbox, comp.bbox) > 0.3:
                matched.add(comp.refdes)
    assert len(matched) >= 20
