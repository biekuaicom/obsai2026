from __future__ import annotations

import math

import cv2
import numpy as np

from pcb_ar.core.seam_elements import extract_elements, match_elements
from pcb_ar.core.synthetic import make_synthetic_board, synthetic_view


def test_extract_and_match_elements():
    board, _, _ = make_synthetic_board()
    frame, _ = synthetic_view(board, (1600, 1100), 0.9, 1.5, (1200, 800), seed=0)
    source = extract_elements(frame)
    counts = source.counts()
    assert counts["component"] + counts["pad"] + counts["hole"] > 0
    assert counts["text"] > 0
    assert counts["line"] > 0

    s = 0.9
    a = math.radians(3.0)
    matrix = np.array(
        [[s * math.cos(a), -s * math.sin(a), 60.0], [s * math.sin(a), s * math.cos(a), 40.0], [0, 0, 1]],
        dtype=np.float64,
    )
    h, w = frame.shape[:2]
    target = cv2.warpAffine(frame, matrix[:2, :].astype(np.float32), (w, h))
    target_elements = extract_elements(target)
    match = match_elements(source, target_elements, matrix)
    assert match.ok, match.message
    assert match.inliers >= 6
    assert match.matrix is not None
