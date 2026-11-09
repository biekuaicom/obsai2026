from __future__ import annotations

import cv2
import numpy as np

from pcb_ar.core.board_rectify import detect_board_quad, rectify_board
from pcb_ar.core.synthetic import make_synthetic_board


def test_rotated_board_is_rectified_to_landscape(board):
    image = cv2.resize(board, (1000, 688), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((900, 1400, 3), dtype=np.uint8)
    canvas[80 : 80 + image.shape[0], 150 : 150 + image.shape[1]] = image
    matrix = cv2.getRotationMatrix2D((700, 450), 7.0, 1.0)
    rotated = cv2.warpAffine(canvas, matrix, (1400, 900), borderValue=(0, 0, 0))

    quad, message = detect_board_quad(rotated)
    assert quad is not None, message
    result = rectify_board(rotated)
    assert result.applied, result.message
    assert result.image.shape[1] > result.image.shape[0]
    assert result.homography is not None
    assert result.image.shape[1] > 700


def test_blank_image_skips_rectification():
    result = rectify_board(np.zeros((400, 600, 3), dtype=np.uint8))
    assert not result.applied
    assert result.image.shape == (400, 600, 3)
