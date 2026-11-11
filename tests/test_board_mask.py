from __future__ import annotations

import numpy as np

from pcb_ar.core.board_mask import foreground_mask
from pcb_ar.core.synthetic import make_synthetic_board, synthetic_view


def test_foreground_mask_excludes_white_paper():
    board, _, _ = make_synthetic_board()
    frame, _ = synthetic_view(board, (1600, 1100), 0.9, 1.5, (1000, 700), seed=0)
    h, w = frame.shape[:2]
    paper = np.full((h, w, 3), 245, dtype=np.uint8)
    content = np.max(frame, axis=2) > 7
    paper[content] = frame[content]
    mask = foreground_mask(paper)
    assert mask.shape == (h, w)
    assert mask.mean() > 20
    # 纯白纸区域不应被算成板面。
    white = np.all(paper > 235, axis=2)
    assert np.count_nonzero((mask > 0) & white) < int(0.01 * h * w)
