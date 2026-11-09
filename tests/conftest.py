from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if _font_dir.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_font_dir))


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def board_bundle():
    from pcb_ar.core.synthetic import make_synthetic_board

    return make_synthetic_board()


@pytest.fixture(scope="session")
def board(board_bundle):
    return board_bundle[0]


@pytest.fixture(scope="session")
def sample_components(board_bundle):
    return board_bundle[1]
