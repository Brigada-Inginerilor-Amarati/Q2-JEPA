import importlib.util
from pathlib import Path

import numpy as np
import pytest

from q2jepa.data.chess import move_state

pytest.importorskip("chess")
spec = importlib.util.spec_from_file_location(
    "chess_baselines", Path(__file__).parents[1] / "scripts/chess_baselines.py"
)
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_legal_squares_start_position():
    legal = cb.legal_squares(START)
    assert len(legal) == 20 and "e2e4" in legal and "e2e5" not in legal


def test_baselines_ceiling_and_legal_rate():
    truth = np.array([[move_state(m) for m in ("e2e4", "d2d4", "g1f3")]], float)
    m = cb.baselines([START], truth, k=4000, seed=0)
    assert m["true moves (ceiling)"]["validity"] == 1.0 and m["true moves (ceiling)"]["coverage"] == 1.0
    assert abs(m["random legal move"]["validity"] - 3 / 20) < 0.03  # 3 of the 20 legal moves are true
    assert m["random squares"]["validity"] < 0.01
