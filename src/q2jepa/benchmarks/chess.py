"""Chess (data/chess.py): a position, its 3 equally good engine moves as futures. State: the move's squares; a sample
matches a move only if it decodes to exactly its from and to squares (tolerance half a square)."""

import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.chess import ChessPositions


class ChessBenchmark(FuturesBenchmark):
    tol = (0.5,) * 4

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = ChessPositions(split, self.cfg.dataset.root)
        return ds.obs, ds.fut, ds.state
