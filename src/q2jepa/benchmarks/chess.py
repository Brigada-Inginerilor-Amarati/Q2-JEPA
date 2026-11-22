"""Chess (data/chess.py): a position, its 3 equally good engine moves as futures. State: the move's squares
(from file, from rank, to file, to rank). Validity: metrics.match_futures, in units of one square, capped at the
exact squares."""

import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.chess import ChessPositions, render_codes


class ChessBenchmark(FuturesBenchmark):
    src_dims, scale = 2, 1.0  # source: piece codes (8, 8)
    max_radius = 0.25  # RMS over 4 squares <= 0.25 implies every coordinate within 0.5: the exact squares

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = ChessPositions(split, self.cfg.dataset.root)
        return ds.obs, ds.fut, ds.state

    def render(self, src: torch.Tensor) -> torch.Tensor:
        return render_codes(src)
