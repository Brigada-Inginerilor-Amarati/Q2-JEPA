"""Chess (data/chess.py): a position, its 3 equally good engine moves as futures. State: the move's squares
(from file, from rank, to file, to rank). The readout classifies the from- and to-square (readout.futures.square_loss)
and decodes the argmax, so decoded moves are exact squares. Validity: metrics.match_futures, in units of one square,
capped at the exact squares."""

import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.chess import ChessPositions, render_codes
from q2jepa.readout.futures import SQUARES, decode_squares, square_loss


class ChessBenchmark(FuturesBenchmark):
    src_dims, scale = 2, 1.0  # source: piece codes (8, 8)
    max_radius = 0.25  # RMS over 4 squares <= 0.25 implies every coordinate within 0.5: the exact squares

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = ChessPositions(split, self.cfg.dataset.root)
        return ds.obs, ds.fut, ds.state

    def render(self, src: torch.Tensor) -> torch.Tensor:
        return render_codes(src)

    def readout_dim(self) -> int:
        return 2 * SQUARES

    def readout_fit_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return square_loss(out, target)

    def decode(self, out: torch.Tensor) -> torch.Tensor:
        return decode_squares(out)
