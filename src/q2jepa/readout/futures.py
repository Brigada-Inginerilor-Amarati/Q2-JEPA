"""Frozen state readout for the multi-future benchmarks (charged particles, chess): [h; t] -> future state.

A small MLP on the observation feature h and a (predicted or encoded) future feature t, like MovingMNIST's readout
(App. A.2). For charged particles it regresses the state (ball positions in px): squared error in units of the state scale,
minimised over the state permutations that describe the same scene (the 3 balls as a set), matching
metrics.futures_metrics. For chess it classifies the from- and to-square (2 x 64 logits, cross-entropy) and decodes
the argmax squares: regressing 4 coordinates to within 0.25 squares capped readout_validity at 1-18% (2026-11-23).
"""

import torch
from torch import nn

HIDDEN = 512  # 128 underfit the readout diagnostic (2026-11-22); 512 fits it without overfitting at 20k scenes


class StateReadout(nn.Module):
    def __init__(self, d: int, state_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, state_dim))  # fmt: skip

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, t], -1))


SQUARES = 64


def square_index(state: torch.Tensor) -> torch.Tensor:
    """Chess move states (..., 4) = (from file, from rank, to file, to rank) -> square indices (..., 2), rank*8+file."""
    s = state.long()
    return torch.stack([s[..., 1] * 8 + s[..., 0], s[..., 3] * 8 + s[..., 2]], -1)


def square_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of from- and to-square logits (B, 2 * 64) against move states (B, 4)."""
    return nn.functional.cross_entropy(logits.reshape(-1, SQUARES), square_index(target).reshape(-1))


def decode_squares(logits: torch.Tensor) -> torch.Tensor:
    """Logits (..., 2 * 64) -> move states (..., 4) of the most likely from- and to-square."""
    idx = logits.unflatten(-1, (2, SQUARES)).argmax(-1)  # (..., 2)
    return torch.stack([idx % 8, idx // 8], -1).flatten(-2).float()  # (file, rank) per square


def readout_loss(pred: torch.Tensor, target: torch.Tensor, scale: float, perms: torch.Tensor) -> torch.Tensor:
    """pred, target (B, m); perms (P, m) index permutations (identity only: (1, m))."""
    err = ((pred[:, perms] - target[:, None]) / scale) ** 2  # (B, P, m)
    return err.mean(-1).amin(-1).mean()
