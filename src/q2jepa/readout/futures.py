"""Frozen state readout for the multi-future benchmarks (charged particles, chess): [h; t] -> future state.

A small MLP on the observation feature h and a (predicted or encoded) future feature t, like MovingMNIST's readout
(App. A.2). It outputs the benchmark's state vector directly (ball positions in px, or a move's squares). The loss
is the squared error in units of the state scale, minimised over the state permutations that describe the same
scene (charged particles: the 3 balls as a set), matching metrics.futures_metrics.
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


def readout_loss(pred: torch.Tensor, target: torch.Tensor, scale: float, perms: torch.Tensor) -> torch.Tensor:
    """pred, target (B, m); perms (P, m) index permutations (identity only: (1, m))."""
    err = ((pred[:, perms] - target[:, None]) / scale) ** 2  # (B, P, m)
    return err.mean(-1).amin(-1).mean()
