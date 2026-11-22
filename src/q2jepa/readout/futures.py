"""Frozen state readout for the multi-future benchmarks (charged particles, chess): [h; t] -> future state.

Same shape as MovingMNIST's readout (App. A.2): a small MLP on the observation feature h and a (predicted or
encoded) future feature t. It outputs the benchmark's state vector directly (ball positions in px, or a move's
squares), and the loss is the squared error in units of the matching tolerance.
"""

import torch
from torch import nn


class StateReadout(nn.Module):
    def __init__(self, d: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, state_dim))  # fmt: skip

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, t], -1))


def readout_loss(pred: torch.Tensor, target: torch.Tensor, tol: torch.Tensor) -> torch.Tensor:
    return (((pred - target) / tol) ** 2).mean()
