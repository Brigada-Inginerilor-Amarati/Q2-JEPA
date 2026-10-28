"""Shared code predictor P_psi(h, z), Q-JEPA Eqs. 5-6 (MovingMNIST: N = LayerNorm)."""

import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, d: int = 64, n_bits: int = 3, width: int = 96):
        super().__init__()
        self.code = nn.Sequential(nn.Linear(n_bits, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.context = nn.Linear(d, width)
        self.transition = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u = self.code(2 * z - 1)  # (B, K, width)
        c = self.context(h)[:, None].expand_as(u)
        return self.norm(h[:, None] + self.transition(torch.cat([c, u, c * u], dim=-1)))
