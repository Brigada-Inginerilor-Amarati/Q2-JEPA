"""Frozen-JEPA translation readout (Q-JEPA App. A.2): [h; t] -> 3 bounded 2-D displacements."""

import torch
from torch import nn

from q2jepa.data.mmnist import translate


class TranslationReadout(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 96, frames: int = 3, bound: float = 18.0):
        super().__init__()
        self.frames, self.bound = frames, bound
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, 2 * frames))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.net(torch.cat([h, t], -1))
        return (self.bound * torch.tanh(a / self.bound)).reshape(-1, self.frames, 2)


def render(last_frame: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    return translate(last_frame[:, None], disp)


def readout_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    l1 = ((1 + 4 * target) * (pred - target).abs()).mean()
    inter = (pred * target).sum((-2, -1))
    dice = (2 * inter + smooth) / (pred.sum((-2, -1)) + target.sum((-2, -1)) + smooth)
    return l1 + 0.5 * (1 - dice.mean())
