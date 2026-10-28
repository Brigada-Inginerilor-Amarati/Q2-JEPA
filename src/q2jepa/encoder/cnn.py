"""Q-JEPA App. A.2 MovingMNIST encoder (3-frame clip stacked as channels)."""

import torch
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, res: int, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * (res // 8) ** 2, 96), nn.SiLU(),
            nn.Linear(96, d),
            nn.LayerNorm(d),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
