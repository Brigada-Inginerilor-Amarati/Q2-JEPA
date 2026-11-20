"""HEP18 frozen trajectory readout (Q-JEPA App. A.2, F.3): one agent's 16-dim feature -> speeds -> positions.

A shared per-agent MLP 16 -> 64 -> 3 (SiLU) gives the speeds at t = 3, 6, 9 s, bounded by 12 sigmoid(a) m/s.
Positions follow by trapezoidal integration from the observed (s, v) at t = 0, in 3 s steps. The MLP sees no
observation features; the observed state only starts the integration.
"""

import torch
from torch import nn

STEP_S = 3.0
SPEED_MAX = 12.0
POS_SCALE, SPEED_SCALE = 20.0, 5.0  # loss scales (App. A.2)


class SpeedReadout(nn.Module):
    def __init__(self, d_agent: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_agent, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(..., d_agent) -> speeds (..., 3) in m/s."""
        return SPEED_MAX * torch.sigmoid(self.net(feat))


def integrate(s0: torch.Tensor, v0: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
    """Positions (..., 3) at t = 3, 6, 9 s from s0, v0 (...) at t = 0 and speeds (..., 3), trapezoidal rule."""
    prev = torch.cat([v0.expand(speeds.shape[:-1])[..., None], speeds[..., :-1]], -1)
    return s0[..., None] + torch.cumsum(STEP_S / 2 * (prev + speeds), -1)


def readout_loss(s: torch.Tensor, v: torch.Tensor, s_true: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    return (((s - s_true) / POS_SCALE) ** 2).mean() + (((v - v_true) / SPEED_SCALE) ** 2).mean()
