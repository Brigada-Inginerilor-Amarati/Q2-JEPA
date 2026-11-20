"""D3IL frozen spline readout (Q-JEPA App. B.2): [h; t] -> 8 control points of a clamped cubic B-spline.

MLP 128 -> 128 -> 128 -> 16 (SiLU). The spline is evaluated at u = k/24, k = 1..24 (start excluded, endpoint
included), the same grid as the data's future segments. Coordinates are normalized, as in data/d3il.py.
"""

import numpy as np
import torch
from torch import nn

CONTROL_POINTS, DEGREE, POINTS = 8, 3, 24


def bspline_basis(n_ctrl: int = CONTROL_POINTS, degree: int = DEGREE, points: int = POINTS) -> np.ndarray:
    """(points, n_ctrl) clamped, uniform-interior B-spline basis at u = k / points, k = 1..points (Cox-de Boor)."""
    interior = np.linspace(0, 1, n_ctrl - degree + 1)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    u = np.arange(1, points + 1) / points
    basis = np.zeros((points, len(knots) - 1))
    for i in range(len(knots) - 1):
        basis[:, i] = (knots[i] <= u) & (u < knots[i + 1])
    basis[u == 1.0, n_ctrl - 1] = 1.0  # clamped end: the curve ends on the last control point
    for p in range(1, degree + 1):
        nxt = np.zeros((points, len(knots) - 1 - p))
        for i in range(len(knots) - 1 - p):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                nxt[:, i] += (u - knots[i]) / left * basis[:, i]
            if right > 0:
                nxt[:, i] += (knots[i + p + 1] - u) / right * basis[:, i + 1]
        basis = nxt
    return basis


class SplineReadout(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * CONTROL_POINTS),
        )  # fmt: skip
        self.register_buffer("basis", torch.tensor(bspline_basis(), dtype=torch.float32))

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(..., d), (..., d) -> segment points (..., 24, 2), normalized coordinates."""
        ctrl = self.net(torch.cat([h, t], -1)).unflatten(-1, (CONTROL_POINTS, 2))
        return self.basis.to(ctrl.dtype) @ ctrl
