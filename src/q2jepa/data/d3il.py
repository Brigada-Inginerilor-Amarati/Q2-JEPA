"""D3IL Avoiding (Q-JEPA §5.4, App. B): planar demonstrations rendered as 3-frame clips.

World to pixel: points are in normalized coordinates (xy_m - NORM_CENTRE) / NORM_SCALE. The fixed window (WINDOW_X,
WINDOW_Y) is an 80 cm square, y centred on the start and the goal line and x on the start, about 1.27 cm/pixel. Then
col = (x - x_min) / 2 * 63 and row = (y_max - y) / 2 * 63 on pixel centres 0..63, so +x points right and +y points up.
"""

import numpy as np
import torch

NORM_CENTRE = (0.5, -0.28)
NORM_SCALE = 0.4
WINDOW_X = (-0.9375, 1.0625)  # x in [0.125, 0.925] m
WINDOW_Y = (-0.2125, 1.7875)  # y in [-0.365, 0.435] m
SIGMA = 0.85  # px; provisional, final value set by the render check (dataset spec section 8)
START_POINT = (0.52492847, -0.27973673)  # App. B.4, metres


def normalize(xy_m: np.ndarray) -> np.ndarray:
    return (np.asarray(xy_m) - NORM_CENTRE) / NORM_SCALE


def to_pixels(points: torch.Tensor) -> torch.Tensor:
    """(col, row) of normalized points (..., 2), clamped to the window edge."""
    x = points[..., 0].clamp(*WINDOW_X)
    y = points[..., 1].clamp(*WINDOW_Y)
    return torch.stack([(x - WINDOW_X[0]) / 2 * 63, (WINDOW_Y[1] - y) / 2 * 63], -1)


def render_segment(points: torch.Tensor) -> torch.Tensor:
    """Clip (..., 3, 64, 64) from a segment (..., 24, 2) in normalized coordinates, on its device.

    Points 1-8, 9-16 and 17-24 become frames 0, 1 and 2, which keeps the direction of travel. Each frame is the
    polyline through its points: Gaussian falloff of the distance to each sub-segment, max-combined, peak amplitude 1.
    Out-of-window points clamp to the edge, so failed rollouts never give a blank clip.
    """
    if not torch.isfinite(points).all():
        raise ValueError("segment points must be finite")
    frames = to_pixels(points.float()).unflatten(-2, (3, 8))  # (..., 3, 8, 2)
    start = frames[..., :-1, None, None, :]  # (..., 3, 7, 1, 1, 2)
    along = frames[..., 1:, None, None, :] - start
    grid = torch.arange(64, dtype=torch.float32, device=points.device)
    rows, cols = torch.meshgrid(grid, grid, indexing="ij")
    to_pixel = torch.stack([cols, rows], -1) - start  # (..., 3, 7, 64, 64, 2)
    t = ((to_pixel * along).sum(-1) / (along**2).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    dist2 = ((to_pixel - t[..., None] * along) ** 2).sum(-1)
    return torch.exp(-dist2 / (2 * SIGMA**2)).amax(-3)
