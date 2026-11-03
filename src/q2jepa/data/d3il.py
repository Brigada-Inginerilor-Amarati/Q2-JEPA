"""D3IL Avoiding (Q-JEPA §5.4, App. B): planar demonstrations rendered as 3-frame clips.

World to pixel: points are in normalized coordinates (xy_m - NORM_CENTRE) / NORM_SCALE. The fixed window (WINDOW_X,
WINDOW_Y) is an 80 cm square, y centred on the start and the goal line and x on the start, about 1.27 cm/pixel. Then
col = (x - x_min) / 2 * 63 and row = (y_max - y) / 2 * 63 on pixel centres 0..63, so +x points right and +y points up.

Data: paths are the recorded c_pos, cut at the first upward crossing of the goal line with an interpolated endpoint.
Recordings run on to y = 0.488 m; App. B.1 does not mention the cut, but without it the count is 1,161 segments, and
with it 1,029 against the paper's 1,030. Without it demos also leave the render window. The 96 recorded starts, the
24 routes and their 22 x 4 + 3 + 5 distribution all match App. B.4.
"""

from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

NORM_CENTRE = (0.5, -0.28)
NORM_SCALE = 0.4
WINDOW_X = (-0.9375, 1.0625)  # x in [0.125, 0.925] m
WINDOW_Y = (-0.2125, 1.7875)  # y in [-0.365, 0.435] m
SIGMA = 0.85  # px; provisional, final value set by the render check (dataset spec section 8)
START_POINT = (0.52492847, -0.27973673)  # App. B.4, metres

# ALRhub/d3il @ 1d9c718, environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py
# (get_obj_list: Cylinder size[0] is the radius; finish_line gives the goal). Metres. Rows are in crossing order.
ROW_Y = (-0.1, 0.08, 0.26)
ROW_OBSTACLE_X = ((0.5,), (0.425, 0.575), (0.35, 0.5, 0.65))
ROW_RADIUS = (0.03, 0.025, 0.025)
GOAL_Y = 0.35
INFLATION = 0.01  # App. B.4: manipulator tip
REACH = 1.0913  # App. B.4; the paper does not name its base point
REACH_TOL = 1e-6
REACH_BASE = (0.0, 0.0)  # environments/d3il/d3il_sim/core/Robots.py:33, the robot's default base_position

SEGMENT_M = 0.08
EXPECTED_SEGMENTS = 1029  # App. B.1 reports 1,030; see the module docstring


def start_point() -> np.ndarray:
    return np.array(START_POINT)


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


def route_label(path_xy: np.ndarray) -> int | None:
    """Route 0..23 = 12 g0 + 4 g1 + g2, or None if a row is never crossed upward.

    g_r is the gap at the first upward crossing of row r's centre line, numbered by increasing x.
    """
    y, gaps = path_xy[:, 1], []
    for row_y, obstacles in zip(ROW_Y, ROW_OBSTACLE_X, strict=True):
        crossings = np.flatnonzero((y[:-1] < row_y) & (y[1:] >= row_y))
        if not len(crossings):
            return None
        i = crossings[0]
        x = path_xy[i, 0] + (row_y - y[i]) / (y[i + 1] - y[i]) * (path_xy[i + 1, 0] - path_xy[i, 0])
        gaps.append(int(np.searchsorted(obstacles, x)))
    return 12 * gaps[0] + 4 * gaps[1] + gaps[2]


def to_goal(path: np.ndarray) -> np.ndarray:
    """Path up to its first upward crossing of GOAL_Y, ending exactly on the line; unchanged if it never crosses."""
    crossings = np.flatnonzero((path[:-1, 1] < GOAL_Y) & (path[1:, 1] >= GOAL_Y))
    if not len(crossings):
        return path
    i = crossings[0]
    end = path[i] + (GOAL_Y - path[i, 1]) / (path[i + 1, 1] - path[i, 1]) * (path[i + 1] - path[i])
    return np.vstack([path[: i + 1], end])


def first_observation(path: np.ndarray) -> np.ndarray:
    """Recorded frames 4-8, 1-indexed, interpolated to 24 points uniformly in frame index (App. B.1)."""
    frames, t = path[3:8], np.linspace(0, 4, 24)
    return np.stack([np.interp(t, np.arange(5), frames[:, axis]) for axis in range(2)], -1)


def segments(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split into 8 cm arc-length segments: points (K, 24, 2) and true lengths (K,), keeping a shorter final one.

    Points sit at arc-length fractions k/24, k = 1..24, of each segment: the start is excluded and the endpoint included,
    matching the readout's spline grid (App. B.2).
    """
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    moving = np.concatenate([[True], steps > 0])
    arc, path = np.concatenate([[0.0], np.cumsum(steps[steps > 0])]), path[moving]
    count = int(np.ceil((arc[-1] - 1e-9) / SEGMENT_M))
    starts = SEGMENT_M * np.arange(count)
    lengths = np.append(np.full(max(count - 1, 0), SEGMENT_M), arc[-1] - starts[-1:])[:count]
    at = starts[:, None] + lengths[:, None] * np.arange(1, 25) / 24
    return np.stack([np.interp(at, arc, path[:, axis]) for axis in range(2)], -1), lengths


def make_folds(routes: np.ndarray, seed: int) -> np.ndarray:
    """Each demo's eval fold in {0, 1, 2}: shuffle within routes, then deal in turn, so folds are equal and stratified."""
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.flatnonzero(routes == r)) for r in np.unique(routes)])
    folds = np.empty(len(routes), dtype=np.int64)
    folds[order] = np.arange(len(routes)) % 3
    return folds


def _load(root: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Recorded paths (metres) and route labels, failing loudly if the data doesn't match App. B.4."""
    npz = Path(root) / "d3il" / "avoiding.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} is missing; run scripts/fetch_d3il.py")
    with np.load(npz, allow_pickle=False) as data:
        xy, offsets = data["xy"], data["offsets"]
    paths = [xy[a:b] for a, b in pairwise(offsets)]
    labels = [route_label(p) for p in paths]
    if len(paths) != 96 or None in labels:
        raise ValueError(f"expected 96 demos crossing every row, got {len(paths)} with labels {labels}")
    if np.abs(np.array([p[0] for p in paths]) - START_POINT).max() > 1e-6:
        raise ValueError(f"demo starts differ from {START_POINT}")
    routes = np.array(labels)
    if sorted(np.bincount(routes, minlength=24).tolist()) != [3, *[4] * 22, 5]:
        raise ValueError(f"route counts differ from App. B.4: {np.bincount(routes, minlength=24)}")
    return paths, routes


def _check_args(split: str, fold: int) -> None:
    if split not in ("train", "test"):
        raise ValueError(
            f"split must be train or test, got {split!r}: D3IL has no dev split (dataset spec section 5.6)"
        )
    if fold not in (0, 1, 2):
        raise ValueError(f"fold must be 0, 1 or 2, got {fold}")


def _demo_ids(routes: np.ndarray, split: str, fold: int, seed: int) -> np.ndarray:
    folds = make_folds(routes, seed)
    return np.flatnonzero(folds == fold if split == "test" else folds != fold)


def demos(fold: int, split: str, root: str = "data", seed: int = 0) -> list[np.ndarray]:
    """The fold's paths in metres, from the start to the goal line, for the reference panel."""
    _check_args(split, fold)
    paths, routes = _load(root)
    return [to_goal(paths[i]) for i in _demo_ids(routes, split, fold, seed)]


class D3ILAvoiding(Dataset):
    """One item per observation-future segment pair, states only; render with render_segment. "test" is the eval demos."""

    def __init__(self, split: Literal["train", "test"], fold: int, root: str = "data", seed: int = 0):
        _check_args(split, fold)
        paths, routes = _load(root)
        self.items = []
        for demo in _demo_ids(routes, split, fold, seed):
            pts, lengths = segments(to_goal(paths[demo])[7:])
            obs = first_observation(paths[demo])
            for k, (fut, length) in enumerate(zip(pts, lengths, strict=True)):
                self.items.append((normalize(obs), normalize(fut), length, demo, k, routes[demo]))
                obs = fut

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        obs, fut, length, demo, k, route = self.items[i]
        return {
            "obs_pts": torch.tensor(obs, dtype=torch.float32),
            "fut_pts": torch.tensor(fut, dtype=torch.float32),
            "fut_len": torch.tensor(length, dtype=torch.float32),
            "demo_id": torch.tensor(demo),
            "seg_idx": torch.tensor(k),
            "route": torch.tensor(route),
        }
