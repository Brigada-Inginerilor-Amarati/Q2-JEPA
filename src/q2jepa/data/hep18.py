"""HEP18: HighwayEnv priority intersection (Q-JEPA §5.3, App. F.3), rendered per car.

Four cars approach from N, W, S, E and drive straight through. Each observation has 18 equally likely joint futures:
axis (NS or EW) has priority, each priority car picks schedule A, B or C, and both yielding cars use D.

d0 is the distance from the centre at t = 0, the end of the observation, so s(t) = -d0 + v0 t on [-2, 0]. Measured at
t = -2 instead, yielding cars would stop between 7 m before and 2 m past the centre, inside the intersection.

Priority order in j = 9 axis + 3 a_first + a_second: axis 0 (NS) has first = N, second = S; axis 1 (EW) has first = W,
second = E. Schedules are numbered A, B, C = 0, 1, 2.

Road frame is HighwayEnv's: +x east, +y south (screen coordinates). Each car's s is 0 on the centre line and increases
along its driving direction, in the right-hand lane.

Mean L2 norm of a rendered train frame at SIGMA = 1.25: 2.2156, so encoder.input_scale = 0.0346. Rerun
scripts/render_examples.py if SIGMA changes.
"""

from typing import ClassVar, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

CARS = ("N", "W", "S", "E")
SCHEDULE_A, SCHEDULE_B, SCHEDULE_C, SCHEDULE_D = 0, 1, 2, 3

# highway-env 1.12.1: road/lane.py:16 (DEFAULT_WIDTH), vehicle/kinematics.py:21,23 (LENGTH, WIDTH),
# envs/intersection_env.py:160 (outer_distance), :173 (incoming lanes: rotation @ [lane_width / 2, ...]).
LANE_WIDTH = 4.0
VEHICLE_LENGTH = 5.0
VEHICLE_WIDTH = 2.0
INTERSECTION_HALF = 11.0
HEADING = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])  # per car, in CARS order
LANE_POINT = LANE_WIDTH / 2 * np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])  # lane position at s = 0
_HALF_EXTENT = np.abs(HEADING) * VEHICLE_LENGTH / 2 + np.abs(HEADING[:, ::-1]) * VEHICLE_WIDTH / 2  # axis-aligned cars

STAGE = 3.0
_ACC_FIXED = np.array([[-0.5, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])  # App. F.3 Eq. 17
_ACC_PER_V0 = np.array([[0.0, 0.0, 0.0]] * 3 + [[-1 / 5, -1 / 10, -1 / 30]])
_OBS_T = np.array([-2.0, -1.0, 0.0])
_FUT_T = np.array([3.0, 6.0, 9.0])
_CHECK_T = np.linspace(-2.0, 9.0, 221)  # 0.05 s

# Positions span [-65, 49.25] m (section 4.3-4.4 bounds); padded by 3 sigma at sigma = 1.5 px. About 2.1 m/pixel.
S_WINDOW = (-75.0, 60.0)
SIGMA = 1.25  # px; provisional, final value set by the render check (dataset spec section 8)
COLUMN = 31.5  # px; provisional, same check


def schedule_acc(schedule, v0: np.ndarray) -> np.ndarray:
    """Accelerations (..., 3) in m/s^2 per stage; schedule (int or array) and v0 broadcast."""
    return _ACC_FIXED[schedule] + _ACC_PER_V0[schedule] * np.asarray(v0)[..., None]


def car_state(d0, v0, acc, t) -> tuple[np.ndarray, np.ndarray]:
    """Exact (s, v), shape (..., T), at times t (T,) in [-2, 9]: constant speed before 0, then acc (..., 3) per stage."""
    d0, v0, acc, t = np.asarray(d0), np.asarray(v0), np.asarray(acc), np.asarray(t)
    in_stage = np.clip(t[:, None] - STAGE * np.arange(3), 0.0, STAGE)  # (T, 3): time spent in each stage
    v_start = v0[..., None] + STAGE * np.concatenate([np.zeros_like(acc[..., :1]), np.cumsum(acc, -1)[..., :-1]], -1)
    v_start, acc = v_start[..., None, :], acc[..., None, :]
    s = -d0[..., None] + v0[..., None] * np.minimum(t, 0.0)
    return s + (v_start * in_stage + 0.5 * acc * in_stage**2).sum(-1), v0[..., None] + (acc * in_stage).sum(-1)


def future_index(axis: int, a_first: int, a_second: int) -> int:
    return 9 * axis + 3 * a_first + a_second


def future_schedules(j: int) -> np.ndarray:
    """Each car's schedule (4,), in CARS order, for joint future j."""
    axis, first, second = j // 9, (j % 9) // 3, j % 3
    return np.array([first, SCHEDULE_D, second, SCHEDULE_D] if axis == 0 else [SCHEDULE_D, first, SCHEDULE_D, second])


_SCHEDULES = np.stack([future_schedules(j) for j in range(18)])  # (18, 4)


def to_xy(s, v, car) -> np.ndarray:
    """The paper's stored state (..., 6) = (x/50, y/50, vx/10, vy/10, cos phi, sin phi); car (int or array) broadcasts."""
    s, v, heading = np.asarray(s), np.asarray(v), HEADING[car]
    xy = (LANE_POINT[car] + s[..., None] * heading) / 50
    vel = v[..., None] * heading / 10
    return np.concatenate([xy, vel, np.broadcast_to(heading, xy.shape)], -1)


def any_overlap(xy: np.ndarray) -> bool:
    """Whether any two car rectangles overlap; xy (..., 4, 2) in metres, cars in CARS order."""
    gap = np.abs(xy[..., :, None, :] - xy[..., None, :, :])
    hit = (gap < _HALF_EXTENT[:, None] + _HALF_EXTENT[None, :]).all(-1)
    return bool((hit & ~np.eye(4, dtype=bool)).any())


def collides(d0: np.ndarray, v0: np.ndarray) -> bool:
    """Whether any of the observation's 18 futures has overlapping cars at 0.05 s steps over [-2, 9] s."""
    s, _ = car_state(d0, v0, schedule_acc(_SCHEDULES, v0), _CHECK_T)  # (18, 4, T)
    xy = LANE_POINT[:, None] + s[..., None] * HEADING[:, None]  # (18, 4, T, 2)
    return any_overlap(np.moveaxis(xy, 1, 2))


def generate(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(d0, v0, resampled): n observations (n, 4) from one stream, redrawing any that collide."""
    rng = np.random.default_rng(seed)
    d0, v0, resampled = np.zeros((n, 4)), np.zeros((n, 4)), 0
    for i in range(n):
        v0[i], d0[i] = rng.uniform(8.0, 9.0, 4), rng.uniform(43.0, 47.0, 4)
        while collides(d0[i], v0[i]):
            resampled += 1
            v0[i], d0[i] = rng.uniform(8.0, 9.0, 4), rng.uniform(43.0, 47.0, 4)
    return d0, v0, resampled


def s_to_row(s: torch.Tensor) -> torch.Tensor:
    """Image row of longitudinal position s (m): row 0 is the top, so a car moves up the image."""
    return (S_WINDOW[1] - s) / (S_WINDOW[1] - S_WINDOW[0]) * 63


def render_agent_clip(s: torch.Tensor) -> torch.Tensor:
    """Clip (..., 3, 64, 64) of one car from s (..., 3) in metres, on s's device.

    World to pixel: row = s_to_row(s) on pixel centres 0..63, column fixed at COLUMN. Frame f is one Gaussian blob of
    peak amplitude 1 at s[..., f], evaluated analytically, so sub-pixel position survives. Used for observations
    (t = -2, -1, 0) and futures (t = 3, 6, 9) alike.
    """
    grid = torch.arange(64, dtype=torch.float32, device=s.device)
    rows = (grid - s_to_row(s.float())[..., None]) ** 2
    return torch.exp(-(rows[..., None] + (grid - COLUMN) ** 2) / (2 * SIGMA**2))


class HEP18(Dataset):
    """One item per observation, with all 18 futures. Items hold states only; render with render_agent_clip."""

    _SLICES: ClassVar[dict[str, slice]] = {"train": slice(0, 256), "dev": slice(256, 320), "test": slice(320, 448)}

    def __init__(self, split: Literal["train", "dev", "test"], root: str = "data", seed: int = 0):
        if split not in self._SLICES:
            raise ValueError(f"unknown split {split!r}")
        d0, v0, self.resampled = generate(448, seed)  # root is unused: nothing is downloaded
        self.d0, self.v0 = d0[self._SLICES[split]], v0[self._SLICES[split]]
        d, v = self.d0, self.v0
        self.obs_state = np.stack(car_state(d, v, np.zeros((*d.shape, 3)), _OBS_T), -1)  # (n, 4, 3, 2)
        d, v = d[:, None], v[:, None]
        self.fut_state = np.stack(car_state(d, v, schedule_acc(_SCHEDULES, v), _FUT_T), -1)  # (n, 18, 4, 3, 2)
        cars = np.arange(4)[:, None]
        self.obs_xy = to_xy(self.obs_state[..., 0], self.obs_state[..., 1], cars)
        self.fut_xy = to_xy(self.fut_state[..., 0], self.fut_state[..., 1], cars)

    def __len__(self) -> int:
        return len(self.d0)

    def __getitem__(self, i: int) -> dict:
        arrays = {k: getattr(self, k)[i] for k in ("obs_state", "fut_state", "obs_xy", "fut_xy")}
        item = {k: torch.tensor(a, dtype=torch.float32) for k, a in arrays.items()}
        return item | {"future_id": torch.arange(18), "obs_idx": torch.tensor(i)}
