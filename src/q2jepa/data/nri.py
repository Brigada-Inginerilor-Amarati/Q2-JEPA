"""Charged particles (NRI-style, Kipf et al. 2018, "charged" simulation), rendered as 64x64 clips.

Three balls move freely during the 3 observed frames (t = -2, -1, 0). At t = 0 their hidden charges switch on:
q = (+1, q1, q2), q1, q2 in {+1, -1}, with Coulomb forces between each pair (attract if opposite signs, repel if
equal). Ball 0's charge is fixed at +1 because only products q_i q_j act, so a global sign flip gives the same
motion. That leaves 4 equally likely joint futures per observation, which the observation cannot tell apart.
The future frames are t = 2, 4, 6.

Observations whose 4 futures are not at least MIN_SEP px apart (max over balls, at t = 6) are redrawn, as HEP18
redraws colliding ones. The balls have different blob widths so their identity is visible in every frame.
Coordinates are pixels, (x, y) with +y down; walls reflect at WALL px from the border.
"""

from typing import ClassVar, Literal

import numpy as np
import torch

BALLS = 3
CHARGES = np.array([[1.0, a, b] for a in (1.0, -1.0) for b in (1.0, -1.0)])  # (4, 3): the 4 joint futures
FUTURES = len(CHARGES)
STRENGTH = 250.0  # force constant, px^3 / time^2
SOFTEN = 2.0  # px, keeps close passes finite
DT, SUBSTEPS = 0.05, 20  # 20 x 0.05 = one time unit per observed frame
FUT_EVERY = 2  # future frames every 2 time units: t = 2, 4, 6
WALL = 4.0
MIN_SEP = 5.0  # px between any two futures at t = 6 (max over balls and axes)
MIN_DIST = 8.0  # px between balls at t = 0
SIGMAS = (1.2, 1.8, 2.4)  # blob width per ball, px
SPEED = 1.0  # px per time unit, std of each velocity component
N_TRAIN = 20000  # train scenes (dev 80, test 128)

# Mean L2 norm of a rendered train frame (seed 0): 5.7438, so encoder.input_scale = 0.0897.
# Rerun tests/test_futures_benchmarks.py::test_input_scale if SIGMAS change.


def step(p: np.ndarray, v: np.ndarray, q: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Advance positions and velocities (..., 3, 2) by n substeps; q (..., 3) charges, None = free flight."""
    p, v = p.copy(), v.copy()
    for _ in range(n):
        if q is not None:
            d = p[..., :, None, :] - p[..., None, :, :]
            r2 = (d**2).sum(-1) + SOFTEN**2
            f = STRENGTH * (q[..., :, None] * q[..., None, :])[..., None] * d / r2[..., None] ** 1.5
            v = v + f.sum(-2) * DT
        p = p + v * DT
        out = (p < WALL) | (p > 63 - WALL)
        v = np.where(out, -v, v)
        p = np.clip(p, WALL, 63 - WALL)
    return p, v


def rollout(p0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From positions and velocities at t = 0 (3, 2): observed (3 frames, 3, 2) and the 4 futures (4, 3, 3, 2)."""
    obs = [p0 - 2 * v0, p0 - v0, p0]  # free flight before t = 0 (start away from the walls, so no bounce)
    p, v = np.broadcast_to(p0, (FUTURES, BALLS, 2)), np.broadcast_to(v0, (FUTURES, BALLS, 2))
    fut = []
    for _ in range(3):
        p, v = step(p, v, CHARGES, FUT_EVERY * SUBSTEPS)
        fut.append(p)
    return np.stack(obs), np.stack(fut, 1)


def separation(fut: np.ndarray) -> float:
    """Smallest distance between two futures at the last frame: max over balls and axes, px."""
    last = fut[:, -1]
    gap = np.abs(last[:, None] - last[None]).max((-1, -2))
    return float(gap[np.triu_indices(FUTURES, 1)].min())


def generate(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(obs (n, 3, 3, 2), fut (n, 4, 3, 3, 2), redrawn) from one stream."""
    rng = np.random.default_rng(seed)
    obs, fut, redrawn = np.zeros((n, 3, BALLS, 2)), np.zeros((n, FUTURES, 3, BALLS, 2)), 0
    for i in range(n):
        while True:
            p0, v0 = rng.uniform(20.0, 44.0, (BALLS, 2)), rng.normal(0.0, SPEED, (BALLS, 2))
            dist = np.linalg.norm(p0[:, None] - p0[None], axis=-1)[np.triu_indices(BALLS, 1)]
            if dist.min() >= MIN_DIST:
                o, f = rollout(p0, v0)
                if separation(f) >= MIN_SEP:
                    break
            redrawn += 1
        obs[i], fut[i] = o, f
    return obs, fut, redrawn


def render(pos: torch.Tensor) -> torch.Tensor:
    """Frames (..., 64, 64) from ball positions (..., 3, 2) in px: one Gaussian blob per ball, peak 1, summed."""
    grid = torch.arange(64, dtype=torch.float32, device=pos.device)
    pos = pos.float()
    sig = torch.tensor(SIGMAS, device=pos.device)
    gx = torch.exp(-((grid - pos[..., 0, None]) ** 2) / (2 * sig[:, None] ** 2))  # (..., 3, 64)
    gy = torch.exp(-((grid - pos[..., 1, None]) ** 2) / (2 * sig[:, None] ** 2))
    return (gy[..., :, None] * gx[..., None, :]).sum(-3).clamp(max=1.0)


class ChargedParticles:
    """Ball positions per split: obs (n, 3, 3, 2), fut (n, 4, 3, 3, 2) in px. Render with render().

    Each split is its own stream (seed, seed + 1, seed + 2). Train has N_TRAIN scenes: with 800 the readout
    memorised them (train error 0.1 px, dev 6-10 px; diagnostic 2026-11-22). Generating 20000 takes ~70 s.
    """

    _SPLITS: ClassVar[dict[str, tuple[int, int]]] = {"train": (N_TRAIN, 0), "dev": (80, 1), "test": (128, 2)}

    def __init__(self, split: Literal["train", "dev", "test"], seed: int = 0):
        if split not in self._SPLITS:
            raise ValueError(f"unknown split {split!r}")
        n, offset = self._SPLITS[split]
        self.obs, self.fut, self.redrawn = generate(n, seed + offset)

    def __len__(self) -> int:
        return len(self.obs)
