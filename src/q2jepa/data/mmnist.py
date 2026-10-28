"""Stochastic single-digit MovingMNIST (Q-JEPA App. F.2).

Observation: 3 frames, digit centre at x = 18, 21, 24, fixed y ~ U{24..40}.
Future: 3 frames continuing from the last observed frame, 4 px/frame in one of 8 directions.
Image coordinates: +x right, +y down; direction k moves along (cos 45k°, sin 45k°).
"""

from typing import ClassVar, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

ANGLES_DEG = np.arange(8) * 45
DIRECTION_PROBS = np.array([0.15, 0.10] * 4)  # even k = cardinal
RECORDS_PER_DIRECTION = [3, 2] * 4
SIZE = 64
SPRITE_OFFSET = 18  # 28x28 sprite pasted at (18, 18) -> treated as centred at (32, 32)
CANVAS_CENTRE = 32.0
OBS_X = (18.0, 21.0, 24.0)
STEP = 4.0


def true_trajectories() -> np.ndarray:
    """(8, 3, 2) displacements from the last observed centre, frames f = 1..3."""
    rad = np.deg2rad(ANGLES_DEG)
    unit = np.stack([np.cos(rad), np.sin(rad)], axis=-1)
    return STEP * np.arange(1, 4)[None, :, None] * unit[:, None, :]


def translate(images: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Shift images (..., H, W) by disp (..., 2) = (dx, dy) pixels; bilinear, zero padding."""
    h, w = images.shape[-2:]
    lead = torch.broadcast_shapes(images.shape[:-2], disp.shape[:-1])
    imgs = images.expand(*lead, h, w).reshape(-1, 1, h, w)
    d = disp.to(images.dtype).expand(*lead, 2).reshape(-1, 2)
    theta = torch.zeros(d.shape[0], 2, 3, dtype=images.dtype, device=images.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * d[:, 0] / (w - 1)
    theta[:, 1, 2] = -2.0 * d[:, 1] / (h - 1)
    grid = F.affine_grid(theta, list(imgs.shape), align_corners=True)
    out = F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(*lead, h, w)


class StochasticMovingMNIST(Dataset):
    _SPLIT_ID: ClassVar[dict[str, int]] = {"train": 0, "dev": 1, "test": 2}
    _DEFAULT_N: ClassVar[dict[str, int]] = {"train": 800, "dev": 80, "test": 80}

    def __init__(
        self, split: Literal["train", "dev", "test"], root: str = "data", n_obs: int | None = None, seed: int = 0
    ):
        n = n_obs or self._DEFAULT_N[split]
        mnist = MNIST(root, train=split != "test", download=True)
        perm = np.random.default_rng(seed).permutation(len(mnist))
        # train takes the head of the MNIST-train permutation, dev the tail: disjoint by construction
        self.mnist_indices = (perm[-n:] if split == "dev" else perm[:n]).tolist()
        rng = np.random.default_rng([seed, self._SPLIT_ID[split]])

        sprites = mnist.data[self.mnist_indices].float() / 255.0
        canvas = torch.zeros(n, SIZE, SIZE)
        canvas[:, SPRITE_OFFSET : SPRITE_OFFSET + 28, SPRITE_OFFSET : SPRITE_OFFSET + 28] = sprites
        ys = torch.tensor(rng.integers(24, 41, size=n), dtype=torch.float32)
        disp = torch.stack(
            [torch.tensor(OBS_X).expand(n, 3) - CANVAS_CENTRE, (ys - CANVAS_CENTRE)[:, None].expand(n, 3)], -1
        )
        self._obs = translate(canvas[:, None], disp).clamp(0, 1)  # (n, 3, 64, 64)
        self._traj = torch.tensor(true_trajectories(), dtype=torch.float32)

        if split == "train":
            dirs = np.repeat(np.arange(8), RECORDS_PER_DIRECTION)
            self.records = [(i, int(k)) for i in range(n) for k in rng.permutation(dirs)]
        else:
            self.records = [(i, k) for i in range(n) for k in range(8)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        i, k = self.records[idx]
        obs = self._obs[i]
        fut = translate(obs[2], self._traj[k]).clamp(0, 1)
        return {"obs": obs, "fut": fut, "direction": k, "obs_idx": i}

    def unique_obs(self) -> torch.Tensor:
        return self._obs
