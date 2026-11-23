"""TaxiBJ crowd flow (Zhang et al., ST-ResNet, AAAI 2017), OpenSTL's preprocessed release, rendered as 64x64 clips.

Each sample: 4 observed and 4 future half-hour maps of Beijing taxi inflow and outflow on a 32 x 32 grid, values
in [-1, 1] (OpenSTL `dataset.npz`: X_train/Y_train 20461, X_test/Y_test 500). We keep the last 3 observed and the
first 3 future maps (our encoders take 3-frame clips), rescaled to [0, 1] as OpenSTL does.
A frame is rendered as inflow | outflow side by side (32 x 64), rows doubled to 64 x 64.
One future per observation: this benchmark tests prediction error, not multi-future coverage.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch

PATH = Path("taxibj/dataset.npz")  # under dataset.root; scripts/fetch_taxibj.py
URL = "https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/taxibj_dataset.zip"
SHA256 = "9a7bc62983e3706f7fad5448848ab58aa0576084e53238810566c2ac669232a8"  # dataset.npz, fetched 2026-11-23
N_DEV = 80  # the last 80 train samples; train keeps the rest (20381)


def render(maps: torch.Tensor) -> torch.Tensor:
    """Frames (..., 64, 64) from maps (..., 2, 32, 32) in [0, 1]: inflow | outflow, rows doubled."""
    side = torch.cat([maps[..., 0, :, :], maps[..., 1, :, :]], -1)  # (..., 32, 64)
    return side.float().repeat_interleave(2, -2)


class TaxiBJ:
    """obs (n, 3, 2, 32, 32), fut (n, 3, 2, 32, 32) in [0, 1], float16 to keep the train split small on the GPU."""

    def __init__(self, split: Literal["train", "dev", "test"], root: str = "data"):
        path = Path(root) / PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python scripts/fetch_taxibj.py --root {root}")
        z = np.load(path)
        if split == "test":
            x, y = z["X_test"], z["Y_test"]
        elif split in ("train", "dev"):
            x, y = z["X_train"], z["Y_train"]
            x, y = (x[:-N_DEV], y[:-N_DEV]) if split == "train" else (x[-N_DEV:], y[-N_DEV:])
        else:
            raise ValueError(f"unknown split {split!r}")
        self.obs = torch.tensor((x[:, 1:4] + 1) / 2, dtype=torch.float16)
        self.fut = torch.tensor((y[:, :3] + 1) / 2, dtype=torch.float16)

    def __len__(self) -> int:
        return len(self.obs)
