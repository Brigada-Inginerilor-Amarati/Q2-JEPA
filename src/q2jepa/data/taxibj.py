"""TaxiBJ crowd flow (Zhang et al., ST-ResNet, AAAI 2017), OpenSTL's preprocessed release, rendered as 64x64 clips.

Each sample: 4 observed and 4 future half-hour maps of Beijing taxi inflow and outflow on a 32 x 32 grid, values
in [-1, 1] (OpenSTL `dataset.npz`: X_train/Y_train 20461, X_test/Y_test 500). We keep the last 3 observed and the
first 3 future maps (our encoders take 3-frame clips), rescaled to [0, 1] as OpenSTL does.
A frame is rendered as inflow | outflow side by side (32 x 64), rows doubled to 64 x 64.
One future per observation: this benchmark tests prediction error, not multi-future coverage.

scripts/fetch_taxibj.py writes `maps.npz` (float16, the kept frames per split, ~0.5 GB) next to the 2.6 GB
float64 `dataset.npz`, so each run loads the small file (several runs share the VM's RAM).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch

PATH = Path("taxibj/dataset.npz")  # under dataset.root; scripts/fetch_taxibj.py
MAPS = Path("taxibj/maps.npz")
URL = "https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/taxibj_dataset.zip"
SHA256 = "9a7bc62983e3706f7fad5448848ab58aa0576084e53238810566c2ac669232a8"  # dataset.npz, fetched 2026-11-23
N_DEV = 80  # the last 80 train windows
GAP = 8  # train windows dropped before dev: a window spans 8 half-hours, so neighbours would share maps with dev
SPLITS = ("train", "dev", "test")


def compact(src: Path, out: Path) -> None:
    """dataset.npz -> maps.npz: <split>_obs, <split>_fut (n, 3, 2, 32, 32) float16 in [0, 1]."""
    z = np.load(src)
    arrays = {}
    for split in SPLITS:
        if split == "test":
            x, y = z["X_test"], z["Y_test"]
        else:
            x, y = z["X_train"], z["Y_train"]
            x, y = (x[: -(N_DEV + GAP)], y[: -(N_DEV + GAP)]) if split == "train" else (x[-N_DEV:], y[-N_DEV:])
        arrays[f"{split}_obs"] = ((x[:, 1:4] + 1) / 2).astype(np.float16)
        arrays[f"{split}_fut"] = ((y[:, :3] + 1) / 2).astype(np.float16)
    np.savez(out, **arrays)


def render(maps: torch.Tensor) -> torch.Tensor:
    """Frames (..., 64, 64) from maps (..., 2, 32, 32) in [0, 1]: inflow | outflow, rows doubled."""
    side = torch.cat([maps[..., 0, :, :], maps[..., 1, :, :]], -1)  # (..., 32, 64)
    return side.float().repeat_interleave(2, -2)


class TaxiBJ:
    """obs (n, 3, 2, 32, 32), fut (n, 3, 2, 32, 32) in [0, 1], float16 to keep the train split small on the GPU."""

    def __init__(self, split: Literal["train", "dev", "test"], root: str = "data"):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        path = Path(root) / MAPS
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python scripts/fetch_taxibj.py --root {root}")
        z = np.load(path)
        self.obs, self.fut = torch.from_numpy(z[f"{split}_obs"]), torch.from_numpy(z[f"{split}_fut"])

    def __len__(self) -> int:
        return len(self.obs)
