"""Reference errors for the TaxiBJ results table: copy the last observed map, and the mean train future map.

python scripts/taxibj_baselines.py [--root data] [--split test]

Error: mean squared error per value on the [0, 1] scale, as metrics.flow_metrics reports it. No model, no training.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.data.taxibj import TaxiBJ


def copy_last_mse(obs: torch.Tensor, fut: torch.Tensor) -> float:
    """The last observed map (n, 3, 2, 32, 32)[:, -1] as the forecast of all 3 future maps."""
    return float(((obs[:, -1:].double() - fut.double()) ** 2).mean())


def train_mean_mse(train_fut: torch.Tensor, fut: torch.Tensor) -> float:
    """The mean train future (one map per horizon step, cell and channel) as the forecast for every sample."""
    total = sum(c.double().sum(0) for c in train_fut.split(1024))  # chunked: float64 train futures are ~1 GB
    return float(((total / len(train_fut) - fut.double()) ** 2).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    a = p.parse_args()
    ds, train = TaxiBJ(a.split, a.root), TaxiBJ("train", a.root)
    print(f"{a.split}: {len(ds)} samples")
    print(f"copy last map   mse={copy_last_mse(ds.obs, ds.fut):.6f}")
    print(f"train mean map  mse={train_mean_mse(train.fut, ds.fut):.6f}")


if __name__ == "__main__":
    main()
