"""Seeding and checkpoint helpers shared by train.py and the benchmarks.

Every iteration reseeds from (seed, it), so a checkpoint needs only model and optimizer state.
"""

import os
from pathlib import Path

import torch

READOUT_SEED_OFFSET = 10_000_000
EVAL_SEED_OFFSET = 20_000_000
FINETUNE_SEED_OFFSET = 30_000_000


def reseed(seed: int, it: int) -> None:
    torch.manual_seed(seed * 1_000_003 + it)


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
