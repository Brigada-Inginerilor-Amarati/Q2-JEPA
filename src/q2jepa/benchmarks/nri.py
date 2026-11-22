"""Charged particles (data/nri.py): 3 balls, 4 hidden-charge futures, whole-scene clips. State: the balls' positions
in the 3 future frames (18 numbers, px). Balls are matched as a set (any of the 3! orders), because the readout
can confuse them: they differ only in blob width. Validity: metrics.match_futures."""

from itertools import permutations

import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.nri import BALLS, ChargedParticles, render

# state index = frame * 6 + ball * 2 + axis; one index list per ball order
BALL_PERMS = [
    [f * 2 * BALLS + 2 * b + a for f in range(3) for b in p for a in range(2)] for p in permutations(range(BALLS))
]


class ChargedParticlesBenchmark(FuturesBenchmark):
    src_dims, scale, perms = 3, 1.0, BALL_PERMS  # source: positions (3 frames, 3 balls, 2), px

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = ChargedParticles(split)
        obs, fut = torch.tensor(ds.obs, dtype=torch.float32), torch.tensor(ds.fut, dtype=torch.float32)
        return obs, fut, fut.flatten(2)  # (n, 3, 3, 2), (n, 4, 3, 3, 2), (n, 4, 18)

    def render(self, src: torch.Tensor) -> torch.Tensor:
        return render(src)  # (..., 3, 3, 2) -> (..., 3, 64, 64)
