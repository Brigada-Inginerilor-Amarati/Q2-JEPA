"""Charged particles (data/nri.py): 3 balls, 4 hidden-charge futures, whole-scene clips. State: the balls' positions
in the 3 future frames (18 numbers, px); a sample matches a future if every coordinate is within POS_TOL px."""

import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.nri import ChargedParticles, render

POS_TOL = 2.0  # px; futures are generated at least MIN_SEP = 5 px apart


class ChargedParticlesBenchmark(FuturesBenchmark):
    tol = (POS_TOL,) * 18

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = ChargedParticles(split)
        obs, fut = torch.tensor(ds.obs, dtype=torch.float32), torch.tensor(ds.fut, dtype=torch.float32)
        return render(obs), render(fut), fut.flatten(2)  # (n, 3, 64, 64), (n, 4, 3, 64, 64), (n, 4, 18)
