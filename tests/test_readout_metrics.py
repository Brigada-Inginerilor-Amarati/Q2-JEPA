import numpy as np
import torch

from q2jepa.data.mmnist import true_trajectories
from q2jepa.metrics import mmnist_metrics
from q2jepa.readout.mmnist import TranslationReadout, render


def test_zero_init_identity_and_bound():
    r = TranslationReadout()
    d = r(torch.randn(2, 64), torch.randn(2, 64))
    assert torch.equal(d, torch.zeros(2, 3, 2))
    last = torch.rand(2, 64, 64)
    assert torch.allclose(render(last, d), last[:, None].expand(2, 3, 64, 64), atol=1e-4)
    with torch.no_grad():
        r.net[-1].bias.fill_(1e6)
    assert r(torch.randn(1, 64), torch.randn(1, 64)).abs().max() <= 18.0


def test_metrics_perfect_distribution():
    t = true_trajectories()
    samples = np.concatenate([np.repeat(t[k : k + 1], 3 if k % 2 == 0 else 2, 0) for k in range(8)])
    m = mmnist_metrics(samples)
    assert m["validity"] == 1.0 and m["coverage"] == 1.0 and abs(m["tv"]) < 1e-9 and abs(m["min_ade"]) < 1e-9


def test_metrics_single_mode_and_invalid():
    t = true_trajectories()
    m = mmnist_metrics(np.repeat(t[0:1], 128, 0))
    assert m["coverage"] == 1 / 8 and abs(m["tv"] - 0.85) < 1e-9
    far = mmnist_metrics(np.full((128, 3, 2), 50.0))
    assert far["validity"] == 0.0 and far["coverage"] == 0.0 and abs(far["tv"] - 1.0) < 1e-9
