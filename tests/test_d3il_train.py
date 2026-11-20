"""D3IL training pieces: spline readout, route statistics, path acceptance and smoke runs on synthetic demos."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.benchmarks.d3il import accept
from q2jepa.data import d3il
from q2jepa.metrics import ROUTES, expected_route_stats
from q2jepa.readout.d3il import SplineReadout, bspline_basis
from q2jepa.train import run

CONF = str(Path(__file__).resolve().parents[1] / "conf")
GAPS = ((0.42, 0.58), (0.35, 0.5, 0.65), (0.28, 0.425, 0.575, 0.72))  # gap centres per row, increasing x


def synthetic_path(route: int, jitter: float = 0.0) -> np.ndarray:
    """A demo through the gaps of `route`, ~5 mm steps, starting at START_POINT and running past the goal."""
    g = (route // 12, (route // 4) % 3, route % 4)
    sx, sy = d3il.START_POINT
    way = [(sx, sy), (sx + 0.002, sy + 0.01)]
    for (row_y, gaps), gi in zip(zip(d3il.ROW_Y, GAPS, strict=True), g, strict=True):
        way += [(gaps[gi] + jitter, row_y - 0.04), (gaps[gi] + jitter, row_y + 0.04)]
    way.append((way[-1][0], 0.45))
    pts = [np.array(way[0])]
    for a, b in itertools.pairwise(np.array(way)):
        n = max(2, int(np.linalg.norm(b - a) / 0.005))
        pts += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.array(pts)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """data/d3il/avoiding.npz with 96 synthetic demos and App. B.4's route distribution (22 x 4 + 3 + 5)."""
    counts = [3, 5] + [4] * 22
    paths = [synthetic_path(r, 0.003 * (k - 1.5)) for r, c in enumerate(counts) for k in range(c)]
    out = tmp_path_factory.mktemp("data")
    (out / "d3il").mkdir()
    offsets = np.cumsum([0, *map(len, paths)])
    np.savez(out / "d3il" / "avoiding.npz", xy=np.concatenate(paths), offsets=offsets)
    return out


def _cfg(root, extra=()):
    tiny = [
        "+experiment=d3il", f"dataset.root={root}", "trainer.iters=4", "trainer.warmup_repr=1",
        "trainer.warmup_pred=2", "trainer.batch=4", "trainer.k_train=4", "trainer.finetune_iters=2",
        "trainer.finetune_windows=2", "readout.iters=2", "readout.iters_low=1", "readout.batch=8",
        "eval.rollouts=4", "eval.max_calls=2", "eval.subset=3", "eval.n_obs=4", "wandb.mode=disabled",
    ]  # fmt: skip
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=tiny + list(extra))


def test_bspline_basis_is_clamped_partition_of_unity():
    b = bspline_basis()
    assert b.shape == (24, 8) and (b >= 0).all()
    np.testing.assert_allclose(b.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(b[-1], np.eye(8)[-1])  # ends on the last control point
    out = SplineReadout(64)(torch.zeros(5, 64), torch.zeros(5, 64))
    assert out.shape == (5, 24, 2)


def test_expected_route_stats_match_enumeration():
    counts = np.zeros(ROUTES, dtype=int)
    counts[:3] = [3, 1, 2]
    paths = np.repeat(np.arange(ROUTES), counts)
    covs, tvs = [], []
    for sub in itertools.combinations(range(len(paths)), 4):
        c = np.bincount(paths[list(sub)], minlength=ROUTES)
        covs.append((c > 0).sum())
        tvs.append(0.5 * np.abs(c / 4 - 1 / ROUTES).sum())
    cov, tv = expected_route_stats(counts, 4)
    assert cov == pytest.approx(np.mean(covs)) and tv == pytest.approx(np.mean(tvs))
    assert expected_route_stats(np.zeros(ROUTES, dtype=int), 60)[0] == 0.0


def test_acceptance_rules():
    good = d3il.to_goal(synthetic_path(13))
    assert accept(good) == 13 == d3il.route_label(good)
    assert accept(good[: len(good) // 2]) is None  # never reaches the goal
    through = good.copy()
    through[:, 0] = np.clip(through[:, 0], 0.5, 0.5)  # straight up x = 0.5: hits the first row's obstacle
    assert accept(through) is None
    far = good.copy()
    far[5] = [1.2, 0.0]  # outside the reach envelope
    assert accept(far) is None


def test_synthetic_demos_pass_the_loader(root):
    ds = d3il.D3ILAvoiding("train", 0, str(root))
    assert len(ds) > 0 and ds[0]["obs_pts"].shape == (24, 2)


@pytest.mark.parametrize(
    "extra",
    [[], ["prior=entangled_qcbm", "quantum.shots=8", "quantum.shift_obs=2", "quantum.shift_coords=3"],
     ["encoder=hqvit", "encoder.depth=1", "quantum.hqvit_train=torch"]],
    ids=["cnn", "qcbm", "hqvit"],
)  # fmt: skip
def test_d3il_smoke_and_resume(root, tmp_path, extra):
    cfg = _cfg(root, extra)
    m = run(cfg, tmp_path)
    assert {"routes", "coverage", "tv", "acceptance", "n_accepted", "readout_ade_m"} <= set(m)
    assert 0 <= m["acceptance"] <= 1 and np.isfinite(m["readout_ade_m"])
    assert (tmp_path / "finetune.pt").exists()
    again = run(cfg, tmp_path)  # finished run: reloads ckpt, readout and stage-3 weights, re-evaluates
    assert json.dumps(again, sort_keys=True) == json.dumps(m, sort_keys=True)


def test_d3il_preset_is_paper_app_b(root):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        full = compose("config", overrides=["+experiment=d3il"])
    t, r, q, p = full.trainer, full.readout, full.quantum, full.prior
    assert (t.iters, t.batch, t.k_train, t.finetune_iters, t.lr_finetune) == (12000, 64, 32, 1000, 1e-4)
    assert (r.iters, r.iters_low, r.lr, r.lr_low, r.weight_decay) == (8000, 2000, 2e-3, 2e-4, 1e-5)
    assert (p.n_bits, p.layers, q.shots, q.shift_obs, q.shift_coords) == (8, 4, 512, 8, 12)
    assert (full.predictor.kind, full.predictor.width, full.eval.rollouts, full.eval.subset) == ("scaled", 128, 384, 60)
