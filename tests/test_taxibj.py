"""TaxiBJ: splits, rendering, single-future metrics and an end-to-end smoke run on fake maps."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.data import taxibj
from q2jepa.metrics import flow_metrics
from q2jepa.train import run
from tests.test_futures_benchmarks import QCBM, TINY

CONF = str(Path(__file__).resolve().parents[1] / "conf")


def _fake_root(tmp_path, n_train=96, n_test=6):
    rng = np.random.default_rng(0)
    arrays = {k: rng.uniform(-1, 1, (n, 4, 2, 32, 32)) for k, n in
              (("X_train", n_train), ("Y_train", n_train), ("X_test", n_test), ("Y_test", n_test))}  # fmt: skip
    (tmp_path / "taxibj").mkdir(parents=True)
    np.savez(tmp_path / "taxibj" / "dataset.npz", **arrays)
    return tmp_path, arrays


def test_splits_keep_the_last_observed_and_first_future_maps(tmp_path):
    root, a = _fake_root(tmp_path)
    train, dev, test = (taxibj.TaxiBJ(s, str(root)) for s in ("train", "dev", "test"))
    assert (len(train), len(dev), len(test)) == (96 - taxibj.N_DEV, taxibj.N_DEV, 6)
    assert train.obs.shape == (16, 3, 2, 32, 32) and train.fut.shape == (16, 3, 2, 32, 32)
    np.testing.assert_allclose(dev.obs[0].float().numpy(), (a["X_train"][16, 1:4] + 1) / 2, atol=1e-3)
    np.testing.assert_allclose(test.fut[5].float().numpy(), (a["Y_test"][5, :3] + 1) / 2, atol=1e-3)
    with pytest.raises(FileNotFoundError, match="fetch_taxibj"):
        taxibj.TaxiBJ("dev", str(tmp_path / "missing"))


def test_render_puts_inflow_left_and_outflow_right():
    maps = torch.zeros(2, 32, 32)
    maps[0, 3, 5], maps[1, 7, 9] = 1.0, 0.5
    frame = taxibj.render(maps)
    assert frame.shape == (64, 64)
    assert frame[6, 5] == 1.0 and frame[7, 5] == 1.0  # rows doubled
    assert frame[14, 32 + 9] == 0.5 and frame.sum() == pytest.approx(2 * 1.5)


def test_flow_metrics():
    truth = np.zeros(6144)
    m = flow_metrics(np.stack([truth, truth + 0.1]), truth)
    assert m["mse_min"] == 0.0 and m["mse_mean"] == pytest.approx(0.005) and m["mae_min"] == 0.0
    assert m["es"] == pytest.approx(0.05 - 0.5 * 0.1)  # mean RMS 0.05, pair RMS 0.1
    same = flow_metrics(np.repeat(truth[None] + 0.2, 4, 0), truth)
    assert same["es"] == pytest.approx(0.2) and same["mse_min"] == pytest.approx(0.04)


@pytest.mark.parametrize("extra", [[], QCBM])
def test_taxibj_smoke(tmp_path, extra):
    root, _ = _fake_root(tmp_path / "data")
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose("config", overrides=[*TINY, "+experiment=taxibj", f"dataset.root={root}", *extra])
    m = run(cfg, tmp_path / "run")
    assert set(m) == {"mse_min", "mse_mean", "mae_min", "es", "readout_mse"}
    assert all(np.isfinite(list(m.values())))
    assert json.loads((tmp_path / "run" / "metrics_dev.json").read_text()) == m
