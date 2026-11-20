"""HEP18 training pieces: readout integration, metrics, scene targets, and end-to-end smoke runs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.benchmarks.hep18 import _encode_cars, _Split, scene_targets
from q2jepa.data.hep18 import render_agent_clip
from q2jepa.encoder import prepare
from q2jepa.encoder.cnn import CNNEncoder
from q2jepa.metrics import hep18_metrics
from q2jepa.readout.hep18 import integrate
from q2jepa.train import _identity, run

CONF = str(Path(__file__).resolve().parents[1] / "conf")
TINY = [
    "+experiment=hep18", "trainer.iters=4", "trainer.warmup_repr=1", "trainer.warmup_pred=2", "trainer.batch=4",
    "trainer.k_train=4", "readout.iters=3", "readout.batch=8", "eval.n_obs=2", "eval.k_eval=8", "eval.split=dev",
    "wandb.mode=disabled",
]  # fmt: skip
QCBM = ["prior=entangled_qcbm", "quantum.shots=8", "quantum.shift_obs=2", "quantum.shift_coords=3"]


def _cfg(extra=()):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=TINY + list(extra))


def test_trapezoidal_integration():
    s0, v0 = torch.tensor([10.0]), torch.tensor([8.0])
    torch.testing.assert_close(integrate(s0, v0, torch.full((1, 3), 8.0)), torch.tensor([[34.0, 58.0, 82.0]]))
    # v: 8 -> 6 -> 4 -> 2: s gains 1.5 (8 + 6), 1.5 (6 + 4), 1.5 (4 + 2)
    torch.testing.assert_close(integrate(s0, v0, torch.tensor([[6.0, 4.0, 2.0]])), torch.tensor([[31.0, 46.0, 55.0]]))


def test_metrics_on_the_true_futures():
    data = _Split("dev", "data", torch.device("cpu"))
    s_true, v_true = data.fut[0, ..., 0].double().numpy(), data.fut[0, ..., 1].double().numpy()
    s, v = np.repeat(s_true, 4, 0), np.repeat(v_true, 4, 0)  # every future, equally often
    m = hep18_metrics(s, v, s_true, v_true, np.random.default_rng(0))
    assert m["coverage"] == 1.0 and m["validity"] == 1.0 and m["local_validity"] == 1.0
    assert m["tv"] == pytest.approx(0.0, abs=1e-12) and m["min_ade"] == 0.0 and m["min_fde"] == 0.0
    assert m["shuffled_validity"] < 0.5  # re-pairing cars breaks the joint futures
    off = hep18_metrics(s + 2.5, v, s_true, v_true, np.random.default_rng(0))  # 2.5 m off: beyond the 2 m tolerance
    assert off["validity"] == 0.0 and off["tv"] == 1.0 and off["min_ade"] == pytest.approx(2.5)


def test_scene_targets_equal_encoding_every_future():
    torch.manual_seed(0)
    enc = CNNEncoder(64, 16).eval()
    data = _Split("dev", "data", torch.device("cpu"))
    cfg = _cfg()
    e = cfg.encoder
    with torch.no_grad():
        fast = scene_targets(_encode_cars(enc, data.uniq[:2], cfg))  # 16 clips per observation
        clips = render_agent_clip(data.fut[:2, ..., 0]).reshape(-1, 3, 64, 64)  # all 72
        brute = enc(prepare(clips, 64, e.input_norm, e.input_scale)).reshape(2, 18, 64)
    torch.testing.assert_close(fast, brute)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["prior=mixture"],
        QCBM,
        ["encoder=vit", "encoder.depth=1"],
        ["encoder=hqvit", "encoder.depth=1", "quantum.hqvit_train=torch"],
    ],
    ids=["cnn", "mixture", "qcbm", "vit", "hqvit"],
)
def test_hep18_smoke(tmp_path, extra):
    m = run(_cfg(extra), tmp_path)
    assert {"coverage", "validity", "tv", "min_ade", "es", "shuffled_validity", "readout_validity"} <= set(m)
    assert all(np.isfinite(list(m.values())))
    assert json.loads((tmp_path / "metrics_dev.json").read_text()) == m


def test_hep18_resume_is_exact(tmp_path):
    cfg = _cfg([*QCBM, "trainer.ckpt_every=2"])
    ref = run(cfg, tmp_path / "ref")
    assert run(_cfg([*QCBM, "trainer.ckpt_every=2", "trainer.stop_after=2"]), tmp_path / "split") == {}
    assert run(cfg, tmp_path / "split") == ref


def test_hep18_preset_is_paper_table5():
    cfg = _cfg()
    assert (cfg.trainer.iters, cfg.trainer.batch, cfg.readout.iters, cfg.readout.batch) == (4, 4, 3, 8)  # TINY
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        full = compose("config", overrides=["+experiment=hep18"])
    t, q, p = full.trainer, full.quantum, full.prior
    assert (t.iters, t.batch, t.k_train, full.readout.iters, full.readout.batch) == (6000, 32, 32, 16000, 128)
    assert (p.n_bits, p.layers, p.layout, q.shots, q.shift_obs, q.shift_coords) == (12, 4, "hep18_grid", 512, 8, 12)
    assert (t.energy_beta, t.energy_eps, full.predictor.width, full.encoder.d * full.dataset.agents) == (
        0.5, 1e-4, 64, 64,
    )  # fmt: skip
    assert _identity(full) != _identity(_cfg())
