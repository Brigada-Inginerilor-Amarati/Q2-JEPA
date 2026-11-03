import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.train import run, stage

CONF = str(Path(__file__).resolve().parents[1] / "conf")
TINY = [
    "trainer.iters=6", "trainer.warmup_repr=2", "trainer.warmup_pred=4", "trainer.batch=8", "trainer.k_train=4",
    "readout.iters=2", "readout.batch=8", "eval.n_obs=2", "eval.k_eval=8",
    "dataset.n_train=2", "dataset.n_test=2", "dataset.n_dev=2", "wandb.mode=disabled",
]  # fmt: skip


def _cfg(extra=()):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=TINY + list(extra))


def test_stage_boundaries():
    assert [stage(i, 200, 300) for i in (1, 200, 201, 300, 301)] == ["repr", "repr", "pred", "pred", "full"]


def test_smoke_and_reproducible(tmp_path):
    cfg = _cfg()
    m1 = run(cfg, tmp_path / "a")
    run(cfg, tmp_path / "b")
    assert set(m1) >= {"coverage", "validity", "tv", "min_ade", "es"}
    assert (tmp_path / "a" / "metrics.json").read_text() == (tmp_path / "b" / "metrics.json").read_text()
    assert json.loads((tmp_path / "a" / "metrics.json").read_text()) == m1


@pytest.mark.parametrize(
    "extra",
    [
        ["prior=entangled_qcbm"],
        ["prior=product_qcbm"],
        ["prior=frozen_qcbm"],
        ["prior=entangled_qcbm", "quantum.grad=ideal", "quantum.readout_flip=0"],
    ],
    ids=["entangled", "product", "frozen", "ideal"],
)
def test_qcbm_smoke(tmp_path, extra):
    m = run(_cfg([*extra, "quantum.shots=8", "quantum.shift_obs=2", "quantum.shift_coords=3"]), tmp_path)
    assert set(m) >= {"coverage", "validity", "tv", "min_ade", "es"}


def test_hqvit_smoke_and_inherits_twin_config(tmp_path):
    cfg = _cfg(["encoder=hqvit", "encoder.patch=4", "encoder.depth=1"])
    twin = _cfg(["encoder=vit"]).encoder
    assert {k: v for k, v in cfg.encoder.items() if k not in ("name", "patch", "depth")} == {
        k: v for k, v in twin.items() if k not in ("name", "patch", "depth")
    }
    assert set(run(cfg, tmp_path)) >= {"coverage", "validity", "tv", "min_ade", "es"}


def test_ideal_gradient_rejects_readout_noise(tmp_path):
    with pytest.raises(ValueError, match="readout_flip=0"):
        run(_cfg(["prior=entangled_qcbm", "quantum.grad=ideal"]), tmp_path)


def test_new_default_keys_keep_checkpoint_identity():
    from q2jepa.train import _identity

    base = _identity(_cfg())
    assert _identity(_cfg(["encoder.input_scale=1.0", "quantum.grad=shift"])) == base
    assert _identity(_cfg(["encoder.input_scale=0.5"])) != base


def test_qcbm_resume_is_exact(tmp_path):
    q = ["prior=entangled_qcbm", "quantum.shots=8", "quantum.shift_obs=2", "trainer.ckpt_every=2"]
    ref = run(_cfg(q), tmp_path / "ref")
    assert run(_cfg([*q, "trainer.stop_after=5"]), tmp_path / "res") == {}
    assert run(_cfg(q), tmp_path / "res") == ref


def test_frozen_qcbm_is_noise_free_and_never_updates(tmp_path):
    import torch

    cfg = _cfg(["prior=frozen_qcbm"])
    assert cfg.quantum.readout_flip == 0.0
    run(cfg, tmp_path)
    ck = torch.load(tmp_path / "ckpt.pt", weights_only=True)["prior"]
    assert torch.equal(ck["net.2.weight"], torch.zeros_like(ck["net.2.weight"]))  # zero init never moved
    assert torch.equal(ck["terminal"], torch.zeros(3))


def test_dev_split_eval_reuses_checkpoint(tmp_path):
    test = run(_cfg(), tmp_path)
    dev = run(_cfg(["eval.split=dev"]), tmp_path)  # finished run: only re-evaluates
    assert (tmp_path / "metrics_dev.json").exists() and dev != test
    assert json.loads((tmp_path / "metrics.json").read_text()) == test
    assert "split: dev" in (tmp_path / "eval_config_dev.yaml").read_text()


def test_resume_rejects_different_model(tmp_path):
    run(_cfg(), tmp_path)
    with pytest.raises(ValueError, match="different model"):
        run(_cfg(["trainer.iters=8"]), tmp_path)


def test_tune_presets_select_on_dev():
    for s, lrs in {"s1": (3e-4, 1e-5), "s2": (3e-4, 3e-4), "s3": (1e-3, 1e-4)}.items():
        c = _cfg(["+experiment=phase2", f"+tune={s}", "encoder=vit"])
        assert (c.trainer.lr_enc, c.trainer.lr_enc_later, c.eval.split, c.encoder.res) == (*lrs, "dev", 64)


@pytest.mark.parametrize("enc", [[], ["encoder=vit"]], ids=["cnn", "vit"])
def test_interrupted_run_resumes_to_same_metrics(tmp_path, enc):
    full = _cfg(["trainer.ckpt_every=2", *enc])
    part = _cfg(["trainer.ckpt_every=2", "trainer.stop_after=3", *enc])
    ref = run(full, tmp_path / "ref")
    assert run(part, tmp_path / "res") == {} and (tmp_path / "res" / "ckpt.pt").exists()
    assert not (tmp_path / "res" / "metrics.json").exists()
    assert run(full, tmp_path / "res") == ref


def test_resume_during_readout_matches_uninterrupted(tmp_path):
    cfg = _cfg(["trainer.ckpt_every=2"])
    ref = run(cfg, tmp_path / "r")
    (tmp_path / "r" / "readout.pt").unlink()  # session died after world model, before readout finished
    (tmp_path / "r" / "metrics.json").unlink()
    assert run(cfg, tmp_path / "r") == ref
