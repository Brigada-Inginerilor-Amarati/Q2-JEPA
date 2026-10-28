import json
from pathlib import Path

from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.train import run, stage

CONF = str(Path(__file__).resolve().parents[1] / "conf")
TINY = [
    "trainer.iters=6", "trainer.warmup_repr=2", "trainer.warmup_pred=4", "trainer.batch=8", "trainer.k_train=4",
    "readout.iters=2", "readout.batch=8", "eval.n_obs=2", "eval.k_eval=8",
    "dataset.n_train=2", "dataset.n_test=2", "wandb.mode=disabled",
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


def test_interrupted_run_resumes_to_same_metrics(tmp_path):
    full = _cfg(["trainer.ckpt_every=2"])
    part = _cfg(["trainer.ckpt_every=2", "trainer.stop_after=3"])
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
