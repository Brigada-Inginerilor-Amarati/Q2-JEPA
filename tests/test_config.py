from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

import q2jepa.config  # noqa: F401  registers schema

CONF = str(Path(__file__).resolve().parents[1] / "conf")


def _cfg(overrides=()):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=list(overrides))


def test_table5_defaults():
    c = _cfg()
    assert (c.encoder.d, c.predictor.width, c.prior.hidden, c.prior.n_bits, c.prior.layers) == (64, 96, 48, 3, 2)
    t = c.trainer
    assert (t.iters, t.batch, t.k_train, t.warmup_repr, t.warmup_pred) == (4000, 64, 32, 200, 300)
    assert (t.lr_enc, t.lr_enc_later, t.lr_pred, t.lr_prior, t.weight_decay, t.clip) == (
        3e-4,
        1e-5,
        1e-3,
        1e-3,
        1e-5,
        5.0,
    )
    assert t.lambda_reg == pytest.approx(4 / 3)
    assert (c.visreg.n_proj, c.visreg.w_center, c.visreg.w_scale, c.visreg.shape_ramp) == (128, 1.0, 10.0, 100)
    assert (c.quantum.shots, c.quantum.readout_flip, c.quantum.shift_obs, c.quantum.shift_coords) == (256, 0.005, 8, 9)
    assert (c.readout.iters, c.readout.batch, c.readout.lr, c.readout.weight_decay) == (4000, 64, 2e-3, 0.01)
    assert (c.eval.n_obs, c.eval.k_eval) == (80, 128)


def test_group_switches():
    assert _cfg(["quantum=gpu"]).quantum.target == "nvidia"
    assert _cfg(["prior=mixture"]).prior.name == "mixture"


def test_typo_rejected():
    with pytest.raises(ConfigCompositionException):
        _cfg(["trainer.itters=5"])
