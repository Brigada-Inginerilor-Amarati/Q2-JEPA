"""The TaxiBJ clip script shows the draws the metrics scored."""

import json

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("matplotlib")

from q2jepa.metrics import flow_metrics
from q2jepa.train import run
from scripts import show_taxibj
from tests.test_futures_benchmarks import TINY
from tests.test_taxibj import CONF, _fake_root


def test_show_taxibj_shows_the_scored_draws(tmp_path):
    root, _ = _fake_root(tmp_path / "data")
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose("config", overrides=[*TINY, "+experiment=taxibj", f"dataset.root={root}"])
    rd = tmp_path / "run"
    (rd / ".hydra").mkdir(parents=True)
    (rd / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    run(cfg, rd)
    d = show_taxibj.sample(rd)
    assert d["draws"].shape == (2, 8, 3, 2, 32, 32)
    per_obs = [flow_metrics(d["draws"][i].reshape(8, -1).astype(float), d["fut"][i].reshape(-1)) for i in range(2)]
    saved = json.loads((rd / "metrics_dev.json").read_text())
    for key in per_obs[0]:
        assert np.mean([m[key] for m in per_obs]) == pytest.approx(saved[key], rel=1e-4)
    recorded, copy, _, mean = show_taxibj.forecasts(d, 0)
    np.testing.assert_array_equal(copy[2], d["obs"][0][-1])
    assert recorded.shape == mean.shape == (3, 2, 32, 32)
    for step in (0, 5):
        assert show_taxibj.to_image(show_taxibj.figure(d, 2, step)).size[0] > 0
