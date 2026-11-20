import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import q2jepa.config  # noqa: F401
from q2jepa.metrics import mmnist_metrics
from q2jepa.train import run
from scripts.show_futures import grid, sample_futures
from tests.test_train import TINY

CONF = str(Path(__file__).resolve().parents[1] / "conf")


def test_show_futures_shows_the_scored_samples(tmp_path):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose("config", overrides=[*TINY, "eval.split=dev"])
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    run(cfg, tmp_path)

    obs, sampled, truth, disp = sample_futures(tmp_path)  # defaults: the run's eval settings
    n, k = cfg.eval.n_obs, cfg.eval.k_eval
    assert obs.shape == (n, 3, 64, 64) and sampled.shape == (n, k, 3, 64, 64) and truth.shape == (n, 8, 3, 64, 64)

    # the same draws the evaluation scored: recomputing the metrics from them gives metrics_dev.json
    per_obs = [mmnist_metrics(d) for d in disp.double().numpy()]
    saved = json.loads((tmp_path / "metrics_dev.json").read_text())
    for key, value in saved.items():
        assert np.mean([m[key] for m in per_obs]) == pytest.approx(value)

    img = grid(obs, sampled, truth, frame=2)
    assert img.size == ((3 + max(k, 8)) * 66 + 2, 2 * n * 66 + 2)  # 2 rows per observation, 64 px + 2 px padding
