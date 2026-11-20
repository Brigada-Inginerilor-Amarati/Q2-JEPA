"""Plot scripts show the draws the metrics scored, and the results summary aggregates correctly."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("matplotlib")

from q2jepa.benchmarks.d3il import accept
from q2jepa.metrics import hep18_metrics
from q2jepa.train import run
from scripts import plot_results, show_d3il, show_hep18
from tests import test_d3il_train, test_hep18_train


def _finished(cfg, path):
    (path / ".hydra").mkdir(parents=True)
    (path / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    return run(cfg, path)


def test_show_hep18_shows_the_scored_samples(tmp_path):
    cfg = test_hep18_train._cfg()
    saved = _finished(cfg, tmp_path)
    d = show_hep18.sample(tmp_path)
    rng = np.random.default_rng([cfg.seed, 7])  # the evaluation's shuffle stream
    per_obs = [hep18_metrics(d["s"][i], d["v"][i], d["fut"][i, ..., 0], d["fut"][i, ..., 1], rng)
               for i in range(len(d["s"]))]  # fmt: skip
    for key in per_obs[0]:
        assert np.mean([m[key] for m in per_obs]) == pytest.approx(saved[key])
    show_hep18.figure(d, [0, 1]).savefig(tmp_path / "hep18.png")
    assert (tmp_path / "hep18.png").stat().st_size > 0


def test_show_d3il_shows_the_scored_rollouts(tmp_path):
    root = tmp_path / "data"
    counts = [3, 5] + [4] * 22
    paths = [test_d3il_train.synthetic_path(r, 0.003 * (k - 1.5)) for r, c in enumerate(counts) for k in range(c)]
    (root / "d3il").mkdir(parents=True)
    np.savez(root / "d3il" / "avoiding.npz", xy=np.concatenate(paths), offsets=np.cumsum([0, *map(len, paths)]))
    cfg = test_d3il_train._cfg(root)
    saved = _finished(cfg, tmp_path / "run")
    d = show_d3il.rollouts(tmp_path / "run")
    assert d["routes"] == [accept(p) for p in d["paths"]]
    assert np.mean([r is not None for r in d["routes"]]) == pytest.approx(saved["acceptance"])
    show_d3il.figure(d).savefig(tmp_path / "d3il.png")
    assert (tmp_path / "d3il.png").stat().st_size > 0


def test_plot_results_aggregates_seeds(tmp_path, capsys):
    for enc, prior, seed, cov in [("cnn", "softmax_categorical", 0, 0.9), ("cnn", "softmax_categorical", 1, 0.8),
                                  ("vit", "entangled_qcbm", 0, 0.5)]:  # fmt: skip
        d = tmp_path / f"{enc}-{prior}-s{seed}"
        d.mkdir()
        cfg = {"dataset": {"name": "mmnist"}, "encoder": {"name": enc}, "prior": {"name": prior}, "seed": seed,
               "trainer": {"lr_enc": 3e-4, "lr_enc_later": 1e-5}, "quantum": {"hqvit_train": "torch", "target": "nvidia"}}  # fmt: skip
        (d / "eval_config_test.yaml").write_text(OmegaConf.to_yaml(cfg))
        (d / "metrics.json").write_text(json.dumps({"coverage": cov, "validity": 1.0, "tv": 0.1, "min_ade": 1.0,
                                                    "es": 9.0}))  # fmt: skip
    runs = plot_results.collect([tmp_path], "test")
    groups = plot_results.group(runs)
    assert list(groups) == [
        ("cnn", "softmax_categorical", "s1", "", "nvidia"),
        ("vit", "entangled_qcbm", "s1", "", "nvidia"),
    ]
    assert plot_results.stats(groups[("cnn", "softmax_categorical", "s1", "", "nvidia")], "coverage") == pytest.approx(
        (0.85, np.std([0.9, 0.8], ddof=1))
    )
    assert "0.850 ± 0.071" in plot_results.table("mmnist", groups)
    plot_results.figure("mmnist", groups).savefig(tmp_path / "r.png")
    cfg["encoder"]["name"] = "state"  # diagnostic encoder: never a results row
    (tmp_path / "state-run").mkdir()
    (tmp_path / "state-run" / "eval_config_test.yaml").write_text(OmegaConf.to_yaml(cfg))
    (tmp_path / "state-run" / "metrics.json").write_text("{}")
    with pytest.raises(SystemExit, match="encoder=state"):
        plot_results.collect([tmp_path], "test")
