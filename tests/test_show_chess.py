"""The chess clip script shows the draws the metrics scored, and counts legal ones."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("matplotlib")
pytest.importorskip("chess")

from q2jepa.benchmarks.chess import ChessBenchmark
from q2jepa.data import chess as data
from q2jepa.metrics import futures_metrics
from q2jepa.train import run
from scripts import show_chess
from tests.test_futures_benchmarks import _cfg


def _fake_root(tmp_path, monkeypatch):
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    n = data.N
    (tmp_path / "chess").mkdir()
    np.savez(
        tmp_path / "chess" / "positions.npz", fen=np.array([start] * n), moves=np.array([["e2e4", "d2d4", "g1f3"]] * n)
    )
    monkeypatch.setattr(data, "SHA256", "PENDING")
    return tmp_path


def test_show_chess_shows_the_scored_draws(tmp_path, monkeypatch):
    root = _fake_root(tmp_path, monkeypatch)
    cfg = _cfg(["+experiment=chess", f"dataset.root={root}", "eval.n_obs=4"])
    rd = tmp_path / "run"
    (rd / ".hydra").mkdir(parents=True)
    (rd / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    run(cfg, rd)
    d = show_chess.sample(rd)
    assert d["draws"].shape == (4, cfg.eval.k_eval, 4)
    per_obs = [futures_metrics(d["draws"][i], d["state"][i], 1.0, None, ChessBenchmark.max_radius) for i in range(4)]
    saved = json.loads((rd / "metrics_dev.json").read_text())
    for key in per_obs[0]:
        assert np.mean([m[key] for m in per_obs]) == pytest.approx(saved[key])
    for step in range(3):
        assert show_chess.to_image(show_chess.figure(d, [0, 2], step)).size[0] > 0


def test_legal_share_and_spacing():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    draws = np.array([[[4, 1, 4, 3], [4, 1, 4, 4], [0, 0, 0, 0], [6, 0, 5, 2]]], float)  # e2e4, e2e5, a1a1, g1f3
    assert show_chess.uci(draws[0, 0]) == "e2e4"
    assert show_chess.legal_share({"fens": [start], "draws": draws}) == 0.5
    assert show_chess.spaced(128, 3) == [21, 64, 106]
