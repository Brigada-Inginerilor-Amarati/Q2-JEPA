"""Charged particles and chess: data, rendering, metrics and end-to-end smoke runs (fast-iteration benchmarks)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401
from q2jepa.benchmarks.nri import BALL_PERMS
from q2jepa.data import chess, nri
from q2jepa.metrics import futures_metrics, match_futures
from q2jepa.readout.futures import decode_squares, square_index, square_loss
from q2jepa.train import run

CONF = str(Path(__file__).resolve().parents[1] / "conf")
TINY = [
    "trainer.iters=4", "trainer.warmup_repr=1", "trainer.warmup_pred=2", "trainer.batch=4", "trainer.k_train=4",
    "readout.iters=3", "readout.batch=8", "eval.n_obs=2", "eval.k_eval=8", "eval.split=dev", "wandb.mode=disabled",
]  # fmt: skip
QCBM = ["prior=entangled_qcbm", "quantum.shots=8", "quantum.shift_obs=2", "quantum.shift_coords=3"]


def _cfg(extra):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=TINY + list(extra))


def test_futures_metrics():
    truth = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])  # nearest-other gap 10 (RMS 7.07): radius 3.54
    m = futures_metrics(np.repeat(truth, 4, 0), truth, 1.0)
    assert (m["coverage"], m["validity"], m["min_ade"]) == (1.0, 1.0, 0.0) and m["tv"] == pytest.approx(0.0)
    one = futures_metrics(np.repeat(truth[:1] + 3.0, 6, 0), truth, 1.0)  # RMS 3.0 from future 0 only
    assert one["validity"] == 1.0 and one["coverage"] == pytest.approx(1 / 3) and one["tv"] == pytest.approx(2 / 3)
    off = futures_metrics(np.repeat(truth[:1] + 4.0, 6, 0), truth, 1.0)  # nearest is still 0, but beyond the radius
    assert (off["validity"], off["coverage"], off["tv"]) == (0.0, 0.0, 1.0)
    assert futures_metrics(np.repeat(truth[:1] + 4.0, 6, 0), truth, 2.0)["validity"] == 0.0  # uncapped: scale-free
    assert futures_metrics(np.repeat(truth[:1] + 3.0, 6, 0), truth, 1.0, max_radius=2.0)["validity"] == 0.0  # capped


def test_particle_balls_match_as_a_set():
    _, fut, _ = nri.generate(4, seed=3)
    truth = fut[0].reshape(4, 18)
    swapped = fut[0][:, :, [1, 0, 2]].reshape(4, 18)  # balls 0 and 1 relabelled
    perms = np.array(BALL_PERMS)
    assert futures_metrics(swapped, truth, 1.0, perms)["validity"] == 1.0
    assert futures_metrics(swapped, truth, 1.0, perms)["min_ade"] == pytest.approx(0.0)
    assert futures_metrics(swapped, truth, 1.0)["min_ade"] > 1.0  # without perms the order matters
    valid, nearest, _ = match_futures(swapped, truth, 1.0, perms)
    assert valid.all() and (nearest == np.arange(4)).all()


def test_particles_have_four_separated_futures():
    obs, fut, _ = nri.generate(32, seed=1)
    assert obs.shape == (32, 3, 3, 2) and fut.shape == (32, 4, 3, 3, 2)
    assert min(nri.separation(f) for f in fut) >= nri.MIN_SEP
    np.testing.assert_allclose(obs[:, 1] - obs[:, 0], obs[:, 2] - obs[:, 1])  # free flight before t = 0
    assert fut.min() >= nri.WALL and fut.max() <= 63 - nri.WALL


def test_charge_sign_flip_gives_the_same_motion():
    p0, v0 = np.array([[25.0, 30.0], [35.0, 30.0], [30.0, 38.0]]), np.zeros((3, 2))
    q = np.array([1.0, -1.0, 1.0])
    a, _ = nri.step(p0, v0, q, 40)
    b, _ = nri.step(p0, v0, -q, 40)
    np.testing.assert_allclose(a, b)
    c, _ = nri.step(p0, v0, np.array([1.0, 1.0, 1.0]), 40)
    assert np.abs(a - c).max() > 1.0  # but a different charge pattern moves differently


def test_particle_render_peaks_at_the_balls():
    pos = torch.tensor([[10.0, 20.0], [40.0, 12.0], [50.0, 50.0]])
    frame = nri.render(pos)
    assert frame.shape == (64, 64)
    for x, y in pos.long().tolist():
        assert frame[y, x] == pytest.approx(1.0, abs=1e-6)


def test_input_scale():
    """conf/experiment/nri.yaml input_scale = mean rendered train-frame L2 norm / 64."""
    obs, _, _ = nri.generate(800, seed=0)
    norm = nri.render(torch.tensor(obs).reshape(-1, 3, 2)).flatten(1).norm(dim=1).mean().item()
    assert norm / 64 == pytest.approx(0.0897, abs=1e-3)


def test_chess_moves_and_render():
    board = chess.parse_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
    after = chess.apply_move(board, "e5d6")  # en passant removes the d5 pawn
    assert after[2][3] == "P" and after[3][3] == "" and after[3][4] == ""
    castled = chess.apply_move(board, "e1g1")
    assert castled[7][6] == "K" and castled[7][5] == "R" and castled[7][7] == ""
    assert chess.apply_move(board, "e1h1") == castled and chess.move_state("e1h1") == chess.move_state("e1g1")
    promo = chess.apply_move(chess.parse_fen("8/4P3/8/8/8/8/8/k6K w - - 0 1"), "e7e8q")
    assert promo[0][4] == "Q"
    assert chess.move_state("e2e4") == [4, 1, 4, 3]
    clip = chess.render(board)
    assert clip.shape == (3, 64, 64)
    assert (
        clip[0, 7 * 8 + 3, 4 * 8 + 3] == chess.INTENSITY["k"] and clip[1, 0 * 8 + 3, 0 * 8 + 3] == chess.INTENSITY["r"]
    )
    assert clip[0].sum() > 0 and clip[1].sum() > 0 and np.allclose(clip[2], np.maximum(clip[0], clip[1]))
    start = chess.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    for b in (board, castled, promo, start):  # the GPU renderer equals the reference one
        assert torch.equal(chess.render_codes(torch.tensor(chess.codes(b))), torch.tensor(chess.render(b)))


def test_square_readout_round_trips_moves():
    state = torch.tensor([chess.move_state(m) for m in ("e2e4", "g1f3", "e1h1", "a7a8q")])
    idx = square_index(state)
    assert idx[0].tolist() == [1 * 8 + 4, 3 * 8 + 4]
    logits = torch.full((4, 128), -10.0)
    logits[torch.arange(4), idx[:, 0]] = 10.0
    logits[torch.arange(4), 64 + idx[:, 1]] = 10.0
    assert torch.equal(decode_squares(logits), state)
    assert square_loss(logits, state) < 1e-6 < square_loss(-logits, state)


def _fake_chess_root(tmp_path):
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * chess.N
    moves = [["e2e4", "d2d4", "g1f3"]] * chess.N
    (tmp_path / "chess").mkdir(parents=True)
    np.savez(tmp_path / "chess" / "positions.npz", fen=np.array(fens), moves=np.array(moves))
    return tmp_path


@pytest.mark.parametrize("extra", [[], QCBM])
def test_particles_smoke(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(nri.ChargedParticles, "_SPLITS", {"train": (16, 0), "dev": (4, 1), "test": (4, 2)})
    m = run(_cfg(["+experiment=nri", *extra]), tmp_path)
    assert {"coverage", "validity", "tv", "min_ade", "es", "readout_validity"} <= set(m)
    assert all(np.isfinite(list(m.values())))
    assert json.loads((tmp_path / "metrics_dev.json").read_text()) == m


def test_chess_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(chess, "SHA256", "PENDING")  # fake positions, not the pinned fetch
    root = _fake_chess_root(tmp_path / "data")
    m = run(_cfg(["+experiment=chess", f"dataset.root={root}"]), tmp_path / "run")
    assert {"coverage", "validity", "tv", "readout_validity"} <= set(m) and all(np.isfinite(list(m.values())))


def test_chess_needs_the_fetched_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_chess"):
        chess.ChessPositions("dev", str(tmp_path))
