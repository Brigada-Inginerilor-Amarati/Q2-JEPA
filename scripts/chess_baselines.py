"""Reference rows for the chess results table, scored with the benchmark's own metric (metrics.futures_metrics,
scale 1, ChessBenchmark.max_radius), K samples per position as eval.k_eval. No model, no training.

pip install chess && python scripts/chess_baselines.py [--root data] [--split test] [--k 128] [--seed 0]

Rows: the 3 true moves drawn uniformly (ceiling), a uniformly random legal move, uniformly random from/to squares
(floor). Promotions to different pieces share squares, so they count as one legal move.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.benchmarks.chess import ChessBenchmark
from q2jepa.data.chess import ChessPositions, move_state
from q2jepa.metrics import futures_metrics


def legal_squares(fen: str) -> list[str]:
    """Distinct from/to squares (UCI without promotion) of the legal moves of a position."""
    import chess

    return sorted({m.uci()[:4] for m in chess.Board(fen).legal_moves})


def baselines(fens, state: np.ndarray, k: int, seed: int) -> dict[str, dict[str, float]]:
    """Mean futures_metrics per baseline over positions. state: (n, 3, 4) true moves as from/to squares."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict[str, float]]] = {
        "true moves (ceiling)": [],
        "random legal move": [],
        "random squares": [],
    }
    for fen, truth in zip(fens, state):
        legal = legal_squares(str(fen))
        samples = {
            "true moves (ceiling)": truth[rng.integers(len(truth), size=k)],
            "random legal move": np.array([move_state(legal[i]) for i in rng.integers(len(legal), size=k)], float),
            "random squares": rng.integers(8, size=(k, 4)).astype(float),
        }
        for name, pred in samples.items():
            rows[name].append(futures_metrics(pred, truth, 1.0, None, ChessBenchmark.max_radius))
    return {name: {key: float(np.mean([r[key] for r in rs])) for key in rs[0]} for name, rs in rows.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ds = ChessPositions(a.split, a.root)
    n_legal = [len(legal_squares(str(f))) for f in ds.fens]
    print(f"{a.split}: {len(ds.fens)} positions, {np.mean(n_legal):.1f} legal moves per position on average")
    for name, m in baselines(ds.fens, ds.state.numpy().astype(float), a.k, a.seed).items():
        print(f"{name:22s} " + " ".join(f"{key}={v:.4f}" for key, v in m.items()))


if __name__ == "__main__":
    main()
