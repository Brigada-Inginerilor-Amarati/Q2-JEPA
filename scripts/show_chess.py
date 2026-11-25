"""Show a trained chess run's sampled moves on the board, and measure how many are legal. Needs python-chess.

Rows are fixed by rule, not selected: the middle of each of --n-obs equal parts of the scored positions. The GIF has 3
frames: the position; the 3 true moves (green); the model's --top most sampled moves of the eval.k_eval draws, with
the share of draws (blue: one of the true moves, orange: another legal move, red: illegal). The draws are the ones
the run's metrics scored (with its own split and device). Printed: the share of draws over all scored positions that
are legal moves (from/to squares only). That share is a diagnostic, not a metric of the results tables.

pip install chess && python scripts/show_chess.py runs/chess/grid/hqvit-entangled_qcbm-s1-s0 --out chess
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.data.chess import ChessPositions, parse_fen
from q2jepa.runs import load_trained
from scripts.chess_baselines import legal_squares

GLYPHS = dict(zip("KQRBNPkqrbnp", "♔♕♖♗♘♙♚♛♜♝♞♟", strict=True))
TRUE, OTHER, ILLEGAL = "#1a7f4b", "#e07b00", "#c92a2a"
MATCH = "#1c6fd1"


def spaced(n: int, rows: int) -> list[int]:
    """The middle index of each of `rows` equal parts of range(n)."""
    rows = min(rows, n)
    return [(2 * i + 1) * n // (2 * rows) for i in range(rows)]


def uci(state: np.ndarray) -> str:
    """(from file, from rank, to file, to rank) -> 'e2e4'."""
    f0, r0, f1, r1 = (int(v) for v in np.rint(state))
    return f"{chr(97 + f0)}{r0 + 1}{chr(97 + f1)}{r1 + 1}"


@torch.no_grad()
def sample(run_dir: Path, split: str | None = None, device: torch.device | None = None) -> dict:
    """fens, states (n, 3, 4) and decoded draws (n, K, 4) for every scored position, as ChessBenchmark.evaluate."""
    t = load_trained(run_dir, split, device)
    obs, _, state = t.bench.load(t.cfg.eval.split)
    n, k = min(t.cfg.eval.n_obs, len(obs)), t.cfg.eval.k_eval
    h = t.bench.encode_src(t.encoder, obs[:n].to(t.bench.device))
    pred = t.predictor(h, t.prior.sample(h, k))
    dec = t.bench.decode(t.readout(h[:, None].expand(-1, k, -1), pred))
    fens = ChessPositions(t.cfg.eval.split, t.cfg.dataset.root).fens[:n]
    return {
        "fens": [str(f) for f in fens],
        "state": state[:n].numpy().astype(float),
        "draws": dec.cpu().double().numpy(),
    }


def legal_share(d: dict) -> float:
    """Share of all draws whose from/to squares are a legal move of their position."""
    return float(
        np.mean(
            [
                [uci(m) in legal for m in draws]
                for legal, draws in zip(map(set, map(legal_squares, d["fens"])), d["draws"])
            ]
        )
    )


def true_share(d: dict) -> float:
    """Share of all draws equal to one of the true moves; equals the run's recorded validity."""
    return float(
        np.mean(
            [[uci(m) in set(map(uci, st)) for m in draws] for st, draws in zip(d["state"], d["draws"], strict=True)]
        )
    )


def draw_board(ax, fen: str) -> None:
    for r, row in enumerate(parse_fen(fen)):  # row 0 = rank 8
        for c, piece in enumerate(row):
            ax.add_patch(Rectangle((c, 7 - r), 1, 1, color="#eeeed2" if (r + c) % 2 == 0 else "#7fa35f"))
            if piece:
                ax.text(c + 0.5, 7 - r + 0.46, GLYPHS[piece], ha="center", va="center", fontsize=20, color="#111111")
    ax.set(xlim=(0, 8), ylim=(0, 8), aspect="equal", xticks=[], yticks=[])


def arrow(ax, move: str, color: str, width: float) -> None:
    a = (ord(move[0]) - 96.5, int(move[1]) - 0.5)
    b = (ord(move[2]) - 96.5, int(move[3]) - 0.5)
    if a == b:  # from-square = to-square: never a move
        ax.add_patch(plt.Circle(a, 0.35, fill=False, color=color, lw=width))
        return
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=width, alpha=0.85))


def figure(d: dict, rows: list[int], step: int, top: int = 5) -> plt.Figure:
    """step 0: position; 1: true moves; 2: the `top` most sampled moves."""
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 5.2), squeeze=False)
    for ax, i in zip(axes[0], rows, strict=True):
        draw_board(ax, d["fens"][i])
        true = [uci(s) for s in d["state"][i]]
        if step == 1:
            for m in true:
                arrow(ax, m, TRUE, 3)
        note = ""
        if step == 2:
            legal, k = set(legal_squares(d["fens"][i])), len(d["draws"][i])
            for m, c in reversed(Counter(uci(s) for s in d["draws"][i]).most_common(top)):
                color = MATCH if m in true else OTHER if m in legal else ILLEGAL
                arrow(ax, m, color, 1 + 6 * c / k)
            lg = np.mean([uci(s) in legal for s in d["draws"][i]])
            note = f"\nlegal draws {100 * lg:.0f}%"
        ax.set_title(f"position {i}{note}", fontsize=10)
    label = ("position", "the 3 equally good moves", f"{top} most sampled moves")[step]
    key = "blue: a true move, orange: other legal move, red: illegal" if step == 2 else ""
    fig.suptitle(f"Chess, white to move: {label}\n{key}", fontsize=11)
    fig.tight_layout()
    return fig


def to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-obs", type=int, default=3, help="boards")
    ap.add_argument("--top", type=int, default=5, help="sampled moves drawn per board")
    ap.add_argument("--split", choices=("dev", "test"), default=None, help="default: the run's eval.split")
    ap.add_argument("--device", default=None, help="default: the run's trainer.device")
    ap.add_argument("--out", type=Path, default=None, help="output path without extension (default: run_dir/chess)")
    args = ap.parse_args()
    out = args.out or args.run_dir / "chess"
    d = sample(args.run_dir, args.split, torch.device(args.device) if args.device else None)
    rows = spaced(len(d["fens"]), args.n_obs)
    frames = [to_image(figure(d, rows, s, args.top)) for s in range(3)]
    frames[-1].save(out.with_suffix(".png"))
    frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=1500, loop=0)
    print(f"legal draws {100 * legal_share(d):.1f}% over {len(d['fens'])} positions x {d['draws'].shape[1]} draws")
    saved = args.run_dir / "metrics.json"
    recorded = f", recorded validity {json.loads(saved.read_text())['validity']:.5f}" if saved.exists() else ""
    print(f"draws equal to a true move {true_share(d):.5f}{recorded}")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
