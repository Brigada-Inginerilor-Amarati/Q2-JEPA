"""Plot a trained D3IL run's rollouts. Visual only: computes no metric, changes nothing.

Left: the eval.rollouts paths from the common start over the obstacle map, accepted paths in blue and rejected ones
in grey. Middle: the fold's eval demos for reference. Right: how often each of the 24 routes was taken by accepted
rollouts and by the demos, against the uniform target 1/24. The rollouts are the ones the run's metrics scored.

uv run --group dev python scripts/show_d3il.py outputs/my-run --out d3il
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from q2jepa.benchmarks.d3il import accept
from q2jepa.data import d3il
from q2jepa.metrics import ROUTES
from q2jepa.runs import load_trained

INK, MUTED, GREY, BLUE, ORANGE, SURFACE = "#0b0b0b", "#52514e", "#7b7a74", "#2a78d6", "#eb6834", "#fcfcfb"


def route_label(r: int) -> str:
    """Gap per row, left to right, numbered from 1: e.g. 1-3-2."""
    return f"{r // 12 + 1}-{(r // 4) % 3 + 1}-{r % 4 + 1}"


@torch.no_grad()
def rollouts(run_dir: Path, device=None) -> dict:
    """The scored rollouts (metres), their routes (None = rejected), and the fold's eval demos."""
    t = load_trained(run_dir, device=device)
    paths = t.bench.rollouts(t.cfg, t.encoder, t.predictor, t.prior, t.readout)
    demos = d3il.demos(t.cfg.dataset.fold, "test", t.cfg.dataset.root)
    return {"paths": paths, "routes": [accept(p) for p in paths], "demos": demos, "fold": t.cfg.dataset.fold}


def _map(ax, title: str) -> None:
    for row_y, xs, radius in zip(d3il.ROW_Y, d3il.ROW_OBSTACLE_X, d3il.ROW_RADIUS, strict=True):
        for x in xs:
            ax.add_patch(plt.Circle((x, row_y), radius, color=INK, alpha=0.8))
            ax.add_patch(plt.Circle((x, row_y), radius + d3il.INFLATION, fill=False, color=INK, ls=":", lw=0.8))
    ax.axhline(d3il.GOAL_Y, color=INK, lw=1, ls="--")
    ax.plot(*d3il.START_POINT, marker="o", color=INK, ms=6)
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(-0.32, 0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
    ax.set_ylabel("y (m)", fontsize=8, color=MUTED)


def figure(d: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.9]}, facecolor=SURFACE)
    accepted = [r for r in d["routes"] if r is not None]
    ax = axes[0]
    for p, r in zip(d["paths"], d["routes"], strict=True):
        ax.plot(p[:, 0], p[:, 1], color=GREY if r is None else BLUE, lw=0.8, alpha=0.35)
    _map(ax, f"rollouts: {len(accepted)}/{len(d['paths'])} accepted")
    ax = axes[1]
    for p in d["demos"]:
        ax.plot(p[:, 0], p[:, 1], color=ORANGE, lw=0.8, alpha=0.6)
    _map(ax, f"fold {d['fold']} eval demos ({len(d['demos'])})")

    ax = axes[2]
    model = np.bincount(np.array(accepted, dtype=int), minlength=ROUTES) / max(len(accepted), 1)
    demo_routes = np.array([d3il.route_label(p) for p in d["demos"]])
    demo = np.bincount(demo_routes, minlength=ROUTES) / len(demo_routes)
    x = np.arange(ROUTES)
    ax.bar(x - 0.2, model, width=0.38, color=BLUE, label="accepted rollouts")
    ax.bar(x + 0.2, demo, width=0.38, color=ORANGE, label="eval demos")
    ax.axhline(1 / ROUTES, color=INK, lw=1, ls=":", label="uniform 1/24")
    ax.set_xticks(x, [route_label(r) for r in x], rotation=90, fontsize=7)
    ax.set_xlabel("route (gap per row, left to right)", fontsize=8, color=MUTED)
    ax.set_ylabel("share of paths", fontsize=8, color=MUTED)
    ax.set_title(f"routes: {(model > 0).sum()}/24 taken by rollouts", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.tick_params(labelsize=7, colors=MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--device", default=None, help="default: the run's trainer.device")
    ap.add_argument("--out", type=Path, default=None, help="output path without extension (default: run_dir/d3il)")
    args = ap.parse_args()
    d = rollouts(args.run_dir, torch.device(args.device) if args.device else None)
    out = (args.out or args.run_dir / "d3il").with_suffix(".png")
    figure(d).savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
