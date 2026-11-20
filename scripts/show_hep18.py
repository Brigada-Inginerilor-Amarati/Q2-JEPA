"""Plot a trained HEP18 run's sampled futures. Visual only: computes no metric, changes nothing.

One row per observation. The first four panels are the cars (N, W, S, E): position along the lane s (m) over time,
with the observed motion and the car's distinct true futures in black, samples that match a joint future in blue,
and samples that match none in grey. The last panel shows how often each of the 18 joint futures was sampled,
against the uniform target 1/18. The draws are the ones the run's metrics scored.

uv run --group dev python scripts/show_hep18.py outputs/my-run --obs 0 1 2 --out hep18
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from q2jepa.data.hep18 import _FUT_T, _OBS_T, CARS
from q2jepa.metrics import _joint_valid
from q2jepa.runs import load_trained

INK, MUTED, GREY, BLUE, SURFACE = "#0b0b0b", "#52514e", "#7b7a74", "#2a78d6", "#fcfcfb"
FUTURES = 18


def future_label(j: int) -> str:
    axis, first, second = j // 9, (j // 3) % 3, j % 3
    return f"{('NS', 'EW')[axis]} {'ABC'[first]}{'ABC'[second]}"


@torch.no_grad()
def sample(run_dir: Path, device=None) -> dict[str, np.ndarray]:
    """The scored draws: obs (n, 4, 3, 2), s and v (n, K, 4, 3), true futures (n, 18, 4, 3, 2), matches (n, K)."""
    t = load_trained(run_dir, device=device)
    data, s, v = t.bench.sample(t.cfg, t.encoder, t.predictor, t.prior, t.readout)
    n = len(s)
    obs, fut = data.obs[:n].cpu().double().numpy(), data.fut[:n].cpu().double().numpy()
    match = []
    for i in range(n):
        valid, nearest = _joint_valid(s[i], v[i], fut[i, ..., 0], fut[i, ..., 1])
        match.append(np.where(valid, nearest, FUTURES))  # FUTURES = matches none
    return {"obs": obs, "s": s, "v": v, "fut": fut, "match": np.array(match)}


def figure(d: dict[str, np.ndarray], rows: list[int]) -> plt.Figure:
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 2.9 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]}, facecolor=SURFACE)  # fmt: skip
    t_obs, t_fut = _OBS_T, np.concatenate([[0.0], _FUT_T])
    for r, i in enumerate(rows):
        valid = d["match"][i] < FUTURES
        for c, car in enumerate(CARS):
            ax = axes[r, c]
            s0 = d["obs"][i, c, -1, 0]
            for s in d["s"][i][~valid, c]:
                ax.plot(t_fut, [s0, *s], color=GREY, lw=1, alpha=0.25)
            for s in d["s"][i][valid, c]:
                ax.plot(t_fut, [s0, *s], color=BLUE, lw=1, alpha=0.25)
            for s in np.unique(d["fut"][i, :, c, :, 0], axis=0):
                ax.plot(t_fut, [s0, *s], color=INK, lw=1.5, ls="--")
            ax.plot(t_obs, d["obs"][i, c, :, 0], color=INK, lw=2, marker="o", ms=4)
            ax.set_title(f"obs {i}, car {car}", fontsize=9, color=INK)
            ax.set_xlabel("t (s)", fontsize=8, color=MUTED)
            if c == 0:
                ax.set_ylabel("s (m)", fontsize=8, color=MUTED)
        ax = axes[r, 4]
        k = len(d["match"][i])
        freq = np.bincount(d["match"][i], minlength=FUTURES + 1) / k
        ax.bar(np.arange(FUTURES), freq[:FUTURES], color=BLUE, width=0.8)
        ax.bar([FUTURES], freq[FUTURES:], color=GREY, width=0.8)
        ax.axhline(1 / FUTURES, color=INK, lw=1, ls=":")
        ax.set_xticks(range(FUTURES + 1), [*map(future_label, range(FUTURES)), "none"], rotation=90, fontsize=6)
        covered = (freq[:FUTURES] > 0).sum()
        ax.set_title(f"futures sampled: {covered}/18 covered, {valid.mean():.0%} valid (dotted: 1/18)",
                     fontsize=9, color=INK)  # fmt: skip
        ax.set_ylabel("share of samples", fontsize=8, color=MUTED)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], color=INK, lw=2, marker="o", ms=4, label="observed"),
               plt.Line2D([], [], color=INK, lw=1.5, ls="--", label="true futures (car's own)"),
               plt.Line2D([], [], color=BLUE, lw=2, label="sample matching a joint future"),
               plt.Line2D([], [], color=GREY, lw=2, label="sample matching none")]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (2.9 * len(rows))))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--obs", type=int, nargs="+", default=[0, 1, 2, 3], help="observation indices to plot")
    ap.add_argument("--device", default=None, help="default: the run's trainer.device")
    ap.add_argument("--out", type=Path, default=None, help="output path without extension (default: run_dir/hep18)")
    args = ap.parse_args()
    d = sample(args.run_dir, torch.device(args.device) if args.device else None)
    out = (args.out or args.run_dir / "hep18").with_suffix(".png")
    figure(d, [i for i in args.obs if i < len(d["s"])]).savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
