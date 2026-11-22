"""Summarize finished runs: a markdown table (mean ± s.d. over seeds and folds) and one bar chart per benchmark.

Reads every run dir under the given roots that has metrics for the split (metrics.json for test,
metrics_<split>.json otherwise) and its eval_config_<split>.yaml. Rows group runs by encoder, prior and encoder
learning-rate schedule (s1-s3); each bar is the mean, with one dot per seed (and fold).

uv run --group dev python scripts/plot_results.py runs/hep18 --split test --out results/hep18
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# fixed hue per prior (validated categorical slots 1-5), so a prior keeps its colour across every chart
PRIOR_COLORS = {"softmax_categorical": "#2a78d6", "entangled_qcbm": "#eb6834", "product_qcbm": "#1baf7a",
                "frozen_qcbm": "#eda100", "mixture": "#e87ba4"}  # fmt: skip
ENCODER_ORDER = ("cnn", "vit", "hqvit")
SCHEDULES = {(3e-4, 1e-5): "s1", (3e-4, 3e-4): "s2", (1e-3, 1e-4): "s3"}
METRICS = {
    "mmnist": ("coverage", "validity", "tv", "min_ade", "es"),
    "hep18": ("coverage", "validity", "tv", "min_ade", "es", "shuffled_validity", "readout_validity"),
    "d3il": ("coverage", "tv", "acceptance", "readout_ade_m"),
    "nri": ("coverage", "validity", "tv", "min_ade", "es", "readout_validity"),
    "chess": ("coverage", "validity", "tv", "min_ade", "es", "readout_validity"),
}
PLOTTED = {"mmnist": ("coverage", "validity", "tv"), "hep18": ("coverage", "validity", "tv"),
           "d3il": ("coverage", "tv", "acceptance"), "nri": ("coverage", "validity", "tv"),
           "chess": ("coverage", "validity", "tv")}  # fmt: skip


def collect(roots: list[Path], split: str) -> list[dict]:
    """One record per finished run: dataset, encoder, prior, schedule, hqvit_train, seed, fold and its metrics."""
    name = "metrics.json" if split == "test" else f"metrics_{split}.json"
    runs = []
    for root in roots:
        for path in sorted(root.rglob(name)):
            cfg_path = path.parent / f"eval_config_{split}.yaml"
            if not cfg_path.exists():
                continue
            cfg = OmegaConf.load(cfg_path)
            if cfg.encoder.name == "state":  # results are on rendered images only (AGENTS.md §3)
                raise SystemExit(f"{path.parent} uses encoder=state: a diagnostic, never a results row")
            t = cfg.trainer
            commit = path.parent / "commit.txt"
            runs.append({
                "dir": str(path.parent),
                "dataset": cfg.dataset.name,
                "encoder": cfg.encoder.name,
                "prior": cfg.prior.name,
                "schedule": SCHEDULES.get((t.lr_enc, t.lr_enc_later), f"lr {t.lr_enc:g}->{t.lr_enc_later:g}"),
                "hqvit_train": cfg.quantum.hqvit_train if cfg.encoder.name == "hqvit" else "",
                "platform": cfg.quantum.target,
                "commit": commit.read_text().split()[0] if commit.exists() else "unknown",
                "seed": cfg.seed,
                "fold": cfg.dataset.get("fold", 0) if cfg.dataset.name == "d3il" else None,
                "metrics": json.loads(path.read_text()),
            })  # fmt: skip
    return runs


def group(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["encoder"], r["prior"], r["schedule"], r["hqvit_train"], r["platform"])].append(r)

    def order(key):
        enc = ENCODER_ORDER.index(key[0]) if key[0] in ENCODER_ORDER else len(ENCODER_ORDER)
        pri = list(PRIOR_COLORS).index(key[1]) if key[1] in PRIOR_COLORS else len(PRIOR_COLORS)
        return enc, key[0], key[2], pri, key[1]

    return dict(sorted(groups.items(), key=lambda kv: order(kv[0])))


def stats(rows: list[dict], metric: str) -> tuple[float, float]:
    x = np.array([r["metrics"][metric] for r in rows], dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def table(dataset: str, groups: dict[tuple, list[dict]]) -> str:
    metrics = [m for m in METRICS.get(dataset, ()) if any(m in r["metrics"] for g in groups.values() for r in g)]
    head = ["encoder", "prior", "schedule", "platform", "commit", "n", *metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (enc, prior, sched, hq, platform), rows in groups.items():
        commits = sorted({r["commit"][:7] for r in rows})  # more than one: rerun those rows on one commit
        cells = [enc + (f" ({hq})" if hq else ""), prior, sched, platform, ", ".join(commits), str(len(rows))]
        for m in metrics:
            mean, sd = stats(rows, m)
            cells.append(f"{mean:.3f} ± {sd:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(dataset: str, groups: dict[tuple, list[dict]]) -> plt.Figure:
    metrics = PLOTTED.get(dataset, ("coverage", "tv"))
    schedules = {k[2] for k in groups}
    columns = list(dict.fromkeys((k[0], k[2]) for k in groups))  # (encoder, schedule) in order
    priors = list(dict.fromkeys(k[1] for k in groups))
    width = 0.8 / len(priors)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), facecolor=SURFACE)
    for ax, m in zip(np.atleast_1d(axes), metrics, strict=True):
        for (enc, prior, sched, _, _), rows in groups.items():
            x = columns.index((enc, sched)) + (priors.index(prior) - (len(priors) - 1) / 2) * width
            mean, sd = stats(rows, m)
            ax.bar(x, mean, width * 0.92, color=PRIOR_COLORS.get(prior, MUTED), yerr=sd, capsize=3,
                   error_kw={"ecolor": INK, "lw": 1}, label=prior)  # fmt: skip
            ax.scatter(np.full(len(rows), x), [r["metrics"][m] for r in rows], s=10, color=INK, zorder=3)
        labels = [enc + (f"\n{sched}" if len(schedules) > 1 else "") for enc, sched in columns]
        ax.set_xticks(range(len(columns)), labels, fontsize=8)
        ax.set_title(m + (" (lower is better)" if m == "tv" else ""), fontsize=10, color=INK)
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(axis="y", alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PRIOR_COLORS.get(p, MUTED)) for p in priors]
    fig.legend(handles, priors, loc="upper center", ncol=len(priors), frameon=False, fontsize=9)
    fig.suptitle(f"{dataset}: mean ± s.d., dots = seeds", y=0.02, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", type=Path, nargs="+", help="directories searched recursively for finished runs")
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--out", type=Path, default=Path("results"), help="output prefix; writes <out>_<dataset>.png")
    args = ap.parse_args()
    runs = collect(args.roots, args.split)
    if not runs:
        raise SystemExit(f"no finished {args.split} runs under {args.roots}")
    for dataset in sorted({r["dataset"] for r in runs}):
        groups = group([r for r in runs if r["dataset"] == dataset])
        print(f"## {dataset} ({args.split})\n\n{table(dataset, groups)}\n")
        out = args.out.parent / f"{args.out.name}_{dataset}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        figure(dataset, groups).savefig(out, dpi=130)
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
