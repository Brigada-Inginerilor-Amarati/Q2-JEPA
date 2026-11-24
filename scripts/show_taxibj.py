"""Show a trained TaxiBJ run's predicted maps next to the true ones. Visual only: computes no reported metric.

One row per observation, inflow or outflow (--channel). Rows are fixed by rule, not selected: the middle of each of --n-obs equal parts of the scored
observations. Columns: the
recorded maps, copying the last observed map, the first prior draw, and the mean of the run's eval.k_eval draws. The
GIF steps through the 3 observed and 3 future maps; the PNG shows the last future map. With the run's own split and
device the draws are the ones its metrics scored. The number under each forecast is its mean squared error (x 1e-3)
over all 3 future maps and both channels.

python scripts/show_taxibj.py runs/taxibj/grid/hqvit-entangled_qcbm-s1-s0 --out taxibj
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q2jepa.runs import load_trained

CHUNK = 16  # observations per call in TaxiBJBenchmark.evaluate: the same chunks give the same draws
COLUMNS = ("recorded", "copy last map", "one draw", "mean of draws")


def spaced(n: int, rows: int) -> list[int]:
    """The middle index of each of `rows` equal parts of range(n)."""
    rows = min(rows, n)
    return [(2 * i + 1) * n // (2 * rows) for i in range(rows)]


@torch.no_grad()
def sample(run_dir: Path, rows: int = 3, split: str | None = None, device: torch.device | None = None) -> dict:
    """index (rows,), obs, fut (rows, 3, 2, 32, 32) and draws (rows, K, 3, 2, 32, 32) of the spaced observations.

    Every chunk is sampled in the evaluation's order, so each row keeps the draws its metrics scored."""
    t = load_trained(run_dir, split, device)
    obs, fut, _ = (x.to(t.bench.device) for x in t.bench.load(t.cfg.eval.split))
    n, k = min(t.cfg.eval.n_obs, len(obs)), t.cfg.eval.k_eval
    pick, draws = spaced(n, rows), []
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        h = t.bench.encode_src(t.encoder, obs[start:stop])
        pred = t.predictor(h, t.prior.sample(h, k))
        dec = t.readout(h[:, None].expand(-1, k, -1), pred)  # whole chunk, as TaxiBJBenchmark.evaluate
        draws += [dec[i - start].reshape(k, *fut.shape[2:]).cpu().numpy() for i in pick if start <= i < stop]
    return {
        "index": pick,
        "obs": obs[pick].float().cpu().numpy(),
        "fut": fut[pick, 0].float().cpu().numpy(),
        "draws": np.stack(draws),
    }


def forecasts(d: dict, i: int) -> list[np.ndarray]:
    """The 4 columns' future maps (3, 2, 32, 32) for observation i."""
    copy = np.repeat(d["obs"][i][-1:], 3, 0)
    return [d["fut"][i], copy, d["draws"][i, 0], d["draws"][i].mean(0)]


def figure(d: dict, n_obs: int, step: int, channel: int = 0) -> plt.Figure:
    """step 0-2: observed maps (the same in every column); 3-5: future maps."""
    n = min(n_obs, len(d["obs"]))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n + 0.6), squeeze=False)
    for i in range(n):
        cols = forecasts(d, i)
        vmax = max(d["obs"][i][:, channel].max(), d["fut"][i][:, channel].max())
        for j, (ax, maps) in enumerate(zip(axes[i], cols, strict=True)):
            img = d["obs"][i][step, channel] if step < 3 else maps[step - 3, channel]
            ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
            ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(COLUMNS[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"window {d['index'][i]}", fontsize=8)
            if j:
                ax.set_xlabel(f"MSE {1e3 * ((maps - d['fut'][i]) ** 2).mean():.2f}", fontsize=8)
    phase = f"observed {step + 1}/3" if step < 3 else f"future {step - 2}/3"
    fig.suptitle(f"TaxiBJ {('inflow', 'outflow')[channel]}: {phase}", fontsize=11)
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
    ap.add_argument("--n-obs", type=int, default=3, help="rows")
    ap.add_argument("--channel", type=int, default=0, choices=(0, 1), help="0 inflow, 1 outflow")
    ap.add_argument("--split", choices=("dev", "test"), default=None, help="default: the run's eval.split")
    ap.add_argument("--device", default=None, help="default: the run's trainer.device")
    ap.add_argument("--out", type=Path, default=None, help="output path without extension (default: run_dir/taxibj)")
    args = ap.parse_args()
    out = args.out or args.run_dir / "taxibj"
    d = sample(args.run_dir, args.n_obs, args.split, torch.device(args.device) if args.device else None)
    frames = [to_image(figure(d, args.n_obs, s, args.channel)) for s in range(6)]
    frames[-1].save(out.with_suffix(".png"))
    frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
