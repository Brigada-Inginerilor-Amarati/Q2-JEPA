"""Show a trained MovingMNIST run's sampled futures as pixels. Visual only: computes no metric, changes nothing.

For each observation, the top row holds the 3 observed frames and K futures sampled from the prior, rendered by the
frozen readout. The bottom row holds the 8 true futures. The PNG shows the last future frame; the GIF steps through
all 3. By default it uses the run's eval settings (eval.n_obs, eval.k_eval, eval.split, trainer.device) and seed,
so it shows exactly the samples the metrics scored. Overriding --n-obs, --k, --split or --device changes the draws.

uv run python scripts/show_futures.py outputs/my-run --out futures
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from q2jepa.benchmarks.mmnist import encode_clips
from q2jepa.data.mmnist import StochasticMovingMNIST, true_trajectories
from q2jepa.readout.mmnist import render
from q2jepa.runs import load_trained


@torch.no_grad()
def sample_futures(run_dir: Path, n_obs=None, k=None, split=None, device=None):
    """(obs (N, 3, 64, 64), sampled (N, K, 3, 64, 64), truth (N, 8, 3, 64, 64), disp (N, K, 3, 2)).

    Arguments left as None take the run's eval settings, which reproduce the evaluation's samples exactly.
    """
    t = load_trained(run_dir, split, device)
    cfg, readout = t.cfg, t.readout
    n_obs, k, split = n_obs or cfg.eval.n_obs, k or cfg.eval.k_eval, cfg.eval.split
    device = t.bench.device
    n_split = cfg.dataset.n_test if split == "test" else cfg.dataset.n_dev
    obs = StochasticMovingMNIST(split, cfg.dataset.root, n_obs=n_split).unique_obs()[:n_obs].to(device)
    h = encode_clips(t.encoder, obs, cfg.encoder)
    pred = t.predictor(h, t.prior.sample(h, k))
    disp = readout(h.repeat_interleave(k, 0), pred.reshape(-1, pred.shape[-1])).reshape(len(obs), k, 3, 2)
    last = obs[:, 2]
    sampled = render(last, disp.reshape(len(obs), -1, 2)).reshape(len(obs), k, 3, 64, 64)
    truth_disp = torch.tensor(true_trajectories(), dtype=obs.dtype, device=device).reshape(1, -1, 2)
    truth = render(last, truth_disp.expand(len(obs), -1, -1)).reshape(len(obs), 8, 3, 64, 64)
    return obs.cpu(), sampled.clamp(0, 1).cpu(), truth.clamp(0, 1).cpu(), disp.cpu()


def grid(obs: torch.Tensor, sampled: torch.Tensor, truth: torch.Tensor, frame: int) -> Image.Image:
    """One image: per observation, [obs x3 | K samples] over [blank x3 | 8 truths], padded to equal width."""
    k = sampled.shape[1]
    width = 3 + max(k, 8)
    blank = torch.zeros(64, 64)
    tiles = []
    for o, s, t in zip(obs, sampled, truth, strict=True):
        top = [*o, *s[:, frame]]
        bottom = [blank] * 3 + list(t[:, frame])
        for row in (top, bottom):
            tiles += row + [blank] * (width - len(row))
    img = make_grid(torch.stack(tiles)[:, None], nrow=width, padding=2, pad_value=0.5)
    return Image.fromarray((img[0] * 255).round().byte().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-obs", type=int, default=None, help="default: the run's eval.n_obs")
    ap.add_argument("--k", type=int, default=None, help="futures per observation; default: the run's eval.k_eval")
    ap.add_argument("--split", choices=("dev", "test"), default=None, help="default: the run's eval.split")
    ap.add_argument("--device", default=None, help="default: the run's trainer.device")
    ap.add_argument("--out", type=Path, default=None, help="output path without extension (default: run_dir/futures)")
    args = ap.parse_args()
    out = args.out or args.run_dir / "futures"
    device = torch.device(args.device) if args.device else None
    obs, sampled, truth, _ = sample_futures(args.run_dir, args.n_obs, args.k, args.split, device)
    frames = [grid(obs, sampled, truth, f) for f in range(3)]
    frames[-1].save(out.with_suffix(".png"))
    frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=600, loop=0)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
