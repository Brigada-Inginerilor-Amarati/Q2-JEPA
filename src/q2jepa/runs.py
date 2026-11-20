"""Reload a finished run in the exact state `train.run()` evaluates it in, for plots and inspection."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf

from q2jepa.benchmarks import build_benchmark
from q2jepa.config import Config
from q2jepa.quantum import set_target
from q2jepa.seeding import EVAL_SEED_OFFSET, reseed
from q2jepa.train import build_models, resolve_device, use_hqvit_backend


class Trained(NamedTuple):
    cfg: object
    bench: object
    encoder: torch.nn.Module
    predictor: torch.nn.Module
    prior: torch.nn.Module
    readout: torch.nn.Module


def load_config(run_dir: Path, split: str | None = None):
    """The run's config for `split` (default: the run's eval split). eval_config_<split>.yaml wins when present,
    since re-evaluating via the CLI overwrites .hydra/."""
    run_dir = Path(run_dir)
    split = split or OmegaConf.load(run_dir / ".hydra" / "config.yaml").eval.split
    path = run_dir / f"eval_config_{split}.yaml"
    path = path if path.exists() else run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path))
    cfg.eval.split = split
    return cfg


class _NoLog:
    def log(self, *_):
        pass


@torch.no_grad()
def load_trained(run_dir: Path, split: str | None = None, device: torch.device | None = None) -> Trained:
    """Models, readout and stage-3 weights of a finished run, with the CUDA-Q backend and the evaluation seed set,
    so the next draws are the ones the run's metrics scored (with the run's own split and device)."""
    run_dir = Path(run_dir)
    for name in ("ckpt.pt", "readout.pt"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} is missing: the run has not finished")
    cfg = load_config(run_dir, split)
    if cfg.trainer.finetune_iters and not (run_dir / "finetune.pt").exists():
        raise FileNotFoundError(f"{run_dir / 'finetune.pt'} is missing: the run has not finished")
    set_target(cfg.quantum.target, cfg.quantum.precision)
    np.random.seed(cfg.seed)
    device = device or resolve_device(cfg.trainer.device)
    groups, _ = build_models(cfg, device)
    ck = torch.load(run_dir / "ckpt.pt", map_location=device, weights_only=True)
    for name, (module, _) in groups.items():
        module.load_state_dict(ck[name])
        module.requires_grad_(False)
    encoder, predictor, prior = (groups[n][0] for n in ("encoder", "predictor", "prior"))
    bench = build_benchmark(cfg, device)
    readout = bench.fit_readout(cfg, encoder, run_dir, _NoLog())  # loads readout.pt
    bench.post_readout(cfg, groups, readout, run_dir, _NoLog())  # loads finetune.pt (D3IL)
    use_hqvit_backend(encoder, "cudaq")
    reseed(cfg.seed, EVAL_SEED_OFFSET)
    return Trained(cfg, bench, encoder, predictor, prior, readout)
