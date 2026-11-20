"""3-stage JEPA training, frozen readout and evaluation (Q-JEPA §3, App. A.3), for every benchmark.

Resumable for Colab: every iteration reseeds from (seed, it), so a checkpoint only needs model and
optimizer state; batches are drawn with torch.randint, not from a DataLoader iterator. Benchmark-specific
batches, readouts and metrics live in q2jepa/benchmarks/.
"""

import json
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from q2jepa.benchmarks import build_benchmark, scene_dim
from q2jepa.config import Config  # also registers the schema
from q2jepa.encoder import build_encoder
from q2jepa.losses import energy_score, loo_cost, prior_score_loss, shift_cost, visreg
from q2jepa.predictor import build_predictor
from q2jepa.prior.classical import build_prior
from q2jepa.quantum import set_target
from q2jepa.seeding import EVAL_SEED_OFFSET, READOUT_SEED_OFFSET, reseed, save_atomic

_seed, _save = reseed, save_atomic  # old names, used by scripts
__all__ = ["EVAL_SEED_OFFSET", "READOUT_SEED_OFFSET", "build_models", "resolve_device", "run", "train_step"]


def stage(it: int, warmup_repr: int, warmup_pred: int) -> Literal["repr", "pred", "full"]:
    if it <= warmup_repr:
        return "repr"
    return "pred" if it <= warmup_pred else "full"


def _prior_loss(cfg, prior, predictor, h, z, pred, tgt) -> torch.Tensor:
    """Separate prior update (Q-JEPA App. A.4 step 4): encoder outputs, targets and predictor are held fixed."""
    if not getattr(prior, "trainable", True):
        return torch.zeros((), device=h.device)  # Frozen-QCBM
    dist = (cfg.trainer.energy_beta, cfg.trainer.energy_eps)
    if getattr(prior, "grad", None) == "shift":
        return prior.shift_surrogate(
            h, lambda idx, codes: shift_cost(predictor(h[idx], codes), tgt[idx], pred[idx], *dist)
        )
    return prior_score_loss(prior.log_prob(h, z), loo_cost(pred, tgt, *dist))  # classical priors, ideal QCBM


def _visreg_views(v: torch.Tensor, cfg, it: int) -> torch.Tensor:
    """R over a batch (B, d), or averaged over the J views of (B, J, d) (App. F.3)."""
    if v.dim() == 2:
        return visreg(v, cfg, it)
    return torch.stack([visreg(v[:, j], cfg, it) for j in range(v.shape[1])]).mean()


_NOT_IDENTITY = {"trainer": ("stop_after", "ckpt_every", "device")}
_IDENTITY_GROUPS = ("encoder", "prior", "predictor", "quantum", "trainer", "visreg", "readout", "dataset")


def _identity(cfg) -> str:
    """Everything that defines the trained model, as the values that differ from the schema defaults.

    Resume and re-evaluation must match it. Storing only non-defaults means a new config key with a default value
    doesn't invalidate existing checkpoints.
    """
    defaults = OmegaConf.structured(Config)
    out = {"seed": cfg.seed}
    for g in _IDENTITY_GROUPS:
        cur, ref = OmegaConf.to_container(cfg[g], resolve=True), OmegaConf.to_container(defaults[g])
        out[g] = {k: v for k, v in cur.items() if k not in _NOT_IDENTITY.get(g, ()) and ref.get(k) != v}
    return json.dumps(out, sort_keys=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def use_hqvit_backend(encoder, name: str) -> None:
    """Select HQViT's simulator; a no-op "cudaq" for encoders without quantum blocks."""
    if hasattr(encoder, "set_backend"):
        encoder.set_backend(name)
    elif name != "cudaq":
        raise ValueError(f"quantum.hqvit_train={name!r} only applies to encoder=hqvit")


def build_models(cfg, device: torch.device) -> tuple[dict, torch.optim.Optimizer]:
    """Encoder, predictor, prior and their AdamW, initialized from `cfg.seed`. Groups map name -> (module, lr)."""
    t = cfg.trainer
    reseed(cfg.seed, 0)
    d = scene_dim(cfg)
    encoder = build_encoder(cfg.encoder).to(device)
    predictor = build_predictor(cfg.predictor, d, cfg.prior.n_bits, cfg.dataset.agents).to(device)
    prior = build_prior(cfg.prior, d, cfg.quantum).to(device)
    groups = {"encoder": (encoder, t.lr_enc), "predictor": (predictor, t.lr_pred), "prior": (prior, t.lr_prior)}
    opt = torch.optim.AdamW(
        [{"params": m.parameters(), "lr": lr, "name": n} for n, (m, lr) in groups.items()], weight_decay=t.weight_decay
    )
    return groups, opt


def train_step(cfg, groups: dict, opt: torch.optim.Optimizer, bench, batch, it: int):
    """One world-model update at iteration `it` (Q-JEPA App. A.3). Seeding and the learning-rate schedule are
    the caller's; `scripts/bench_encoder_step.py` times exactly this function."""
    (encoder, _), (predictor, _), (prior, _) = groups["encoder"], groups["predictor"], groups["prior"]
    t = cfg.trainer
    phase = stage(it, t.warmup_repr, t.warmup_pred)
    h, tgt = bench.encode(encoder, batch)
    loss = t.lambda_reg * (visreg(h, cfg.visreg, it) + _visreg_views(tgt, cfg.visreg, it)) / 2
    if t.lambda_change:  # HEP18: R(t - h) per target view (App. F.3)
        change = tgt - (h if tgt.dim() == 2 else h[:, None])
        loss = loss + t.lambda_change * _visreg_views(change, cfg.visreg, it)
    if phase != "repr":
        z = prior.sample(h.detach(), t.k_train)
        pred = predictor(h, z)
        loss = loss + energy_score(pred, tgt.detach(), t.energy_beta, t.energy_eps)
        if phase == "full":
            loss = loss + _prior_loss(cfg, prior, predictor, h.detach(), z, pred.detach(), tgt.detach())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for module, _ in groups.values():
        torch.nn.utils.clip_grad_norm_(module.parameters(), t.clip)
    opt.step()
    return loss


def run(cfg, out_dir: Path) -> dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_target(cfg.quantum.target, cfg.quantum.precision)
    np.random.seed(cfg.seed)
    t = cfg.trainer
    device = resolve_device(t.device)
    groups, opt = build_models(cfg, device)
    encoder, predictor, prior = (groups[n][0] for n in ("encoder", "predictor", "prior"))
    use_hqvit_backend(encoder, cfg.quantum.hqvit_train)  # training and readout; evaluation switches to cudaq

    ckpt_path = out_dir / "ckpt.pt"
    identity = _identity(cfg)
    start = 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ck.get("identity", identity) != identity:  # pre-phase-2 checkpoints have no identity; accept them
            raise ValueError(f"{ckpt_path} was trained with a different model/training config; use a new run dir")
        for name, (module, _) in groups.items():
            module.load_state_dict(ck[name])
        opt.load_state_dict(ck["opt"])
        start = ck["it"] + 1

    def checkpoint(it: int) -> None:
        save_atomic(
            {
                **{n: m.state_dict() for n, (m, _) in groups.items()},
                "opt": opt.state_dict(),
                "it": it,
                "identity": identity,
            },
            ckpt_path,
        )

    log = wandb.init(mode=cfg.wandb.mode, project=cfg.wandb.project, dir=str(out_dir),
                     config=OmegaConf.to_container(cfg, resolve=True))  # fmt: skip
    bench = build_benchmark(cfg, device)

    for it in range(start, t.iters + 1):
        reseed(cfg.seed, it)
        opt.param_groups[0]["lr"] = t.lr_enc if it <= t.warmup_repr else t.lr_enc_later
        loss = train_step(cfg, groups, opt, bench, bench.batch(t.batch), it)
        if it % 50 == 0:
            log.log({"loss": loss.item(), "it": it})
        if it == t.iters or it % t.ckpt_every == 0:
            checkpoint(it)
        if t.stop_after and it - start + 1 >= t.stop_after and it < t.iters:
            checkpoint(it)
            log.finish()
            return {}

    for module, _ in groups.values():
        module.requires_grad_(False)
    readout = bench.fit_readout(cfg, encoder, out_dir, log)
    bench.post_readout(cfg, groups, readout, out_dir, log)

    use_hqvit_backend(encoder, "cudaq")  # every reported number comes from the CUDA-Q circuits
    reseed(cfg.seed, EVAL_SEED_OFFSET)
    split = cfg.eval.split
    if split not in ("test", "dev"):
        raise ValueError(f"unknown eval split {split!r}")
    metrics = bench.evaluate(cfg, encoder, predictor, prior, readout)
    # Keep each split's provenance next to its metrics: re-evaluating via the CLI overwrites .hydra/.
    (out_dir / f"eval_config_{split}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (out_dir / ("metrics.json" if split == "test" else f"metrics_{split}.json")).write_text(
        json.dumps(metrics, sort_keys=True, indent=2)
    )
    log.log(metrics)
    log.finish()
    return metrics


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(run(cfg, Path(HydraConfig.get().runtime.output_dir)))


if __name__ == "__main__":
    main()
