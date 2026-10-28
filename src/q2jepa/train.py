"""3-stage JEPA training, frozen readout and evaluation (Q-JEPA §3, App. A.3).

Resumable for Colab: every iteration reseeds from (seed, it), so a checkpoint only needs model and
optimizer state; batches are drawn with torch.randint, not from a DataLoader iterator.
"""

import json
import os
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import q2jepa.config  # noqa: F401  registers schema
from q2jepa.data.mmnist import StochasticMovingMNIST
from q2jepa.encoder import build_encoder, prepare
from q2jepa.losses import energy_score, loo_cost, prior_score_loss, visreg
from q2jepa.metrics import mmnist_metrics
from q2jepa.predictor import Predictor
from q2jepa.prior.classical import build_prior
from q2jepa.quantum import set_target
from q2jepa.readout.mmnist import TranslationReadout, readout_loss, render

READOUT_SEED_OFFSET = 10_000_000
EVAL_SEED_OFFSET = 20_000_000


def stage(it: int, warmup_repr: int, warmup_pred: int) -> Literal["repr", "pred", "full"]:
    if it <= warmup_repr:
        return "repr"
    return "pred" if it <= warmup_pred else "full"


def _seed(seed: int, it: int) -> None:
    torch.manual_seed(seed * 1_000_003 + it)


def _batch(ds: StochasticMovingMNIST, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    items = [ds[i] for i in torch.randint(len(ds), (size,)).tolist()]
    obs = torch.stack([x["obs"] for x in items]).to(device)
    fut = torch.stack([x["fut"] for x in items]).to(device)
    return obs, fut


def _save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(cfg, out_dir: Path) -> dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_target(cfg.quantum.target)
    np.random.seed(cfg.seed)
    t = cfg.trainer
    device = torch.device("cuda" if t.device == "auto" and torch.cuda.is_available() else
                          "cpu" if t.device == "auto" else t.device)  # fmt: skip

    _seed(cfg.seed, 0)
    encoder = build_encoder(cfg.encoder).to(device)
    predictor = Predictor(cfg.encoder.d, cfg.prior.n_bits, cfg.predictor.width).to(device)
    prior = build_prior(cfg.prior, cfg.encoder.d).to(device)
    groups = {"encoder": (encoder, t.lr_enc), "predictor": (predictor, t.lr_pred), "prior": (prior, t.lr_prior)}
    opt = torch.optim.AdamW(
        [{"params": m.parameters(), "lr": lr, "name": n} for n, (m, lr) in groups.items()], weight_decay=t.weight_decay
    )

    ckpt_path = out_dir / "ckpt.pt"
    start = 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        for name, (module, _) in groups.items():
            module.load_state_dict(ck[name])
        opt.load_state_dict(ck["opt"])
        start = ck["it"] + 1

    def checkpoint(it: int) -> None:
        _save({**{n: m.state_dict() for n, (m, _) in groups.items()}, "opt": opt.state_dict(), "it": it}, ckpt_path)

    log = wandb.init(mode=cfg.wandb.mode, project=cfg.wandb.project, dir=str(out_dir),
                     config=OmegaConf.to_container(cfg, resolve=True))  # fmt: skip
    train_ds = StochasticMovingMNIST("train", cfg.dataset.root, n_obs=cfg.dataset.n_train)
    res = cfg.encoder.res

    for it in range(start, t.iters + 1):
        _seed(cfg.seed, it)
        phase = stage(it, t.warmup_repr, t.warmup_pred)
        opt.param_groups[0]["lr"] = t.lr_enc if it <= t.warmup_repr else t.lr_enc_later
        obs, fut = _batch(train_ds, t.batch, device)
        h, tgt = encoder(prepare(obs, res)), encoder(prepare(fut, res))
        loss = t.lambda_reg * (visreg(h, cfg.visreg, it) + visreg(tgt, cfg.visreg, it)) / 2
        if phase != "repr":
            z = prior.sample(h.detach(), t.k_train)
            pred = predictor(h, z)
            loss = loss + energy_score(pred, tgt.detach())
            if phase == "full":
                cost = loo_cost(pred, tgt.detach())
                loss = loss + prior_score_loss(prior.log_prob(h.detach(), z), cost)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for module, _ in groups.values():
            torch.nn.utils.clip_grad_norm_(module.parameters(), t.clip)
        opt.step()
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
    _seed(cfg.seed, READOUT_SEED_OFFSET)  # readout init must not depend on whether the loop above ran
    readout = TranslationReadout(cfg.encoder.d).to(device)
    readout_path = out_dir / "readout.pt"
    if readout_path.exists():
        readout.load_state_dict(torch.load(readout_path, map_location=device, weights_only=True))
    else:
        r = cfg.readout
        ropt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        for it in range(1, r.iters + 1):
            _seed(cfg.seed, READOUT_SEED_OFFSET + it)
            obs, fut = _batch(train_ds, r.batch, device)
            with torch.no_grad():
                h, tgt = encoder(prepare(obs, res)), encoder(prepare(fut, res))
            rloss = readout_loss(render(obs[:, 2], readout(h, tgt)), fut)
            ropt.zero_grad(set_to_none=True)
            rloss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), r.clip)
            ropt.step()
            if it % 50 == 0:
                log.log({"readout_loss": rloss.item(), "readout_it": it})
        _save(readout.state_dict(), readout_path)

    _seed(cfg.seed, EVAL_SEED_OFFSET)
    test_obs = StochasticMovingMNIST("test", cfg.dataset.root, n_obs=cfg.dataset.n_test).unique_obs()
    obs = test_obs[: cfg.eval.n_obs].to(device)
    k = cfg.eval.k_eval
    with torch.no_grad():
        h = encoder(prepare(obs, res))
        pred = predictor(h, prior.sample(h, k))  # (B, K, d)
        disp = readout(h.repeat_interleave(k, 0), pred.reshape(-1, pred.shape[-1])).reshape(len(obs), k, 3, 2)
    per_obs = [mmnist_metrics(d) for d in disp.cpu().double().numpy()]
    metrics = {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True, indent=2))
    log.log(metrics)
    log.finish()
    return metrics


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(run(cfg, Path(HydraConfig.get().runtime.output_dir)))


if __name__ == "__main__":
    main()
