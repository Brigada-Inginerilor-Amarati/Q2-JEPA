"""Q-JEPA objectives: energy score (Eq. 7), LOO cost (Eq. 12), VISReg (App. A.3), prior update (Eq. 13)."""

import math

import torch

from q2jepa.config import VISRegCfg


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with value 0 and gradient 0 at x == 0 (collapsed predictions / constant features)."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), torch.zeros_like(x))


def _rho(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _safe_sqrt(((u - v) ** 2).sum(-1)) / math.sqrt(u.shape[-1])


def _pairwise(pred: torch.Tensor) -> torch.Tensor:
    """(B, K, K) rho between predictions, zero diagonal."""
    k = pred.shape[1]
    return _rho(pred[:, :, None], pred[:, None]) * (1 - torch.eye(k, device=pred.device))


def energy_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    k = pred.shape[1]
    fidelity = _rho(pred, target[:, None]).mean(1)
    diversity = _pairwise(pred).sum((1, 2)) / (2 * k * (k - 1))
    return (fidelity - diversity).mean()


def loo_cost(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    k = pred.shape[1]
    return _rho(pred, target[:, None]) - _pairwise(pred).sum(2) / (k - 1)


def visreg_terms(v: torch.Tensor, n_proj: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, d = v.shape
    mu = v.mean(0)
    s = _safe_sqrt(v.var(0, unbiased=False))  # population std
    center = (mu**2).sum() / d
    scale = ((s - 1) ** 2).sum() / d
    ev = (v - mu) / (s.detach() + 1e-6)
    w = torch.randn(d, n_proj, device=v.device, dtype=v.dtype)
    proj, _ = torch.sort(ev @ (w / w.norm(dim=0)), dim=0)  # (B, J)
    q = torch.special.ndtri(torch.arange(1, b + 1, device=v.device, dtype=v.dtype) / (b + 1))
    shape = ((proj - q[:, None]) ** 2).mean()
    return center, scale, shape


def visreg(v: torch.Tensor, cfg: VISRegCfg, it: int) -> torch.Tensor:
    center, scale, shape = visreg_terms(v, cfg.n_proj)
    return cfg.w_center * center + cfg.w_scale * scale + min(1.0, it / cfg.shape_ramp) * shape


def prior_score_loss(log_prob: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    return (log_prob * cost.detach()).mean()
