"""Q-JEPA objectives: energy score (Eq. 7), LOO cost (Eq. 12), shifted cost (Eq. 11), VISReg (App. A.3), prior update (Eq. 13)."""

import math

import torch

from q2jepa.config import VISRegCfg


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with value 0 and gradient 0 at x == 0 (collapsed predictions / constant features)."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), torch.zeros_like(x))


def _rho(u: torch.Tensor, v: torch.Tensor, beta: float = 1.0, eps: float = 0.0) -> torch.Tensor:
    """rho_{beta,eps}(u, v) = (|u - v|^2 / d + eps^2)^(beta/2) - eps^beta (Eq. 18); beta = 1, eps = 0 is |u - v| / sqrt(d)."""
    if beta == 1.0 and eps == 0.0:
        return _safe_sqrt(((u - v) ** 2).sum(-1)) / math.sqrt(u.shape[-1])
    return (((u - v) ** 2).sum(-1) / u.shape[-1] + eps**2) ** (beta / 2) - eps**beta


def _to_targets(pred: torch.Tensor, target: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    """(B, K) rho from each prediction to its target (B, d), or the mean over several targets (B, J, d)."""
    if target.dim() == 2:
        return _rho(pred, target[:, None], beta, eps)
    return _rho(pred[:, :, None], target[:, None], beta, eps).mean(-1)


def _pairwise(pred: torch.Tensor, beta: float = 1.0, eps: float = 0.0) -> torch.Tensor:
    """(B, K, K) rho between predictions, zero diagonal."""
    k = pred.shape[1]
    return _rho(pred[:, :, None], pred[:, None], beta, eps) * (1 - torch.eye(k, device=pred.device))


def energy_score(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0, eps: float = 0.0) -> torch.Tensor:
    """Eq. 7 with distance rho_{beta,eps}; target (B, d) or several futures (B, J, d), averaged (App. F.3)."""
    k = pred.shape[1]
    fidelity = _to_targets(pred, target, beta, eps).mean(1)
    diversity = _pairwise(pred, beta, eps).sum((1, 2)) / (2 * k * (k - 1))
    return (fidelity - diversity).mean()


def loo_cost(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0, eps: float = 0.0) -> torch.Tensor:
    k = pred.shape[1]
    return _to_targets(pred, target, beta, eps) - _pairwise(pred, beta, eps).sum(2) / (k - 1)


def shift_cost(
    u: torch.Tensor, target: torch.Tensor, ref: torch.Tensor, beta: float = 1.0, eps: float = 0.0
) -> torch.Tensor:
    """Eq. 11 cost of predictions u (b, M, d): rho(u, t) - E_{V~Q} rho(u, V), Q = reference predictions (b, K, d)."""
    return _to_targets(u, target, beta, eps) - _rho(u[:, :, None], ref[:, None], beta, eps).mean(-1)


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
