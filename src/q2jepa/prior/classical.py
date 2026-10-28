"""Classical latent-action priors (Q-JEPA App. A.1). Interface: sample(h, k) and log_prob(h, z)."""

import torch
import torch.nn.functional as F
from torch import nn

from q2jepa.prior.codes import bits_to_index, index_to_bits


def _conditioner(d: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, out))


class SoftmaxCategorical(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 48, n_bits: int = 3):
        super().__init__()
        self.n_bits = n_bits
        self.net = _conditioner(d, hidden, 2**n_bits)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.distributions.Categorical(logits=self.net(h)).sample((k,)).T  # (B, k)
        return index_to_bits(idx, self.n_bits)

    def log_prob(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(h), -1).gather(1, bits_to_index(z))


class MixturePrior(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 48, n_bits: int = 3, components: int = 2):
        super().__init__()
        self.n_bits, self.components = n_bits, components
        self.net = _conditioner(d, hidden, components * (n_bits + 1))

    def _logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(h)
        r = self.components
        return out[:, :r], out[:, r:].reshape(-1, r, self.n_bits)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, k: int) -> torch.Tensor:
        comp_logits, bit_logits = self._logits(h)
        comp = torch.distributions.Categorical(logits=comp_logits).sample((k,)).T  # (B, k)
        chosen = bit_logits.gather(1, comp[..., None].expand(-1, -1, self.n_bits))  # (B, k, n)
        return torch.bernoulli(torch.sigmoid(chosen))

    def log_prob(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        comp_logits, bit_logits = self._logits(h)
        zb = z[:, :, None, :]  # (B, k, 1, n)
        lb = bit_logits[:, None]  # (B, 1, R, n)
        per_comp = (zb * F.logsigmoid(lb) + (1 - zb) * F.logsigmoid(-lb)).sum(-1)  # (B, k, R)
        return torch.logsumexp(F.log_softmax(comp_logits, -1)[:, None] + per_comp, -1)


def build_prior(cfg, d: int) -> nn.Module:
    if cfg.name == "softmax_categorical":
        return SoftmaxCategorical(d, cfg.hidden, cfg.n_bits)
    if cfg.name == "mixture":
        return MixturePrior(d, cfg.hidden, cfg.n_bits, cfg.components)
    raise ValueError(f"unknown prior {cfg.name!r}")
