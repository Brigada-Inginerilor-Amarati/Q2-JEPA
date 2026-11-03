"""Conditional Born-machine priors (Q-JEPA §3.2, Eqs. 3-4, 14; App. A.4, C).

alpha_lj(h) = g_eta(h)_lj + theta_lj + (pi/2) 1[l = 1] for the L body RY layers, then terminal RY(phi_j) shared
across observations; p(z | h) = |<z|U|0^n>|^2. g_eta is a two-layer SiLU MLP with a zero-initialized output, so
the initial prior is input-independent with broad support.

Prior updates (`quantum.grad`):
- "shift": measurement-based parameter shift (Eq. 14) with b-of-B observation and r-of-G angle subsampling,
  shots per shifted circuit and the readout-flip channel; never differentiates a state. Reported rows use this.
- "ideal": score-function update (Eq. 13) with exact Born log-probabilities differentiated through CUDA-Q.
"""

import math
from collections.abc import Callable

import torch
from torch import nn

from q2jepa.prior.codes import bits_to_index
from q2jepa.quantum import bridge
from q2jepa.quantum.born import brickwork_layers


def _seed() -> int:
    return int(torch.randint(2**31 - 1, ()))  # follows the per-iteration torch seed, so resume is exact


class QCBMPrior(nn.Module):
    def __init__(
        self,
        d: int = 64,
        hidden: int = 48,
        n_bits: int = 3,
        layers: int = 2,
        entangled: bool = True,
        trainable: bool = True,
        grad: str = "shift",
        shots: int = 256,
        flip: float = 0.005,
        shift_obs: int = 8,
        shift_coords: int = 9,
        offset_init: float = 0.1,
    ):
        super().__init__()
        if grad not in ("shift", "ideal"):
            raise ValueError(f"unknown quantum.grad {grad!r}")
        if grad == "ideal" and flip > 0:
            # Flipped codes scored with noise-free Born log-probs bias the score-function gradient (App. D).
            raise ValueError("quantum.grad=ideal is the noise-free reference: set quantum.readout_flip=0")
        self.n_bits, self.layers, self.trainable, self.grad = n_bits, layers, trainable, grad
        self.shots, self.flip, self.shift_obs, self.shift_coords = shots, flip, shift_obs, shift_coords
        self.cnots = brickwork_layers(n_bits, layers) if entangled else [[] for _ in range(layers)]
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, layers * n_bits))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.offset = nn.Parameter(offset_init * torch.randn(layers, n_bits))  # theta
        self.terminal = nn.Parameter(torch.zeros(n_bits))  # phi
        if not trainable:
            self.requires_grad_(False)

    def angles(self, h: torch.Tensor) -> torch.Tensor:
        """(B, L+1, n)."""
        body = self.net(h).reshape(-1, self.layers, self.n_bits) + self.offset
        body = body + torch.tensor([math.pi / 2] + [0.0] * (self.layers - 1), device=h.device)[:, None]
        return torch.cat([body, self.terminal.expand(body.shape[0], 1, self.n_bits)], 1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, k: int) -> torch.Tensor:
        return bridge.sample(self.angles(h), self.cnots, k, self.flip, _seed())

    def log_prob(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Exact, noise-free Born log-probabilities (ideal reference and diagnostics)."""
        p = bridge.BornProbs.apply(self.angles(h), self.cnots)
        return p.gather(1, bits_to_index(z)).clamp_min(1e-12).log()

    def shift_surrogate(
        self, h: torch.Tensor, cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Zero-valued loss whose gradient is the Eq. 14 estimate of d L_pred / d(prior params).

        cost(obs_idx (b,), codes (b, M, n)) -> (b, M) predictive cost C(P(h, z); t, Q) with fixed predictor, target
        and reference predictions. Angle gradients for the b sampled observations and r sampled coordinates are
        scaled by G/r (unsampled coordinates) and averaged over b (unsampled observations; the loss is a batch
        mean), then backpropagated through the conditioner, offsets and terminal angles.
        """
        big_b, g = h.shape[0], (self.layers + 1) * self.n_bits
        b, r = min(self.shift_obs, big_b), min(self.shift_coords, g)
        obs = torch.randperm(big_b, device=h.device)[:b]
        coords = torch.stack([torch.randperm(g)[:r] for _ in range(b)])
        a = self.angles(h[obs])
        codes = bridge.sample_shifted(a, coords, self.cnots, self.shots, self.flip, _seed())  # (b, r, 2, S, n)
        with torch.no_grad():
            c = cost(obs, codes.reshape(b, -1, self.n_bits)).reshape(b, r, 2, self.shots).mean(-1)
            est = 0.5 * (c[..., 0] - c[..., 1]) * g / r  # (b, r)
            grad = (
                torch.zeros(b, g, dtype=a.dtype, device=h.device).scatter(1, coords.to(h.device), est.to(a.dtype)) / b
            )
        s = (a.reshape(b, -1) * grad).sum()
        return s - s.detach()


def build_qcbm(cfg, d: int, quantum) -> QCBMPrior:
    kinds = {"entangled_qcbm": (True, True), "product_qcbm": (False, True), "frozen_qcbm": (True, False)}
    entangled, trainable = kinds[cfg.name]
    return QCBMPrior(
        d, cfg.hidden, cfg.n_bits, cfg.layers, entangled, trainable, quantum.grad, quantum.shots,
        quantum.readout_flip, quantum.shift_obs, quantum.shift_coords, cfg.offset_init,
    )  # fmt: skip
