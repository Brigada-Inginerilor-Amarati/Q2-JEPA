"""torch <-> CUDA-Q glue for the conditional Born prior (Q-JEPA §3.4, App. A.4).

Angles have shape (B, L+1, n): rows 0..L-1 body RY layers, row L terminal RY (born.py convention).
Every circuit runs through CUDA-Q; torch only sees probabilities, gradients and measured codes.
"""

import numpy as np
import torch

from q2jepa.quantum.born import apply_readout_flips, born_probs, sample_codes

SHIFT = np.pi / 2  # each angle drives one RY = exp(-i a Y / 2): the two-term shift rule is exact


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


class BornProbs(torch.autograd.Function):
    """p(z | angles), (B, 2**n) little-endian; backward = exact parameter shift on probabilities via CUDA-Q.

    Used only by the ideal-gradient reference mode (App. A.4); the reported mode never differentiates states.
    """

    @staticmethod
    def forward(ctx, angles: torch.Tensor, cnot_layers) -> torch.Tensor:
        ctx.save_for_backward(angles)
        ctx.cnot_layers = cnot_layers
        p = np.stack([born_probs(a, cnot_layers) for a in _np(angles)])
        return torch.as_tensor(p, dtype=angles.dtype, device=angles.device)

    @staticmethod
    def backward(ctx, grad_p: torch.Tensor):
        (angles,) = ctx.saved_tensors
        a, g = _np(angles), _np(grad_p)
        out = np.zeros_like(a)
        for b in range(a.shape[0]):
            for idx in np.ndindex(a.shape[1:]):
                plus, minus = a[b].copy(), a[b].copy()
                plus[idx] += SHIFT
                minus[idx] -= SHIFT
                dp = 0.5 * (born_probs(plus, ctx.cnot_layers) - born_probs(minus, ctx.cnot_layers))
                out[(b, *idx)] = g[b] @ dp
        return torch.as_tensor(out, dtype=angles.dtype, device=angles.device), None


def sample(angles: torch.Tensor, cnot_layers, shots: int, flip: float, seed: int) -> torch.Tensor:
    """Measured codes (B, shots, n) as float {0, 1}, with independent readout flips of probability `flip`."""
    rng = np.random.default_rng(seed)
    bits = sample_codes(_np(angles), cnot_layers, shots, rng, seed=seed)
    if flip > 0:
        bits = apply_readout_flips(bits, flip, rng)
    return torch.as_tensor(bits, dtype=torch.float32, device=angles.device)


def sample_shifted(
    angles: torch.Tensor, coords: torch.Tensor, cnot_layers, shots: int, flip: float, seed: int
) -> torch.Tensor:
    """Codes from the ±π/2-shifted circuits of Eq. 14.

    angles (b, L+1, n); coords (b, r) flat angle indices to shift, per observation.
    Returns (b, r, 2, shots, n): [..., 0, :, :] from the +π/2 shift, [..., 1, :, :] from −π/2.
    """
    b, r = coords.shape
    shifted = angles.detach()[:, None, None].expand(b, r, 2, *angles.shape[1:]).clone()
    flat = shifted.reshape(b, r, 2, -1)
    rows, cols = torch.arange(b)[:, None], torch.arange(r)[None]
    flat[rows, cols, 0, coords] += SHIFT
    flat[rows, cols, 1, coords] -= SHIFT
    codes = sample(flat.reshape(-1, *angles.shape[1:]), cnot_layers, shots, flip, seed)
    return codes.reshape(b, r, 2, shots, angles.shape[-1])
