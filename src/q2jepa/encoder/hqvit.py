"""HQViT encoder (HQViT §4, phase-2 spec §5.4): the twin's skeleton with swap-test quantum attention.

Everything except `attend` is inherited from the classical twin (`encoder/vit.py`), so patching, positional and
time embeddings, FFN, LayerNorm, pooling and fusion are identical by construction (phase-2 F4).

Per block, the tokens of each clip are amplitude-encoded into one register (whole-clip normalization, HQViT Eq. 5;
the token count is zero-padded to a power of two) and evolved by I_idx ⊗ W(θ) for Q, K and V in CUDA-Q
(`quantum.bridge.RegisterEvolve`, exact gradients). With q_i = √w_i q̂_i:
- ACM_ij = |⟨q_i|k_j⟩|² = w_i w_j |⟨q̂_i|k̂_j⟩|² (HQViT Eq. 22; equal to the swap-test probabilities, see
  tests/test_attention_equivalence.py);
- V_il = |v_il|², the probability of measuring (i, l) on the V register;
- output = softmax(ACM)·V over real tokens. ACM and V are rescaled by fixed constants (T², T·d with T the real,
  unpadded token count) so their entries are O(1) — information-free, but a choice the paper doesn't state.

θ starts near zero, where W is the fixed CNOT-ring permutation of the pixel basis (not the identity): ACM starts
as patch-content similarity (overlaps are permutation-invariant), while V starts pixel-permuted. θ is drawn from
its own generator, so at the same seed every shared parameter is initialized exactly like the twin's (F6 pairing).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from q2jepa.encoder.vit import Block, ViTEncoder
from q2jepa.quantum.bridge import RegisterEvolve

THETA_INIT_STD = 0.1


def n_ansatz_params(m: int, depth: int) -> int:
    return 3 * m * depth + m  # same layout as quantum/attention.py


class QuantumBlock(Block):
    def __init__(self, dim: int, hidden: int, ansatz_depth: int = 2):
        super().__init__(dim, hidden)
        del self.q, self.k, self.v
        self.m, self.ansatz_depth = int(math.log2(dim)), ansatz_depth
        if 2**self.m != dim:
            raise ValueError(f"token dim {dim} must be a power of two (patch*patch pixels on m qubits)")
        n = n_ansatz_params(self.m, ansatz_depth)
        self.theta_q, self.theta_k, self.theta_v = (nn.Parameter(torch.zeros(n)) for _ in range(3))  # see encoder

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        n, t, dim = x.shape
        tp = 1 << (t - 1).bit_length()
        flat = F.pad(x, (0, 0, 0, tp - t)).reshape(n, tp * dim)
        amp = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)  # amplitude encoding of each clip
        q, k, v = (
            RegisterEvolve.apply(amp, th, self.m, self.ansatz_depth).reshape(n, tp, dim)
            for th in (self.theta_q, self.theta_k, self.theta_v)
        )
        acm = (q.conj() @ k.transpose(-2, -1)).abs() ** 2 * t**2
        acm = acm.masked_fill(torch.arange(tp, device=x.device) >= t, float("-inf"))  # padded keys
        values = v.abs() ** 2 * (t * dim)
        return (acm.softmax(-1) @ values)[:, :t].to(x.dtype)


class HQViTEncoder(ViTEncoder):
    def __init__(self, *args, ansatz_depth: int = 2, **kwargs):
        super().__init__(*args, block=lambda dim, hidden: QuantumBlock(dim, hidden, ansatz_depth), **kwargs)
        # QuantumBlock still builds (then drops) the twin's Q/K/V layers, so the global RNG advances exactly as in
        # the twin; θ comes from a separate generator keyed on the current seed.
        gen = torch.Generator().manual_seed(torch.initial_seed() + 7919)
        with torch.no_grad():
            for blk in self.blocks:
                for th in (blk.theta_q, blk.theta_k, blk.theta_v):
                    th.copy_(THETA_INIT_STD * torch.randn(th.shape, generator=gen))
