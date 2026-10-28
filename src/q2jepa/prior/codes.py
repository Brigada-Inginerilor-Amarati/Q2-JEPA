"""n-bit code <-> integer index. Bit j = qubit j (cudaq little-endian): idx = sum_j z_j 2^j."""

import torch


def index_to_bits(idx: torch.Tensor, n: int) -> torch.Tensor:
    return ((idx[..., None] >> torch.arange(n, device=idx.device)) & 1).float()


def bits_to_index(z: torch.Tensor) -> torch.Tensor:
    return (z.long() << torch.arange(z.shape[-1], device=z.device)).sum(-1)
