"""Classical twin of HQViT: same patching, block stack, pooling and temporal fusion; only attention differs.

Each frame is encoded separately (HQViT amplitude-encodes one image per circuit) and fused after pooling.
Patch layout matches quantum/attention.py: patch i row-major over the grid, pixel l row-major in the patch.

With pool="mean" and pos_emb=False the encoder is invariant to permuting patches, exactly like HQViT's
I_idx ⊗ W ansatz followed by mean pooling; pool="flatten" or pos_emb=True restore patch positions.
"""

import math

import torch
from torch import nn


def to_patches(x: torch.Tensor, p: int) -> torch.Tensor:
    """(..., H, W) -> (..., T, p*p)."""
    *lead, h, w = x.shape
    x = x.reshape(*lead, h // p, p, w // p, p).transpose(-3, -2)
    return x.reshape(*lead, (h // p) * (w // p), p * p)


class Block(nn.Module):
    """Single-head post-norm transformer block; HQViT replaces `attend` with swap-test attention."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-2, -1) / math.sqrt(x.shape[-1])
        return scores.softmax(-1) @ self.v(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attend(x))
        return self.ln2(x + self.ffn(x))


class ViTEncoder(nn.Module):
    def __init__(
        self,
        res: int,
        d: int = 64,
        patch: int = 4,
        depth: int = 3,
        ffn_hidden: int | None = None,
        pool: str = "mean",
        pos_emb: bool = False,
        frames: int = 3,
    ):
        super().__init__()
        if pool not in ("mean", "flatten"):
            raise ValueError(f"unknown pool {pool!r}")
        if res % patch:
            raise ValueError(f"res {res} is not a multiple of patch {patch}")
        self.patch, self.pool = patch, pool
        tokens, dim = (res // patch) ** 2, patch * patch
        self.pos = nn.Parameter(torch.zeros(tokens, dim)) if pos_emb else None
        self.blocks = nn.ModuleList(Block(dim, ffn_hidden or dim) for _ in range(depth))
        per_frame = dim if pool == "mean" else tokens * dim
        # Temporal fusion: same head widths as the CNN encoder (Q-JEPA App. A.2).
        self.head = nn.Sequential(
            nn.Linear(frames * per_frame, 96), nn.SiLU(), nn.Linear(96, d), nn.LayerNorm(d)
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = to_patches(x, self.patch).flatten(0, 1)  # (B*frames, T, dim)
        if self.pos is not None:
            t = t + self.pos
        for block in self.blocks:
            t = block(t)
        t = t.mean(1) if self.pool == "mean" else t.flatten(1)
        return self.head(t.reshape(b, -1))
