"""Classical twin of HQViT: same patching, block stack, pooling and temporal fusion; only attention differs.

tokens="frame": each frame is encoded separately and fused after pooling (HQViT paper: one image per circuit).
tokens="spacetime": the patches of all frames form one token set, so attention compares patches across frames
(HQViT: the index register runs over (frame, patch)); a sinusoidal frame embedding marks time, and tokens are
still pooled per frame (mean, mean + soft-argmax, or flatten) before the shared fusion head.
Patch layout matches quantum/attention.py: patch i row-major over the grid, pixel l row-major in the patch.

With pool="mean" and pos_emb="none" the encoder is invariant to permuting patches, exactly like HQViT's
I_idx ⊗ W ansatz followed by mean pooling; pool="flatten" or a positional embedding restore patch positions.
pool="softargmax" (phase-2 spec §5.6) appends, per channel, the soft-argmax (x, y) over the patch grid to the mean:
mean pooling alone dilutes a small object's position among empty patches. It acts after the attention blocks,
so HQViT's circuits are unchanged.
The embedding is added to the pixel tokens before block 1, so HQViT amplitude-encodes the same tokens.
"""

import math
from collections.abc import Callable

import torch
from torch import nn


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2-D sinusoidal embedding (grid*grid, dim), row-major token order; half the channels per axis."""
    if dim % 4:
        raise ValueError(f"sincos embedding needs dim divisible by 4, got {dim}")
    q = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(q) / q)
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")

    def enc(pos: torch.Tensor) -> torch.Tensor:
        a = pos.reshape(-1, 1).float() * omega
        return torch.cat([a.sin(), a.cos()], 1)

    return torch.cat([enc(rows), enc(cols)], 1)


def sincos_1d(n: int, dim: int) -> torch.Tensor:
    """Fixed 1-D sinusoidal embedding (n, dim)."""
    omega = 1.0 / 10000 ** (torch.arange(dim // 2) / (dim // 2))
    a = torch.arange(n).reshape(-1, 1).float() * omega
    return torch.cat([a.sin(), a.cos()], 1)


def soft_argmax(t: torch.Tensor, grid: int) -> torch.Tensor:
    """(N, grid*grid, C) row-major tokens -> (N, 2C): per channel, softmax over tokens, then expected (x, y) in [-1, 1]."""
    lin = torch.linspace(-1.0, 1.0, grid, device=t.device, dtype=t.dtype)
    ys, xs = torch.meshgrid(lin, lin, indexing="ij")
    w = t.softmax(1)  # (N, T, C)
    x = (w * xs.reshape(1, -1, 1)).sum(1)
    y = (w * ys.reshape(1, -1, 1)).sum(1)
    return torch.cat([x, y], 1)


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
        pos_emb: str = "none",
        frames: int = 3,
        tokens: str = "frame",
        block: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        if pool not in ("mean", "flatten", "softargmax"):
            raise ValueError(f"unknown pool {pool!r}")
        if res % patch:
            raise ValueError(f"res {res} is not a multiple of patch {patch}")
        if tokens not in ("frame", "spacetime"):
            raise ValueError(f"unknown tokens {tokens!r}")
        self.patch, self.pool, self.spacetime, self.frames = patch, pool, tokens == "spacetime", frames
        self.grid = res // patch
        tokens, dim = (res // patch) ** 2, patch * patch
        self.register_buffer("time", sincos_1d(frames, dim)[:, None] if self.spacetime else None)
        if pos_emb == "sincos":
            self.register_buffer("pos", sincos_2d(res // patch, dim))
        elif pos_emb == "learned":
            self.pos = nn.Parameter(torch.zeros(tokens, dim))
        elif pos_emb == "none":
            self.pos = None
        else:
            raise ValueError(f"unknown pos_emb {pos_emb!r}")
        self.blocks = nn.ModuleList((block or Block)(dim, ffn_hidden or dim) for _ in range(depth))
        per_frame = {"mean": dim, "softargmax": 3 * dim, "flatten": tokens * dim}[pool]
        # Temporal fusion: same head widths as the CNN encoder (Q-JEPA App. A.2).
        self.head = nn.Sequential(
            nn.Linear(frames * per_frame, 96), nn.SiLU(), nn.Linear(96, d), nn.LayerNorm(d)
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = to_patches(x, self.patch)  # (B, frames, T, dim)
        if self.pos is not None:
            t = t + self.pos
        if self.spacetime:
            t = (t + self.time).flatten(1, 2)  # (B, frames*T, dim): attention across frames
        else:
            t = t.flatten(0, 1)  # (B*frames, T, dim)
        for block in self.blocks:
            t = block(t)
        if self.spacetime:
            t = t.reshape(b * self.frames, -1, t.shape[-1])  # per-frame pooling keeps frame order
        if self.pool == "mean":
            t = t.mean(1)
        elif self.pool == "softargmax":
            t = torch.cat([t.mean(1), soft_argmax(t, self.grid)], 1)
        else:
            t = t.flatten(1)
        return self.head(t.reshape(b, -1))
