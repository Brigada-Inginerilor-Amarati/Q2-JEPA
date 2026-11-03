import torch
import torch.nn.functional as F
from torch import nn


def prepare(x: torch.Tensor, res: int, norm: str = "none", scale: float = 1.0) -> torch.Tensor:
    """Downsample encoder input only; data, readout and metrics stay 64x64.

    norm="l2" rescales each frame to unit RMS: the information an amplitude-encoded frame keeps (overall
    brightness is lost), up to a constant factor `scale` (information-free; it only sets the input magnitude).
    Use it for every encoder in a fair comparison with HQViT.
    """
    x = x if x.shape[-1] == res else F.interpolate(x, size=(res, res), mode="area")
    if norm == "l2":
        return x * (scale * res / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    if norm != "none":
        raise ValueError(f"unknown input norm {norm!r}")
    return x


def build_encoder(cfg) -> nn.Module:
    from q2jepa.encoder.cnn import CNNEncoder
    from q2jepa.encoder.vit import ViTEncoder

    if cfg.name == "cnn":
        return CNNEncoder(cfg.res, cfg.d)
    if cfg.name == "hqvit":
        from q2jepa.encoder.hqvit import HQViTEncoder

        return HQViTEncoder(
            cfg.res, cfg.d, cfg.patch, cfg.depth, cfg.ffn_hidden, cfg.pool, cfg.pos_emb,
            tokens=cfg.tokens, ansatz_depth=cfg.ansatz_depth,
        )  # fmt: skip
    if cfg.name == "vit":
        return ViTEncoder(
            cfg.res, cfg.d, cfg.patch, cfg.depth, cfg.ffn_hidden, cfg.pool, cfg.pos_emb, tokens=cfg.tokens
        )
    raise ValueError(f"unknown encoder {cfg.name!r}")
