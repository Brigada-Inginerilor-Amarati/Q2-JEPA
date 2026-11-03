import numpy as np
import pytest
import torch

from q2jepa.encoder import build_encoder, prepare
from q2jepa.encoder.vit import ViTEncoder, to_patches
from q2jepa.quantum.attention import image_to_patches


def _permute_patches(x: torch.Tensor, perm: torch.Tensor, p: int = 4) -> torch.Tensor:
    b, f, h, w = x.shape
    tiles = x.reshape(b, f, h // p, p, w // p, p).transpose(3, 4).reshape(b, f, -1, p, p)[:, :, perm]
    return tiles.reshape(b, f, h // p, w // p, p, p).transpose(3, 4).reshape(b, f, h, w)


def test_to_patches_matches_quantum_layout():
    img = torch.arange(256.0).reshape(1, 1, 16, 16)
    patches = to_patches(img, 4)  # (1, 1, 16, 16): patch i row-major over the 4x4 grid, pixel l row-major in the patch
    assert patches.shape == (1, 1, 16, 16)
    assert torch.equal(patches[0, 0, 1], img[0, 0, 0:4, 4:8].reshape(-1))
    assert torch.equal(patches[0, 0, 4], img[0, 0, 4:8, 0:4].reshape(-1))
    for res in (16, 32):
        x = np.random.default_rng(res).random((res, res))
        np.testing.assert_array_equal(to_patches(torch.from_numpy(x), 4).numpy(), image_to_patches(x, 4))


@pytest.mark.parametrize("res,patch", [(16, 4), (32, 4), (64, 8)])
@pytest.mark.parametrize("pool", ["mean", "flatten"])
def test_vit_shapes(res, patch, pool):
    enc = ViTEncoder(res, patch=patch, pool=pool, pos_emb="sincos")
    assert enc(torch.rand(5, 3, res, res)).shape == (5, 64)


def test_mean_pool_without_positions_ignores_patch_layout():
    # The HQViT design (I_idx ⊗ W ansatz + mean pool) has this property; the twin must reproduce it.
    torch.manual_seed(0)
    x, perm = torch.rand(2, 3, 16, 16), torch.randperm(16)
    enc = ViTEncoder(16, pool="mean", pos_emb="none").eval()
    torch.testing.assert_close(enc(_permute_patches(x, perm)), enc(x))


@pytest.mark.parametrize("kwargs", [{"pool": "flatten"}, {"pos_emb": "learned"}, {"pos_emb": "sincos"}])
def test_positional_variants_see_patch_layout(kwargs):
    torch.manual_seed(0)
    x, perm = torch.rand(2, 3, 16, 16), torch.randperm(16)
    enc = ViTEncoder(16, **kwargs).eval()
    if kwargs.get("pos_emb") == "learned":
        torch.nn.init.normal_(enc.pos, std=1.0)  # zero init would hide the embedding
    assert not torch.allclose(enc(_permute_patches(x, perm)), enc(x))


def test_l2_input_norm_is_scale_free():
    x = torch.rand(4, 3, 64, 64)
    a, b = prepare(x, 16, "l2"), prepare(3.7 * x, 16, "l2")
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a.pow(2).mean((-2, -1)), torch.ones(4, 3))  # unit RMS per frame
    assert torch.equal(prepare(x, 16), prepare(x, 16, "none"))
    torch.testing.assert_close(prepare(x, 16, "l2", 0.25), 0.25 * a)


def test_ffn_hidden_defaults_to_patch_dim_and_res_checked():
    assert ViTEncoder(32, patch=8).blocks[0].ffn[0].out_features == 64
    with pytest.raises(ValueError):
        ViTEncoder(18, patch=4)


def test_sincos_is_fixed_and_distinct_per_token():
    enc = ViTEncoder(64, patch=8, pos_emb="sincos")
    assert enc.pos.shape == (64, 64) and not enc.pos.requires_grad
    assert "pos" not in dict(enc.named_parameters())
    assert torch.cdist(enc.pos, enc.pos).add(torch.eye(64)).min() > 0  # every patch position is distinguishable


def test_spacetime_tokens_attend_across_frames_and_keep_time_order():
    torch.manual_seed(0)
    enc = ViTEncoder(16, pos_emb="sincos", tokens="spacetime").eval()
    x = torch.rand(2, 3, 16, 16)
    assert enc(x).shape == (2, 64)
    assert not torch.allclose(enc(x.flip(1)), enc(x))  # frame order matters
    y = x.clone()
    y[:, 0] = torch.rand(2, 16, 16)  # change frame 0 only: frame-2 tokens see it through attention
    frame_only = ViTEncoder(16, pos_emb="sincos").eval()
    with torch.no_grad():

        def per_frame(e, v):
            t = to_patches(v, 4) + e.pos
            t = (t + e.time).flatten(1, 2) if e.spacetime else t.flatten(0, 1)
            for blk in e.blocks:
                t = blk(t)
            return t.reshape(2, 3, -1, 16)[:, 2]

        assert not torch.allclose(per_frame(enc, y), per_frame(enc, x))
        assert torch.allclose(per_frame(frame_only, y), per_frame(frame_only, x))


def test_build_vit():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        name="vit", res=16, d=64, patch=4, depth=3, ffn_hidden=0, pool="mean", pos_emb="none", tokens="frame"
    )
    assert isinstance(build_encoder(cfg), ViTEncoder)
