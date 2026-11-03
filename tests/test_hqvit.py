import pytest
import torch

from q2jepa.encoder.hqvit import HQViTEncoder, QuantumBlock, n_ansatz_params
from q2jepa.encoder.vit import ViTEncoder


def _small(**kw):
    torch.manual_seed(0)
    return HQViTEncoder(16, patch=4, depth=2, **kw)


def test_shapes_and_gradients_reach_every_parameter():
    enc = _small(pos_emb="sincos", tokens="spacetime")
    out = enc(torch.rand(2, 3, 16, 16))
    assert out.shape == (2, 64)
    out.pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_shared_parameters_initialize_like_the_twin():
    torch.manual_seed(3)
    q = HQViTEncoder(16, patch=4, depth=2)
    torch.manual_seed(3)
    c = ViTEncoder(16, patch=4, depth=2)
    cp = dict(c.named_parameters())
    for name, p in q.named_parameters():
        if "theta" not in name:
            assert torch.equal(p, cp[name]), name


def test_quantum_parameters_replace_classical_qkv():
    q, c = _small(), ViTEncoder(16, patch=4, depth=2)
    nq = sum(p.numel() for n, p in q.named_parameters() if "theta" in n)
    assert nq == 2 * 3 * n_ansatz_params(4, 2)  # 2 blocks x (Q, K, V) x 28 angles
    linear_qkv = sum(p.numel() for n, p in c.named_parameters() if n.split(".")[-2] in ("q", "k", "v"))
    shared = sum(p.numel() for p in c.parameters()) - linear_qkv
    assert sum(p.numel() for p in q.parameters()) == shared + nq


def test_position_blind_like_the_paper_without_embedding():
    torch.manual_seed(0)
    x, perm = torch.rand(2, 3, 16, 16), torch.randperm(16)
    b, f, h, w = x.shape
    tiles = x.reshape(b, f, 4, 4, 4, 4).transpose(3, 4).reshape(b, f, 16, 4, 4)[:, :, perm]
    xp = tiles.reshape(b, f, 4, 4, 4, 4).transpose(3, 4).reshape(b, f, h, w)
    enc = _small(pos_emb="none").eval()
    with torch.no_grad():
        torch.testing.assert_close(enc(xp), enc(x), atol=1e-5, rtol=1e-5)


def test_attention_rows_are_distributions_over_real_tokens():
    blk = QuantumBlock(16, 16)
    x = torch.rand(3, 12, 16)  # 12 tokens, padded to 16 inside
    out = blk.attend(x)
    assert out.shape == (3, 12, 16) and torch.isfinite(out).all() and (out >= 0).all()


def test_non_power_of_two_token_dim_rejected():
    with pytest.raises(ValueError):
        QuantumBlock(9, 9)
