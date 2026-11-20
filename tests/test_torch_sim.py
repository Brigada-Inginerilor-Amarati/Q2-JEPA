"""The torch training simulator equals the CUDA-Q circuit (quantum/hqvit.py) on states and gradients."""

import numpy as np
import pytest
import torch

from q2jepa.encoder.hqvit import HQViTEncoder, QuantumBlock, n_ansatz_params
from q2jepa.quantum import is_fp32
from q2jepa.quantum.hqvit import evolve
from q2jepa.quantum.torch_sim import ansatz_matrix

TOL = {"atol": 1e-5, "rtol": 1e-5} if is_fp32() else {"atol": 1e-10, "rtol": 1e-10}


@pytest.mark.parametrize(("m", "depth", "tokens"), [(1, 1, 1), (2, 2, 4), (4, 2, 2), (6, 2, 4)])
def test_register_evolution_matches_cudaq(m, depth, tokens):
    rng = np.random.default_rng(m)
    theta = rng.uniform(-np.pi, np.pi, n_ansatz_params(m, depth))
    x = rng.normal(size=(3, tokens * 2**m))
    ref = evolve(x, theta, m, depth)
    mat = ansatz_matrix(torch.tensor(theta), m, depth)
    out = (torch.tensor(x, dtype=torch.complex128).reshape(3, tokens, 2**m) @ mat).reshape(3, -1)
    torch.testing.assert_close(out, torch.as_tensor(ref).to(torch.complex128), **TOL)


def test_matrix_is_unitary():
    mat = ansatz_matrix(torch.randn(n_ansatz_params(3, 2), dtype=torch.float64), 3, 2)
    torch.testing.assert_close(mat.conj().T @ mat, torch.eye(8, dtype=torch.complex128))


def test_block_backends_agree_on_output_and_gradients():
    torch.manual_seed(0)
    blk = QuantumBlock(16, 16).double()
    with torch.no_grad():
        for th in (blk.theta_q, blk.theta_k, blk.theta_v):
            th.normal_(0, 0.5)
    x = torch.rand(2, 5, 16, dtype=torch.float64)
    grads = {}
    for backend in ("cudaq", "torch"):
        blk.backend = backend
        blk.zero_grad()
        xb = x.clone().requires_grad_()
        out = blk.attend(xb)
        out.pow(2).sum().backward()
        grads[backend] = (out.detach(), xb.grad, *(t.grad.clone() for t in (blk.theta_q, blk.theta_k, blk.theta_v)))
    for a, b in zip(grads["cudaq"], grads["torch"], strict=True):
        torch.testing.assert_close(a, b, **({"atol": 1e-4, "rtol": 1e-4} if is_fp32() else {}))


def test_encoder_backend_switch():
    enc = HQViTEncoder(16, patch=4, depth=2)
    enc.set_backend("torch")
    assert all(b.backend == "torch" for b in enc.blocks)
    with pytest.raises(ValueError):
        enc.set_backend("numpy")
