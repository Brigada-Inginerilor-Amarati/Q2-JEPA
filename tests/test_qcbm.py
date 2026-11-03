import math

import pytest
import torch

from q2jepa.prior.codes import bits_to_index, index_to_bits
from q2jepa.prior.qcbm import QCBMPrior
from q2jepa.quantum import is_fp32
from q2jepa.quantum.bridge import BornProbs

TOL = 1e-6 if is_fp32() else 1e-12  # nvidia simulates in single precision

ALL_CODES = index_to_bits(torch.arange(8), 3)


def _prior(**kw) -> QCBMPrior:
    torch.manual_seed(0)
    return QCBMPrior(**{"flip": 0.0, **kw}).double()


def test_active_param_count_matches_paper():
    # Q-JEPA Table 7, Digits / Entangled-QCBM: 3,423 active (3,435 allocated incl. 12 unused body coordinates).
    assert sum(p.numel() for p in QCBMPrior().parameters()) == 3423


def test_initial_prior_is_input_independent_with_pi_half_first_layer():
    p = _prior()
    a = p.angles(torch.randn(4, 64, dtype=torch.float64))
    assert torch.allclose(a, a[:1].expand_as(a))
    torch.testing.assert_close(a[0, 0], math.pi / 2 + p.offset[0].detach())
    torch.testing.assert_close(a[0, -1], torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize("entangled", [True, False])
def test_log_prob_normalizes_and_matches_samples(entangled):
    p = _prior(entangled=entangled)
    with torch.no_grad():
        p.net[-1].bias.uniform_(-1, 1)  # move away from the near-product initialization
    h = torch.randn(2, 64, dtype=torch.float64)
    probs = p.log_prob(h, ALL_CODES.expand(2, 8, 3).double()).exp()
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, dtype=torch.float64), atol=TOL, rtol=0)
    z = p.sample(h, 20000)
    freq = torch.stack([torch.bincount(bits_to_index(z[b]), minlength=8) / 20000 for b in range(2)])
    assert (0.5 * (freq.double() - probs.detach()).abs().sum(-1) < 0.02).all()


def test_product_circuit_has_independent_bits():
    p = _prior(entangled=False)
    with torch.no_grad():
        p.net[-1].bias.uniform_(-1, 1)
    probs = p.log_prob(torch.randn(1, 64, dtype=torch.float64), ALL_CODES[None].double()).exp()[0].reshape(2, 2, 2)
    marg = [probs.sum(dims) for dims in ((1, 2), (0, 2), (0, 1))]  # [i, j, k] = bits 2, 1, 0 (index 4i + 2j + k)
    torch.testing.assert_close(probs, torch.einsum("i,j,k->ijk", *marg))


def test_born_probs_gradient_is_exact():
    a = torch.empty(2, 3, 3, dtype=torch.float64).uniform_(-math.pi, math.pi).requires_grad_()
    w = torch.randn(2, 8, dtype=torch.float64)
    (BornProbs.apply(a, [[(0, 1)], [(1, 2)]]) * w).sum().backward()
    (eps, tol), num = ((1e-2, 2e-2) if is_fp32() else (1e-6, 1e-5)), torch.zeros_like(a)  # fp32: larger FD step
    for idx in torch.cartesian_prod(*(torch.arange(s) for s in a.shape)):
        d = torch.zeros_like(a)
        d[tuple(idx)] = eps
        f = lambda x: (BornProbs.apply(x, [[(0, 1)], [(1, 2)]]) * w).sum()
        num[tuple(idx)] = (f(a.detach() + d) - f(a.detach() - d)) / (2 * eps)
    torch.testing.assert_close(a.grad, num, atol=tol, rtol=tol)


def test_parameter_shift_estimate_matches_exact_gradient():
    # Eq. 14 with all observations and coordinates and many shots == d/dtheta of mean_i sum_z p_i(z) c(z).
    torch.manual_seed(1)
    table = torch.randn(8, dtype=torch.float64)
    h = torch.randn(3, 64, dtype=torch.float64)
    shifted = _prior(shots=40000, shift_obs=3, shift_coords=9)
    exact = _prior()
    for p in (shifted, exact):
        with torch.no_grad():
            p.net[-1].bias.copy_(torch.linspace(-1, 1, 6))
    shifted.shift_surrogate(h, lambda idx, codes: table[bits_to_index(codes)]).backward()
    (exact.log_prob(h, ALL_CODES.expand(3, 8, 3).double()).exp() @ table).mean().backward()
    for (name, a), b in zip(shifted.named_parameters(), exact.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=0.02, rtol=0.05, msg=name)


def test_subsampled_parameter_shift_is_unbiased():
    # b < B observations and r < G coordinates with the B/b and G/r corrections: the average estimate is exact.
    torch.manual_seed(2)
    table = torch.randn(8, dtype=torch.float64)
    h = torch.randn(4, 64, dtype=torch.float64)
    sub, exact = _prior(shots=2000, shift_obs=2, shift_coords=3), _prior()
    for p in (sub, exact):
        with torch.no_grad():
            p.net[-1].bias.copy_(torch.linspace(-1, 1, 6))
    for _ in range(400):
        sub.shift_surrogate(h, lambda idx, codes: table[bits_to_index(codes)]).backward()
    (exact.log_prob(h, ALL_CODES.expand(4, 8, 3).double()).exp() @ table).mean().backward()
    for (name, a), b in zip(sub.named_parameters(), exact.parameters(), strict=True):
        torch.testing.assert_close(a.grad / 400, b.grad, atol=0.03, rtol=0.1, msg=name)


def test_shift_surrogate_is_zero_valued():
    p = _prior(shots=16)
    s = p.shift_surrogate(torch.randn(10, 64, dtype=torch.float64), lambda idx, codes: codes.sum(-1).double())
    assert s.item() == 0.0 and s.requires_grad


def test_frozen_has_no_trainable_parameters():
    assert not any(q.requires_grad for q in QCBMPrior(trainable=False).parameters())
