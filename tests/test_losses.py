import math

import torch

from q2jepa.config import VISRegCfg
from q2jepa.losses import energy_score, loo_cost, prior_score_loss, visreg, visreg_terms


def test_energy_score_known_values():
    t = torch.zeros(1, 4)
    assert energy_score(t[:, None].expand(1, 3, 4), t).item() == 0.0
    pred = torch.tensor([[[2.0, 0, 0, 0], [0, 2, 0, 0]]])
    # fidelity = 1, diversity = 0.5 * sqrt(8)/2 = sqrt(2)/2
    assert math.isclose(energy_score(pred, t).item(), 1 - math.sqrt(2) / 2, rel_tol=1e-6)


def test_loo_cost_shape_and_value():
    pred = torch.tensor([[[2.0, 0, 0, 0], [0, 2, 0, 0]]])
    c = loo_cost(pred, torch.zeros(1, 4))
    assert c.shape == (1, 2) and torch.allclose(c, torch.full((1, 2), 1 - math.sqrt(2)))


def test_visreg_gaussian_vs_collapsed():
    torch.manual_seed(0)
    c, s, sh = visreg_terms(torch.randn(4096, 64))
    assert c < 1e-3 and s < 1e-3 and sh < 1e-2
    _, s0, _ = visreg_terms(torch.ones(64, 64))
    assert math.isclose(s0.item(), 1.0, rel_tol=1e-5)


def test_shape_ramp():
    v = torch.randn(64, 64)
    a = visreg(v, VISRegCfg(), 0)
    c, s, _ = visreg_terms(v)
    assert torch.isclose(a, 1.0 * c + 10.0 * s, atol=1e-6)


def test_prior_score_loss_detaches_cost():
    lp = torch.zeros(2, 3, requires_grad=True)
    cost = torch.ones(2, 3, requires_grad=True)
    prior_score_loss(lp, cost).backward()
    assert cost.grad is None and torch.allclose(lp.grad, torch.full((2, 3), 1 / 6))


def test_grads_finite_when_collapsed():
    pred = torch.zeros(2, 4, 8, requires_grad=True)
    energy_score(pred, torch.zeros(2, 8)).backward()
    assert torch.isfinite(pred.grad).all()
    v = torch.ones(16, 8, requires_grad=True)
    visreg(v, VISRegCfg(), 200).backward()
    assert torch.isfinite(v.grad).all()
