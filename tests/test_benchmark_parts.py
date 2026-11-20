"""Shared pieces for HEP18 and D3IL: smoothed distance, multi-target energy, CNOT layout, predictor kinds."""

import math

import torch

from q2jepa.losses import _rho, energy_score, loo_cost, shift_cost
from q2jepa.predictor import AgentLocalPredictor, Predictor, ScaledPredictor
from q2jepa.quantum.born import hep18_grid_layers


def test_smoothed_distance_eq18():
    u, v = torch.zeros(1, 4), torch.full((1, 4), 2.0)  # |u - v|^2 / d = 4
    beta, eps = 0.5, 1e-4
    assert math.isclose(_rho(u, v, beta, eps).item(), (4 + eps**2) ** 0.25 - eps**0.5, rel_tol=1e-6)
    assert _rho(u, u, beta, eps).item() == 0.0
    u.requires_grad_()
    _rho(u, u.detach(), beta, eps).sum().backward()  # smooth at coincidence: finite zero gradient
    assert torch.isfinite(u.grad).all() and (u.grad == 0).all()


def test_multi_target_averages_over_futures():
    torch.manual_seed(0)
    pred, tgt = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    per = [energy_score(pred, tgt[:, j]) for j in range(3)]
    torch.testing.assert_close(energy_score(pred, tgt), torch.stack(per).mean())
    loo = torch.stack([loo_cost(pred, tgt[:, j]) for j in range(3)]).mean(0)
    torch.testing.assert_close(loo_cost(pred, tgt), loo)
    sc = torch.stack([shift_cost(pred, tgt[:, j], pred) for j in range(3)]).mean(0)
    torch.testing.assert_close(shift_cost(pred, tgt, pred), sc)
    assert torch.equal(energy_score(pred, tgt[:, 0]), energy_score(pred, tgt[:, 0], 1.0, 0.0))


def test_hep18_grid_layout_app_c():
    layers = hep18_grid_layers()
    assert [len(layer) for layer in layers] == [4, 6, 4, 6]  # 20 CNOTs
    assert (0, 1) in layers[0] and (9, 10) in layers[0]  # row N, row E: columns (0, 1)
    assert (0, 3) in layers[1] and (6, 9) in layers[1]  # column 0: rows (0, 1), (2, 3)
    assert (4, 5) in layers[2]  # row W: columns (1, 2)
    assert (3, 6) in layers[3] and (11, 2) in layers[3]  # column 0 rows (1, 2); column 2 rows (3, 0)


def test_agent_local_predictor_uses_only_own_bits():
    torch.manual_seed(0)
    p = AgentLocalPredictor(d=64, n_bits=12, width=16, agents=4)
    h, z = torch.randn(2, 64), torch.randint(0, 2, (2, 5, 12)).float()
    out = p(h, z).reshape(2, 5, 4, 16)
    z2 = z.clone()
    z2[..., 3:] = 1 - z2[..., 3:]  # flip every bit except agent 0's
    out2 = p(h, z2).reshape(2, 5, 4, 16)
    assert torch.equal(out[:, :, 0], out2[:, :, 0]) and not torch.allclose(out[:, :, 1:], out2[:, :, 1:])
    assert p(h, z).shape == (2, 5, 64)


def test_scaled_predictor_starts_as_shared():
    torch.manual_seed(0)
    shared = Predictor(16, 8, 32)
    torch.manual_seed(0)
    scaled = ScaledPredictor(16, 8, 32)
    with torch.no_grad():
        scaled.norm.weight.uniform_(0.5, 1.5)
        scaled.norm.bias.uniform_(-1, 1)
        shared.norm.load_state_dict(scaled.norm.state_dict())
    h, z = torch.randn(3, 16), torch.randint(0, 2, (3, 4, 8)).float()
    torch.testing.assert_close(scaled(h, z), shared(h, z))  # s = 1, b = 0 at zero init


def test_agent_local_predictor_matches_table7_count():
    assert sum(p.numel() for p in AgentLocalPredictor(64, 12, 64, 4).parameters()) == 22_800
