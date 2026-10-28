import torch

from q2jepa.prior.classical import MixturePrior, SoftmaxCategorical
from q2jepa.prior.codes import bits_to_index, index_to_bits


def test_bit_order_matches_cudaq():
    assert index_to_bits(torch.tensor([1, 6]), 3).tolist() == [[1, 0, 0], [0, 1, 1]]
    assert bits_to_index(torch.tensor([[1.0, 0, 0], [0, 1, 1]])).tolist() == [1, 6]


def test_mixture_param_count_matches_paper():
    assert sum(p.numel() for p in MixturePrior().parameters()) == 3640


def test_categorical_forced_code():
    p = SoftmaxCategorical()
    torch.nn.init.zeros_(p.net[-1].weight)
    with torch.no_grad():
        p.net[-1].bias.copy_(torch.tensor([-50.0, 50, -50, -50, -50, -50, -50, -50]))
    z = p.sample(torch.randn(2, 64), 16)
    assert (z == torch.tensor([1.0, 0, 0])).all()


def test_log_probs_normalize():
    h = torch.randn(3, 64)
    allz = index_to_bits(torch.arange(8), 3).expand(3, 8, 3)
    for prior in (SoftmaxCategorical(), MixturePrior()):
        assert torch.allclose(prior.log_prob(h, allz).exp().sum(-1), torch.ones(3), atol=1e-5)


def test_sample_frequencies_match_log_prob():
    torch.manual_seed(0)
    h = torch.randn(1, 64)
    allz = index_to_bits(torch.arange(8), 3)[None]
    for prior in (SoftmaxCategorical(), MixturePrior()):
        freq = torch.bincount(bits_to_index(prior.sample(h, 20000))[0], minlength=8) / 20000
        assert 0.5 * (freq - prior.log_prob(h, allz).exp()[0]).abs().sum() < 0.03
