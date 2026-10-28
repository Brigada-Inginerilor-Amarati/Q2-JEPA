import torch

from q2jepa.encoder import prepare
from q2jepa.encoder.cnn import CNNEncoder
from q2jepa.predictor import Predictor


def test_encoder_shapes_all_res():
    x = torch.rand(5, 3, 64, 64)
    for r in (16, 32, 64):
        assert CNNEncoder(r)(prepare(x, r)).shape == (5, 64)
    assert torch.equal(prepare(x, 64), x)


def test_predictor_shape_and_code_dependence():
    p = Predictor()
    h = torch.randn(4, 64)
    z = torch.tensor([[0.0, 0, 0], [1, 0, 0]]).expand(4, 2, 3)
    out = p(h, z)
    assert out.shape == (4, 2, 64) and not torch.allclose(out[:, 0], out[:, 1])
