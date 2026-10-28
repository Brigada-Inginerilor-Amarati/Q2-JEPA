import numpy as np
import torch

from q2jepa.data.mmnist import DIRECTION_PROBS, StochasticMovingMNIST, translate, true_trajectories


def test_probs_and_trajectories():
    assert np.isclose(DIRECTION_PROBS.sum(), 1.0)
    t = true_trajectories()
    np.testing.assert_allclose(t[0, 2], [12, 0], atol=1e-9)
    np.testing.assert_allclose(t[2, 0], [0, 4], atol=1e-9)


def test_translate_moves_centroid():
    img = torch.zeros(64, 64)
    img[30, 20] = 1.0
    out = translate(img, torch.tensor([3.0, -2.5]))
    ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    assert abs((out * xs).sum() / out.sum() - 23.0) < 0.05
    assert abs((out * ys).sum() / out.sum() - 27.5) < 0.05


def test_train_records_and_render_convention():
    ds = StochasticMovingMNIST("train", n_obs=2)
    assert len(ds) == 40
    dirs = [ds[i]["direction"] for i in range(40) if ds[i]["obs_idx"] == ds[0]["obs_idx"]]
    assert np.bincount(dirs, minlength=8).tolist() == [3, 2] * 4
    item = next(ds[i] for i in range(40) if ds[i]["direction"] == 0)
    expected = translate(item["obs"][2], torch.tensor([4.0, 0.0]))
    assert torch.allclose(item["fut"][0], expected, atol=1e-5)
    assert item["obs"].min() >= 0 and item["obs"].max() <= 1


def test_eval_split_and_disjointness():
    te = StochasticMovingMNIST("test", n_obs=3)
    assert len(te) == 24 and te.unique_obs().shape == (3, 3, 64, 64)
    tr, dv = StochasticMovingMNIST("train", n_obs=800), StochasticMovingMNIST("dev", n_obs=80)
    assert set(tr.mnist_indices).isdisjoint(dv.mnist_indices)
