import numpy as np
import pytest

from q2jepa.quantum.attention import factored_acm, full_swap_test_probs, image_to_patches, n_ansatz_params


def _setup(seed, img):
    rng = np.random.default_rng(seed)
    k = n_ansatz_params(4, 2)
    return image_to_patches(img, 4), rng.uniform(-np.pi, np.pi, k), rng.uniform(-np.pi, np.pi, k)


def _weights(x):
    return (np.linalg.norm(x, axis=1) ** 2) / (np.linalg.norm(x) ** 2)


def test_param_count():
    assert n_ansatz_params(4, 2) == 28


def test_full_equals_factored_random_image():
    x, tq, tk = _setup(1, np.random.default_rng(0).random((16, 16)))
    w = _weights(x)
    np.testing.assert_allclose(full_swap_test_probs(x, tq, tk), 0.5 * (np.outer(w, w) + factored_acm(x, tq, tk)), atol=1e-6)


def test_empty_patches_no_nan():
    img = np.zeros((16, 16))
    img[5:11, 6:10] = np.random.default_rng(2).random((6, 4))
    x, tq, tk = _setup(3, img)
    f = factored_acm(x, tq, tk)
    assert np.isfinite(f).all()
    empty = np.linalg.norm(x, axis=1) == 0
    assert empty.sum() >= 8 and (f[empty] == 0).all() and (f[:, empty] == 0).all()
    w = _weights(x)
    np.testing.assert_allclose(full_swap_test_probs(x, tq, tk), 0.5 * (np.outer(w, w) + f), atol=1e-6)


def test_black_image_raises():
    x, tq, tk = _setup(0, np.zeros((16, 16)))
    with pytest.raises(ValueError):
        factored_acm(x, tq, tk)


def test_ansatz_parameters_change_attention():
    x, tq, tk = _setup(4, np.random.default_rng(5).random((16, 16)))
    assert not np.allclose(factored_acm(x, tq, tk), factored_acm(x, tq, tk + 0.3))
