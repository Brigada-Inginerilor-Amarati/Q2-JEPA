import numpy as np

from q2jepa.quantum.born import MMNIST_CNOT_LAYERS, apply_readout_flips, born_probs, sample_codes


def _np_born(angles, layers):
    n = angles.shape[1]
    psi = np.zeros(2**n)
    psi[0] = 1.0

    def ry(psi, j, a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        out = psi.copy()
        for i in range(2**n):
            if not (i >> j) & 1:
                k = i | (1 << j)
                out[i] = c * psi[i] - s * psi[k]
                out[k] = s * psi[i] + c * psi[k]
        return out

    def cx(psi, ctl, tgt):
        out = psi.copy()
        for i in range(2**n):
            if (i >> ctl) & 1:
                out[i] = psi[i ^ (1 << tgt)]
        return out

    for layer, pairs in enumerate(layers):
        for j in range(n):
            psi = ry(psi, j, angles[layer, j])
        for c, t in pairs:
            psi = cx(psi, c, t)
    for j in range(n):
        psi = ry(psi, j, angles[-1, j])
    return psi**2


def test_entangled_matches_reference():
    a = np.random.default_rng(0).uniform(-np.pi, np.pi, (3, 3))
    np.testing.assert_allclose(born_probs(a, MMNIST_CNOT_LAYERS), _np_born(a, MMNIST_CNOT_LAYERS), atol=1e-6)


def test_product_marginals():
    a = np.random.default_rng(1).uniform(-np.pi, np.pi, (3, 3))
    p = born_probs(a, [[], []])
    marg = [p[[i for i in range(8) if (i >> j) & 1]].sum() for j in range(3)]
    np.testing.assert_allclose(marg, np.sin(a.sum(0) / 2) ** 2, atol=1e-6)


def test_bit_order_qubit0_is_z0():
    a = np.zeros((3, 3))
    a[-1, 0] = np.pi
    z = sample_codes(a[None], MMNIST_CNOT_LAYERS, 64, np.random.default_rng(0))
    assert z.shape == (1, 64, 3) and (z[0] == [1, 0, 0]).all()


def test_sample_frequencies_match_probs():
    rng = np.random.default_rng(2)
    a = rng.uniform(-np.pi, np.pi, (2, 3, 3))
    z = sample_codes(a, MMNIST_CNOT_LAYERS, 20000, rng)
    for b in range(2):
        idx = (z[b] * [1, 2, 4]).sum(-1)
        freq = np.bincount(idx, minlength=8) / 20000
        assert 0.5 * np.abs(freq - born_probs(a[b], MMNIST_CNOT_LAYERS)).sum() < 0.02


def test_readout_flips():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (4, 1000, 3)).astype(np.uint8)
    assert (apply_readout_flips(bits, 0.0, rng) == bits).all()
    assert (apply_readout_flips(bits, 1.0, rng) == 1 - bits).all()
    assert abs((apply_readout_flips(bits, 0.005, rng) != bits).mean() - 0.005) < 0.002
