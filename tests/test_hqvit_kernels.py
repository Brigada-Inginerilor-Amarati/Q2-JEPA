import numpy as np
import pytest

from q2jepa.quantum import is_fp32
from q2jepa.quantum.attention import factored_acm, image_to_patches, n_ansatz_params, patch_states
from q2jepa.quantum.hqvit import evolve, evolve_inv, theta_grad


def _setup(seed=0, n=3, tokens=16, d=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, tokens * d)) + 1j * rng.normal(size=(n, tokens * d))
    return rng, x, rng.uniform(-np.pi, np.pi, n_ansatz_params(int(np.log2(d)), 2))


def test_batched_register_matches_per_patch_circuits():
    # One batched register == the existing per-patch kernel applied to every patch of every image.
    rng = np.random.default_rng(1)
    imgs = [rng.random((16, 16)) for _ in range(3)]
    theta = rng.uniform(-np.pi, np.pi, n_ansatz_params(4, 2))
    amps = np.stack([image_to_patches(im, 4).reshape(-1) / np.linalg.norm(im) for im in imgs])
    out = evolve(amps, theta, m=4, depth=2).reshape(3, 16, 16)
    for b, im in enumerate(imgs):
        w, states = patch_states(image_to_patches(im, 4), theta)
        np.testing.assert_allclose(out[b], np.sqrt(w)[:, None] * states, atol=1e-6)


def test_batched_attention_matches_factored_acm():
    rng = np.random.default_rng(2)
    img = rng.random((16, 16))
    tq, tk = (rng.uniform(-np.pi, np.pi, n_ansatz_params(4, 2)) for _ in range(2))
    amps = (image_to_patches(img, 4).reshape(-1) / np.linalg.norm(img))[None]
    q, k = (evolve(amps, t, 4, 2).reshape(16, 16) for t in (tq, tk))
    np.testing.assert_allclose(np.abs(q.conj() @ k.T) ** 2, factored_acm(image_to_patches(img, 4), tq, tk), atol=1e-6)


@pytest.mark.parametrize("m", [4, 6])
def test_inverse_undoes_forward_any_norm(m):
    _, x, _ = _setup(3, n=5, tokens=4, d=2**m)
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, n_ansatz_params(m, 2))
    atol = 1e-4 if is_fp32() else 1e-6  # |x| ~ 25: fp32 round-trip error ~ 1e-6 relative
    np.testing.assert_allclose(evolve_inv(evolve(x, theta, m, 2), theta, m, 2), x, atol=atol)


def test_theta_gradient_matches_finite_differences():
    rng, x, theta = _setup(5, n=2, tokens=4, d=16)
    g = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    loss = lambda t: float(np.real(np.sum(np.conj(g) * evolve(x, t, 4, 2))))
    eps, tol = (1e-2, 2e-2) if is_fp32() else (1e-6, 1e-5)  # fp32 (nvidia): rounding swamps tiny steps
    num = np.array([(loss(theta + eps * e) - loss(theta - eps * e)) / (2 * eps) for e in np.eye(len(theta))])
    np.testing.assert_allclose(theta_grad(x, g, theta, 4, 2), num, atol=tol, rtol=tol)


def test_theta_gradient_matches_parameter_shift():
    # Spec §6 reference: for L = sum_z w(z) |psi(z)|^2 (a diagonal expectation) the two-term shift rule is exact.
    import torch

    from q2jepa.quantum.bridge import RegisterEvolve

    rng, x, theta = _setup(7, n=3, tokens=4, d=16)
    x = torch.tensor(np.real(x))
    w = torch.tensor(rng.normal(size=x.shape))
    loss = lambda th: float((RegisterEvolve.apply(x, torch.tensor(th), 4, 2).abs() ** 2 * w).sum())
    t = torch.tensor(theta, requires_grad=True)
    (RegisterEvolve.apply(x, t, 4, 2).abs() ** 2 * w).sum().backward()
    shift = np.array([0.5 * (loss(theta + np.pi / 2 * e) - loss(theta - np.pi / 2 * e)) for e in np.eye(len(theta))])
    tol = 1e-3 if is_fp32() else 1e-8
    np.testing.assert_allclose(t.grad.numpy(), shift, atol=tol, rtol=tol)


def test_register_evolve_autograd_matches_dense_reference():
    # Reference: the same linear map as an explicit matrix (built from CUDA-Q on basis states, test only).
    import torch

    from q2jepa.quantum.bridge import RegisterEvolve

    rng = np.random.default_rng(6)
    m, d, tokens = 2, 4, 2
    theta = rng.uniform(-np.pi, np.pi, n_ansatz_params(m, 2))
    mat = torch.as_tensor(evolve(np.eye(tokens * d), theta, m, 2).T).to(torch.complex128)  # column j = image of e_j

    x = torch.tensor(rng.normal(size=(3, tokens * d)), requires_grad=True)
    t = torch.tensor(theta, requires_grad=True)
    w = torch.tensor(rng.normal(size=(3, tokens * d)) + 1j * rng.normal(size=(3, tokens * d)))
    loss = lambda psi: (psi.abs() ** 2 * w.real).sum() + (psi * w).real.sum()
    loss(RegisterEvolve.apply(x, t, m, 2)).backward()

    xr = x.detach().clone().requires_grad_()
    loss(xr.to(torch.complex128) @ mat.T).backward()
    gtol = 1e-4 if is_fp32() else 1e-7
    torch.testing.assert_close(x.grad, xr.grad, atol=gtol, rtol=gtol)

    (eps, tol), num = ((1e-2, 2e-2) if is_fp32() else (1e-6, 1e-5)), []
    for e in np.eye(len(theta)):
        f = lambda th: float(loss(RegisterEvolve.apply(x.detach(), torch.tensor(th), m, 2)))
        num.append((f(theta + eps * e) - f(theta - eps * e)) / (2 * eps))
    torch.testing.assert_close(t.grad, torch.tensor(num, dtype=torch.float64), atol=tol, rtol=tol)
