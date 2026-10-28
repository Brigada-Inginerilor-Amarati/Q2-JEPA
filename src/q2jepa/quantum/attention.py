"""HQViT swap-test self-attention (HQViT §4.2, Eqs. 11-22).

Whole-image amplitude encoding: flat index = i*d + l (patch i, pixel l), so under cudaq's
little-endian indexing the m = log2(d) patch qubits are the low qubits of each register.

- factored: per-patch m-qubit circuits, exact; the training path.
- full: one (2n+1)-qubit swap-test circuit, exact statevector; equivalence + HPC benchmark.
  Layout: RQ = qubits 0..n-1, RK = n..2n-1, ancilla = 2n.
"""

import cudaq
import numpy as np


@cudaq.kernel
def _ansatz(qs: cudaq.qview, theta: list[float], depth: int):
    m = qs.size()
    for u in range(depth):
        for j in range(m):
            base = 3 * (u * m + j)
            rx(theta[base], qs[j])
            ry(theta[base + 1], qs[j])
            rx(theta[base + 2], qs[j])
        if m > 1:
            for j in range(m):
                x.ctrl(qs[j], qs[(j + 1) % m])
    for j in range(m):
        ry(theta[3 * depth * m + j], qs[j])


@cudaq.kernel
def _patch(amps: list[complex], theta: list[float], depth: int):
    q = cudaq.qvector(amps)
    _ansatz(q, theta, depth)


@cudaq.kernel
def _full(init: cudaq.State, n: int, m: int, theta_q: list[float], theta_k: list[float], depth: int):
    # ponytail: init passed as a cudaq.State (|0>_anc ⊗ |psi>_K ⊗ |psi>_Q). cudaq 0.16 caps list
    # init at 10 qubits and segfaults on two list-initialized registers in one kernel.
    q = cudaq.qvector(init)
    _ansatz(q[0:m], theta_q, depth)
    _ansatz(q[n : n + m], theta_k, depth)
    h(q[2 * n])
    for r in range(m):
        swap.ctrl(q[2 * n], q[r], q[n + r])
    h(q[2 * n])


def image_to_patches(img: np.ndarray, p: int = 4) -> np.ndarray:
    h, w = img.shape
    return img.reshape(h // p, p, w // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)


def n_ansatz_params(m: int, depth: int) -> int:
    return 3 * m * depth + m


def _whole_image_norm(patches: np.ndarray) -> float:
    norm = float(np.linalg.norm(patches))
    if norm == 0.0:
        raise ValueError("all-zero image cannot be amplitude-encoded")
    return norm


def patch_states(patches: np.ndarray, theta: np.ndarray, depth: int = 2) -> tuple[np.ndarray, np.ndarray]:
    norm = _whole_image_norm(patches)
    t, d = patches.shape
    w = np.linalg.norm(patches, axis=1) ** 2 / norm**2
    states = np.zeros((t, d), dtype=complex)
    states[:, 0] = 1.0  # |0...0> placeholder for empty patches (weight 0, never simulated)
    theta_l = np.asarray(theta, dtype=float).tolist()
    for i in np.flatnonzero(w > 0):
        amps = (patches[i] / np.linalg.norm(patches[i])).astype(complex).tolist()
        states[i] = np.array(cudaq.get_state(_patch, amps, theta_l, depth))
    return w, states


def factored_acm(patches: np.ndarray, theta_q: np.ndarray, theta_k: np.ndarray, depth: int = 2) -> np.ndarray:
    """ACM_ij = |<q_i|k_j>|^2 with unnormalized q_i (HQViT Eq. 22) = w_i w_j |<q^_i|k^_j>|^2."""
    w, q = patch_states(patches, theta_q, depth)
    _, k = patch_states(patches, theta_k, depth)
    return np.outer(w, w) * np.abs(q.conj() @ k.T) ** 2


def full_swap_test_probs(patches: np.ndarray, theta_q: np.ndarray, theta_k: np.ndarray, depth: int = 2) -> np.ndarray:
    """Pr(i, j, anc=0) from the full swap-test circuit (HQViT Eq. 21)."""
    psi = (patches.reshape(-1) / _whole_image_norm(patches)).astype(complex)
    t, d = patches.shape
    init = cudaq.State.from_data(np.concatenate([np.kron(psi, psi), np.zeros(psi.size**2, dtype=complex)]))
    theta_q_l, theta_k_l = np.asarray(theta_q, float).tolist(), np.asarray(theta_k, float).tolist()
    state = cudaq.get_state(_full, init, int(np.log2(t * d)), int(np.log2(d)), theta_q_l, theta_k_l, depth)
    probs = (np.abs(np.array(state)) ** 2).reshape(2, t, d, t, d)  # (anc, iK, lK, iQ, lQ)
    return probs[0].sum(axis=(1, 3)).T
