"""Batched HQViT register evolution and its exact gradients, all through CUDA-Q (phase-2 spec §5.4, §6).

HQViT's ansatz is I_idx ⊗ W(θ): it acts only on the m low (pixel) qubits. A batch of N amplitude-encoded clips
therefore fits in ONE register of log2(N) batch qubits + log2(tokens) index qubits + m pixel qubits, and one
CUDA-Q call evolves every token of every clip. Gradients:
- w.r.t. θ_k: the derivative state (-i/2) U_{>k} P_k U_{≤k} |ψ⟩ (P_k = X or Y, the gate's generator), one circuit
  per angle for the whole batch;
- w.r.t. the input state: U† applied to the incoming gradient, one inverse circuit.
Gate order and parameter layout are exactly quantum/attention.py's `_ansatz`.
"""

import cudaq
import numpy as np


@cudaq.kernel
def _ansatz_d(qs: cudaq.qview, theta: list[float], depth: int, deriv: int):
    # _ansatz with the generator of gate `deriv` inserted after it (deriv = -1: plain ansatz). RX/RY commute with
    # their generators, so "after" and "before" give the same state.
    m = qs.size()
    for u in range(depth):
        for j in range(m):
            base = 3 * (u * m + j)
            rx(theta[base], qs[j])
            if base == deriv:
                x(qs[j])
            ry(theta[base + 1], qs[j])
            if base + 1 == deriv:
                y(qs[j])
            rx(theta[base + 2], qs[j])
            if base + 2 == deriv:
                x(qs[j])
        if m > 1:
            for j in range(m):
                x.ctrl(qs[j], qs[(j + 1) % m])
    for j in range(m):
        ry(theta[3 * depth * m + j], qs[j])
        if 3 * depth * m + j == deriv:
            y(qs[j])


@cudaq.kernel
def _ansatz_inv(qs: cudaq.qview, theta: list[float], depth: int):
    m = qs.size()
    for jj in range(m):
        j = m - 1 - jj
        ry(-theta[3 * depth * m + j], qs[j])
    for uu in range(depth):
        u = depth - 1 - uu
        if m > 1:
            for jj in range(m):
                j = m - 1 - jj
                x.ctrl(qs[j], qs[(j + 1) % m])
        for jj in range(m):
            j = m - 1 - jj
            base = 3 * (u * m + j)
            rx(-theta[base + 2], qs[j])
            ry(-theta[base + 1], qs[j])
            rx(-theta[base], qs[j])


@cudaq.kernel
def _evolve(init: cudaq.State, m: int, theta: list[float], depth: int, deriv: int):
    q = cudaq.qvector(init)
    _ansatz_d(q[0:m], theta, depth, deriv)


@cudaq.kernel
def _evolve_inv(init: cudaq.State, m: int, theta: list[float], depth: int):
    q = cudaq.qvector(init)
    _ansatz_inv(q[0:m], theta, depth)


def _dtype() -> type:
    return np.complex64 if cudaq.get_target().get_precision() == cudaq.SimulationPrecision.fp32 else np.complex128


def _run(kernel, states: np.ndarray, m: int, *args) -> np.ndarray:
    """Apply a pixel-qubit kernel to every row of `states` (N, D) in one register; rows may have any norm."""
    n, dim = states.shape
    rows = 1 << max(0, (n - 1).bit_length())  # pad the batch to a power of two (zero rows)
    flat = np.zeros((rows, dim), dtype=_dtype())
    flat[:n] = states
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        return np.zeros_like(states, dtype=_dtype())
    out = np.array(cudaq.get_state(kernel, cudaq.State.from_data((flat / norm).reshape(-1)), m, *args))
    return (norm * out.reshape(rows, dim))[:n]


def evolve(states: np.ndarray, theta: np.ndarray, m: int, depth: int, deriv: int = -1) -> np.ndarray:
    """(I ⊗ W(θ)) applied to each row (N, D); with deriv = k, the un-scaled derivative circuit P_k inserted."""
    return _run(_evolve, states, m, np.asarray(theta, float).tolist(), depth, deriv)


def evolve_inv(states: np.ndarray, theta: np.ndarray, m: int, depth: int) -> np.ndarray:
    """(I ⊗ W(θ))† applied to each row."""
    return _run(_evolve_inv, states, m, np.asarray(theta, float).tolist(), depth)


def theta_grad(states: np.ndarray, grad_out: np.ndarray, theta: np.ndarray, m: int, depth: int) -> np.ndarray:
    """dL/dθ_k = Re Σ conj(g) ∂_k ψ, with ∂_k ψ = (-i/2) × (derivative circuit applied to the input states)."""
    g = np.conj(grad_out)
    return np.array(
        [float(np.real(np.sum(g * (-0.5j) * evolve(states, theta, m, depth, k)))) for k in range(len(theta))]
    )
