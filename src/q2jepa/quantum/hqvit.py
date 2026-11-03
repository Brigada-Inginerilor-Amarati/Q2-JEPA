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
    return _run_raw(kernel, states, m, *args)


def _run_raw(kernel, states: np.ndarray, *args) -> np.ndarray:
    """Run `kernel(init, *args)` with every row of `states` (N, D) in one register; rows may have any norm."""
    n, dim = states.shape
    rows = 1 << max(0, (n - 1).bit_length())  # pad the batch to a power of two (zero rows)
    flat = np.zeros((rows, dim), dtype=_dtype())
    flat[:n] = states
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        return np.zeros_like(states, dtype=_dtype())
    out = np.array(cudaq.get_state(kernel, cudaq.State.from_data((flat / norm).reshape(-1)), *args))
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


# --- Adjoint (reverse-sweep) gradients ------------------------------------------------------------------------
# The ansatz is a sequence of segments: per repeated unit u, an RX layer, an RY layer, an RX layer and the CNOT ring;
# then the terminal RY layer. Segment s of 4u + r (r = 0, 1, 2 rotations, r = 3 ring); segment 4 * depth = terminal.
# Sweeping back from the output, with φ the state after a segment and λ the incoming gradient there,
# dL/dθ_(s, j) = Re⟨λ|(−i/2) P_j φ⟩ (P_j = X or Y on pixel qubit j commutes with the whole rotation layer); then
# φ and λ are stepped back through the segment's inverse. λ at the start is U†g, the input gradient. Every gate
# runs in CUDA-Q: ~2 segment inverses per segment plus one Pauli per angle, instead of one full circuit per angle.


@cudaq.kernel
def _segment_inv(init: cudaq.State, m: int, theta: list[float], depth: int, seg: int):
    q = cudaq.qvector(init)
    if seg == 4 * depth:
        for j in range(m):
            ry(-theta[3 * depth * m + j], q[j])
    else:
        u = seg // 4
        r = seg % 4
        if r == 3:
            if m > 1:
                for jj in range(m):
                    j = m - 1 - jj
                    x.ctrl(q[j], q[(j + 1) % m])
        else:
            for j in range(m):
                a = -theta[3 * (u * m + j) + r]
                if r == 1:
                    ry(a, q[j])
                else:
                    rx(a, q[j])


@cudaq.kernel
def _pauli(init: cudaq.State, j: int, kind: int):
    q = cudaq.qvector(init)
    if kind == 1:
        y(q[j])
    else:
        x(q[j])


@cudaq.kernel
def _pauli_each(init: cudaq.State, m: int, c: int, kind: int):
    # Rows are stacked as `2**c` copies (copy index = the c highest qubits); copy j < m gets P = X or Y on pixel
    # qubit j, applied as a gate controlled on the copy index being j. One call yields P_j φ for every j.
    q = cudaq.qvector(init)
    top = q.size() - c
    for j in range(m):
        for b in range(c):
            if (j >> b) % 2 == 0:
                x(q[top + b])
        if kind == 1:
            y.ctrl(q[top : top + c], q[j])
        else:
            x.ctrl(q[top : top + c], q[j])
        for b in range(c):
            if (j >> b) % 2 == 0:
                x(q[top + b])


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _pad_rows(a: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows, a.shape[1]), dtype=_dtype())
    out[: len(a)] = a
    return out


# Two equivalent call layouts for the sweep (both exact, both in CUDA-Q):
# - per-qubit (default): per segment, one Pauli call per angle plus two inverse calls; best where gates dominate
#   (qpp-cpu, batch 4: 3.2 s backward vs 64 s for derivative circuits);
# - batched: two calls per segment on larger registers (copy-indexed Paulis, stacked φ/λ); 10x slower on qpp-cpu,
#   meant for GPUs where per-call overhead dominates. Pick per target after timing (phase-2 spec §6).
BATCHED_CALLS = False


def adjoint_grads(
    out: np.ndarray, grad_out: np.ndarray, theta: np.ndarray, m: int, depth: int, batched: bool | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(dL/dθ, U† g) from the forward output ψ = U x and the incoming gradient g, in one reverse sweep."""
    batched = BATCHED_CALLS if batched is None else batched
    th = np.asarray(theta, float).tolist()
    n = len(out)
    rows = _pow2(n)
    c = max(1, (m - 1).bit_length())  # copy-index qubits for the batched Paulis
    phi, lam = _pad_rows(out, rows), _pad_rows(grad_out, rows)
    grads = np.zeros(len(th))
    for seg in reversed(range(4 * depth + 1)):
        r = 1 if seg == 4 * depth else seg % 4  # terminal layer is RY
        if r != 3:
            kind = 1 if r == 1 else 0
            if batched:
                copies = _run_raw(_pauli_each, np.tile(phi, (1 << c, 1)), m, c, kind).reshape(1 << c, rows, -1)
            for j in range(m):
                idx = 3 * depth * m + j if seg == 4 * depth else 3 * ((seg // 4) * m + j) + r
                p_phi = copies[j] if batched else _run_raw(_pauli, phi, j, kind)
                grads[idx] = float(np.real(np.sum(np.conj(lam) * (-0.5j) * p_phi)))
        if batched:
            nphi, nlam = float(np.linalg.norm(phi)), float(np.linalg.norm(lam))
            scale = nphi / nlam if nlam > 0 else 1.0  # keep both halves at one magnitude for fp32
            both = _run(_segment_inv, np.concatenate([phi, lam * scale]), m, th, depth, seg)
            phi, lam = both[:rows], both[rows:] / scale
        else:
            phi = _run(_segment_inv, phi, m, th, depth, seg)
            lam = _run(_segment_inv, lam, m, th, depth, seg)
    return grads, lam[:n]
