"""Torch simulation of HQViT's ansatz W(θ), for training only (phase-2 spec §6, option B).

W acts on the m pixel qubits alone, so it is a 2**m x 2**m matrix applied to every token. Building it from the same
gates as quantum/hqvit.py lets autograd differentiate it. Evaluation always runs the CUDA-Q circuits;
tests/test_torch_sim.py checks this matches them, bit order included.

Bit order: CUDA-Q is little-endian within the register (qubit j is bit j of the amplitude index).
"""

import torch


def _gate(state: torch.Tensor, g: torch.Tensor, j: int, m: int) -> torch.Tensor:
    """Apply the 2x2 gate g to qubit j of each row of state (R, 2**m)."""
    r = state.shape[0]
    s = state.reshape(r, 2 ** (m - 1 - j), 2, 2**j)
    return torch.einsum("ab,xhbl->xhal", g, s).reshape(r, 2**m)


def _cnot(state: torch.Tensor, c: int, t: int, m: int) -> torch.Tensor:
    idx = torch.arange(2**m, device=state.device)
    return state[:, idx ^ (((idx >> c) & 1) << t)]


def _rx(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def _ry(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(c.dtype.to_complex())


def ansatz_matrix(theta: torch.Tensor, m: int, depth: int) -> torch.Tensor:
    """Wᵀ as a (2**m, 2**m) complex matrix: a row of amplitudes x maps to x @ ansatz_matrix = W x.

    Gate order is quantum/attention.py's `_ansatz`: per layer RX, RY, RX on each qubit, then a CNOT ring; a final RY.
    """
    ctype = theta.dtype.to_complex()
    state = torch.eye(2**m, dtype=ctype, device=theta.device)  # row i = basis vector e_i, evolved in place
    for u in range(depth):
        for j in range(m):
            base = 3 * (u * m + j)
            state = _gate(state, _rx(theta[base]).to(ctype), j, m)
            state = _gate(state, _ry(theta[base + 1]), j, m)
            state = _gate(state, _rx(theta[base + 2]).to(ctype), j, m)
        if m > 1:
            for j in range(m):
                state = _cnot(state, j, (j + 1) % m, m)
    for j in range(m):
        state = _gate(state, _ry(theta[3 * depth * m + j]), j, m)
    return state  # row i = W e_i, i.e. Wᵀ
