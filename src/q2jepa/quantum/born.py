"""Conditional Born-machine (QCBM) kernel, Q-JEPA Eq. 4.

angles has shape (L+1, n): rows 0..L-1 are body RY angles, row L the terminal RY angles.
Bit convention: z[..., j] is qubit j; probability vectors are indexed little-endian.
"""

import cudaq
import numpy as np

# Q-JEPA App. C: layer 1 CNOT(0,1), layer 2 CNOT(1,2); pairs are (control, target).
MMNIST_CNOT_LAYERS: list[list[tuple[int, int]]] = [[(0, 1)], [(1, 2)]]


@cudaq.kernel
def _born(n: int, angles: list[float], cnots: list[int], offsets: list[int]):
    q = cudaq.qvector(n)
    n_layers = len(offsets) - 1
    for layer in range(n_layers):
        for j in range(n):
            ry(angles[layer * n + j], q[j])
        for c in range(offsets[layer], offsets[layer + 1]):
            x.ctrl(q[cnots[2 * c]], q[cnots[2 * c + 1]])
    for j in range(n):
        ry(angles[n_layers * n + j], q[j])


def _kernel_args(angles: np.ndarray, cnot_layers) -> tuple[int, list[float], list[int], list[int]]:
    n = angles.shape[-1]
    assert angles.shape[-2] == len(cnot_layers) + 1, "angles must be (L+1, n)"
    cnots = [q for layer in cnot_layers for pair in layer for q in pair]
    offsets = np.concatenate([[0], np.cumsum([len(layer) for layer in cnot_layers])]).tolist()
    return n, angles.reshape(-1).astype(float).tolist(), cnots, [int(o) for o in offsets]


def born_probs(angles: np.ndarray, cnot_layers) -> np.ndarray:
    n, flat, cnots, offsets = _kernel_args(angles, cnot_layers)
    state = np.array(cudaq.get_state(_born, n, flat, cnots, offsets))
    return np.abs(state) ** 2


def brickwork_layers(n: int, layers: int) -> list[list[tuple[int, int]]]:
    """Q-JEPA App. B/C: even layers couple (0,1),(2,3),..., odd layers (1,2),(3,4),...; lower index controls."""
    return [[(j, j + 1) for j in range(layer % 2, n - 1, 2)] for layer in range(layers)]


def sample_codes(
    angles: np.ndarray, cnot_layers, shots: int, rng: np.random.Generator, seed: int | None = None
) -> np.ndarray:
    """(B, L+1, n) angles -> (B, shots, n) uint8 codes from one broadcast cudaq.sample call.

    `seed` seeds CUDA-Q's sampler so runs (and resumed runs) are reproducible; shot order is shuffled by `rng`.
    """
    if seed is not None:
        cudaq.set_random_seed(seed)
    args = [_kernel_args(a, cnot_layers) for a in angles]
    results = cudaq.sample(_born, *(list(col) for col in zip(*args, strict=True)), shots_count=shots)
    results = results if isinstance(results, list) else [results]
    out = np.empty((angles.shape[0], shots, angles.shape[-1]), dtype=np.uint8)
    for b, counts in enumerate(results):
        bits = [[int(ch) for ch in bitstring] for bitstring, k in counts.items() for _ in range(k)]
        out[b] = rng.permutation(np.array(bits, dtype=np.uint8))
    return out


def apply_readout_flips(bits: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    return bits ^ (rng.random(bits.shape) < p).astype(bits.dtype)
