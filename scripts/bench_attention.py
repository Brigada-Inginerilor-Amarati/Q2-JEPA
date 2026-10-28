"""Swap-test attention benchmark: factored m-qubit kernels vs the full (2n+1)-qubit circuit.

uv run python scripts/bench_attention.py --target qpp-cpu --res 16 --res 32 --images 16
"""

import argparse
import time

import numpy as np

from q2jepa.quantum import set_target
from q2jepa.quantum.attention import factored_acm, full_swap_test_probs, image_to_patches, n_ansatz_params

PATCH, DEPTH = 4, 2


def run_bench(target: str, res: int, n_images: int, seed: int = 0) -> list[dict]:
    set_target(target)
    rng = np.random.default_rng(seed)
    m, k = int(np.log2(PATCH * PATCH)), n_ansatz_params(int(np.log2(PATCH * PATCH)), DEPTH)
    images = [image_to_patches(rng.random((res, res)), PATCH) for _ in range(n_images)]
    tq, tk = rng.uniform(-np.pi, np.pi, k), rng.uniform(-np.pi, np.pi, k)
    rows = []
    for mode, fn, qubits in (
        ("factored", factored_acm, m),
        ("full", full_swap_test_probs, 2 * int(np.log2(res * res)) + 1),
    ):
        fn(images[0], tq, tk, DEPTH)  # warm-up (JIT compile)
        start = time.perf_counter()
        for x in images:
            fn(x, tq, tk, DEPTH)
        ms = 1000 * (time.perf_counter() - start) / n_images
        rows.append({"target": target, "res": res, "mode": mode, "qubits": qubits, "ms_per_image": ms})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qpp-cpu")
    ap.add_argument("--res", type=int, action="append")
    ap.add_argument("--images", type=int, default=16)
    args = ap.parse_args()
    print("| target | res | mode | qubits | ms/image |\n|---|---|---|---|---|")
    for res in args.res or [16]:
        for r in run_bench(args.target, res, args.images):
            print(f"| {r['target']} | {r['res']} | {r['mode']} | {r['qubits']} | {r['ms_per_image']:.2f} |")


if __name__ == "__main__":
    main()
