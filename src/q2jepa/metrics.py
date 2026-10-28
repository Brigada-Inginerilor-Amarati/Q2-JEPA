"""Q-JEPA App. F.4 MovingMNIST metrics for one observation's sampled displacements (S, 3, 2), pixels."""

import numpy as np

from q2jepa.data.mmnist import DIRECTION_PROBS, true_trajectories

ENDPOINT_TOL = 4.0


def mmnist_metrics(disp: np.ndarray) -> dict[str, float]:
    truth = true_trajectories()  # (8, 3, 2)
    s = disp.shape[0]
    end_err = np.linalg.norm(disp[:, None, -1] - truth[None, :, -1], axis=-1)  # (S, 8)
    nearest = end_err.argmin(1)
    valid = end_err[np.arange(s), nearest] < ENDPOINT_TOL
    cats = np.where(valid, nearest, 8)
    freq = np.bincount(cats, minlength=9) / s
    target = np.append(DIRECTION_PROBS, 0.0)

    ade = np.linalg.norm(disp[:, None] - truth[None], axis=-1).mean(-1)  # (S, 8)
    flat = disp.reshape(s, -1)
    to_truth = np.linalg.norm(flat[:, None] - truth.reshape(8, -1)[None], axis=-1).mean(0)  # (8,)
    pair = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum() / (s * (s - 1))
    return {
        "coverage": float(np.isin(np.arange(8), cats[valid]).mean()),
        "validity": float(valid.mean()),
        "tv": float(0.5 * np.abs(freq - target).sum()),
        "min_ade": float((DIRECTION_PROBS * ade.min(0)).sum()),
        "es": float((DIRECTION_PROBS * to_truth).sum() - 0.5 * pair),
    }
