"""Q-JEPA App. F.4 metrics: MovingMNIST (displacements, pixels) and HEP18 (four-agent trajectories, metres)."""

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


POS_TOL, SPEED_TOL = 2.0, 1.0  # HEP18 joint match: metres, m/s (App. F.4)


def _match_err(s: np.ndarray, v: np.ndarray, s_true: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    """(S, J) largest tolerance-normalized error over the trailing axes (agents, times) of each sample-target pair."""
    ds = np.abs(s[:, None] - s_true[None]) / POS_TOL
    dv = np.abs(v[:, None] - v_true[None]) / SPEED_TOL
    return np.maximum(ds, dv).reshape(ds.shape[0], ds.shape[1], -1).max(-1)


def _joint_valid(s, v, s_true, v_true) -> tuple[np.ndarray, np.ndarray]:
    err = _match_err(s, v, s_true, v_true)
    return err.min(1) <= 1.0, err.argmin(1)


def hep18_metrics(
    s: np.ndarray, v: np.ndarray, s_true: np.ndarray, v_true: np.ndarray, rng: np.random.Generator, shuffles: int = 10
) -> dict[str, float]:
    """Q-JEPA App. F.4 HEP18 metrics for one observation.

    s, v: sampled positions and speeds (S, agents, 3); s_true, v_true: the 18 futures (18, agents, 3).
    """
    n_s, n_fut, agents = s.shape[0], s_true.shape[0], s.shape[1]
    valid, nearest = _joint_valid(s, v, s_true, v_true)
    cats = np.where(valid, nearest, n_fut)
    freq = np.bincount(cats, minlength=n_fut + 1) / n_s
    target = np.append(np.full(n_fut, 1 / n_fut), 0.0)

    pos_err = np.abs(s[:, None] - s_true[None])  # (S, 18, agents, 3)
    flat, flat_true = s.reshape(n_s, -1), s_true.reshape(n_fut, -1)
    to_truth = np.linalg.norm(flat[:, None] - flat_true[None], axis=-1).mean()
    pair = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum() / (n_s * (n_s - 1))

    shuffled = []
    for _ in range(shuffles):  # re-pair agents independently: keeps marginals, breaks coordination
        perm = np.stack([rng.permutation(n_s) for _ in range(agents)], 1)
        idx = (perm, np.arange(agents)[None])
        shuffled.append(_joint_valid(s[idx], v[idx], s_true, v_true)[0].mean())

    local = []  # each agent matches any of its own distinct futures (its 4 schedules)
    for a in range(agents):
        own_s, own_v = np.unique(np.stack([s_true[:, a], v_true[:, a]], -1), axis=0).transpose(2, 0, 1)
        local.append(_match_err(s[:, a], v[:, a], own_s, own_v).min(1) <= 1.0)
    return {
        "coverage": float(np.isin(np.arange(n_fut), cats[valid]).mean()),
        "validity": float(valid.mean()),
        "tv": float(0.5 * np.abs(freq - target).sum()),
        "min_ade": float(pos_err.mean((2, 3)).min(0).mean()),
        "min_fde": float(pos_err[..., -1].mean(-1).min(0).mean()),
        "es": float(to_truth - 0.5 * pair),
        "shuffled_validity": float(np.mean(shuffled)),
        "local_validity": float(np.mean(local)),
    }


def state_distance(a: np.ndarray, b: np.ndarray, perms: np.ndarray | None = None) -> np.ndarray:
    """RMS distance (A, B) between states a (A, m) and b (B, m), minimised over index permutations perms (P, m)."""
    idx = np.arange(a.shape[-1])[None] if perms is None else perms
    return np.sqrt(((a[:, None][..., idx] - b[None, :, None]) ** 2).mean(-1)).min(-1)  # (A, B, P, m) -> (A, B)


def match_futures(
    pred: np.ndarray, truth: np.ndarray, scale: float, perms: np.ndarray | None = None, max_radius: float = np.inf
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(valid (S,), nearest future (S,), distances (S, J) in units of scale) of samples pred (S, m) to futures (J, m).

    A sample is valid if it lies within half the gap from its nearest future to that future's nearest other
    future, so it matches exactly one future, and within max_radius of it, so widely spaced futures do not accept
    large errors. perms: optional (P, m) state permutations describing the same scene.
    """
    n_s, n_fut = len(pred), len(truth)
    to_fut = state_distance(pred / scale, truth / scale, perms)
    gap = state_distance(truth / scale, truth / scale, perms) + np.diag(np.full(n_fut, np.inf))
    nearest = to_fut.argmin(1)
    radius = np.minimum(gap.min(1) / 2, max_radius)
    return to_fut[np.arange(n_s), nearest] <= radius[nearest], nearest, to_fut


def futures_metrics(
    pred: np.ndarray, truth: np.ndarray, scale: float, perms: np.ndarray | None = None, max_radius: float = np.inf
) -> dict[str, float]:
    """Multi-future metrics for one observation (charged particles, chess), J equally likely futures.

    pred: decoded samples (S, m); truth: the J futures (J, m); scale: the state unit (px, squares); perms: optional
    (P, m) index permutations of the state that describe the same scene (charged particles: balls as a set).
    A valid sample (match_futures) counts for the one future it matches. min_ade and es use the same distance; with
    perms that distance is a minimum over orders, so es is descriptive (not a proper score).
    """
    n_s, n_fut = len(pred), len(truth)
    valid, nearest, to_fut = match_futures(pred, truth, scale, perms, max_radius)
    p = pred / scale
    cats = np.where(valid, nearest, n_fut)
    freq = np.bincount(cats, minlength=n_fut + 1) / n_s
    target = np.append(np.full(n_fut, 1 / n_fut), 0.0)
    pair = state_distance(p, p, perms).sum() / (n_s * (n_s - 1))
    return {
        "coverage": float(np.isin(np.arange(n_fut), cats[valid]).mean()),
        "validity": float(valid.mean()),
        "tv": float(0.5 * np.abs(freq - target).sum()),
        "min_ade": float(to_fut.min(0).mean()),
        "es": float(to_fut.mean() - 0.5 * pair),
    }


def flow_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Single-future metrics for one observation (TaxiBJ): pred (S, m) decoded samples, truth (m,), values in [0, 1].

    mse_min / mae_min: error of the best of the S samples (oracle, as minADE); mse_mean: average over samples;
    es: energy score with the RMS distance (proper for one future; lower is better). Units: per value, [0, 1] scale.
    """
    err = pred - truth[None]
    mse, mae = (err**2).mean(-1), np.abs(err).mean(-1)
    n_s = len(pred)
    rms = np.sqrt(mse)
    sq = (pred**2).sum(-1)  # pairwise distances via the Gram matrix: S x S x m would not fit for m = 6144
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * pred @ pred.T, 0.0) / pred.shape[-1]
    np.fill_diagonal(d2, 0.0)
    pair = np.sqrt(d2).sum() / (n_s * (n_s - 1))
    return {
        "mse_min": float(mse.min()),
        "mse_mean": float(mse.mean()),
        "mae_min": float(mae.min()),
        "es": float(rms.mean() - 0.5 * pair),
    }


ROUTES = 24  # D3IL: 2 x 3 x 4 gaps (App. B.4)


def expected_route_stats(counts: np.ndarray, subset: int) -> tuple[float, float]:
    """Exact expected (routes covered, TV vs uniform) of a uniform `subset`-path draw from accepted paths (App. B.4).

    counts: accepted paths per route (ROUTES,). Each route's count in the subset is hypergeometric, so both
    expectations are sums over its pmf; no resampling.
    """
    from math import comb

    total = int(counts.sum())
    m = min(subset, total)
    if m == 0:
        return 0.0, float("nan")
    denom = comb(total, m)
    covered = sum(1 - comb(total - int(c), m) / denom for c in counts)
    tv = 0.0
    for c in map(int, counts):
        for x in range(max(0, m - (total - c)), min(c, m) + 1):
            tv += comb(c, x) * comb(total - c, m - x) / denom * abs(x / m - 1 / ROUTES)
    return float(covered), 0.5 * tv
