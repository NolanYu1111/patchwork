"""Statistical primitives for Patchwork.

Everything here works on plain NumPy arrays so it can be tested without the web app.
AUROC is computed with the Mann-Whitney rank formulation (ties count half), which
needs no scikit-learn dependency and matches ``sklearn.metrics.roc_auc_score``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y_true, scores) -> float:
    """Area under the ROC curve. NaN if only one class is present."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve_points(y_true, scores, max_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates for plotting, thinned to ``max_points``."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(y) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return fpr, tpr


def bootstrap_auroc(y_true, scores, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile-bootstrap interval for AUROC.

    Returns ``(point, low, high, samples)``. Resamples that lose a class are skipped.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    point = auroc(y, s)
    if np.isnan(point):
        return point, float("nan"), float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = auroc(y[idx], s[idx])
        if not np.isnan(v):
            samples[k] = v
            k += 1
    samples = samples[:k]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), samples


def gap_interval(samples_a: np.ndarray, samples_b: np.ndarray, alpha: float = 0.05, seed: int = 1):
    """Interval for AUROC(a) − AUROC(b) from two independent bootstrap distributions.

    Groups are bootstrapped independently (they contain different people), so the
    difference distribution is formed by pairing shuffled draws.
    """
    if len(samples_a) == 0 or len(samples_b) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = min(len(samples_a), len(samples_b))
    diff = rng.permutation(samples_a)[:m] - rng.permutation(samples_b)[:m]
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """ECE with equal-width bins on predicted probability of the positive class."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 0, 1)
    if len(y) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def balanced_accuracy(y_true, probs, threshold: float = 0.5) -> float:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float(((pred[pos] == 1).mean() + (pred[neg] == 0).mean()) / 2)


def majority_floor(y_true) -> float:
    """Accuracy a model gets by always predicting the more common class."""
    y = np.asarray(y_true).astype(int)
    if len(y) == 0:
        return float("nan")
    p = y.mean()
    return float(max(p, 1 - p))
