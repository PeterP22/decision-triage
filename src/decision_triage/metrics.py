from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 1001), 3)


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    accuracy: float
    confidence: float


def accuracy(correct: ArrayLike) -> float:
    c = np.asarray(correct, dtype=bool)
    return float(c.mean()) if c.size else float("nan")


def reliability_bins(confidence: ArrayLike, correct: ArrayLike, n_bins: int = 15) -> list[ReliabilityBin]:
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lo=b / n_bins,
                hi=(b + 1) / n_bins,
                count=n,
                accuracy=float(corr[mask].mean()) if n else float("nan"),
                confidence=float(conf[mask].mean()) if n else float("nan"),
            )
        )
    return bins


def ece(confidence: ArrayLike, correct: ArrayLike, n_bins: int = 15) -> float:
    n = np.asarray(confidence).size
    if n == 0:
        return float("nan")
    bins = reliability_bins(confidence, correct, n_bins)
    return float(sum(b.count / n * abs(b.accuracy - b.confidence) for b in bins if b.count))


def coverage_curve(
    confidence: ArrayLike, correct: ArrayLike, refused: ArrayLike | None = None, thresholds: np.ndarray = THRESHOLDS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, coverage, accuracy on covered). Refusals are never auto-routed."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    eligible = ~np.asarray(refused, dtype=bool) if refused is not None else np.ones(conf.size, dtype=bool)
    covered = (conf[None, :] >= thresholds[:, None]) & eligible[None, :]
    n_cov = covered.sum(axis=1)
    coverage = n_cov / conf.size if conf.size else np.zeros(thresholds.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(n_cov > 0, (covered & corr[None, :]).sum(axis=1) / n_cov, np.nan)
    return thresholds, coverage, acc


def choose_threshold(
    confidence: ArrayLike, correct: ArrayLike, refused: ArrayLike | None = None, target: float = 0.98
) -> float | None:
    t, _, acc = coverage_curve(confidence, correct, refused)
    ok = np.nonzero(np.nan_to_num(acc, nan=-1.0) >= target)[0]
    return float(t[ok[0]]) if ok.size else None


def at_threshold(
    threshold: float, confidence: ArrayLike, correct: ArrayLike, refused: ArrayLike | None = None
) -> tuple[float, float]:
    _, cov, acc = coverage_curve(confidence, correct, refused, thresholds=np.array([threshold]))
    return float(cov[0]), float(acc[0])


def aurc(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Area under the risk-coverage curve (lower is better).

    Mean over k = 1..n of the error rate among the k most confident predictions. Tied confidences are scored
    by their expectation under random tie-breaking, so the result does not depend on input order.
    """
    conf = np.asarray(confidence, dtype=float)
    err = ~np.asarray(correct, dtype=bool)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    conf, err = conf[order], err[order].astype(float)
    starts = np.flatnonzero(np.r_[True, conf[1:] != conf[:-1]])
    sizes = np.diff(np.r_[starts, n])
    group_rate = np.add.reduceat(err, starts) / sizes
    errors_before = np.r_[0.0, np.cumsum(err)][starts]
    group = np.repeat(np.arange(starts.size), sizes)
    k = np.arange(1, n + 1)
    expected_errors = errors_before[group] + (k - starts[group]) * group_rate[group]
    return float((expected_errors / k).mean())


def latency_percentiles(latencies: ArrayLike) -> tuple[float, float]:
    lat = np.asarray(latencies, dtype=float)
    if lat.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


def cost_per_1k(costs: ArrayLike) -> float:
    c = np.asarray(costs, dtype=float)
    return float(c.mean() * 1000) if c.size else float("nan")
