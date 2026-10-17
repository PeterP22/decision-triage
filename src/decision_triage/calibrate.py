"""Post-hoc temperature scaling for methods that return a full probability distribution.

Only Decisions returns one. Responses gives a single verbalised confidence (the other queues are a uniform
spread of the remainder), so rescaling it would only re-map one number; it is reported raw.
"""

import numpy as np
from numpy.typing import ArrayLike

P_MIN = 1e-6
_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0


def scale(probs: ArrayLike, temperature: float) -> np.ndarray:
    """softmax(log(p) / T) row-wise, with p clipped to [P_MIN, 1]."""
    logits = np.log(np.clip(np.asarray(probs, dtype=float), P_MIN, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs: ArrayLike, gold: ArrayLike, temperature: float) -> float:
    p = scale(probs, temperature)
    g = np.asarray(gold, dtype=int)
    return float(-np.log(p[np.arange(g.size), g]).mean())


def fit_temperature(
    probs: ArrayLike, gold: ArrayLike, t_min: float = 0.05, t_max: float = 20.0, tol: float = 1e-6
) -> float:
    """Minimise dev NLL over T by golden-section search on beta = 1/T.

    NLL is convex in beta (logsumexp of a linear function minus a linear term), so the 1-D search is exact.
    """
    p = np.asarray(probs, dtype=float)
    g = np.asarray(gold, dtype=int)
    if g.size == 0:
        return 1.0

    def f(beta: float) -> float:
        return nll(p, g, 1.0 / beta)

    a, b = 1.0 / t_max, 1.0 / t_min
    c, d = b - _INV_PHI * (b - a), a + _INV_PHI * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = f(d)
    return float(2.0 / (a + b))
