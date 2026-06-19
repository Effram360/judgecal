"""False discovery rate control via Benjamini-Hochberg."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (monotone adjusted p-values).

    For m p-values sorted ascending, ``p_(1) <= ... <= p_(m)``, the
    step-up adjusted value is::

        q_(i) = min_{j >= i} ( m · p_(j) / j ),  clipped to [0, 1]

    computed as a reverse cumulative minimum and mapped back to the input
    order. Rejecting all hypotheses with ``q <= alpha`` controls the
    false discovery rate at level ``alpha`` for independent or positively
    dependent tests (Benjamini & Hochberg 1995). Matches
    ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``
    corrected p-values exactly.

    Properties guaranteed by construction: ``q_i >= p_i`` elementwise,
    ``q`` is monotone non-decreasing in ``p``, and ``q`` is bounded in
    [0, 1].

    Args:
        pvals: Raw p-values, each in [0, 1]. May be empty.

    Returns:
        Array of q-values aligned with the input order (empty input →
        empty array).

    Raises:
        ValueError: If any p-value is non-finite or outside [0, 1].

    References:
        Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false
        discovery rate: a practical and powerful approach to multiple
        testing." *JRSS-B* 57(1), 289-300.
    """
    p = np.asarray(list(pvals), dtype=float)
    if p.size == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("all p-values must be finite and within [0, 1]")

    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q
