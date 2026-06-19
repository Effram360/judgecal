"""Exact and mid-p McNemar tests for paired binary outcomes."""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import binom

from judgecal.stats.types import McNemarResult


def mcnemar_test(b: int, c: int, method: Literal["exact", "midp"] = "midp") -> McNemarResult:
    """McNemar test of marginal symmetry from discordant pair counts.

    Only the discordant pairs are informative: under the null of symmetry
    (no systematic direction), ``b ~ Binomial(b + c, 1/2)``. With
    ``n = b + c`` and ``k = min(b, c)``:

    * ``"exact"``: two-sided exact binomial p-value
      ``p = min(1, 2 · P(X <= k))`` where ``X ~ Binomial(n, 1/2)`` —
      identical to ``statsmodels``' ``mcnemar(..., exact=True)``.
    * ``"midp"``: the mid-p variant subtracts half the point mass at the
      observed statistic, ``p = min(1, 2 · P(X <= k) − P(X = k))``. Mid-p
      is less conservative than the exact test while retaining close to
      nominal size (Fagerland, Lydersen & Laake 2013, BMC Med Res
      Methodol 13:91); it is the recommended default for the positional
      consistency probe.

    Edge cases (documented behavior):

    * ``b + c == 0``: no discordant pairs → no evidence against symmetry;
      ``p_value = 1.0`` and ``estimate = nan``.
    * ``b == c``: exact p capped at 1.0; mid-p equals exactly 1.0.
    * ``k == 0`` (all discordant pairs in one direction): the smallest
      attainable p, ``2 · 0.5**n`` (exact).

    Args:
        b: Discordant count in the first direction (>= 0).
        c: Discordant count in the second direction (>= 0).
        method: ``"exact"`` or ``"midp"`` (default).

    Returns:
        A :class:`~judgecal.stats.types.McNemarResult` with
        ``estimate = b / (b + c)`` (null value 0.5).

    Raises:
        ValueError: If ``b`` or ``c`` is negative, or ``method`` invalid.

    References:
        McNemar, Q. (1947). "Note on the sampling error of the difference
        between correlated proportions or percentages." *Psychometrika*
        12(2), 153-157.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative, got b={b}, c={c}")
    if method not in ("exact", "midp"):
        raise ValueError(f"method must be 'exact' or 'midp', got {method!r}")

    n = b + c
    k = min(b, c)
    if n == 0:
        return McNemarResult(
            b=b,
            c=c,
            n_discordant=0,
            statistic=0.0,
            estimate=math.nan,
            p_value=1.0,
            method=method,
        )

    # Mid-p subtracts the point mass from the UNcapped doubled CDF before
    # capping: at b == c this yields exactly 1.0 (2·CDF(k) = 1 + pmf(k)
    # by symmetry, so 2·CDF(k) − pmf(k) = 1).
    two_cdf = 2.0 * float(binom.cdf(k, n, 0.5))
    midp_correction = 0.0 if method == "exact" else float(binom.pmf(k, n, 0.5))
    p = min(1.0, two_cdf - midp_correction)
    return McNemarResult(
        b=b,
        c=c,
        n_discordant=n,
        statistic=float(k),
        estimate=b / n,
        p_value=max(0.0, p),
        method=method,
    )
