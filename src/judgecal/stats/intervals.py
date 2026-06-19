"""Closed-form confidence intervals for binomial proportions."""

from __future__ import annotations

import math

from scipy.stats import norm


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Inverts the normal-approximation score test (Wilson 1927). With
    ``z = Phi^{-1}(1 - alpha/2)`` and ``p̂ = k/n``::

        center = (p̂ + z²/(2n)) / (1 + z²/n)
        half   = z / (1 + z²/n) · sqrt( p̂(1−p̂)/n + z²/(4n²) )
        CI     = [center − half, center + half]

    Unlike the Wald interval it never collapses to zero width at
    ``k = 0`` or ``k = n`` (the bound at the boundary is exactly 0 or 1,
    the other bound is interior) and has good small-n coverage.

    Args:
        k: Number of successes, ``0 <= k <= n``.
        n: Number of trials, ``n > 0``.
        alpha: Two-sided miscoverage level; 0.05 yields a 95% CI.

    Returns:
        ``(ci_low, ci_high)``, clipped to [0, 1] against floating-point
        round-off (analytically the Wilson interval lies within [0, 1]).

    Raises:
        ValueError: If ``n <= 0``, ``k`` is outside ``[0, n]``, or
            ``alpha`` is outside ``(0, 1)``.

    References:
        Wilson, E. B. (1927). "Probable inference, the law of succession,
        and statistical inference." *JASA* 22(158), 209-212.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n] = [0, {n}], got {k}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
