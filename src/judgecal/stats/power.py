"""Power analysis: minimum detectable effects, design effects, and ICC.

Pre-registered power analysis is a project non-negotiable: every
null-bearing metric reports the minimum detectable effect (MDE) at the
realized effective sample size, so "no bias detected" is never silently
conflated with "underpowered to detect bias" (Miller, "Adding Error Bars
to Evals", arXiv:2411.00640, motivates the clustering correction).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def mde_proportion(
    n_eff: float,
    p0: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Two-sided minimum detectable effect for a one-sample proportion.

    Normal-approximation MDE: the smallest ``|p − p0|`` detectable with
    the given power by a two-sided level-``alpha`` test on ``n_eff``
    effective observations::

        MDE = (z_{1−α/2} + z_{power}) · sqrt( p0 · (1 − p0) / n_eff )

    The variance is evaluated at the null ``p0`` for both terms — the
    standard simplification, slightly conservative near ``p0 = 0.5``
    (where the variance is maximal). ``n_eff`` should already be
    deflated for clustering: ``n_eff = n / design_effect(...)``.

    Note: for very small ``n_eff`` the returned value can exceed the
    feasible range (e.g. > 1 − p0); it is reported as-is so that
    "underpowered" flags trigger naturally.

    Args:
        n_eff: Effective sample size (> 0); ``n / DEFF`` under clustering.
            ``n_eff == 0`` returns ``inf`` (nothing is detectable).
        p0: Null proportion in (0, 1); 0.5 for pick-rate probes.
        alpha: Two-sided test level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        MDE on the proportion scale (``inf`` when ``n_eff == 0``).

    Raises:
        ValueError: If ``n_eff < 0`` or any of ``p0``/``alpha``/``power``
            lies outside (0, 1).
    """
    if n_eff < 0:
        raise ValueError(f"n_eff must be non-negative, got {n_eff}")
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if n_eff == 0:
        return math.inf
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    return (z_alpha + z_power) * math.sqrt(p0 * (1.0 - p0) / n_eff)


def mde_from_se(se: float, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Normal-approximation MDE for an estimator with standard error ``se``.

    ::

        MDE = (z_{1−α/2} + z_{power}) · se

    This is the *realized-SE* MDE: ``se`` should be the standard error the
    inference actually uses (e.g. ``BootstrapResult.boot_se`` from the
    cluster bootstrap), so the MDE is consistent by construction with the
    reported CI — clustering, negative within-cluster correlation, and
    any other dependence are already baked into the SE. Probes use this
    for every cluster-bootstrap-CI'd metric; :func:`mde_proportion` is
    the *planning* counterpart (no data yet, variance from the null).

    Args:
        se: Realized standard error of the estimator.
        alpha: Two-sided test level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        MDE on the estimator's own scale, or ``None`` when ``se`` is
        non-finite or non-positive (degenerate bootstrap distribution).
    """
    if not math.isfinite(se) or se <= 0.0:
        return None
    return float((norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)) * se)


def mde_mcnemar(n_discordant: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable asymmetry ``|P(b) − 0.5|`` among discordant pairs.

    The McNemar test reduces to a one-sample binomial test with null
    proportion 0.5 on the ``n_discordant`` discordant pairs, so the MDE
    is :func:`mde_proportion` evaluated there::

        MDE = (z_{1−α/2} + z_{power}) · 0.5 / sqrt(n_discordant)

    Args:
        n_discordant: Number of discordant pairs (``b + c``).
            0 returns ``inf`` — with no discordant pairs no asymmetry is
            detectable at any size.
        alpha: Two-sided test level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        MDE on the discordant-proportion scale (``inf`` when
        ``n_discordant == 0``).

    Raises:
        ValueError: If ``n_discordant < 0`` or ``alpha``/``power`` lie
            outside (0, 1).
    """
    return mde_proportion(float(n_discordant), p0=0.5, alpha=alpha, power=power)


def design_effect(cluster_sizes: Sequence[int], icc: float) -> float:
    """Kish design effect for cluster sampling.

    ::

        DEFF = 1 + (m̄ − 1) · ICC,   m̄ = mean cluster size

    The variance inflation of a cluster sample relative to a simple
    random sample of equal size; divide n by DEFF to obtain the
    effective sample size used in :func:`mde_proportion` (Miller,
    arXiv:2411.00640, documents >3x SE inflation in clustered evals).

    Args:
        cluster_sizes: Sizes of each cluster (all >= 1, non-empty).
        icc: Intraclass correlation in [0, 1] (see :func:`estimate_icc`).

    Returns:
        The design effect (>= 1.0 for non-negative ICC).

    Raises:
        ValueError: On empty ``cluster_sizes``, any size < 1, or ``icc``
            outside [0, 1].
    """
    sizes = np.asarray(list(cluster_sizes), dtype=float)
    if sizes.size == 0:
        raise ValueError("cluster_sizes must be non-empty")
    if np.any(sizes < 1):
        raise ValueError("all cluster sizes must be >= 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [0, 1], got {icc}")
    m_bar = float(sizes.mean())
    return 1.0 + (m_bar - 1.0) * icc


def estimate_icc(df: pd.DataFrame, value_col: str, cluster_col: str) -> float:
    """One-way ANOVA (method-of-moments) intraclass correlation estimate.

    With k clusters of sizes ``n_g`` (N total), between/within mean
    squares ``MSB = SSB/(k−1)``, ``MSW = SSW/(N−k)``, and the unbalanced-
    design average cluster size ``n0 = (N − Σ n_g² / N) / (k − 1)``::

        ICC = (MSB − MSW) / (MSB + (n0 − 1) · MSW)

    (the standard ANOVA estimator, e.g. Donner 1986), clipped to [0, 1] —
    the method-of-moments estimate can be negative under sampling noise
    and a negative ICC has no design-effect interpretation.

    Degenerate cases return 0.0: fewer than 2 clusters, all clusters of
    size 1 (within variance unidentifiable), or zero total variance.

    Args:
        df: Input frame.
        value_col: Numeric column (binary outcomes are fine — the ANOVA
            estimator remains a consistent method-of-moments ICC).
        cluster_col: Column whose distinct values define clusters.

    Returns:
        Estimated ICC in [0, 1].

    Raises:
        ValueError: If columns are missing or ``value_col`` contains
            non-finite values.
    """
    if value_col not in df.columns:
        raise ValueError(f"value_col {value_col!r} not in df columns")
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col {cluster_col!r} not in df columns")
    values = df[value_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{value_col!r} contains non-finite values")

    codes, uniques = pd.factorize(df[cluster_col].to_numpy())
    k = int(len(uniques))
    n_total = int(values.size)
    if k < 2 or n_total <= k:
        # <2 clusters, or all-singleton clusters: within variance unidentifiable.
        return 0.0

    counts = np.bincount(codes, minlength=k).astype(float)
    cluster_means = np.bincount(codes, weights=values, minlength=k) / counts
    grand_mean = float(values.mean())

    ssb = float(np.sum(counts * (cluster_means - grand_mean) ** 2))
    ssw = float(np.sum((values - cluster_means[codes]) ** 2))
    msb = ssb / (k - 1)
    msw = ssw / (n_total - k)
    n0 = (n_total - float(np.sum(counts**2)) / n_total) / (k - 1)

    denom = msb + (n0 - 1.0) * msw
    if denom <= 0.0:
        return 0.0
    icc = (msb - msw) / denom
    return float(min(max(icc, 0.0), 1.0))
