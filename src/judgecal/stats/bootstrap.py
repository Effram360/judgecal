"""Cluster (block) bootstrap confidence intervals and p-values.

Judgments of the same item are dependent (the same pair is judged under
swap, padding, template, and repeat conditions). Naive row-level
resampling understates this dependence; clustered standard errors can
exceed naive ones by more than 3x in eval settings (Miller, "Adding
Error Bars to Evals", arXiv:2411.00640). Resampling whole clusters
(items) with replacement is the nonparametric analogue of clustered SEs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from judgecal.stats.types import BootstrapResult

#: Below this many clusters the percentile cluster bootstrap is known to
#: undercover and a warning is attached to the result (see
#: :func:`cluster_bootstrap_ci`, "Few-cluster regime").
FEW_CLUSTER_WARNING_THRESHOLD = 15


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    cluster_col: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    method: Literal["percentile", "basic"] = "percentile",
    null_value: float | None = None,
) -> BootstrapResult:
    """Cluster bootstrap CI (and optional p-value) for an arbitrary statistic.

    Algorithm: let the G distinct values of ``df[cluster_col]`` define
    clusters. For each of ``n_boot`` replicates, draw G cluster ids
    uniformly *with replacement*, concatenate all rows of each drawn
    cluster (rows of a cluster drawn twice appear twice), and recompute
    ``stat_fn`` on the resampled frame. This preserves within-cluster
    dependence, unlike row-level resampling (Miller, arXiv:2411.00640).

    CI construction (equal-tailed, level ``1 - alpha``):

    * ``"percentile"``: ``[Q*(alpha/2), Q*(1 - alpha/2)]`` — empirical
      quantiles of the bootstrap replicates.
    * ``"basic"`` (reflected): ``[2·t̂ − Q*(1 − alpha/2), 2·t̂ − Q*(alpha/2)]``
      where ``t̂`` is the full-sample statistic.

    Bootstrap p-value — percentile-rank CI inversion. When ``null_value``
    is supplied, the p-value is the two-sided percentile rank of the null
    within the B valid replicates ``t*`` (finite-sample-corrected)::

        p = min(1, 2 · min(#{t* <= null} + 1, #{t* >= null} + 1) / (B + 1))

    The +1/(B+1) correction guarantees ``p >= 1/(B+1)`` — a bootstrap
    p-value is never exactly 0. For the ``"basic"`` method the identical
    rank computation is applied at the *reflected* point
    ``2·t̂ − null_value``, because the basic interval excludes the null
    exactly when the percentile interval excludes that reflection.

    The p-value agrees with the CI's exclusion decision up to the
    quantile-interpolation discreteness of order 1/B: the rank-based
    p-value and the linearly interpolated ``np.quantile`` CI can disagree
    in a hairline band at small B (e.g. at B≈500, occasional cases where
    the 95% CI excludes the null but p ≈ 0.052–0.056 — always in the
    conservative direction). At the default ``n_boot=2000`` no
    disagreement was observed over 1200 randomized trials.

    **Few-cluster regime (warning attached, no estimator switch):** the
    percentile cluster bootstrap is anti-conservative with few clusters.
    Measured on a binary clustered DGP (true mean 0.5, ICC 0.5, m=4, 600
    sims), coverage of the nominal 95% CI / type-I error of the nominal
    5% test were: G=3 → 0.823/0.172, G=5 → 0.863/0.132, G=8 →
    0.915/0.078, G=12 → 0.942/0.057, G=20 → 0.950/0.048. When
    ``n_clusters < 15`` the returned result carries a warning in
    ``BootstrapResult.warnings``; treat such CIs and p-values as
    optimistic.

    Performance: cluster row-index lists are computed once via a stable
    argsort; each replicate assembles its row indices with vectorized
    numpy (no per-iteration ``groupby``). 2000 replicates over a
    ~1000-row frame with a cheap ``stat_fn`` complete in well under a
    second on a laptop.

    Args:
        df: Input frame; passed (resampled) to ``stat_fn``. The original
            index labels are preserved in resampled frames and may repeat.
        stat_fn: Maps a DataFrame to a scalar statistic. Must tolerate
            duplicated rows/index labels. Replicates where it returns a
            non-finite value are dropped (recorded via ``n_boot``).
        cluster_col: Column whose distinct values define the resampling
            clusters (typically ``item_id``).
        n_boot: Number of bootstrap replicates to draw.
        alpha: Two-sided miscoverage level; 0.05 yields a 95% CI.
        seed: Seed for ``numpy.random.default_rng`` (fully deterministic).
        method: ``"percentile"`` or ``"basic"`` (see above).
        null_value: Optional null hypothesis value; when given, the
            returned result carries the inversion p-value described above.

    Returns:
        A :class:`~judgecal.stats.types.BootstrapResult`.

    Raises:
        ValueError: On empty input, fewer than 2 clusters, invalid
            ``alpha``/``n_boot``/``method``, a non-finite full-sample
            statistic, or if more than half the replicates fail.
    """
    if len(df) == 0:
        raise ValueError("df is empty")
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col {cluster_col!r} not in df columns")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    if method not in ("percentile", "basic"):
        raise ValueError(f"method must be 'percentile' or 'basic', got {method!r}")

    estimate = float(stat_fn(df))
    if not np.isfinite(estimate):
        raise ValueError(f"stat_fn returned a non-finite value on the full data: {estimate}")

    codes, uniques = pd.factorize(df[cluster_col].to_numpy())
    n_clusters = int(len(uniques))
    if n_clusters < 2:
        raise ValueError(f"need >= 2 clusters to bootstrap, got {n_clusters}")

    # Precompute per-cluster row positions once: `order` lists row positions
    # grouped by cluster; cluster g occupies order[starts[g] : starts[g] + counts[g]].
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_clusters)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    all_singletons = bool(counts.max() == 1)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        if all_singletons:
            idx = order[draw]
        else:
            sizes = counts[draw]
            ends = np.cumsum(sizes)
            # Position within the output run of each drawn cluster (0..size-1),
            # offset by that cluster's start in `order` — fully vectorized.
            within = np.arange(ends[-1]) - np.repeat(ends - sizes, sizes)
            idx = order[np.repeat(starts[draw], sizes) + within]
        boot[i] = stat_fn(df.take(idx))

    valid = boot[np.isfinite(boot)]
    if valid.size < max(2, n_boot // 2):
        raise ValueError(
            f"only {valid.size}/{n_boot} bootstrap replicates produced a finite "
            "statistic; stat_fn is too fragile for these data"
        )

    q_low, q_high = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
    if method == "percentile":
        ci_low, ci_high = float(q_low), float(q_high)
    else:  # basic / reflected
        ci_low, ci_high = 2.0 * estimate - float(q_high), 2.0 * estimate - float(q_low)

    p_value: float | None = None
    if null_value is not None:
        # Percentile-rank inversion (see docstring). For "basic", rank the
        # reflected point in the same percentile distribution.
        point = null_value if method == "percentile" else 2.0 * estimate - null_value
        n_le = int(np.sum(valid <= point))
        n_ge = int(np.sum(valid >= point))
        p_value = min(1.0, 2.0 * min(n_le + 1, n_ge + 1) / (valid.size + 1))

    warnings: list[str] = []
    if n_clusters < FEW_CLUSTER_WARNING_THRESHOLD:
        warnings.append(
            f"only {n_clusters} clusters: percentile cluster-bootstrap CIs are "
            f"anti-conservative below ~{FEW_CLUSTER_WARNING_THRESHOLD} clusters "
            "(measured coverage of nominal 95% CIs: ~0.82 at G=3, ~0.86 at G=5, "
            "~0.92 at G=8, ~0.94 at G=12; type-I error up to ~3.4x nominal); "
            "interpret the CI and p-value as optimistic"
        )

    return BootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_clusters=n_clusters,
        boot_se=float(np.std(valid, ddof=1)),
        n_boot=int(valid.size),
        method=method,
        alpha=alpha,
        null_value=null_value,
        p_value=p_value,
        warnings=warnings,
    )
