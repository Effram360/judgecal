"""Result dataclasses for :mod:`judgecal.stats`.

Small, serialization-friendly containers returned by the estimators. They
carry everything a probe needs to populate a
:class:`judgecal.core.Estimate` (point estimate, interval, SE, p-value).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a cluster bootstrap (:func:`judgecal.stats.cluster_bootstrap_ci`).

    Attributes:
        estimate: Statistic computed on the full (non-resampled) data.
        ci_low: Lower confidence bound.
        ci_high: Upper confidence bound.
        n_clusters: Number of distinct clusters resampled.
        boot_se: Standard deviation (ddof=1) of the valid bootstrap
            replicates — the bootstrap standard error of the statistic.
        n_boot: Number of *valid* (finite) bootstrap replicates actually
            used for the interval; at most the requested ``n_boot``.
        method: CI construction method, ``"percentile"`` or ``"basic"``.
        alpha: Two-sided miscoverage level of the interval (0.05 → 95% CI).
        null_value: Null value the p-value tests against, if supplied.
        p_value: Two-sided bootstrap p-value obtained by percentile-rank
            inversion of the CI (smallest alpha at which the interval
            excludes ``null_value``); ``None`` when no null was supplied.
        warnings: Caveats about the inference (e.g. the few-cluster
            anti-conservative regime). Callers should surface these to
            the user alongside the interval.
    """

    estimate: float
    ci_low: float
    ci_high: float
    n_clusters: int
    boot_se: float
    n_boot: int
    method: str
    alpha: float
    null_value: float | None = None
    p_value: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of an exact / mid-p McNemar test on discordant pair counts.

    Attributes:
        b: Count of pairs discordant in the first direction.
        c: Count of pairs discordant in the second direction.
        n_discordant: ``b + c``.
        statistic: ``min(b, c)``, the exact binomial test statistic.
        estimate: ``b / (b + c)``, the proportion of discordant pairs in
            the first direction (``nan`` when ``b + c == 0``). Null value
            under symmetry is 0.5.
        p_value: Two-sided p-value (1.0 when ``b + c == 0``: no discordant
            pairs carry no evidence against symmetry).
        method: ``"exact"`` or ``"midp"``.
    """

    b: int
    c: int
    n_discordant: int
    statistic: float
    estimate: float
    p_value: float
    method: str


@dataclass(frozen=True, eq=False)
class LogisticFit:
    """Outcome of an IRLS logistic regression (:func:`judgecal.stats.logistic_fit`).

    ``eq=False`` because ndarray fields make field-wise ``==`` ambiguous.

    Attributes:
        coef: Coefficient vector. When the model was fit with
            ``add_intercept=True`` the intercept is ``coef[0]`` and the
            slopes follow in column order of ``X``.
        se: Wald standard errors, ``sqrt(diag((X' W X)^{-1}))`` at the
            converged coefficients (model-based, NOT sandwich/clustered —
            use :func:`judgecal.stats.cluster_bootstrap_ci` re-fitting for
            cluster-robust inference).
        converged: Whether IRLS reached the coefficient-change tolerance
            within ``max_iter`` iterations.
        n_iter: Number of IRLS iterations performed.
        n_obs: Number of observations.
        loglik: Bernoulli log-likelihood at the final coefficients.
    """

    coef: np.ndarray
    se: np.ndarray
    converged: bool
    n_iter: int
    n_obs: int
    loglik: float
