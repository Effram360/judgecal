"""Logistic regression via iteratively reweighted least squares (IRLS)."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, xlogy

from judgecal.stats.types import LogisticFit

_MU_EPS = 1e-10


def logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogisticFit:
    """Maximum-likelihood logistic regression via IRLS (Newton-Raphson).

    Model: ``P(y = 1 | x) = expit(x' β)``. Starting from ``β = 0``, each
    iteration performs the Newton step on the Bernoulli log-likelihood::

        μ = expit(X β);   W = diag(μ (1 − μ))
        β ← β + (X' W X)^{-1} X' (y − μ)

    until the largest absolute coefficient change falls below ``tol`` or
    ``max_iter`` is reached (then ``converged=False``). Weights are
    floored at 1e-10 to keep the Fisher information invertible when
    fitted probabilities saturate; a singular information matrix falls
    back to a least-squares (pseudoinverse) Newton step. Note that under
    perfect separation the MLE does not exist and IRLS will fail to
    converge — check ``converged``.

    Standard errors are model-based Wald, ``sqrt(diag((X' W X)^{-1}))``
    at the converged β — deliberately NOT sandwich or cluster-robust.
    For probe inference, CIs on coefficients come from
    :func:`judgecal.stats.cluster_bootstrap_ci` re-fitting the model on
    cluster resamples (Miller, arXiv:2411.00640); the Wald SEs here serve
    the observational verbosity GLM and cross-validation against
    ``statsmodels.Logit``.

    Args:
        X: Design matrix of shape (n, p) — or (n,) for one predictor.
        y: Binary responses of shape (n,), values in {0, 1}.
        add_intercept: If True (default), prepend a column of ones;
            ``coef[0]`` is then the intercept (statsmodels
            ``add_constant`` ordering).
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance on the max absolute coefficient change.

    Returns:
        A :class:`~judgecal.stats.types.LogisticFit` with ``coef``,
        ``se``, ``converged``, ``n_iter``, ``n_obs``, ``loglik``.

    Raises:
        ValueError: On empty input, shape mismatch, non-finite values,
            non-binary ``y``, or ``n <= p`` (underdetermined).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim={X_arr.ndim}")
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n == 0:
        raise ValueError("y is empty")
    if X_arr.shape[0] != n:
        raise ValueError(f"X has {X_arr.shape[0]} rows but y has {n} entries")
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("X and y must be finite")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("y must contain only 0 and 1")

    if add_intercept:
        X_arr = np.column_stack([np.ones(n), X_arr])
    p = X_arr.shape[1]
    if n <= p:
        raise ValueError(f"need n > p, got n={n}, p={p}")

    beta = np.zeros(p)
    converged = False
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        mu = expit(X_arr @ beta)
        w = np.maximum(mu * (1.0 - mu), _MU_EPS)
        info = (X_arr.T * w) @ X_arr  # X' W X
        score = X_arr.T @ (y_arr - mu)
        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(info, score, rcond=None)[0]
        beta = beta + delta
        if float(np.max(np.abs(delta))) < tol:
            converged = True
            n_iter = it
            break

    mu = expit(X_arr @ beta)
    w = np.maximum(mu * (1.0 - mu), _MU_EPS)
    info = (X_arr.T * w) @ X_arr
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    mu_safe = np.clip(mu, _MU_EPS, 1.0 - _MU_EPS)
    loglik = float(np.sum(xlogy(y_arr, mu_safe) + xlogy(1.0 - y_arr, 1.0 - mu_safe)))

    return LogisticFit(
        coef=beta,
        se=se,
        converged=converged,
        n_iter=n_iter,
        n_obs=n,
        loglik=loglik,
    )
