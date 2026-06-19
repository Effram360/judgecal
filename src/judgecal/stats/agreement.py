"""Inter-rater agreement statistics."""

from __future__ import annotations

import math

import numpy as np

_DEGENERATE_EPS = 1e-12


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for agreement among a fixed number of raters.

    ``table[i, j]`` is the number of raters assigning item ``i`` to
    category ``j``; every row must sum to the same number of raters
    ``n >= 2`` (balanced design — same requirement as statsmodels'
    ``fleiss_kappa``). With N items::

        P_i  = ( Σ_j table[i,j]² − n ) / ( n (n − 1) )   per-item agreement
        P̄    = mean_i P_i                                observed agreement
        p_j  = Σ_i table[i,j] / (N n)                    category shares
        P̄_e  = Σ_j p_j²                                  chance agreement
        κ    = ( P̄ − P̄_e ) / ( 1 − P̄_e )

    (Fleiss 1971). κ = 1 is perfect agreement; κ ≈ 0 is chance-level;
    κ < 0 is worse than chance.

    Degenerate case: if every rating falls in a single category then
    ``P̄_e = 1`` and κ is 0/0. By convention this returns 1.0 when
    observed agreement is also perfect (it always is in that case, but
    checked explicitly) and ``nan`` otherwise. Callers (the template and
    stability probes) should treat ``nan`` as "kappa undefined".

    Args:
        table: Integer-valued array of shape (n_items, n_categories) with
            ``n_items >= 1``, ``n_categories >= 2``, and constant row sums
            ``n >= 2``.

    Returns:
        Fleiss' kappa (see degenerate case above).

    Raises:
        ValueError: If the table is not 2-D, has fewer than 2 categories,
            contains negative or non-finite entries, or has unequal row
            sums / fewer than 2 ratings per item.

    References:
        Fleiss, J. L. (1971). "Measuring nominal scale agreement among
        many raters." *Psychological Bulletin* 76(5), 378-382.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"table must be 2-D (items x categories), got ndim={arr.ndim}")
    n_items, n_cat = arr.shape
    if n_items < 1:
        raise ValueError("table must contain at least one item")
    if n_cat < 2:
        raise ValueError(f"table must have >= 2 categories, got {n_cat}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("table entries must be finite and non-negative")

    row_sums = arr.sum(axis=1)
    n = float(row_sums[0])
    if not np.all(row_sums == n):
        raise ValueError("all rows must sum to the same number of raters (balanced design)")
    if n < 2:
        raise ValueError(f"need >= 2 ratings per item, got {n:g}")

    p_i = (np.sum(arr * arr, axis=1) - n) / (n * (n - 1.0))
    p_bar = float(p_i.mean())
    p_j = arr.sum(axis=0) / (n_items * n)
    p_e = float(np.sum(p_j * p_j))

    if 1.0 - p_e <= _DEGENERATE_EPS:
        # All ratings in one category: kappa is 0/0; see docstring.
        return 1.0 if p_bar >= 1.0 - _DEGENERATE_EPS else math.nan
    return (p_bar - p_e) / (1.0 - p_e)
