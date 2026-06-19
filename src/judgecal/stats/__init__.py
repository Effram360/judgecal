"""judgecal.stats — the statistical inference core.

Standalone-importable (pure numpy/scipy/pandas; the only judgecal import
permitted is ``judgecal.core.Estimate``). The five closed-form estimators
(Wilson CI, McNemar exact, BH-FDR, Fleiss' kappa, logistic GLM) are
cross-validated against statsmodels reference implementations in
``tests/test_stats_crossval.py``; the bootstrap/MDE machinery is
validated by a 200-seed frequentist coverage scenario in the validation
suite plus behavioral tests in ``tests/test_stats.py`` — the target
audience checks the math.

Key references: Wilson (1927) score intervals; Benjamini & Hochberg
(1995) FDR; Fleiss (1971) multi-rater kappa; Miller, "Adding Error Bars
to Evals" (arXiv:2411.00640) for clustered standard errors.
"""

from __future__ import annotations

from judgecal.stats.agreement import fleiss_kappa
from judgecal.stats.bootstrap import cluster_bootstrap_ci
from judgecal.stats.fdr import bh_fdr
from judgecal.stats.glm import logistic_fit
from judgecal.stats.intervals import wilson_ci
from judgecal.stats.mcnemar import mcnemar_test
from judgecal.stats.power import (
    design_effect,
    estimate_icc,
    mde_from_se,
    mde_mcnemar,
    mde_proportion,
)
from judgecal.stats.types import BootstrapResult, LogisticFit, McNemarResult

__all__ = [
    "BootstrapResult",
    "LogisticFit",
    "McNemarResult",
    "bh_fdr",
    "cluster_bootstrap_ci",
    "design_effect",
    "estimate_icc",
    "fleiss_kappa",
    "logistic_fit",
    "mcnemar_test",
    "mde_from_se",
    "mde_mcnemar",
    "mde_proportion",
    "wilson_ci",
]
