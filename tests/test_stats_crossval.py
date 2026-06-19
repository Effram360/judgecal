"""Cross-validation of judgecal.stats against statsmodels references.

statsmodels is a dev/test-only dependency (never imported from src/);
the whole module is skipped when it is unavailable. Tolerances per the
implementation contract: 1e-6 for closed-form estimators, 1e-4 for IRLS
coefficients/SEs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar  # noqa: E402
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from judgecal.stats import (  # noqa: E402
    bh_fdr,
    fleiss_kappa,
    logistic_fit,
    mcnemar_test,
    wilson_ci,
)

ATOL_CLOSED_FORM = 1e-6
ATOL_IRLS = 1e-4


# ---------------------------------------------------------------------------
# wilson_ci vs proportion_confint(method="wilson")
# ---------------------------------------------------------------------------


class TestWilsonVsStatsmodels:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    @pytest.mark.parametrize("n", [5, 10, 50, 200, 1000])
    def test_matches_over_grid(self, n: int, alpha: float) -> None:
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            ours = wilson_ci(k, n, alpha=alpha)
            ref = proportion_confint(k, n, alpha=alpha, method="wilson")
            np.testing.assert_allclose(ours, ref, atol=ATOL_CLOSED_FORM)


# ---------------------------------------------------------------------------
# mcnemar_test(method="exact") vs statsmodels mcnemar(exact=True)
# ---------------------------------------------------------------------------


class TestMcNemarVsStatsmodels:
    @pytest.mark.parametrize(
        ("b", "c"),
        [(0, 5), (1, 1), (3, 3), (10, 2), (2, 10), (25, 25), (40, 12), (0, 1), (7, 19)],
    )
    def test_exact_pvalue_matches(self, b: int, c: int) -> None:
        # statsmodels reads discordant counts off the 2x2 anti-diagonal:
        # b = table[0, 1], c = table[1, 0]; the diagonal is irrelevant.
        table = np.array([[11, b], [c, 13]])
        ref = sm_mcnemar(table, exact=True)
        ours = mcnemar_test(b, c, method="exact")
        assert ours.p_value == pytest.approx(float(ref.pvalue), abs=ATOL_CLOSED_FORM)
        assert ours.statistic == pytest.approx(float(ref.statistic), abs=ATOL_CLOSED_FORM)


# ---------------------------------------------------------------------------
# bh_fdr vs multipletests(method="fdr_bh")
# ---------------------------------------------------------------------------


class TestBhFdrVsStatsmodels:
    @pytest.mark.parametrize("size", [1, 2, 7, 50, 200])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_random_vectors(self, size: int, seed: int) -> None:
        rng = np.random.default_rng(100 * seed + size)
        p = rng.uniform(0.0, 1.0, size)
        ref = multipletests(p, method="fdr_bh")[1]
        np.testing.assert_allclose(bh_fdr(p), ref, atol=ATOL_CLOSED_FORM)

    def test_matches_with_ties_and_extremes(self) -> None:
        p = np.array([0.0, 0.01, 0.01, 0.5, 0.5, 0.5, 1.0, 1.0, 0.2])
        ref = multipletests(p, method="fdr_bh")[1]
        np.testing.assert_allclose(bh_fdr(p), ref, atol=ATOL_CLOSED_FORM)


# ---------------------------------------------------------------------------
# fleiss_kappa vs statsmodels.stats.inter_rater.fleiss_kappa
# ---------------------------------------------------------------------------


class TestFleissVsStatsmodels:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_random_tables(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet(np.ones(4))
        table = rng.multinomial(6, probs, size=30)  # 30 items, 6 raters, 4 cats
        ours = fleiss_kappa(table)
        ref = float(sm_fleiss_kappa(table, method="fleiss"))
        assert ours == pytest.approx(ref, abs=ATOL_CLOSED_FORM)

    def test_matches_concentrated_table(self) -> None:
        rng = np.random.default_rng(9)
        # high-agreement table: most mass on a per-item modal category
        modal = rng.integers(0, 3, size=25)
        table = np.zeros((25, 3), dtype=int)
        for i, m in enumerate(modal):
            table[i, m] = 4
            table[i, (m + 1) % 3] = 1
        ours = fleiss_kappa(table)
        ref = float(sm_fleiss_kappa(table, method="fleiss"))
        assert ours == pytest.approx(ref, abs=ATOL_CLOSED_FORM)


# ---------------------------------------------------------------------------
# logistic_fit vs sm.Logit
# ---------------------------------------------------------------------------


class TestLogitVsStatsmodels:
    @staticmethod
    def _simulate(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.normal(0.0, 1.0, size=(n, 2))
        eta = -0.3 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
        y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        return X, y

    @pytest.mark.parametrize("seed", [0, 7, 21])
    def test_coefficients_and_se_match(self, seed: int) -> None:
        X, y = self._simulate(seed)
        ours = logistic_fit(X, y, add_intercept=True)
        assert ours.converged
        ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        np.testing.assert_allclose(ours.coef, np.asarray(ref.params), atol=ATOL_IRLS)
        np.testing.assert_allclose(ours.se, np.asarray(ref.bse), atol=ATOL_IRLS)

    def test_loglik_matches(self) -> None:
        X, y = self._simulate(3)
        ours = logistic_fit(X, y)
        ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        assert ours.loglik == pytest.approx(float(ref.llf), abs=1e-6)

    def test_no_intercept_matches(self) -> None:
        X, y = self._simulate(5)
        ours = logistic_fit(X, y, add_intercept=False)
        ref = sm.Logit(y, X).fit(disp=0)
        np.testing.assert_allclose(ours.coef, np.asarray(ref.params), atol=ATOL_IRLS)
        np.testing.assert_allclose(ours.se, np.asarray(ref.bse), atol=ATOL_IRLS)
