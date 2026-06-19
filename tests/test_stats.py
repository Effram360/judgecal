"""Behavioral tests for judgecal.stats.

Cross-validation against statsmodels lives in test_stats_crossval.py;
these tests check statistical *behavior*: clustering widens CIs, BH
q-values are monotone and bounded, MDEs shrink with n, bootstrap CIs
cover planted truths, and documented edge cases hold exactly.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from judgecal.stats import (
    BootstrapResult,
    LogisticFit,
    McNemarResult,
    bh_fdr,
    cluster_bootstrap_ci,
    design_effect,
    estimate_icc,
    fleiss_kappa,
    logistic_fit,
    mcnemar_test,
    mde_from_se,
    mde_mcnemar,
    mde_proportion,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mean_value(df: pd.DataFrame) -> float:
    return float(df["value"].mean())


def _clustered_frame(
    n_clusters: int = 40,
    m: int = 25,
    cluster_sd: float = 1.0,
    noise_sd: float = 0.3,
    seed: int = 3,
) -> pd.DataFrame:
    """High-ICC frame: big shared cluster effects, small within noise."""
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, cluster_sd, n_clusters)
    cluster = np.repeat(np.arange(n_clusters), m)
    value = effects[cluster] + rng.normal(0.0, noise_sd, n_clusters * m)
    return pd.DataFrame({"cluster": [f"c{g}" for g in cluster], "value": value})


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_contains_point_estimate(self) -> None:
        lo, hi = wilson_ci(37, 100)
        assert lo < 0.37 < hi

    def test_k_zero_edge(self) -> None:
        lo, hi = wilson_ci(0, 20)
        assert lo == 0.0
        assert 0.0 < hi < 0.5

    def test_k_equals_n_edge(self) -> None:
        lo, hi = wilson_ci(20, 20)
        assert hi == 1.0
        assert 0.5 < lo < 1.0

    def test_symmetry_around_half(self) -> None:
        lo1, hi1 = wilson_ci(30, 100)
        lo2, hi2 = wilson_ci(70, 100)
        assert lo1 == pytest.approx(1.0 - hi2, abs=1e-12)
        assert hi1 == pytest.approx(1.0 - lo2, abs=1e-12)

    def test_narrows_with_n(self) -> None:
        w_small = np.diff(wilson_ci(10, 20))[0]
        w_large = np.diff(wilson_ci(500, 1000))[0]
        assert w_large < w_small

    def test_lower_alpha_widens(self) -> None:
        w95 = np.diff(wilson_ci(40, 100, alpha=0.05))[0]
        w99 = np.diff(wilson_ci(40, 100, alpha=0.01))[0]
        assert w99 > w95

    @pytest.mark.parametrize(
        ("k", "n", "alpha"),
        [(0, 0, 0.05), (-1, 10, 0.05), (11, 10, 0.05), (5, 10, 0.0), (5, 10, 1.0)],
    )
    def test_invalid_inputs_raise(self, k: int, n: int, alpha: float) -> None:
        with pytest.raises(ValueError):
            wilson_ci(k, n, alpha)


# ---------------------------------------------------------------------------
# cluster_bootstrap_ci
# ---------------------------------------------------------------------------


class TestClusterBootstrap:
    def test_clustered_wider_than_shuffled(self) -> None:
        """High-ICC data: cluster bootstrap must report wider CIs than the
        same data with cluster labels shuffled (which destroys the
        within-cluster correlation). Theoretical width ratio here is
        ~sqrt(DEFF) ≈ 4.5; we assert a conservative 2x."""
        df = _clustered_frame(seed=3)
        rng = np.random.default_rng(11)
        shuffled = df.assign(cluster=rng.permutation(df["cluster"].to_numpy()))

        res_clu = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=500, seed=1)
        res_shf = cluster_bootstrap_ci(shuffled, _mean_value, "cluster", n_boot=500, seed=1)
        width_clu = res_clu.ci_high - res_clu.ci_low
        width_shf = res_shf.ci_high - res_shf.ci_low
        assert width_clu > 2.0 * width_shf
        assert res_clu.boot_se > 2.0 * res_shf.boot_se

    def test_covers_true_value_across_seeds(self) -> None:
        """95% percentile CI on iid Bernoulli(0.6) means should cover the
        truth in the vast majority of seeds (binomial bound: P(<23/30
        covered) is ~1e-3 even at 92% per-seed coverage)."""
        true_p = 0.6
        covered = 0
        for seed in range(30):
            rng = np.random.default_rng(1000 + seed)
            df = pd.DataFrame(
                {
                    "cluster": [f"i{j}" for j in range(200)],
                    "value": rng.binomial(1, true_p, 200).astype(float),
                }
            )
            res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=300, seed=seed)
            if res.ci_low <= true_p <= res.ci_high:
                covered += 1
        assert covered >= 23

    def test_estimate_is_full_sample_stat(self) -> None:
        df = _clustered_frame(n_clusters=10, m=5, seed=5)
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=100, seed=0)
        assert res.estimate == pytest.approx(_mean_value(df))
        assert res.n_clusters == 10
        assert res.ci_low <= res.estimate <= res.ci_high

    def test_p_value_small_when_null_far(self) -> None:
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "cluster": [f"i{j}" for j in range(400)],
                "value": rng.binomial(1, 0.65, 400).astype(float),
            }
        )
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=500, seed=3, null_value=0.5)
        assert res.p_value is not None
        assert res.p_value < 0.01
        # never exactly zero by construction: p >= 1/(B+1)
        assert res.p_value >= 1.0 / (res.n_boot + 1)

    def test_p_value_large_when_null_at_estimate(self) -> None:
        df = _clustered_frame(n_clusters=50, m=4, seed=7)
        est = _mean_value(df)
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=500, seed=3, null_value=est)
        assert res.p_value is not None
        assert res.p_value > 0.3

    def test_p_value_none_without_null(self) -> None:
        df = _clustered_frame(n_clusters=10, m=3, seed=1)
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=100, seed=0)
        assert res.p_value is None
        assert res.null_value is None

    def test_deterministic_given_seed(self) -> None:
        df = _clustered_frame(n_clusters=20, m=5, seed=9)
        r1 = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=200, seed=42)
        r2 = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=200, seed=42)
        assert r1 == r2
        r3 = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=200, seed=43)
        assert r3.boot_se != r1.boot_se

    def test_basic_method_reflects_percentile(self) -> None:
        df = _clustered_frame(n_clusters=30, m=4, seed=13)
        per = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=300, seed=5)
        bas = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=300, seed=5, method="basic")
        assert bas.ci_low == pytest.approx(2 * per.estimate - per.ci_high, abs=1e-12)
        assert bas.ci_high == pytest.approx(2 * per.estimate - per.ci_low, abs=1e-12)
        assert bas.ci_low <= bas.ci_high

    def test_result_type_and_counts(self) -> None:
        df = _clustered_frame(n_clusters=15, m=2, seed=4)
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=150, seed=0)
        assert isinstance(res, BootstrapResult)
        assert res.n_boot == 150
        assert res.method == "percentile"

    def test_errors(self) -> None:
        df = _clustered_frame(n_clusters=5, m=2, seed=1)
        with pytest.raises(ValueError, match="empty"):
            cluster_bootstrap_ci(df.iloc[:0], _mean_value, "cluster")
        with pytest.raises(ValueError, match="cluster_col"):
            cluster_bootstrap_ci(df, _mean_value, "nope")
        with pytest.raises(ValueError, match="alpha"):
            cluster_bootstrap_ci(df, _mean_value, "cluster", alpha=1.5)
        with pytest.raises(ValueError, match="method"):
            cluster_bootstrap_ci(df, _mean_value, "cluster", method="bca")  # type: ignore[arg-type]
        one = df[df["cluster"] == "c0"]
        with pytest.raises(ValueError, match="clusters"):
            cluster_bootstrap_ci(one, _mean_value, "cluster", n_boot=50)

        def bad_stat(_: pd.DataFrame) -> float:
            return math.nan

        with pytest.raises(ValueError, match="non-finite"):
            cluster_bootstrap_ci(df, bad_stat, "cluster", n_boot=50)

    def test_performance_2000_boot_1000_rows(self) -> None:
        """Contract target: n_boot=2000 over ~1000 rows well under a second.
        Asserted at 5s to keep slow CI runners from flaking."""
        df = _clustered_frame(n_clusters=250, m=4, seed=21)
        assert len(df) == 1000
        t0 = time.perf_counter()
        cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=2000, seed=0)
        assert time.perf_counter() - t0 < 5.0

    def test_few_cluster_warning_attached(self) -> None:
        """Regression (stats review F1): below 15 clusters the result must
        carry the anti-conservative-regime warning; at >= 15 it must not."""
        few = _clustered_frame(n_clusters=8, m=4, seed=2)
        res = cluster_bootstrap_ci(few, _mean_value, "cluster", n_boot=100, seed=0)
        assert any("anti-conservative" in w for w in res.warnings)
        assert any("only 8 clusters" in w for w in res.warnings)
        enough = _clustered_frame(n_clusters=15, m=4, seed=2)
        res = cluster_bootstrap_ci(enough, _mean_value, "cluster", n_boot=100, seed=0)
        assert res.warnings == []


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------


class TestMcNemar:
    def test_no_discordant_pairs_documented_edge(self) -> None:
        for method in ("exact", "midp"):
            res = mcnemar_test(0, 0, method=method)  # type: ignore[arg-type]
            assert res.p_value == 1.0
            assert math.isnan(res.estimate)
            assert res.n_discordant == 0

    def test_balanced_counts_p_is_one(self) -> None:
        assert mcnemar_test(7, 7, method="exact").p_value == 1.0
        assert mcnemar_test(7, 7, method="midp").p_value == pytest.approx(1.0)

    def test_symmetric_in_b_c(self) -> None:
        assert mcnemar_test(12, 3).p_value == mcnemar_test(3, 12).p_value

    def test_one_sided_extreme(self) -> None:
        # k = 0: exact p = 2 * 0.5**10
        res = mcnemar_test(0, 10, method="exact")
        assert res.p_value == pytest.approx(2 * 0.5**10)
        assert res.statistic == 0.0

    def test_strong_imbalance_significant(self) -> None:
        assert mcnemar_test(30, 5, method="exact").p_value < 0.01

    def test_midp_no_larger_than_exact(self) -> None:
        for b, c in [(0, 10), (3, 12), (8, 20), (5, 5), (1, 1)]:
            assert mcnemar_test(b, c, "midp").p_value <= mcnemar_test(b, c, "exact").p_value

    def test_estimate_is_b_over_n(self) -> None:
        res = mcnemar_test(9, 3)
        assert isinstance(res, McNemarResult)
        assert res.estimate == pytest.approx(9 / 12)

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            mcnemar_test(-1, 3)
        with pytest.raises(ValueError):
            mcnemar_test(1, 3, method="chi2")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bh_fdr
# ---------------------------------------------------------------------------


class TestBhFdr:
    def test_known_example(self) -> None:
        # m*p/rank = [.05,.05,.05,.05,.05]: all q-values tie at 0.05.
        q = bh_fdr([0.01, 0.02, 0.03, 0.04, 0.05])
        np.testing.assert_allclose(q, [0.05] * 5)

    def test_monotone_and_bounded(self) -> None:
        rng = np.random.default_rng(8)
        p = rng.uniform(0, 1, 60)
        p[:5] = rng.uniform(0, 1e-4, 5)  # plant some signal
        q = bh_fdr(p)
        order = np.argsort(p, kind="stable")
        assert np.all(np.diff(q[order]) >= -1e-15), "q must be monotone in p"
        assert np.all((q >= 0.0) & (q <= 1.0))
        assert np.all(q >= p), "BH q-values never fall below raw p-values"

    def test_single_p_unchanged(self) -> None:
        np.testing.assert_allclose(bh_fdr([0.123]), [0.123])

    def test_preserves_input_order(self) -> None:
        p = [0.9, 0.001, 0.5]
        q = bh_fdr(p)
        assert q[1] == min(q)
        assert q[0] == max(q)

    def test_empty(self) -> None:
        assert bh_fdr([]).size == 0

    @pytest.mark.parametrize("bad", [[0.5, -0.1], [0.5, 1.1], [0.5, math.nan]])
    def test_invalid_pvalues_raise(self, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            bh_fdr(bad)


# ---------------------------------------------------------------------------
# power: mde_proportion, mde_mcnemar, design_effect, estimate_icc
# ---------------------------------------------------------------------------


class TestPower:
    def test_mde_closed_form_value(self) -> None:
        # (z_.975 + z_.80) * 0.5 / 10 = 2.8015852... * 0.05
        assert mde_proportion(100.0) == pytest.approx(0.14007926, rel=1e-5)

    def test_mde_from_se_closed_form_and_guards(self) -> None:
        # (z_.975 + z_.80) * 0.05 — same constant as mde_proportion(100).
        assert mde_from_se(0.05) == pytest.approx(0.14007926, rel=1e-5)
        assert mde_from_se(0.0) is None
        assert mde_from_se(-1.0) is None
        assert mde_from_se(math.nan) is None
        assert mde_from_se(math.inf) is None
        base = mde_from_se(0.05)
        assert base is not None
        stricter = mde_from_se(0.05, alpha=0.01)
        more_power = mde_from_se(0.05, power=0.9)
        assert stricter is not None and stricter > base
        assert more_power is not None and more_power > base

    def test_mde_from_se_matches_planning_mde_on_independent_data(self) -> None:
        """m7 cross-check: on independent (singleton-cluster) Bernoulli(0.5)
        data, the realized-SE MDE and the planning-formula MDE must agree
        within 10% — the two views coincide when there is no clustering."""
        rng = np.random.default_rng(5)
        n = 400
        df = pd.DataFrame(
            {
                "cluster": [f"r{i}" for i in range(n)],
                "value": rng.integers(0, 2, size=n).astype(float),
            }
        )
        res = cluster_bootstrap_ci(df, _mean_value, "cluster", n_boot=2000, seed=1)
        realized = mde_from_se(res.boot_se)
        planning = mde_proportion(float(n), p0=0.5)
        assert realized is not None
        assert realized == pytest.approx(planning, rel=0.10)

    def test_mde_decreases_with_n(self) -> None:
        assert mde_proportion(400.0) < mde_proportion(100.0)
        # sqrt scaling: 4x the n halves the MDE
        assert mde_proportion(400.0) == pytest.approx(mde_proportion(100.0) / 2)

    def test_mde_zero_n_is_inf(self) -> None:
        assert mde_proportion(0.0) == math.inf
        assert mde_mcnemar(0) == math.inf

    def test_mde_mcnemar_matches_proportion_at_half(self) -> None:
        assert mde_mcnemar(150) == pytest.approx(mde_proportion(150.0, p0=0.5))

    def test_mde_mcnemar_decreases_with_discordant_n(self) -> None:
        assert mde_mcnemar(200) < mde_mcnemar(50)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_eff": -1.0},
            {"n_eff": 100.0, "p0": 0.0},
            {"n_eff": 100.0, "alpha": 1.0},
            {"n_eff": 100.0, "power": 0.0},
        ],
    )
    def test_mde_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            mde_proportion(**kwargs)

    def test_design_effect_values(self) -> None:
        assert design_effect([5] * 10, 0.0) == 1.0
        assert design_effect([1] * 10, 0.9) == 1.0
        assert design_effect([10] * 5, 0.5) == pytest.approx(5.5)
        assert design_effect([10] * 5, 0.9) > design_effect([10] * 5, 0.1)

    def test_design_effect_errors(self) -> None:
        with pytest.raises(ValueError):
            design_effect([], 0.5)
        with pytest.raises(ValueError):
            design_effect([0, 3], 0.5)
        with pytest.raises(ValueError):
            design_effect([3, 3], 1.5)

    def test_icc_high_when_clusters_dominate(self) -> None:
        # true ICC = 1 / (1 + 0.3^2) ≈ 0.917
        df = _clustered_frame(n_clusters=40, m=20, cluster_sd=1.0, noise_sd=0.3, seed=17)
        icc = estimate_icc(df, "value", "cluster")
        assert icc > 0.8

    def test_icc_near_zero_for_iid(self) -> None:
        rng = np.random.default_rng(19)
        df = pd.DataFrame(
            {
                "cluster": np.repeat([f"c{g}" for g in range(40)], 20),
                "value": rng.normal(0, 1, 800),
            }
        )
        icc = estimate_icc(df, "value", "cluster")
        assert 0.0 <= icc < 0.15

    def test_icc_degenerate_cases_return_zero(self) -> None:
        singletons = pd.DataFrame({"cluster": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        assert estimate_icc(singletons, "value", "cluster") == 0.0
        one_cluster = pd.DataFrame({"cluster": ["a"] * 5, "value": [1.0, 2, 3, 4, 5]})
        assert estimate_icc(one_cluster, "value", "cluster") == 0.0
        constant = pd.DataFrame({"cluster": ["a", "a", "b", "b"], "value": [2.0] * 4})
        assert estimate_icc(constant, "value", "cluster") == 0.0

    def test_icc_feeds_design_effect(self) -> None:
        df = _clustered_frame(n_clusters=30, m=10, seed=23)
        icc = estimate_icc(df, "value", "cluster")
        deff = design_effect([10] * 30, icc)
        assert deff > 1.0
        assert mde_proportion(300.0 / deff) > mde_proportion(300.0)

    def test_icc_errors(self) -> None:
        df = pd.DataFrame({"cluster": ["a", "b"], "value": [1.0, math.nan]})
        with pytest.raises(ValueError):
            estimate_icc(df, "value", "cluster")
        with pytest.raises(ValueError):
            estimate_icc(df, "missing", "cluster")


# ---------------------------------------------------------------------------
# fleiss_kappa
# ---------------------------------------------------------------------------


class TestFleissKappa:
    def test_perfect_agreement_two_categories(self) -> None:
        table = np.array([[5, 0], [0, 5], [5, 0], [0, 5]])
        assert fleiss_kappa(table) == pytest.approx(1.0)

    def test_uniform_disagreement_hand_computed(self) -> None:
        # 4 raters split 1/1/1/1 over 4 categories on every item:
        # P_i = 0, P_e = 0.25 -> kappa = -1/3 exactly.
        table = np.ones((10, 4))
        assert fleiss_kappa(table) == pytest.approx(-1.0 / 3.0)

    def test_more_agreement_higher_kappa(self) -> None:
        low = np.array([[2, 2], [2, 2], [2, 2], [2, 2]])
        high = np.array([[4, 0], [0, 4], [4, 0], [3, 1]])
        assert fleiss_kappa(high) > fleiss_kappa(low)

    def test_degenerate_single_category_convention(self) -> None:
        # all ratings in one category: P_e = 1; documented convention -> 1.0
        table = np.array([[3, 0], [3, 0], [3, 0]])
        assert fleiss_kappa(table) == 1.0

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            fleiss_kappa(np.array([1, 2, 3]))  # not 2-D
        with pytest.raises(ValueError):
            fleiss_kappa(np.array([[3], [3]]))  # 1 category
        with pytest.raises(ValueError):
            fleiss_kappa(np.array([[2, 1], [2, 2]]))  # unbalanced rows
        with pytest.raises(ValueError):
            fleiss_kappa(np.array([[1, 0], [0, 1]]))  # 1 rater
        with pytest.raises(ValueError):
            fleiss_kappa(np.array([[-1, 4], [2, 1]]))  # negative count


# ---------------------------------------------------------------------------
# logistic_fit
# ---------------------------------------------------------------------------


class TestLogisticFit:
    @staticmethod
    def _simulate(n: int, beta0: float, beta1: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        prob = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * x)))
        y = (rng.uniform(0, 1, n) < prob).astype(float)
        return x[:, None], y

    def test_recovers_planted_coefficients(self) -> None:
        x, y = self._simulate(4000, beta0=-0.4, beta1=0.9, seed=29)
        fit = logistic_fit(x, y)
        assert isinstance(fit, LogisticFit)
        assert fit.converged
        assert fit.coef.shape == (2,)
        assert fit.coef[0] == pytest.approx(-0.4, abs=0.2)
        assert fit.coef[1] == pytest.approx(0.9, abs=0.2)
        assert np.all(fit.se > 0)
        assert fit.n_obs == 4000
        assert fit.loglik < 0

    def test_null_data_gives_near_zero_slope(self) -> None:
        x, y = self._simulate(3000, beta0=0.0, beta1=0.0, seed=31)
        fit = logistic_fit(x, y)
        assert fit.converged
        # slope within ~3 SEs of zero
        assert abs(fit.coef[1]) < 3.5 * fit.se[1]

    def test_one_dim_x_accepted(self) -> None:
        x, y = self._simulate(500, beta0=0.2, beta1=0.5, seed=37)
        fit = logistic_fit(x.ravel(), y)
        assert fit.coef.shape == (2,)

    def test_no_intercept_with_manual_constant(self) -> None:
        x, y = self._simulate(1000, beta0=0.3, beta1=-0.7, seed=41)
        design = np.column_stack([np.ones(len(y)), x])
        fit_auto = logistic_fit(x, y, add_intercept=True)
        fit_manual = logistic_fit(design, y, add_intercept=False)
        np.testing.assert_allclose(fit_auto.coef, fit_manual.coef, atol=1e-10)

    def test_deterministic(self) -> None:
        x, y = self._simulate(800, beta0=0.1, beta1=0.4, seed=43)
        f1, f2 = logistic_fit(x, y), logistic_fit(x, y)
        np.testing.assert_array_equal(f1.coef, f2.coef)

    def test_errors(self) -> None:
        x = np.zeros((10, 1))
        with pytest.raises(ValueError):
            logistic_fit(x, np.full(10, 2.0))  # non-binary y
        with pytest.raises(ValueError):
            logistic_fit(x, np.zeros(9))  # shape mismatch
        with pytest.raises(ValueError):
            logistic_fit(np.zeros((0, 1)), np.zeros(0))  # empty
        with pytest.raises(ValueError):
            logistic_fit(np.zeros((2, 3)), np.array([0.0, 1.0]))  # n <= p
