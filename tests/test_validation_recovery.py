"""Planted-bias recovery tests for the validation suite (contracts §8).

The headline test runs ``run_validation("fast")`` once (module-scoped
fixture, ~3s) and asserts the whole pinned scenario battery passes at the
shipped seed. Cheap per-scenario unit tests exercise determinism and the
exact-unanimity property directly; the 200-seed frequentist coverage
check runs behind the ``slow`` marker.

Zero network, zero real LLMs: everything runs against the deterministic
mock judge.
"""

from __future__ import annotations

import pytest

from judgecal.core import PairwiseItem
from judgecal.executors import MockJudgeExecutor
from judgecal.fixtures import MockJudgeConfig, SyntheticConfig, generate_items
from judgecal.probes import ProbeConfig, analyze_suite, plan_suite
from judgecal.report import build_card
from judgecal.validate import (
    Check,
    ScenarioResult,
    ValidationReport,
    _Params,
    _scenario_position,
    _scenario_stability,
    run_validation,
)

#: Contract §8 scenario names, pinned order (fast level).
EXPECTED_SCENARIOS = [
    "null",
    "position",
    "verbosity",
    "self_preference",
    "template",
    "stability",
    "mixed",
]


@pytest.fixture(scope="module")
def fast_report() -> ValidationReport:
    """One fast-level run at the shipped seed, shared across tests."""
    return run_validation("fast")


# --------------------------------------------------------------------------
# The headline test: the validation suite validates
# --------------------------------------------------------------------------


def test_fast_validation_passes(fast_report: ValidationReport) -> None:
    """All seven planted-bias scenarios pass at the shipped seed."""
    assert fast_report.passed, "\n" + fast_report.render_table()


def test_scenario_names_and_order(fast_report: ValidationReport) -> None:
    assert [s.name for s in fast_report.scenarios] == EXPECTED_SCENARIOS
    assert fast_report.level == "fast"
    assert fast_report.seed == 7


def test_every_scenario_has_diagnosable_checks(fast_report: ValidationReport) -> None:
    """Each check carries a description, an observed value, and a criterion."""
    for scenario in fast_report.scenarios:
        assert scenario.checks, f"scenario {scenario.name!r} has no checks"
        for check in scenario.checks:
            description, passed, observed, expected = check  # NamedTuple unpacks
            assert description and observed and expected
            assert isinstance(passed, bool)


def test_null_scenario_covers_nulls_and_fires_no_bias_flag(
    fast_report: ValidationReport,
) -> None:
    null = fast_report.scenarios[0]
    descriptions = [c.description for c in null.checks]
    assert any("no bias flag" in d for d in descriptions)
    assert any("first_pick_rate CI covers null" in d for d in descriptions)
    assert any("self_error_pick_excess CI covers null" in d for d in descriptions)
    assert null.passed, "\n" + fast_report.render_table()


def test_biased_scenarios_check_analytic_truth_and_flags(
    fast_report: ValidationReport,
) -> None:
    """Each planted scenario compares against analytic truth AND a card flag."""
    by_name = {s.name: s for s in fast_report.scenarios}
    for name, truth_metric, flag in (
        ("position", "first_pick_rate", "position_bias_detected"),
        ("verbosity", "pad_pick_rate", "verbosity_bias_detected"),
        ("self_preference", "self_error_pick_excess", "self_preference_detected"),
    ):
        descriptions = " | ".join(c.description for c in by_name[name].checks)
        assert f"{truth_metric} CI covers" in descriptions
        assert flag in descriptions


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def test_render_table_is_diagnosable(fast_report: ValidationReport) -> None:
    """The table shows observed vs expected vs CI on every check line."""
    table = fast_report.render_table()
    assert "level=fast seed=7" in table
    assert "OVERALL: PASS" in table
    for name in EXPECTED_SCENARIOS:
        assert name in table
    # Observed CIs and acceptance criteria are printed inline.
    assert "CI=[" in table
    assert "CI contains" in table
    n_checks = sum(len(s.checks) for s in fast_report.scenarios)
    # header + column row + 2 rules + footer = 5 fixed lines.
    assert len(table.splitlines()) == n_checks + 5


def test_render_table_marks_failures() -> None:
    """A hand-built failing report renders FAIL and reports not-passed."""
    report = ValidationReport(
        level="fast",
        seed=0,
        scenarios=[
            ScenarioResult(
                name="demo",
                checks=[
                    Check("good", True, "est=1", "1"),
                    Check("bad", False, "est=0", "1"),
                ],
            )
        ],
    )
    assert not report.passed
    assert not report.scenarios[0].passed
    table = report.render_table()
    assert "FAIL" in table
    assert "OVERALL: FAIL (0/1 scenarios, 1/2 checks)" in table


# --------------------------------------------------------------------------
# Individual scenarios (cheap unit runs)
# --------------------------------------------------------------------------


def test_stability_exact_unanimity_at_any_seed() -> None:
    """noise_sigma=0 => unanimity == 1.0 EXACTLY — a property, not luck."""
    scenario = _scenario_stability(_Params(seed=12345, n_items=40, n_boot=200))
    check = next(c for c in scenario.checks if "exactly at zero noise" in c.description)
    assert check.passed, f"{check.observed} vs {check.expected}"


def test_position_scenario_is_deterministic() -> None:
    """Same params => byte-identical checks (everything is seeded)."""
    params = _Params(seed=71, n_items=120, n_boot=200)
    first = _scenario_position(params)
    second = _scenario_position(params)
    assert first == second


def test_run_validation_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="level"):
        run_validation("quick")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Adversarial-review end-to-end regressions (2026-06-10 panel)
# --------------------------------------------------------------------------


def _find_estimate(results, name):
    return next((e for r in results for e in r.estimates if e.name == name), None)


def test_author_blind_judge_with_gap_imbalance_does_not_fire_self_preference() -> None:
    """Regression (stats review F2): the reviewer's author-blind judge with
    a quality-gap imbalance between self and control sets produced a
    17-pp spuriously significant excess. The composition diagnostic must
    suppress the self_preference_detected flag (the metric stays)."""
    items: list[PairwiseItem] = []
    for i in range(150):  # self loses with a SMALL gap (hard pairs)
        items.append(
            PairwiseItem(
                item_id=f"self:{i:04d}",
                prompt=f"q{i}",
                response_a="x" * 400,
                response_b="y" * 400,
                label="B",
                author_a="judge-self",
                author_b="other-model",
                meta={"latent_quality_a": 0.4, "latent_quality_b": 0.6},
            )
        )
    for i in range(150):  # no self side, winner wins with a LARGE gap
        items.append(
            PairwiseItem(
                item_id=f"ctrl:{i:04d}",
                prompt=f"r{i}",
                response_a="x" * 400,
                response_b="y" * 400,
                label="A",
                author_a="model-1",
                author_b="model-2",
                meta={"latent_quality_a": 0.8, "latent_quality_b": 0.2},
            )
        )
    judge = MockJudgeConfig(seed=11, beta_quality=3.0, beta_self=0.0)  # author-blind
    probe_config = ProbeConfig(n_boot=500, seed=11)
    requests = plan_suite(items, ["self_preference"], probe_config)
    judgments = MockJudgeExecutor(judge).execute(requests)
    results = analyze_suite(judgments, ["self_preference"], probe_config)
    card = build_card(results, judge={"model": "mock"})

    est = _find_estimate(results, "self_error_pick_excess")
    assert est is not None
    # The confounded excess is large and "significant" — exactly the trap.
    assert est.estimate > 0.10
    assert est.p_value is not None and est.p_value < 0.05
    # ... but the composition diagnostic catches the imbalance:
    assert est.detail["composition_imbalance"] is True
    assert any("differ in composition" in w for r in results for w in r.warnings)
    assert "self_preference_detected" not in card.overall_flags


def test_per_call_stochastic_judge_reports_instability_not_template_effects() -> None:
    """Regression (stats review F3): a judge whose verdict is independently
    random per call (temperature>0 / nondeterministic serving) depresses
    template kappa AND stability kappa alike; the card must diagnose
    instability, not template sensitivity. (The validation suite's
    template scenario separately asserts the flag still fires for a real
    planted template effect with a noise-free judge.)"""
    import numpy as np
    from tests.test_probes import judge as make_judgments
    from tests.test_probes import make_item

    rng = np.random.default_rng(42)
    items = [make_item(i, label=None) for i in range(200)]
    probe_config = ProbeConfig(n_boot=200, seed=42)
    requests = plan_suite(items, ["template", "stability"], probe_config)
    judgments = make_judgments(
        requests, lambda r: "first" if rng.random() < 0.5 else "second"
    )
    results = analyze_suite(judgments, ["template", "stability"], probe_config)
    card = build_card(results, judge={"model": "per-call-random"})

    tpl_kappa = _find_estimate(results, "template_fleiss_kappa")
    stab_kappa = _find_estimate(results, "stability_fleiss_kappa")
    assert tpl_kappa is not None and stab_kappa is not None
    assert tpl_kappa.estimate < 0.2  # template disagreement is high...
    assert stab_kappa.estimate < 0.2  # ...but so is pure repeat noise
    assert "template_sensitivity_high" not in card.overall_flags
    assert "instability_high" in card.overall_flags


def test_null_judge_n800_shows_adequate_power() -> None:
    """Regression (claims review M4, the n=800 repro): a clean judge at
    n=800 has first_pick_rate MDE below the 0.05 effect-of-interest
    floor and must NOT be labeled underpowered (the old observed-effect
    rule kept it 'underpowered' at any sample size)."""
    items = generate_items(SyntheticConfig(n_items=800, seed=11))
    judge = MockJudgeConfig(seed=11)  # no planted bias
    probe_config = ProbeConfig(n_boot=600, seed=11)
    requests = plan_suite(items, ["position"], probe_config)
    judgments = MockJudgeExecutor(judge).execute(requests)
    results = analyze_suite(judgments, ["position"], probe_config)
    card = build_card(results, judge={"model": "mock"})

    fpr = _find_estimate(results, "first_pick_rate")
    assert fpr is not None
    assert fpr.mde is not None and fpr.mde <= 0.05
    assert "underpowered:first_pick_rate" not in card.overall_flags
    assert "position_bias_detected" not in card.overall_flags


# --------------------------------------------------------------------------
# Full level (slow): 200-seed frequentist coverage
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_full_level_passes_with_coverage_scenario() -> None:
    """Fast scenarios + the 200-seed coverage check all pass."""
    report = run_validation("full", seed=7)
    assert report.passed, "\n" + report.render_table()
    assert [s.name for s in report.scenarios] == [*EXPECTED_SCENARIOS, "coverage"]
    coverage = report.scenarios[-1]
    assert "coverage" in coverage.checks[0].description
