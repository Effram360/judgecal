"""Planted-bias recovery validation — "we test the tester" (contracts §8).

``run_validation`` runs the seven pinned scenarios end to end: synthetic
items (:func:`judgecal.fixtures.generate_items`) → suite plan
(:func:`judgecal.probes.plan_suite`) → deterministic mock judge
(:class:`judgecal.executors.MockJudgeExecutor`) → probe analyses
(:func:`judgecal.probes.analyze_suite`) → reliability card
(:func:`judgecal.report.build_card`) — and compares every recovered
estimate against the mock judge's *analytic* truth helpers
(:func:`judgecal.fixtures.expected_first_pick_rate` and friends), computed
from the same :class:`~judgecal.fixtures.MockJudgeConfig` and the very
requests that were executed.

Statistical conventions (documented decisions):

* **Significance uses p-values, not q-values.** Probes leave
  ``Estimate.q_value`` as ``None``; q-values are filled card-wide by
  ``build_card``. Each scenario plants exactly ONE bias and tests exactly
  one hypothesis per check, so no multiplicity correction is needed at
  scenario level — ``p < alpha`` is the right per-scenario criterion. The
  card-level BH-FDR path *is* still exercised: every scenario builds a
  card and asserts the documented flags (which gate on q-values) fire or
  stay silent as planted.
* **Fixed-seed semantics.** All checks are deterministic given ``seed``:
  the synthetic generator, the mock judge's hash-keyed draws, and every
  bootstrap are seeded. Each scenario runs under its own derived seed
  (``seed * 10 + scenario_index``) so scenarios never share item or
  verdict draws. Each CI-coverage check is a single draw from a
  ~95%-coverage procedure, so for an *arbitrary* seed a check can fail by
  bad luck; the shipped default (``seed=7``) is verified to pass. The
  ``full`` level adds the 200-seed coverage scenario, which checks the
  procedure's *frequentist* coverage (>= 90% of 95% CIs cover the truth)
  rather than any single draw.
* **The null scenario uses a strongly discriminative judge**
  (``beta_quality=8``, ``quality_gap_sd=2``). The mock judge's verdict
  uniform is keyed per request body, so borderline logits yield genuinely
  random verdicts that differ across template paraphrases — intrinsic
  Bernoulli randomness that the agreement metrics cannot distinguish from
  template sensitivity. A near-deterministic null judge keeps the
  agreement metrics clean so "no bias flag fires" is a sharp check.
  Under the exact null the verbosity pad contrast is degenerate (equal
  qualities, zero betas, zero noise => logit 0 => all ties), so
  ``pad_pick_rate`` is legitimately absent there; the scenario instead
  asserts the *other* null-tested metrics exist and their CIs cover their
  nulls.
* **Template scenario reference.** Contract §8 compares the biased kappa
  to "the null-scenario kappa"; the implemented reference is the SAME
  scenario re-run with ``template_sigma=0`` on identical items and seeds,
  which isolates the planted effect exactly (the null scenario proper
  uses a more discriminative judge, so its kappa is not apples-to-apples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

from judgecal.core import Estimate, ProbeResult
from judgecal.executors import MockJudgeExecutor
from judgecal.fixtures import (
    MockJudgeConfig,
    SyntheticConfig,
    expected_first_pick_rate,
    expected_pad_pick_rate,
    expected_self_error_pick_excess,
    generate_items,
)
from judgecal.probes import ProbeConfig, analyze_suite, plan_suite
from judgecal.report import ReliabilityCard, build_card

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import JudgmentRequest, PairwiseItem

# --------------------------------------------------------------------------
# Tuning constants (fixed by contract §8; not user-configurable)
# --------------------------------------------------------------------------

#: Items per scenario at the fast level (§8: "n≈300").
_FAST_N_ITEMS = 300

#: Items for the position scenario (§8 pins scenario 2 at n=400).
_POSITION_N_ITEMS = 400

#: Bootstrap replicates at the fast level (§8: "n_boot 500").
_FAST_N_BOOT = 500

#: Coverage scenario (full level): seeds, per-seed size, replicates, floor.
_COVERAGE_N_SEEDS = 200
_COVERAGE_N_ITEMS = 100
_COVERAGE_N_BOOT = 200
_COVERAGE_FLOOR = 0.90

#: "Unanimity well below 1" threshold for the unstable scenario; mirrors
#: ``judgecal.report.thresholds.STABILITY_UNANIMITY_FLOOR`` so the value
#: check and the ``instability_high`` flag check agree.
_UNANIMITY_FLOOR = 0.80

#: "max_flip elevated" threshold for the template scenario; mirrors
#: ``judgecal.report.thresholds.TEMPLATE_MAX_FLIP_CEILING`` so the value
#: check and the ``template_sensitivity_high`` flag check agree.
_MAX_FLIP_ELEVATED = 0.20

#: Flags that assert a reliability *problem* was detected. The null
#: scenario must fire none of these. ``underpowered:<metric>`` flags are
#: NOT in this set: they are power caveats, not bias detections, and they
#: legitimately fire under the null (observed effects near zero make any
#: finite MDE look large by comparison).
_BIAS_FLAGS = frozenset(
    {
        "position_bias_detected",
        "verbosity_bias_detected",
        "self_preference_detected",
        "template_sensitivity_high",
        "instability_high",
        "high_invalid_rate",
    }
)

#: Null-tested metrics that must actually EXIST in the null scenario
#: (guards the "every CI covers its null" check against vacuous passes).
_NULL_REQUIRED_METRICS = ("first_pick_rate", "length_glm_coef", "self_error_pick_excess")


# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------


class Check(NamedTuple):
    """One pass/fail comparison inside a scenario.

    Attributes:
        description: What is being checked (human-readable).
        passed: Whether the check passed.
        observed: Formatted observed value (estimate, CI, p, flags, ...).
        expected: Formatted acceptance criterion.
    """

    description: str
    passed: bool
    observed: str
    expected: str


@dataclass(frozen=True)
class ScenarioResult:
    """All checks for one planted-bias scenario.

    Attributes:
        name: Scenario name (contract §8 order: "null", "position",
            "verbosity", "self_preference", "template", "stability",
            "mixed", plus "coverage" at the full level).
        checks: The individual comparisons performed.
    """

    name: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        """True iff every check in this scenario passed."""
        return all(c.passed for c in self.checks)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of the full validation suite.

    Attributes:
        level: The level that was run ("fast" or "full").
        seed: The master seed.
        scenarios: Per-scenario results, in contract order.
    """

    level: str
    seed: int
    scenarios: list[ScenarioResult]

    @property
    def passed(self) -> bool:
        """True iff every scenario passed."""
        return all(s.passed for s in self.scenarios)

    def render_table(self) -> str:
        """Aligned plain-text pass/fail table, one line per check.

        Every line carries the observed value (estimate / CI / p-value /
        flags) next to the acceptance criterion, so a failure is
        diagnosable from the table alone.
        """
        headers = ("scenario", "check", "observed", "expected", "status")
        rows: list[tuple[str, str, str, str, str]] = []
        for scenario in self.scenarios:
            for i, check in enumerate(scenario.checks):
                rows.append(
                    (
                        scenario.name if i == 0 else "",
                        check.description,
                        check.observed,
                        check.expected,
                        "PASS" if check.passed else "FAIL",
                    )
                )
        widths = [
            max(len(headers[col]), *(len(r[col]) for r in rows)) if rows else len(headers[col])
            for col in range(len(headers))
        ]

        def fmt(row: tuple[str, ...]) -> str:
            return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)).rstrip()

        rule = "  ".join("-" * w for w in widths)
        n_checks = sum(len(s.checks) for s in self.scenarios)
        n_checks_passed = sum(1 for s in self.scenarios for c in s.checks if c.passed)
        n_scen_passed = sum(1 for s in self.scenarios if s.passed)
        lines = [
            f"judgecal validation — level={self.level} seed={self.seed}",
            fmt(headers),
            rule,
            *(fmt(r) for r in rows),
            rule,
            (
                f"OVERALL: {'PASS' if self.passed else 'FAIL'} "
                f"({n_scen_passed}/{len(self.scenarios)} scenarios, "
                f"{n_checks_passed}/{n_checks} checks)"
            ),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Pipeline + check helpers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pipeline:
    """One executed scenario pipeline: requests, analyses, card."""

    requests: list[JudgmentRequest]
    results: list[ProbeResult]
    card: ReliabilityCard


def _run_pipeline(
    items: Sequence[PairwiseItem],
    probes: Sequence[str],
    probe_config: ProbeConfig,
    judge_config: MockJudgeConfig,
) -> _Pipeline:
    """Plan -> mock-judge -> analyze -> card, end to end."""
    requests = plan_suite(items, probes, probe_config)
    judgments = MockJudgeExecutor(judge_config).execute(requests)
    results = analyze_suite(judgments, probes, probe_config)
    card = build_card(
        results,
        judge={"model": "mock-judge", "planted": repr(judge_config)},
        config={
            "alpha": probe_config.alpha,
            "n_boot": probe_config.n_boot,
            "seed": probe_config.seed,
        },
    )
    return _Pipeline(requests=requests, results=results, card=card)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _fmt_estimate(est: Estimate) -> str:
    return f"est={_fmt(est.estimate)} CI=[{_fmt(est.ci_low)}, {_fmt(est.ci_high)}]"


def _find_estimate(results: Sequence[ProbeResult], name: str) -> Estimate | None:
    for result in results:
        for est in result.estimates:
            if est.name == name:
                return est
    return None


def _all_warnings(results: Sequence[ProbeResult]) -> list[str]:
    return [w for result in results for w in result.warnings]


def _missing_metric_check(
    description: str, name: str, results: Sequence[ProbeResult]
) -> Check:
    return Check(
        description=description,
        passed=False,
        observed=f"metric {name!r} absent; warnings={_all_warnings(results)}",
        expected="metric present",
    )


def _check_ci_covers(
    description: str, est: Estimate | None, target: float, name: str, results: Sequence[ProbeResult]
) -> Check:
    """CI-covers-target check (used for both analytic truths and nulls)."""
    if est is None:
        return _missing_metric_check(description, name, results)
    return Check(
        description=description,
        passed=est.ci_low <= target <= est.ci_high,
        observed=_fmt_estimate(est),
        expected=f"CI contains {_fmt(target)}",
    )


def _check_significant(
    description: str,
    est: Estimate | None,
    alpha: float,
    name: str,
    results: Sequence[ProbeResult],
    *,
    positive: bool = False,
) -> Check:
    """p < alpha (optionally also estimate > 0); see module docstring on p vs q."""
    if est is None:
        return _missing_metric_check(description, name, results)
    if est.p_value is None:
        return Check(
            description=description,
            passed=False,
            observed=f"{_fmt_estimate(est)} p=None",
            expected=f"p < {alpha:g}",
        )
    passed = est.p_value < alpha and (est.estimate > 0.0 if positive else True)
    expected = f"p < {alpha:g}" + (" and estimate > 0" if positive else "")
    return Check(
        description=description,
        passed=passed,
        observed=f"{_fmt_estimate(est)} p={est.p_value:.4g}",
        expected=expected,
    )


def _check_flag(description: str, card: ReliabilityCard, flag: str, *, fires: bool) -> Check:
    present = flag in card.overall_flags
    return Check(
        description=description,
        passed=present == fires,
        observed=f"card flags={sorted(card.overall_flags)}",
        expected=f"{flag!r} {'fires' if fires else 'does not fire'}",
    )


# --------------------------------------------------------------------------
# Scenario parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Params:
    """Shared per-run knobs: master seed, scenario size, bootstrap size."""

    seed: int
    n_items: int
    n_boot: int

    def probe_config(self) -> ProbeConfig:
        return ProbeConfig(n_boot=self.n_boot, seed=self.seed)

    def items(self, **overrides: object) -> list[PairwiseItem]:
        config = SyntheticConfig(n_items=self.n_items, seed=self.seed, **overrides)  # type: ignore[arg-type]
        return generate_items(config)


# --------------------------------------------------------------------------
# Scenarios (contract §8, pinned order)
# --------------------------------------------------------------------------


def _scenario_null(params: _Params) -> ScenarioResult:
    """Scenario 1 — null judge: all bias betas 0, noise 0.

    Every null-tested metric's CI must cover its null and no bias flag may
    fire. Uses a strongly discriminative judge (see module docstring).
    """
    items = params.items(quality_gap_sd=2.0)
    judge = MockJudgeConfig(seed=params.seed, beta_quality=8.0)
    probe_config = params.probe_config()
    pipe = _run_pipeline(
        items,
        ["position", "verbosity", "self_preference", "template", "stability"],
        probe_config,
        judge,
    )

    checks: list[Check] = []
    present = {
        est.name for result in pipe.results for est in result.estimates
    }
    missing_required = [m for m in _NULL_REQUIRED_METRICS if m not in present]
    checks.append(
        Check(
            description="required null-tested metrics present",
            passed=not missing_required,
            observed=f"present={sorted(present)}",
            expected=f"includes all of {list(_NULL_REQUIRED_METRICS)}",
        )
    )
    for result in pipe.results:
        for est in result.estimates:
            if est.null_value is None:
                continue
            checks.append(
                _check_ci_covers(
                    f"{result.probe}.{est.name} CI covers null",
                    est,
                    est.null_value,
                    est.name,
                    pipe.results,
                )
            )
    fired = sorted(set(pipe.card.overall_flags) & _BIAS_FLAGS)
    checks.append(
        Check(
            description="no bias flag fires on the null card",
            passed=not fired,
            observed=f"bias flags fired={fired}; all flags={sorted(pipe.card.overall_flags)}",
            expected="no bias-detection flag",
        )
    )
    return ScenarioResult(name="null", checks=checks)


def _scenario_position(params: _Params) -> ScenarioResult:
    """Scenario 2 — planted position bias (beta_position=0.8, n=400)."""
    items = params.items()
    judge = MockJudgeConfig(seed=params.seed, beta_position=0.8)
    probe_config = params.probe_config()
    pipe = _run_pipeline(items, ["position"], probe_config, judge)

    truth = expected_first_pick_rate(judge, pipe.requests)
    est = _find_estimate(pipe.results, "first_pick_rate")
    checks = [
        _check_ci_covers(
            "first_pick_rate CI covers analytic truth",
            est,
            truth,
            "first_pick_rate",
            pipe.results,
        ),
        _check_significant(
            "first_pick_rate significant vs 0.5",
            est,
            probe_config.alpha,
            "first_pick_rate",
            pipe.results,
        ),
        _check_flag(
            "position_bias_detected fires", pipe.card, "position_bias_detected", fires=True
        ),
    ]
    return ScenarioResult(name="position", checks=checks)


def _scenario_verbosity(params: _Params) -> ScenarioResult:
    """Scenario 3 — planted verbosity bias (beta_length=1.0).

    Plans position too: the verbosity probe's observational GLM consumes
    position judgments (``requires``), and the absence of a false
    ``position_bias_detected`` flag doubles as a no-contamination check
    (the length term negates exactly under the orig/swap symmetrization).
    """
    items = params.items()
    judge = MockJudgeConfig(seed=params.seed, beta_length=1.0)
    probe_config = params.probe_config()
    pipe = _run_pipeline(items, ["position", "verbosity"], probe_config, judge)

    truth = expected_pad_pick_rate(judge, pipe.requests)
    pad = _find_estimate(pipe.results, "pad_pick_rate")
    glm = _find_estimate(pipe.results, "length_glm_coef")
    checks = [
        _check_ci_covers(
            "pad_pick_rate CI covers analytic truth", pad, truth, "pad_pick_rate", pipe.results
        ),
        _check_significant(
            "pad_pick_rate significant vs 0.5",
            pad,
            probe_config.alpha,
            "pad_pick_rate",
            pipe.results,
        ),
        _check_significant(
            "length_glm_coef positive and significant",
            glm,
            probe_config.alpha,
            "length_glm_coef",
            pipe.results,
            positive=True,
        ),
        _check_flag(
            "verbosity_bias_detected fires", pipe.card, "verbosity_bias_detected", fires=True
        ),
        _check_flag(
            "no false position flag under pure length bias",
            pipe.card,
            "position_bias_detected",
            fires=False,
        ),
    ]
    return ScenarioResult(name="verbosity", checks=checks)


def _scenario_self_preference(params: _Params) -> ScenarioResult:
    """Scenario 4 — planted self-preference (beta_self=1.0).

    Items carry author metadata (``self_author_fraction=0.5`` with the
    "judge-self"/"other-model" pool); ``ProbeConfig.judge_author`` and
    ``MockJudgeConfig.self_name`` both default to "judge-self".

    Uses a weaker quality prior (``beta_quality=1.5``) than the mock
    default: the treatment set conditions on items where the self side
    *loses* on quality, so a strongly quality-driven judge sits on the
    flat tail of the sigmoid where a +1.0 self log-odds moves pick
    probabilities very little. The analytic truth uses the same config,
    so the coverage check is unaffected; the weaker prior simply gives
    the planted effect detectable size at the pinned n≈300.
    """
    items = params.items(self_author_fraction=0.5, authors=("judge-self", "other-model"))
    judge = MockJudgeConfig(seed=params.seed, beta_quality=1.5, beta_self=1.0)
    probe_config = params.probe_config()
    pipe = _run_pipeline(items, ["self_preference"], probe_config, judge)

    truth = expected_self_error_pick_excess(judge, pipe.requests)
    est = _find_estimate(pipe.results, "self_error_pick_excess")
    checks = [
        _check_ci_covers(
            "self_error_pick_excess CI covers analytic truth",
            est,
            truth,
            "self_error_pick_excess",
            pipe.results,
        ),
        _check_significant(
            "self_error_pick_excess significant vs 0",
            est,
            probe_config.alpha,
            "self_error_pick_excess",
            pipe.results,
        ),
        _check_flag(
            "self_preference_detected fires", pipe.card, "self_preference_detected", fires=True
        ),
    ]
    return ScenarioResult(name="self_preference", checks=checks)


def _scenario_template(params: _Params) -> ScenarioResult:
    """Scenario 5 — template sensitivity (template_sigma=0.7).

    Reference = identical items and judge with ``template_sigma=0`` (see
    module docstring); the planted offsets must depress Fleiss' kappa,
    elevate the max pairwise flip rate, and fire the card flag.

    Check design notes:

    * The scenario runs the template probe alone (no stability data), so
      the ``template_sensitivity_high`` flag is exercised through its
      no-noise-baseline fallback rule (kappa CI upper bound below the
      floor OR max flip above the ceiling); the judge here has
      ``noise_sigma=0``, so the planted template effect is the only
      disagreement source. The card-level stability-kappa gate (template
      disagreement must exceed repeat noise) is covered by the
      adversarial-review regression tests.
    * Kappa uses CI *separation* (biased upper bound below the reference
      lower bound) — the planted sigma=0.7 effect is large, so the strict
      criterion holds robustly across seeds.
    * "max_flip elevated" is tested against the absolute flag ceiling
      (``_MAX_FLIP_ELEVATED``), not against the sigma=0 reference value:
      the max over template pairs is upward selection-biased, so under
      the mock judge's intrinsic per-body verdict randomness the
      reference *max* fluctuates high and a point comparison against it
      is noise-dominated even when the planted effect is real.
    """
    items = params.items()
    probe_config = params.probe_config()
    biased_judge = MockJudgeConfig(seed=params.seed, template_sigma=0.7)
    null_judge = MockJudgeConfig(seed=params.seed, template_sigma=0.0)
    biased = _run_pipeline(items, ["template"], probe_config, biased_judge)
    reference = _run_pipeline(items, ["template"], probe_config, null_judge)

    checks: list[Check] = []
    kappa_b = _find_estimate(biased.results, "template_fleiss_kappa")
    kappa_0 = _find_estimate(reference.results, "template_fleiss_kappa")
    if kappa_b is None or kappa_0 is None:
        checks.append(
            _missing_metric_check(
                "template_fleiss_kappa below sigma=0 reference",
                "template_fleiss_kappa",
                biased.results if kappa_b is None else reference.results,
            )
        )
    else:
        checks.append(
            Check(
                description="template_fleiss_kappa below sigma=0 reference (CI-separated)",
                passed=kappa_b.ci_high < kappa_0.ci_low,
                observed=f"biased {_fmt_estimate(kappa_b)}",
                expected=(
                    f"CI high < sigma=0 reference CI low {_fmt(kappa_0.ci_low)} "
                    f"(reference est={_fmt(kappa_0.estimate)})"
                ),
            )
        )
    flip_b = _find_estimate(biased.results, "template_max_flip")
    if flip_b is None:
        checks.append(
            _missing_metric_check(
                "template_max_flip elevated", "template_max_flip", biased.results
            )
        )
    else:
        flip_0 = _find_estimate(reference.results, "template_max_flip")
        ref_note = "" if flip_0 is None else f" (sigma=0 reference est={_fmt(flip_0.estimate)})"
        checks.append(
            Check(
                description="template_max_flip elevated",
                passed=flip_b.estimate > _MAX_FLIP_ELEVATED,
                observed=f"biased {_fmt_estimate(flip_b)}{ref_note}",
                expected=f"estimate > {_MAX_FLIP_ELEVATED:g} (flag ceiling)",
            )
        )
    checks.append(
        _check_flag(
            "template_sensitivity_high fires",
            biased.card,
            "template_sensitivity_high",
            fires=True,
        )
    )
    return ScenarioResult(name="template", checks=checks)


def _scenario_stability(params: _Params) -> ScenarioResult:
    """Scenario 6 — instability (noise_sigma=1.0, k repeats).

    Also pins the exact-unanimity property: with ``noise_sigma=0`` the
    mock judge's repeat draws are byte-identical (verdict uniform keyed by
    the custom-id *stem*), so ``unanimity_rate == 1.0`` exactly.
    """
    items = params.items()
    probe_config = params.probe_config()
    noisy = _run_pipeline(
        items, ["stability"], probe_config, MockJudgeConfig(seed=params.seed, noise_sigma=1.0)
    )
    stable = _run_pipeline(
        items, ["stability"], probe_config, MockJudgeConfig(seed=params.seed, noise_sigma=0.0)
    )

    checks: list[Check] = []
    noisy_est = _find_estimate(noisy.results, "unanimity_rate")
    if noisy_est is None:
        checks.append(
            _missing_metric_check(
                "unanimity_rate well below 1 under noise", "unanimity_rate", noisy.results
            )
        )
    else:
        checks.append(
            Check(
                description="unanimity_rate well below 1 under noise",
                passed=noisy_est.estimate < _UNANIMITY_FLOOR,
                observed=_fmt_estimate(noisy_est),
                expected=f"estimate < {_UNANIMITY_FLOOR:g}",
            )
        )
    stable_est = _find_estimate(stable.results, "unanimity_rate")
    if stable_est is None:
        checks.append(
            _missing_metric_check(
                "unanimity_rate == 1 exactly at zero noise", "unanimity_rate", stable.results
            )
        )
    else:
        checks.append(
            Check(
                description="unanimity_rate == 1 exactly at zero noise",
                passed=stable_est.estimate == 1.0,
                observed=_fmt_estimate(stable_est),
                expected="estimate == 1.0 exactly",
            )
        )
    checks.append(
        _check_flag("instability_high fires under noise", noisy.card, "instability_high", fires=True)
    )
    checks.append(
        _check_flag(
            "instability_high silent at zero noise", stable.card, "instability_high", fires=False
        )
    )
    return ScenarioResult(name="stability", checks=checks)


def _scenario_mixed(params: _Params) -> ScenarioResult:
    """Scenario 7 — position AND length bias together, no cross-contamination.

    Both analytic truths are computed under the *mixed* config from the
    executed requests, so CI coverage demonstrates each probe recovers its
    own effect undistorted by the other (the pad contrast pools both
    presentation orders, cancelling position; the position symmetrization
    cancels length).
    """
    items = params.items()
    judge = MockJudgeConfig(seed=params.seed, beta_position=0.8, beta_length=1.0)
    probe_config = params.probe_config()
    pipe = _run_pipeline(items, ["position", "verbosity"], probe_config, judge)

    first_truth = expected_first_pick_rate(judge, pipe.requests)
    pad_truth = expected_pad_pick_rate(judge, pipe.requests)
    first = _find_estimate(pipe.results, "first_pick_rate")
    pad = _find_estimate(pipe.results, "pad_pick_rate")
    checks = [
        _check_ci_covers(
            "first_pick_rate CI covers mixed-config analytic truth",
            first,
            first_truth,
            "first_pick_rate",
            pipe.results,
        ),
        _check_significant(
            "first_pick_rate significant vs 0.5",
            first,
            probe_config.alpha,
            "first_pick_rate",
            pipe.results,
        ),
        _check_ci_covers(
            "pad_pick_rate CI covers mixed-config analytic truth (no contamination)",
            pad,
            pad_truth,
            "pad_pick_rate",
            pipe.results,
        ),
        _check_significant(
            "pad_pick_rate significant vs 0.5",
            pad,
            probe_config.alpha,
            "pad_pick_rate",
            pipe.results,
        ),
        _check_flag(
            "position_bias_detected fires", pipe.card, "position_bias_detected", fires=True
        ),
        _check_flag(
            "verbosity_bias_detected fires", pipe.card, "verbosity_bias_detected", fires=True
        ),
    ]
    return ScenarioResult(name="mixed", checks=checks)


def _scenario_coverage(
    seed: int,
    n_seeds: int = _COVERAGE_N_SEEDS,
    n_items: int = _COVERAGE_N_ITEMS,
    n_boot: int = _COVERAGE_N_BOOT,
) -> ScenarioResult:
    """Full-level scenario — frequentist coverage over ``n_seeds`` seeds.

    For each derived seed: fresh items, fresh position-biased judge
    (beta_position=0.8), position probe end to end; record whether the
    95% ``first_pick_rate`` CI covers that seed's analytic truth. The
    empirical coverage must be at least ``_COVERAGE_FLOOR`` (90%); the
    slack below the nominal 95% absorbs the percentile bootstrap's known
    small-sample undercoverage.

    Args:
        seed: Master seed; rep seeds derive as
            ``(seed * 10 + 7) * n_seeds + rep`` (disjoint from the fast
            scenarios' derived seeds).
        n_seeds: Number of independent replications.
        n_items: Items per replication.
        n_boot: Bootstrap replicates per replication (kept small; the
            check targets CI coverage, not p-value resolution).
    """
    base = seed * 10 + 7
    n_covered = 0
    n_evaluated = 0
    for rep in range(n_seeds):
        rep_seed = base * n_seeds + rep
        items = generate_items(SyntheticConfig(n_items=n_items, seed=rep_seed))
        judge = MockJudgeConfig(seed=rep_seed, beta_position=0.8)
        probe_config = ProbeConfig(n_boot=n_boot, seed=rep_seed)
        requests = plan_suite(items, ["position"], probe_config)
        judgments = MockJudgeExecutor(judge).execute(requests)
        results = analyze_suite(judgments, ["position"], probe_config)
        est = _find_estimate(results, "first_pick_rate")
        if est is None:
            continue
        truth = expected_first_pick_rate(judge, requests)
        n_evaluated += 1
        if est.ci_low <= truth <= est.ci_high:
            n_covered += 1

    coverage = n_covered / n_evaluated if n_evaluated else 0.0
    checks = [
        Check(
            description=f"95% CI coverage of analytic truth over {n_seeds} seeds",
            passed=n_evaluated == n_seeds and coverage >= _COVERAGE_FLOOR,
            observed=(
                f"coverage={coverage:.3f} ({n_covered}/{n_evaluated} CIs; "
                f"{n_seeds - n_evaluated} seed(s) without a usable estimate)"
            ),
            expected=f"all {n_seeds} seeds usable and coverage >= {_COVERAGE_FLOOR:g}",
        )
    ]
    return ScenarioResult(name="coverage", checks=checks)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_validation(level: Literal["fast", "full"], seed: int = 7) -> ValidationReport:
    """Run the planted-bias recovery suite (contract §8).

    Scenarios (each: generate synthetic items -> plan suite ->
    :class:`MockJudgeExecutor` -> analyze -> card -> compare to the mock
    judge's analytic truths):

    1. ``null`` — no planted bias: CIs cover nulls, no bias flag.
    2. ``position`` — ``beta_position=0.8`` at n=400: recovered + flagged.
    3. ``verbosity`` — ``beta_length=1.0``: pad rate at its analytic
       truth, GLM coefficient positive and significant.
    4. ``self_preference`` — ``beta_self=1.0``: excess detected.
    5. ``template`` — ``template_sigma=0.7``: kappa below the sigma=0
       reference, max flip elevated, flag fires.
    6. ``stability`` — ``noise_sigma=1.0``: unanimity well below 1; AND
       exactly 1.0 at ``noise_sigma=0``.
    7. ``mixed`` — position + length together, no cross-contamination.

    The ``full`` level appends the 200-seed frequentist coverage scenario
    (kept behind the ``slow`` pytest marker in the test suite).

    Args:
        level: ``"fast"`` (n≈300, single seed, n_boot=500; CI-friendly,
            <60s) or ``"full"`` (fast scenarios + 200-seed coverage).
        seed: Master seed. Each scenario runs under the derived seed
            ``seed * 10 + scenario_index`` (see module docstring). All
            checks are deterministic given this value; the default (7)
            is the verified shipping seed.

    Returns:
        A :class:`ValidationReport`; ``.passed`` for the verdict and
        ``.render_table()`` for the diagnosable pass/fail table.

    Raises:
        ValueError: If ``level`` is not ``"fast"`` or ``"full"``.
    """
    if level not in ("fast", "full"):
        raise ValueError(f"level must be 'fast' or 'full', got {level!r}")

    def params_for(index: int, n_items: int = _FAST_N_ITEMS) -> _Params:
        return _Params(seed=seed * 10 + index, n_items=n_items, n_boot=_FAST_N_BOOT)

    scenarios = [
        _scenario_null(params_for(0)),
        _scenario_position(params_for(1, _POSITION_N_ITEMS)),
        _scenario_verbosity(params_for(2)),
        _scenario_self_preference(params_for(3)),
        _scenario_template(params_for(4)),
        _scenario_stability(params_for(5)),
        _scenario_mixed(params_for(6)),
    ]
    if level == "full":
        scenarios.append(_scenario_coverage(seed))
    return ValidationReport(level=level, seed=seed, scenarios=scenarios)


__all__ = [
    "Check",
    "ScenarioResult",
    "ValidationReport",
    "run_validation",
]
