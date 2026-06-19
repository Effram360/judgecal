"""Tests for judgecal.report — card construction, FDR scope, flags, rendering.

The BH-FDR reference here is computed by hand in pure Python (``_manual_bh``)
so the card's q-values are checked against an independent implementation,
not against judgecal.stats itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from judgecal.core import Estimate, ProbeResult
from judgecal.report import (
    MetricEntry,
    ReliabilityCard,
    build_card,
    load_card,
    render_markdown,
    save_card,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _manual_bh(pvals: list[float]) -> list[float]:
    """Independent BH q-values: p[i] * m / rank, with monotone enforcement."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        q[i] = running
    return q


def _est(
    name: str,
    estimate: float,
    *,
    n: int = 200,
    null: float | None = 0.5,
    p: float | None = None,
    q: float | None = None,
    mde: float | None = None,
    ci: tuple[float, float] | None = None,
    method: str = "cluster_bootstrap",
) -> Estimate:
    lo, hi = ci if ci is not None else (estimate - 0.04, estimate + 0.04)
    return Estimate(
        name=name,
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        n=n,
        method=method,
        null_value=null,
        p_value=p,
        q_value=q,
        mde=mde,
    )


def _result(
    probe: str,
    estimates: list[Estimate],
    *,
    n_items: int = 100,
    n_judgments: int = 200,
    invalid_rate: float = 0.0,
    warnings: list[str] | None = None,
) -> ProbeResult:
    return ProbeResult(
        probe=probe,
        estimates=estimates,
        n_items=n_items,
        n_judgments=n_judgments,
        invalid_rate=invalid_rate,
        warnings=warnings or [],
    )


def _metric(card: ReliabilityCard, probe: str, name: str) -> MetricEntry:
    for entry in card.probes:
        if entry.probe == probe:
            for metric in entry.metrics:
                if metric.name == name:
                    return metric
    raise AssertionError(f"metric {probe}/{name} not found")


JUDGE: dict[str, Any] = {"model": "mock-judge-v1", "quant": "none"}


# --------------------------------------------------------------------------
# q-value filling: card-wide BH family
# --------------------------------------------------------------------------


class TestQValueFilling:
    def test_q_values_match_manual_bh_across_probes(self) -> None:
        """BH is one family across ALL null-bearing metrics, all probes."""
        results = [
            _result(
                "position",
                [
                    _est("first_pick_rate", 0.61, p=0.005, mde=0.04),
                    _est("positional_mcnemar", 0.60, p=0.05, mde=0.06),
                ],
            ),
            _result(
                "verbosity",
                [
                    _est("pad_pick_rate", 0.58, p=0.03, mde=0.05),
                    # Descriptive metric: no p-value -> outside the family.
                    _est("template_fleiss_kappa", 0.7, null=None, p=None),
                ],
            ),
        ]
        card = build_card(results, JUDGE)

        expected = _manual_bh([0.005, 0.05, 0.03])
        got = [
            _metric(card, "position", "first_pick_rate").q_value,
            _metric(card, "position", "positional_mcnemar").q_value,
            _metric(card, "verbosity", "pad_pick_rate").q_value,
        ]
        for g, e in zip(got, expected, strict=True):
            assert g == pytest.approx(e, abs=1e-9)

        # Cross-probe family check: per-probe BH would give 0.03 for the
        # verbosity metric ({0.03} alone); the joint family gives 0.045.
        assert got[2] == pytest.approx(0.045, abs=1e-9)
        assert got[2] != pytest.approx(0.03, abs=1e-9)

    def test_descriptive_metrics_keep_q_none(self) -> None:
        results = [
            _result(
                "template",
                [
                    _est("template_fleiss_kappa", 0.8, null=None, p=None),
                    _est("template_max_flip", 0.1, null=None, p=None),
                ],
            )
        ]
        card = build_card(results, JUDGE)
        assert _metric(card, "template", "template_fleiss_kappa").q_value is None
        assert _metric(card, "template", "template_max_flip").q_value is None

    def test_single_pvalue_q_equals_p(self) -> None:
        results = [_result("position", [_est("first_pick_rate", 0.6, p=0.02, mde=0.04)])]
        card = build_card(results, JUDGE)
        assert _metric(card, "position", "first_pick_rate").q_value == pytest.approx(0.02)

    def test_no_null_bearing_metrics_is_fine(self) -> None:
        results = [_result("template", [_est("template_fleiss_kappa", 0.9, null=None)])]
        card = build_card(results, JUDGE)  # must not raise
        assert card.probes[0].metrics[0].q_value is None


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


class TestBiasFlags:
    def test_position_bias_fires(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.62, p=0.001, mde=0.04)])],
            JUDGE,
        )
        assert "position_bias_detected" in card.probes[0].flags
        assert "position_bias_detected" in card.overall_flags

    def test_position_bias_not_fired_when_effect_too_small(self) -> None:
        # Significant but |0.52 - 0.5| = 0.02 <= 0.05 floor.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.52, p=0.001, mde=0.01)])],
            JUDGE,
        )
        assert "position_bias_detected" not in card.probes[0].flags

    def test_position_bias_not_fired_when_not_significant(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.62, p=0.4, mde=0.30)])],
            JUDGE,
        )
        assert "position_bias_detected" not in card.probes[0].flags

    def test_position_bias_fires_below_null_too(self) -> None:
        # Effect is two-sided: 0.38 is as biased as 0.62.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.38, p=0.001, mde=0.04)])],
            JUDGE,
        )
        assert "position_bias_detected" in card.probes[0].flags

    def test_verbosity_bias_fires_and_not(self) -> None:
        fired = build_card(
            [_result("verbosity", [_est("pad_pick_rate", 0.65, p=0.002, mde=0.05)])],
            JUDGE,
        )
        assert "verbosity_bias_detected" in fired.probes[0].flags
        quiet = build_card(
            [_result("verbosity", [_est("pad_pick_rate", 0.51, p=0.8, mde=0.02)])],
            JUDGE,
        )
        assert "verbosity_bias_detected" not in quiet.probes[0].flags

    def test_self_preference_fires_and_not(self) -> None:
        fired = build_card(
            [
                _result(
                    "self_preference",
                    [_est("self_error_pick_excess", 0.15, null=0.0, p=0.001, mde=0.06)],
                )
            ],
            JUDGE,
        )
        assert "self_preference_detected" in fired.probes[0].flags
        quiet = build_card(
            [
                _result(
                    "self_preference",
                    [_est("self_error_pick_excess", 0.02, null=0.0, p=0.001, mde=0.01)],
                )
            ],
            JUDGE,
        )
        assert "self_preference_detected" not in quiet.probes[0].flags

    def test_self_preference_flag_suppressed_by_composition_imbalance(self) -> None:
        # Regression (stats review F2): significant + above the effect
        # floor, but the probe's composition diagnostic marked the
        # self/control sets as imbalanced -> the detection flag must NOT
        # fire (the metric and q-value stay on the card).
        def est(imbalance: bool) -> Estimate:
            return Estimate(
                name="self_error_pick_excess",
                estimate=0.17,
                ci_low=0.11,
                ci_high=0.24,
                n=300,
                method="two_sample_cluster_bootstrap",
                null_value=0.0,
                p_value=0.002,
                mde=0.04,
                detail={"composition_imbalance": imbalance},
            )

        confounded = build_card([_result("self_preference", [est(True)])], JUDGE)
        assert "self_preference_detected" not in confounded.overall_flags
        clean = build_card([_result("self_preference", [est(False)])], JUDGE)
        assert "self_preference_detected" in clean.overall_flags

    def test_flag_rules_do_not_cross_probes(self) -> None:
        # A metric named first_pick_rate inside another probe must not
        # trigger the position flag.
        card = build_card(
            [_result("verbosity", [_est("first_pick_rate", 0.62, p=0.001, mde=0.04)])],
            JUDGE,
        )
        assert "position_bias_detected" not in card.overall_flags


class TestTemplateAndStabilityFlags:
    def test_template_sensitivity_fires_on_low_kappa(self) -> None:
        card = build_card(
            [_result("template", [_est("template_fleiss_kappa", 0.4, null=None)])],
            JUDGE,
        )
        assert "template_sensitivity_high" in card.probes[0].flags

    def test_template_sensitivity_fires_on_high_max_flip(self) -> None:
        card = build_card(
            [
                _result(
                    "template",
                    [
                        _est("template_fleiss_kappa", 0.9, null=None),
                        _est("template_max_flip", 0.30, null=None),
                    ],
                )
            ],
            JUDGE,
        )
        assert "template_sensitivity_high" in card.probes[0].flags

    def test_template_sensitivity_quiet_when_stable(self) -> None:
        card = build_card(
            [
                _result(
                    "template",
                    [
                        _est("template_fleiss_kappa", 0.85, null=None),
                        _est("template_max_flip", 0.05, null=None),
                    ],
                )
            ],
            JUDGE,
        )
        assert "template_sensitivity_high" not in card.probes[0].flags
        assert card.probes[0].flags == []

    def test_template_kappa_gate_uses_ci_upper_bound(self) -> None:
        # Point estimate below the 0.6 floor but the CI upper bound above
        # it: the kappa clause must NOT fire on the noisy point estimate.
        card = build_card(
            [
                _result(
                    "template",
                    [
                        _est("template_fleiss_kappa", 0.55, null=None, ci=(0.35, 0.75)),
                        _est("template_max_flip", 0.10, null=None),
                    ],
                )
            ],
            JUDGE,
        )
        assert "template_sensitivity_high" not in card.probes[0].flags

    def test_template_flag_suppressed_when_stability_kappa_equally_low(self) -> None:
        # Regression (stats review F3, stochastic-judge conflation): a
        # temperature>0 / nondeterministically-served judge depresses BOTH
        # kappas the same way (reviewer's construction: template kappa
        # 0.36, max_flip 0.54 with template_sigma=0). When the same card's
        # stability kappa is equally low, the disagreement is verdict
        # noise, not template effects -> no template flag.
        card = build_card(
            [
                _result(
                    "template",
                    [
                        _est("template_fleiss_kappa", 0.36, null=None, ci=(0.25, 0.47)),
                        _est("template_max_flip", 0.54, null=None),
                    ],
                ),
                _result(
                    "stability",
                    [
                        _est("unanimity_rate", 0.55, null=None),
                        _est("stability_fleiss_kappa", 0.37, null=None),
                    ],
                ),
            ],
            JUDGE,
        )
        assert "template_sensitivity_high" not in card.overall_flags
        # The instability is still reported -- as instability.
        assert "instability_high" in card.overall_flags

    def test_template_flag_fires_when_disagreement_exceeds_repeat_noise(self) -> None:
        # Same template metrics, but the judge is repeat-stable (stability
        # kappa 0.95): the disagreement is attributable to templates.
        card = build_card(
            [
                _result(
                    "template",
                    [
                        _est("template_fleiss_kappa", 0.36, null=None, ci=(0.25, 0.47)),
                        _est("template_max_flip", 0.54, null=None),
                    ],
                ),
                _result(
                    "stability",
                    [
                        _est("unanimity_rate", 0.98, null=None),
                        _est("stability_fleiss_kappa", 0.95, null=None),
                    ],
                ),
            ],
            JUDGE,
        )
        assert "template_sensitivity_high" in card.overall_flags

    def test_template_caveat_rendered_only_without_stability_data(self) -> None:
        template_result = _result(
            "template",
            [
                _est("template_fleiss_kappa", 0.40, null=None, ci=(0.30, 0.50)),
                _est("template_max_flip", 0.35, null=None),
            ],
        )
        no_stability = build_card([template_result], JUDGE)
        md = render_markdown(no_stability)
        assert "may reflect verdict instability" in md
        with_stability = build_card(
            [
                template_result,
                _result(
                    "stability",
                    [
                        _est("unanimity_rate", 0.98, null=None),
                        _est("stability_fleiss_kappa", 0.95, null=None),
                    ],
                ),
            ],
            JUDGE,
        )
        md = render_markdown(with_stability)
        assert "template_sensitivity_high" in with_stability.overall_flags
        assert "may reflect verdict instability" not in md

    def test_instability_fires_and_not(self) -> None:
        fired = build_card(
            [_result("stability", [_est("unanimity_rate", 0.6, null=None)])],
            JUDGE,
        )
        assert "instability_high" in fired.probes[0].flags
        quiet = build_card(
            [_result("stability", [_est("unanimity_rate", 0.95, null=None)])],
            JUDGE,
        )
        assert "instability_high" not in quiet.probes[0].flags


class TestUnderpoweredFlag:
    """Design-based power semantics: the MDE is compared
    to the metric's pre-registered MIN_EFFECT_OF_INTEREST floor, NEVER to
    the data-dependent observed effect (post-hoc-power anti-pattern)."""

    def test_fires_when_mde_above_effect_of_interest_floor(self) -> None:
        # first_pick_rate floor is 0.05; mde 0.10 > 0.05; not significant.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.51, p=0.6, mde=0.10)])],
            JUDGE,
        )
        assert "underpowered:first_pick_rate" in card.probes[0].flags

    def test_quiet_when_significant_even_with_big_mde(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.62, p=0.001, mde=0.50)])],
            JUDGE,
        )
        flags = card.probes[0].flags
        assert "underpowered:first_pick_rate" not in flags

    def test_quiet_when_mde_at_or_below_floor(self) -> None:
        # n=800 null-judge case: observed
        # effect ~0.006 with MDE 0.038 <= the 0.05 floor. Under the old
        # observed-effect rule this was labeled underpowered forever; the
        # design-based rule reports adequate power.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.506, p=0.7, mde=0.038)])],
            JUDGE,
        )
        assert "underpowered:first_pick_rate" not in card.probes[0].flags
        md = render_markdown(card)
        row = next(line for line in md.splitlines() if "`first_pick_rate`" in line)
        assert "✓" in row  # no signal at adequate power

    def test_quiet_without_mde_or_null(self) -> None:
        card = build_card(
            [
                _result(
                    "position",
                    [
                        _est("first_pick_rate", 0.51, p=0.6, mde=None),
                        _est("template_fleiss_kappa", 0.5, null=None, mde=0.4),
                    ],
                )
            ],
            JUDGE,
        )
        assert all(not f.startswith("underpowered:") for f in card.probes[0].flags)

    def test_estimate_at_null_with_adequate_power_is_not_underpowered(self) -> None:
        # Estimate exactly at the null used to force "underpowered" at ANY
        # sample size (observed effect 0); design-based semantics: mde 0.03
        # <= 0.05 floor -> adequately powered clean null.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.5, p=1.0, mde=0.03)])],
            JUDGE,
        )
        assert "underpowered:first_pick_rate" not in card.probes[0].flags

    def test_fires_at_null_estimate_when_mde_above_floor(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.5, p=1.0, mde=0.06)])],
            JUDGE,
        )
        assert "underpowered:first_pick_rate" in card.probes[0].flags

    def test_metric_without_registered_floor_never_underpowered(self) -> None:
        # No MIN_EFFECT_OF_INTEREST entry -> power adequacy is undefined;
        # the flag must not fire (and the glyph is "?" rather than a claim).
        card = build_card(
            [_result("position", [_est("some_new_metric", 0.51, p=0.6, mde=0.30)])],
            JUDGE,
        )
        assert all(not f.startswith("underpowered:") for f in card.probes[0].flags)


class TestInvalidRateFlag:
    def test_fires_above_threshold(self) -> None:
        card = build_card(
            [_result("position", [], invalid_rate=0.10)],
            JUDGE,
        )
        assert "high_invalid_rate" in card.probes[0].flags

    def test_quiet_at_or_below_threshold(self) -> None:
        card = build_card([_result("position", [], invalid_rate=0.05)], JUDGE)
        assert "high_invalid_rate" not in card.probes[0].flags
        card2 = build_card([_result("position", [], invalid_rate=0.01)], JUDGE)
        assert "high_invalid_rate" not in card2.probes[0].flags


class TestOverallFlags:
    def test_union_deduplicated_in_probe_order(self) -> None:
        card = build_card(
            [
                _result("position", [_est("first_pick_rate", 0.62, p=0.001, mde=0.04)],
                        invalid_rate=0.2),
                _result("verbosity", [_est("pad_pick_rate", 0.65, p=0.001, mde=0.04)],
                        invalid_rate=0.2),
            ],
            JUDGE,
        )
        assert card.overall_flags == [
            "position_bias_detected",
            "high_invalid_rate",
            "verbosity_bias_detected",
        ]


# --------------------------------------------------------------------------
# created_utc and card metadata
# --------------------------------------------------------------------------


class TestCardMetadata:
    def test_created_utc_defaults_to_none(self) -> None:
        card = build_card([], JUDGE)
        assert card.created_utc is None

    def test_created_utc_is_caller_supplied(self) -> None:
        stamp = "2026-06-10T12:00:00Z"
        card = build_card([], JUDGE, created_utc=stamp)
        assert card.created_utc == stamp

    def test_schema_version_pinned(self) -> None:
        card = build_card([], JUDGE)
        assert card.card_schema_version == "0.1"

    def test_judgecal_version_override(self) -> None:
        card = build_card([], JUDGE, judgecal_version="9.9.9")
        assert card.judgecal_version == "9.9.9"

    def test_warnings_and_counts_carried_over(self) -> None:
        card = build_card(
            [
                _result(
                    "position",
                    [],
                    n_items=42,
                    n_judgments=84,
                    invalid_rate=0.01,
                    warnings=["swap condition missing"],
                )
            ],
            JUDGE,
        )
        entry = card.probes[0]
        assert (entry.n_items, entry.n_judgments) == (42, 84)
        assert entry.invalid_rate == pytest.approx(0.01)
        assert entry.warnings == ["swap condition missing"]


# --------------------------------------------------------------------------
# Persistence: round-trip and tolerant load
# --------------------------------------------------------------------------


def _full_card() -> ReliabilityCard:
    results = [
        _result(
            "position",
            [
                _est("first_pick_rate", 0.61, p=0.004, mde=0.045),
                _est("flip_rate_decisive", 0.18, null=None, ci=(0.14, 0.23), method="wilson"),
                _est("positional_mcnemar", 0.60, p=0.03, mde=0.07, method="mcnemar_midp"),
            ],
            n_items=200,
            n_judgments=400,
            invalid_rate=0.015,
        ),
        _result(
            "verbosity",
            [_est("pad_pick_rate", 0.55, p=0.2, mde=0.12)],
            warnings=["small n for GLM"],
        ),
        _result(
            "stability",
            [_est("unanimity_rate", 0.95, null=None, ci=(0.91, 0.97), method="wilson")],
        ),
    ]
    return build_card(
        results,
        JUDGE,
        datasets=[{"name": "synthetic", "license": "n/a"}],
        config={"alpha": 0.05, "n_boot": 2000, "seed": 7},
        created_utc="2026-06-10T00:00:00Z",
        notes=["Demo card built from the deterministic mock judge."],
    )


class TestPersistence:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        card = _full_card()
        path = tmp_path / "card.json"
        save_card(card, path)
        loaded = load_card(path)
        assert loaded == card
        assert loaded.model_dump() == card.model_dump()

    def test_round_trip_accepts_str_path(self, tmp_path: Path) -> None:
        card = _full_card()
        path = str(tmp_path / "card.json")
        save_card(card, path)
        assert load_card(path) == card

    def test_tolerant_load_ignores_extra_fields(self, tmp_path: Path) -> None:
        card = _full_card()
        path = tmp_path / "card.json"
        save_card(card, path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["future_top_level_field"] = {"x": 1}
        blob["probes"][0]["future_probe_field"] = "y"
        blob["probes"][0]["metrics"][0]["future_metric_field"] = [1, 2, 3]
        path.write_text(json.dumps(blob), encoding="utf-8")
        loaded = load_card(path)
        assert loaded == card

    def test_load_rejects_wrong_schema_version(self, tmp_path: Path) -> None:
        card = _full_card()
        path = tmp_path / "card.json"
        save_card(card, path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["card_schema_version"] = "999"
        path.write_text(json.dumps(blob), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_card(path)


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

# Marketing over-claims that must never appear in a generated report card.
OVERCLAIM_PHRASES = (
    "first toolkit",
    "first judge-reliability toolkit",
    "contamination-proof",
    "empty niche",
)


class TestMarkdown:
    def test_contains_expected_sections_and_numbers(self) -> None:
        card = _full_card()
        md = render_markdown(card)
        assert md.startswith("# Judge Reliability Card")
        assert "mock-judge-v1" in md
        assert "## Summary" in md
        assert "## Probes" in md
        assert "### position" in md
        assert "### verbosity" in md
        assert "`first_pick_rate`" in md
        assert "95% CI" in md
        assert "Benjamini–Hochberg" in md
        assert "MDE" in md
        assert "2026-06-10T00:00:00Z" in md
        assert "Demo card built from the deterministic mock judge." in md
        assert "> Warning: small n for GLM" in md

    def test_summary_describes_position_bias_flag(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.62, p=0.001, mde=0.04)])],
            JUDGE,
        )
        md = render_markdown(card)
        assert "Position bias detected" in md
        assert "62.0%" in md  # the estimate, in plain English
        assert "first-presented" in md

    def test_summary_when_no_flags(self) -> None:
        # mde 0.009 <= the 0.05 effect-of-interest floor: adequately powered.
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.505, p=0.7, mde=0.009)])],
            JUDGE,
        )
        assert card.overall_flags == []
        md = render_markdown(card)
        assert "No reliability flags raised" in md
        assert "MDE" in md

    def test_underpowered_summary_line(self) -> None:
        card = build_card(
            [_result("position", [_est("first_pick_rate", 0.51, p=0.6, mde=0.10)])],
            JUDGE,
        )
        md = render_markdown(card)
        assert "Underpowered" in md
        assert "not evidence of absence" in md
        # Design-based wording: the MDE is compared to the pre-registered
        # floor, never to the observed estimate.
        assert "effect-size-of-interest floor" in md
        assert "observed |effect|" not in md

    def test_observational_glm_gets_obs_verdict_and_footnote(self) -> None:
        # Claims/stats review F4: the observational length GLM must never
        # render a rejected/clear glyph, even when q < 0.05.
        card = build_card(
            [
                _result(
                    "verbosity",
                    [
                        _est("pad_pick_rate", 0.51, p=0.8, mde=0.03),
                        _est(
                            "length_glm_coef",
                            1.99,
                            null=0.0,
                            p=0.002,
                            mde=0.9,
                            ci=(1.07, 3.02),
                            method="cluster_bootstrap_logit (observational)",
                        ),
                    ],
                )
            ],
            JUDGE,
        )
        md = render_markdown(card)
        row = next(line for line in md.splitlines() if "`length_glm_coef`" in line)
        assert "obs." in row
        assert "✗" not in row
        assert "observational association" in md  # the per-metric footnote
        assert "quality–length correlation inflates it" in md
        assert "obs. observational association" in md  # legend entry

    def test_self_preference_summary_surfaces_raw_rates(self) -> None:
        card = build_card(
            [
                _result(
                    "self_preference",
                    [
                        Estimate(
                            name="self_error_pick_excess",
                            estimate=0.15,
                            ci_low=0.08,
                            ci_high=0.22,
                            n=300,
                            method="two_sample_cluster_bootstrap",
                            null_value=0.0,
                            p_value=0.001,
                            mde=0.04,
                            detail={
                                "composition_imbalance": False,
                                "self_error_pick_rate": 0.30,
                                "control_error_pick_rate": 0.15,
                                "n_self_error": 120,
                                "n_control": 180,
                            },
                        )
                    ],
                )
            ],
            JUDGE,
        )
        md = render_markdown(card)
        assert "Self-preference detected" in md
        assert "30.0%" in md and "(n = 120)" in md  # self-set raw rate
        assert "15.0%" in md and "(n = 180)" in md  # control-set raw rate
        assert "unadjusted observational control rate" in md
        assert "matched control" not in md.lower()

    def test_none_values_render_as_dash(self) -> None:
        card = build_card(
            [_result("stability", [_est("unanimity_rate", 0.95, null=None)])],
            JUDGE,
        )
        md = render_markdown(card)
        # The descriptive metric row has no p/q/MDE.
        row = next(line for line in md.splitlines() if "`unanimity_rate`" in line)
        assert "—" in row

    def test_no_overclaim_phrases(self) -> None:
        # Both a heavily flagged card and a clean card.
        cards = [
            _full_card(),
            build_card(
                [
                    _result(
                        "position",
                        [_est("first_pick_rate", 0.7, p=0.0001, mde=0.03)],
                        invalid_rate=0.5,
                    )
                ],
                JUDGE,
            ),
        ]
        for card in cards:
            md = render_markdown(card).lower()
            for phrase in OVERCLAIM_PHRASES:
                assert phrase not in md, f"over-claim phrase in card markdown: {phrase!r}"

    def test_fdr_footer_counts_family(self) -> None:
        card = _full_card()  # 3 null-tested metrics
        md = render_markdown(card)
        assert "across all 3 null-tested metrics" in md

    def test_verdict_glyphs_present(self) -> None:
        card = _full_card()
        md = render_markdown(card)
        assert "✗" in md  # first_pick_rate significant
        assert "–" in md  # descriptive rows
        assert "Verdict key:" in md
