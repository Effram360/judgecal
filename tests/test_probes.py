"""Tests for judgecal.probes: plan correctness and hand-computed analyses.

All analyses run on HAND-CONSTRUCTED judgment sets with known answers;
no executors, no network, small ``n_boot`` for speed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from judgecal.core import REQUIRED_META_KEYS, Estimate, Judgment, JudgmentRequest, PairwiseItem
from judgecal.probes import (
    DEFAULT_TEMPLATE_ID,
    PROBE_REGISTRY,
    TEMPLATE_IDS,
    PositionProbe,
    ProbeConfig,
    SelfPreferenceProbe,
    StabilityProbe,
    TemplateProbe,
    VerbosityProbe,
    analyze_suite,
    get_probe,
    pad_text,
    plan_suite,
    render,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_FILLER = (
    "The method improves accuracy on the benchmark. It uses a simple two-stage "
    "pipeline with careful validation. Results hold across all three datasets. "
    "Limitations are discussed in the final section of the report. "
)


def _text(length: int) -> str:
    """Deterministic sentence-y filler of exactly ``length`` characters."""
    repeated = _FILLER * (length // len(_FILLER) + 2)
    return repeated[:length]


def make_item(
    idx: int,
    *,
    label: str | None = "A",
    len_a: int = 160,
    len_b: int = 120,
    author_a: str | None = None,
    author_b: str | None = None,
) -> PairwiseItem:
    return PairwiseItem(
        item_id=f"it{idx}",
        prompt=f"Question {idx}: what does the report conclude?",
        response_a=_text(len_a),
        response_b=_text(len_b),
        label=label,  # type: ignore[arg-type]
        author_a=author_a,
        author_b=author_b,
    )


def judge(
    requests: Sequence[JudgmentRequest],
    verdict_fn: Callable[[JudgmentRequest], str],
) -> list[Judgment]:
    """Turn planned requests into judgments via a verdict rule on meta."""
    return [
        Judgment(
            custom_id=r.custom_id,
            verdict=verdict_fn(r),  # type: ignore[arg-type]
            raw_text="reasoning... [[A]]",
            meta=dict(r.meta),
        )
        for r in requests
    ]


def get_metric(result, name: str) -> Estimate | None:
    return next((e for e in result.estimates if e.name == name), None)


def cfg(**kwargs) -> ProbeConfig:
    kwargs.setdefault("n_boot", 200)
    return ProbeConfig(**kwargs)


# --------------------------------------------------------------------------
# Registry and suite plumbing
# --------------------------------------------------------------------------


def test_registry_contains_all_five_probes():
    assert set(PROBE_REGISTRY) == {
        "position",
        "verbosity",
        "self_preference",
        "template",
        "stability",
    }
    assert isinstance(get_probe("position"), PositionProbe)
    assert isinstance(get_probe("verbosity"), VerbosityProbe)


def test_get_probe_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown probe"):
        get_probe("nonexistent")


def test_plan_suite_concatenates_and_meta_complete():
    items = [make_item(0), make_item(1)]
    requests = plan_suite(items, ["position", "verbosity"], cfg())
    assert len(requests) == 2 * 2 + 2 * 2  # 2 conditions per probe per item
    for r in requests:
        for key in REQUIRED_META_KEYS:
            assert key in r.meta
        assert r.custom_id.startswith("jc-")


def test_plan_suite_accepts_probe_instances():
    items = [make_item(0)]
    requests = plan_suite(items, [PositionProbe()], cfg())
    assert len(requests) == 2


# --------------------------------------------------------------------------
# Plan correctness
# --------------------------------------------------------------------------


def test_position_plan_swap_coordinates():
    item = make_item(0, label="B", len_a=150, len_b=100, author_a="m1", author_b="m2")
    requests = PositionProbe().plan([item], cfg())
    assert [r.meta["condition"] for r in requests] == ["orig", "swap"]
    orig, swap = requests

    assert orig.meta["first_is_a"] is True
    assert orig.meta["first_len"] == 150
    assert orig.meta["second_len"] == 100
    assert orig.meta["first_author"] == "m1"
    assert orig.meta["label_first"] == "second"  # label B, A presented first

    assert swap.meta["first_is_a"] is False
    assert swap.meta["first_len"] == 100  # response_b presented first
    assert swap.meta["second_len"] == 150
    assert swap.meta["first_author"] == "m2"
    assert swap.meta["label_first"] == "first"  # label B, B presented first

    assert orig.custom_id != swap.custom_id
    assert orig.meta["repeat"] == 0 and swap.meta["repeat"] == 0


def test_template_plan_variants_and_clamp():
    items = [make_item(0)]
    requests = TemplateProbe().plan(items, cfg())
    assert [r.meta["condition"] for r in requests] == list(TEMPLATE_IDS)
    assert len({r.custom_id for r in requests}) == 5  # bodies differ per template

    three = TemplateProbe().plan(items, cfg(n_template_variants=3))
    assert [r.meta["condition"] for r in three] == ["tpl:default", "tpl:v1", "tpl:v2"]

    with pytest.raises(ValueError, match="n_template_variants"):
        TemplateProbe().plan(items, cfg(n_template_variants=6))


def test_stability_plan_repeats_share_body_hash():
    items = [make_item(0)]
    requests = StabilityProbe().plan(items, cfg(stability_k=3))
    assert len(requests) == 3
    assert all(r.body == requests[0].body for r in requests)  # identical bodies
    prefixes = {r.custom_id.rsplit("-r", 1)[0] for r in requests}
    suffixes = [r.custom_id.rsplit("-r", 1)[1] for r in requests]
    assert len(prefixes) == 1  # same content hash
    assert suffixes == ["0", "1", "2"]  # distinct repeat suffixes
    assert [r.meta["repeat"] for r in requests] == [0, 1, 2]


def test_verbosity_plan_padded_lengths_and_determinism():
    items = [make_item(0, len_a=200), make_item(1, len_a=300)]
    requests = VerbosityProbe().plan(items, cfg())
    assert len(requests) == 4
    by_cond = {(r.meta["item_id"], r.meta["condition"]): r for r in requests}

    pad_first = by_cond[("it0", "pad_first")]
    pad_second = by_cond[("it0", "pad_second")]
    assert pad_first.meta["first_len"] > pad_first.meta["second_len"]
    assert pad_second.meta["second_len"] > pad_second.meta["first_len"]
    assert pad_second.meta["first_len"] == 200  # original presented first
    assert pad_first.meta["label_first"] == "tie"  # meaning-preserving contrast
    assert pad_first.meta["first_is_a"] is True

    again = VerbosityProbe().plan(items, cfg())
    assert [r.custom_id for r in again] == [r.custom_id for r in requests]  # deterministic


def test_render_messages_shape():
    messages = render(DEFAULT_TEMPLATE_ID, "Q?", "resp one", "resp two")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "resp one" in messages[1]["content"]
    assert "[[A]]" in messages[1]["content"]  # output convention pinned
    with pytest.raises(KeyError):
        render("tpl:nope", "Q?", "a", "b")


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------


def test_padding_hits_target_ratio_and_is_deterministic():
    texts = [
        _text(180),
        _text(420),
        "One single long sentence without any internal punctuation that just keeps "
        "going to test the no-sentence-boundary path of the padder",
    ]
    for text in texts:
        for ratio in (1.3, 1.6, 2.0):
            padded = pad_text(text, ratio)
            achieved = len(padded) / len(text)
            assert abs(achieved - ratio) <= 0.15 * ratio, (len(text), ratio, achieved)
            assert padded.startswith(text)
            assert pad_text(text, ratio) == padded  # deterministic


def test_padding_noop_cases():
    assert pad_text("", 1.6) == ""
    assert pad_text("Hello there.", 1.0) == "Hello there."


def test_padding_terminator_free_single_token_hits_target():
    """A long single token without sentence terminators (code blob,
    base64, URL) must not overshoot: the final top-up word is truncated
    to the remaining gap (regression for the ratio-2.0 overshoot)."""
    for text in ("A" * 5000, "x" * 487):
        for ratio in (1.3, 1.6, 2.0):
            padded = pad_text(text, ratio)
            achieved = len(padded) / len(text)
            assert abs(achieved - ratio) <= 0.01, (len(text), ratio, achieved)
            assert padded.startswith(text)
            assert pad_text(text, ratio) == padded  # deterministic


# --------------------------------------------------------------------------
# Position analysis
# --------------------------------------------------------------------------


def test_position_always_first_known_answers():
    items = [make_item(i) for i in range(10)]
    probe = PositionProbe()
    judgments = judge(probe.plan(items, cfg()), lambda r: "first")
    result = probe.analyze(judgments, cfg())

    assert result.n_items == 10
    assert result.n_judgments == 20
    assert result.invalid_rate == 0.0

    fpr = get_metric(result, "first_pick_rate")
    assert fpr is not None
    assert fpr.estimate == pytest.approx(1.0)
    assert fpr.null_value == 0.5
    assert fpr.p_value is not None and fpr.p_value < 0.05
    # Every replicate is 1.0 (the judge always picks first), so the
    # realized bootstrap SE is 0 and the SE-based MDE is undefined.
    assert fpr.detail["boot_se"] == 0.0
    assert fpr.mde is None
    assert fpr.n == 20

    flip = get_metric(result, "flip_rate_decisive")
    assert flip is not None
    assert flip.estimate == pytest.approx(1.0)  # mapped winners always disagree
    assert flip.n == 10

    mc = get_metric(result, "positional_mcnemar")
    assert mc is not None
    assert mc.detail["b"] == 10 and mc.detail["c"] == 0
    assert mc.estimate == pytest.approx(1.0)
    assert mc.p_value is not None and mc.p_value < 0.01
    assert mc.mde is not None


def test_position_analyze_is_deterministic():
    items = [make_item(i) for i in range(8)]
    probe = PositionProbe()
    judgments = judge(
        probe.plan(items, cfg()),
        lambda r: "first" if int(r.meta["item_id"][2:]) % 2 else "second",
    )
    r1 = probe.analyze(judgments, cfg())
    r2 = probe.analyze(judgments, cfg())
    e1 = get_metric(r1, "first_pick_rate")
    e2 = get_metric(r2, "first_pick_rate")
    assert e1 is not None and e2 is not None
    assert (e1.ci_low, e1.ci_high, e1.p_value) == (e2.ci_low, e2.ci_high, e2.p_value)


def test_position_missing_swap_condition_warns():
    items = [make_item(i) for i in range(6)]
    probe = PositionProbe()
    requests = [r for r in probe.plan(items, cfg()) if r.meta["condition"] == "orig"]
    result = probe.analyze(judge(requests, lambda r: "first"), cfg())
    assert any("swap" in w for w in result.warnings)
    assert get_metric(result, "first_pick_rate") is not None
    assert get_metric(result, "flip_rate_decisive") is None
    assert get_metric(result, "positional_mcnemar") is None


def test_position_all_invalid_tolerated():
    items = [make_item(i) for i in range(5)]
    probe = PositionProbe()
    result = probe.analyze(judge(probe.plan(items, cfg()), lambda r: "invalid"), cfg())
    assert result.invalid_rate == pytest.approx(1.0)
    assert result.estimates == []
    assert result.warnings  # every metric reported why it was skipped


def test_position_empty_judgments():
    result = PositionProbe().analyze([], cfg())
    assert result.estimates == []
    assert result.n_judgments == 0
    assert result.warnings


# --------------------------------------------------------------------------
# Verbosity analysis
# --------------------------------------------------------------------------


def test_verbosity_always_picks_padded():
    items = [make_item(i) for i in range(10)]
    probe = VerbosityProbe()
    judgments = judge(
        probe.plan(items, cfg()),
        lambda r: "first" if r.meta["condition"] == "pad_first" else "second",
    )
    result = probe.analyze(judgments, cfg())

    ppr = get_metric(result, "pad_pick_rate")
    assert ppr is not None
    assert ppr.estimate == pytest.approx(1.0)
    assert ppr.null_value == 0.5
    assert ppr.p_value is not None and ppr.p_value < 0.05
    # No position judgments were provided: the GLM must be skipped with a warning.
    assert get_metric(result, "length_glm_coef") is None
    assert any("position judgments" in w for w in result.warnings)


def test_verbosity_glm_positive_on_length_following_judge():
    # Lengths span ratios both ways; two mild contrarians at the smallest
    # |log-ratio| prevent perfect separation so the MLE exists.
    sizes = [
        (150, 100),
        (100, 150),
        (180, 90),
        (90, 180),
        (120, 100),
        (100, 120),
        (160, 80),
        (80, 160),
        (140, 70),
        (70, 140),
        (110, 100),
        (100, 110),
    ]
    items = [make_item(i, label=None, len_a=a, len_b=b) for i, (a, b) in enumerate(sizes)]
    contrarian = {"it10", "it11"}

    def verdict(r: JudgmentRequest) -> str:
        longer_first = r.meta["first_len"] > r.meta["second_len"]
        if r.meta["item_id"] in contrarian:
            longer_first = not longer_first
        return "first" if longer_first else "second"

    position_judgments = judge(PositionProbe().plan(items, cfg()), verdict)
    result = VerbosityProbe().analyze(position_judgments, cfg(n_boot=100))

    glm = get_metric(result, "length_glm_coef")
    assert glm is not None
    assert glm.estimate > 0  # longer-presented-first raises pick-first odds
    assert glm.null_value == 0.0
    assert glm.n == 24
    assert glm.detail["controls"] == []  # no labels -> no control column


# --------------------------------------------------------------------------
# Self-preference analysis
# --------------------------------------------------------------------------


def test_self_preference_warns_when_authors_absent():
    items = [make_item(i) for i in range(4)]  # author_a/b None
    probe = SelfPreferenceProbe()
    result = probe.analyze(judge(probe.plan(items, cfg()), lambda r: "first"), cfg())
    assert result.estimates == []
    assert any("author" in w.lower() for w in result.warnings)


def test_self_preference_detects_planted_excess():
    # Treatment: self-authored side A, ground truth says B wins.
    treat = [make_item(i, label="B", author_a="judge-self", author_b="other") for i in range(6)]
    # Control: no self side, labeled winner alternates.
    ctrl = [
        make_item(10 + i, label="A" if i % 2 else "B", author_a="m1", author_b="m2")
        for i in range(6)
    ]
    probe = SelfPreferenceProbe()
    requests = probe.plan(treat + ctrl, cfg())

    def verdict(r: JudgmentRequest) -> str:
        if r.meta["first_author"] == "judge-self":
            return "first"  # always pick own output
        if r.meta["second_author"] == "judge-self":
            return "second"
        return r.meta["label_first"]  # control: pick the true winner

    result = probe.analyze(judge(requests, verdict), cfg())
    est = get_metric(result, "self_error_pick_excess")
    assert est is not None
    assert est.estimate == pytest.approx(1.0)  # 1.0 self-error rate - 0.0 control rate
    assert est.null_value == 0.0
    assert est.p_value is not None and est.p_value < 0.05
    assert est.detail["n_self_error"] == 12
    assert est.detail["n_control"] == 12
    assert est.detail["self_error_pick_rate"] == pytest.approx(1.0)
    assert est.detail["control_error_pick_rate"] == pytest.approx(0.0)


def test_few_cluster_bootstrap_warning_surfaces_into_probe_warnings():
    """Regression (stats review F1): below ~15 items the anti-conservative
    cluster-bootstrap regime must be surfaced as a probe warning."""
    items = [make_item(i) for i in range(8)]
    probe = PositionProbe()
    judgments = judge(
        probe.plan(items, cfg()),
        lambda r: "first" if int(r.meta["item_id"][2:]) % 2 else "second",
    )
    result = probe.analyze(judgments, cfg())
    assert any("anti-conservative" in w for w in result.warnings)

    # At >= 15 items the warning must NOT fire.
    items = [make_item(i) for i in range(20)]
    judgments = judge(
        probe.plan(items, cfg()),
        lambda r: "first" if int(r.meta["item_id"][2:]) % 2 else "second",
    )
    result = probe.analyze(judgments, cfg())
    assert not any("anti-conservative" in w for w in result.warnings)


def test_glm_mde_suppressed_when_unconverged():
    """Regression (stats review F6): a separated/unconverged GLM must not
    report an MDE (previously n=5 produced mde ~1.6e-12 — implied
    infinite power next to a did-not-converge warning)."""
    sizes = [(150, 100), (100, 150), (180, 90), (90, 180), (120, 100)]
    items = [make_item(i, label=None, len_a=a, len_b=b) for i, (a, b) in enumerate(sizes)]
    position_judgments = judge(
        PositionProbe().plan(items, cfg()),
        lambda r: "first" if r.meta["first_len"] > r.meta["second_len"] else "second",
    )
    result = VerbosityProbe().analyze(position_judgments, cfg(n_boot=100))
    glm = get_metric(result, "length_glm_coef")
    assert glm is not None
    assert glm.detail["converged"] is False
    assert glm.mde is None
    assert any("did not converge" in w for w in result.warnings)


def test_glm_method_is_tagged_observational():
    """Stats review F4: the length GLM must self-identify as observational."""
    sizes = [(150, 100), (100, 150), (180, 90), (90, 180), (120, 100), (110, 100)]
    items = [make_item(i, label=None, len_a=a, len_b=b) for i, (a, b) in enumerate(sizes)]

    def verdict(r: JudgmentRequest) -> str:
        longer_first = r.meta["first_len"] > r.meta["second_len"]
        if r.meta["item_id"] == "it5":
            longer_first = not longer_first
        return "first" if longer_first else "second"

    position_judgments = judge(PositionProbe().plan(items, cfg()), verdict)
    result = VerbosityProbe().analyze(position_judgments, cfg(n_boot=100))
    glm = get_metric(result, "length_glm_coef")
    assert glm is not None
    assert "(observational)" in glm.method


def test_self_preference_composition_diagnostic_flags_gap_imbalance():
    """Regression (stats review F2): an author-blind judge with a quality-gap
    imbalance between the self and control sets produces a spuriously
    significant excess; the composition diagnostic must mark it."""
    # Self items: small latent gap (self loses narrowly). Control: large gap.
    treat = [make_item(i, label="B", author_a="judge-self", author_b="other") for i in range(20)]
    for it in treat:
        it.meta.update({"latent_quality_a": 0.4, "latent_quality_b": 0.6})
    ctrl = [make_item(100 + i, label="A", author_a="m1", author_b="m2") for i in range(20)]
    for it in ctrl:
        it.meta.update({"latent_quality_a": 0.8, "latent_quality_b": 0.2})

    # Author-blind, quality-driven verdict rule: pick the loser more often
    # on close pairs (gap 0.2) than on easy pairs (gap 0.6).
    def verdict(r: JudgmentRequest) -> str:
        gap = abs(r.meta["first_latent_q"] - r.meta["second_latent_q"])
        loser = "second" if r.meta["label_first"] == "first" else "first"
        winner = r.meta["label_first"]
        pick_loser = gap < 0.4 and int(r.meta["item_id"][2:]) % 3 == 0
        return loser if pick_loser else winner

    probe = SelfPreferenceProbe()
    result = probe.analyze(judge(probe.plan(treat + ctrl, cfg()), verdict), cfg())
    est = get_metric(result, "self_error_pick_excess")
    assert est is not None
    assert est.detail["composition_imbalance"] is True
    assert est.detail["self_mean_abs_latent_gap"] == pytest.approx(0.2)
    assert est.detail["control_mean_abs_latent_gap"] == pytest.approx(0.6)
    assert any("differ in composition" in w for w in result.warnings)


def test_self_preference_composition_quiet_when_sets_match():
    """Matched gap distributions must not trigger the composition warning."""
    treat = [make_item(i, label="B", author_a="judge-self", author_b="other") for i in range(20)]
    ctrl = [make_item(100 + i, label="A", author_a="m1", author_b="m2") for i in range(20)]
    for it in treat + ctrl:
        it.meta.update({"latent_quality_a": 0.4, "latent_quality_b": 0.6})
    probe = SelfPreferenceProbe()
    result = probe.analyze(
        judge(probe.plan(treat + ctrl, cfg()), lambda r: r.meta["label_first"]), cfg()
    )
    est = get_metric(result, "self_error_pick_excess")
    assert est is not None
    assert est.detail["composition_imbalance"] is False
    assert not any("differ in composition" in w for w in result.warnings)


def test_self_preference_needs_labels():
    items = [make_item(i, label=None, author_a="judge-self", author_b="other") for i in range(4)]
    probe = SelfPreferenceProbe()
    result = probe.analyze(judge(probe.plan(items, cfg()), lambda r: "first"), cfg())
    assert result.estimates == []
    assert any("label" in w.lower() or "ground-truth" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------
# Template analysis
# --------------------------------------------------------------------------


def test_template_kappa_one_when_all_templates_agree():
    items = [make_item(i) for i in range(6)]
    per_item_verdict = {
        "it0": "first",
        "it1": "first",
        "it2": "first",
        "it3": "second",
        "it4": "second",
        "it5": "tie",
    }
    probe = TemplateProbe()
    judgments = judge(probe.plan(items, cfg()), lambda r: per_item_verdict[r.meta["item_id"]])
    result = probe.analyze(judgments, cfg(n_boot=100))

    kappa = get_metric(result, "template_fleiss_kappa")
    assert kappa is not None
    assert kappa.estimate == pytest.approx(1.0)
    assert kappa.n == 6
    assert kappa.p_value is None  # descriptive metric: no null

    max_flip = get_metric(result, "template_max_flip")
    assert max_flip is not None
    assert max_flip.estimate == pytest.approx(0.0)

    acc_range = get_metric(result, "template_accuracy_range")
    assert acc_range is not None  # items carry labels
    assert acc_range.estimate == pytest.approx(0.0)


def test_template_max_flip_detects_deviant_template():
    items = [make_item(i) for i in range(8)]

    def verdict(r: JudgmentRequest) -> str:
        base = "first" if int(r.meta["item_id"][2:]) % 2 else "second"
        if r.meta["condition"] == "tpl:v1":  # v1 always flips the decision
            return "second" if base == "first" else "first"
        return base

    probe = TemplateProbe()
    result = probe.analyze(judge(probe.plan(items, cfg()), verdict), cfg(n_boot=100))

    max_flip = get_metric(result, "template_max_flip")
    assert max_flip is not None
    assert max_flip.estimate == pytest.approx(1.0)
    assert "tpl:v1" in max_flip.detail["pair"]
    assert max_flip.detail["n_pairs_compared"] == 10  # C(5, 2)

    kappa = get_metric(result, "template_fleiss_kappa")
    assert kappa is not None
    assert kappa.estimate < 1.0


def test_template_missing_conditions_warn():
    items = [make_item(i) for i in range(4)]
    probe = TemplateProbe()
    requests = [
        r for r in probe.plan(items, cfg()) if r.meta["condition"] in ("tpl:default", "tpl:v1")
    ]
    result = probe.analyze(judge(requests, lambda r: "first"), cfg(n_boot=50))
    assert any("tpl:v2" in w for w in result.warnings)
    assert get_metric(result, "template_max_flip") is not None


# --------------------------------------------------------------------------
# Stability analysis
# --------------------------------------------------------------------------


def test_stability_known_unanimity_and_flip():
    items = [make_item(i) for i in range(5)]
    flippers = {"it3", "it4"}

    def verdict(r: JudgmentRequest) -> str:
        if r.meta["item_id"] in flippers:
            return "first" if r.meta["repeat"] % 2 == 0 else "second"
        return "first"

    probe = StabilityProbe()
    judgments = judge(probe.plan(items, cfg(stability_k=4)), verdict)
    result = probe.analyze(judgments, cfg(stability_k=4, n_boot=100))

    unanimity = get_metric(result, "unanimity_rate")
    assert unanimity is not None
    assert unanimity.estimate == pytest.approx(3 / 5)
    assert unanimity.n == 5

    # Flippers: verdicts f,s,f,s -> 4 disagreeing pairs of C(4,2)=6 -> 2/3.
    flip = get_metric(result, "mean_pairwise_flip")
    assert flip is not None
    assert flip.estimate == pytest.approx((0 + 0 + 0 + 2 / 3 + 2 / 3) / 5)

    kappa = get_metric(result, "stability_fleiss_kappa")
    assert kappa is not None
    assert -1.0 <= kappa.estimate < 1.0


def test_stability_perfectly_stable_judge():
    items = [make_item(i) for i in range(4)]
    probe = StabilityProbe()
    judgments = judge(probe.plan(items, cfg(stability_k=3)), lambda r: "first")
    result = probe.analyze(judgments, cfg(stability_k=3, n_boot=50))
    unanimity = get_metric(result, "unanimity_rate")
    assert unanimity is not None
    assert unanimity.estimate == pytest.approx(1.0)
    flip = get_metric(result, "mean_pairwise_flip")
    assert flip is not None
    assert flip.estimate == pytest.approx(0.0)


# --------------------------------------------------------------------------
# analyze_suite union passing
# --------------------------------------------------------------------------


def test_analyze_suite_passes_position_judgments_to_verbosity():
    items = [make_item(i) for i in range(6)]
    config = cfg(n_boot=100)
    requests = plan_suite(items, ["position", "verbosity"], config)

    def verdict(r: JudgmentRequest) -> str:
        if r.meta["probe"] == "verbosity":
            return "first" if r.meta["condition"] == "pad_first" else "second"
        return "first"  # position: always presented-first

    results = analyze_suite(judge(requests, verdict), ["position", "verbosity"], config)
    assert [r.probe for r in results] == ["position", "verbosity"]

    verbosity = results[1]
    assert get_metric(verbosity, "pad_pick_rate") is not None
    # Verbosity DID receive position judgments: the GLM was attempted and
    # skipped for lack of outcome variation (all picks "first"), not for
    # missing position judgments.
    assert any("no variation" in w for w in verbosity.warnings)
    assert not any("no position judgments" in w for w in verbosity.warnings)
