"""Markdown rendering for reliability cards.

Produces the user-facing report: a compact header, one metrics table per
probe, a plain-English summary generated from the flags, and a footer
documenting the FDR scope and the MDE ("absence of evidence quantified")
discipline. Pure function of the card — no clock, no I/O.
"""

from __future__ import annotations

from judgecal.report.card import MetricEntry, ProbeEntry, ReliabilityCard
from judgecal.report.thresholds import (
    FLAG_Q_THRESHOLD,
    HIGH_INVALID_RATE_THRESHOLD,
    MIN_EFFECT_OF_INTEREST,
)

# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

_DASH = "—"

#: Verdict glyphs used in the metric tables (legend printed in the footer).
_GLYPH_REJECTED = "✗"  # null rejected at q < FLAG_Q_THRESHOLD
_GLYPH_CLEAR = "✓"  # no signal, MDE at or below the effect-of-interest floor
_GLYPH_UNDERPOWERED = "?"  # no signal, but MDE above the floor (or no floor/MDE)
_GLYPH_DESCRIPTIVE = "–"  # descriptive metric (no null hypothesis)
_GLYPH_OBSERVATIONAL = "obs."  # observational association; never ✗/✓

#: Footnote rendered under any probe table containing an observational
#: metric (method tagged "(observational)").
_OBSERVATIONAL_FOOTNOTE = (
    "`length_glm_coef` is an observational association — quality–length "
    "correlation inflates it; `pad_pick_rate` is the experimental estimate. "
    "p/q are reported (the metric stays in the FDR family because it tests "
    "a null) but no rejected/clear verdict is assigned."
)


def _is_observational(metric: MetricEntry) -> bool:
    return "(observational)" in metric.method


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a number, or an em-dash for None."""
    if value is None:
        return _DASH
    return f"{value:.{digits}f}"


def _fmt_p(value: float | None) -> str:
    """Format a p/q-value; very small values render as '<0.001'."""
    if value is None:
        return _DASH
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    """Format a proportion as a percentage string."""
    if value is None:
        return _DASH
    return f"{value * 100:.{digits}f}%"


def _ci_text(metric: MetricEntry) -> str:
    return f"{_fmt(metric.estimate)} [{_fmt(metric.ci_low)}, {_fmt(metric.ci_high)}]"


def _ci_pct(metric: MetricEntry) -> str:
    return f"{_fmt_pct(metric.ci_low)}–{_fmt_pct(metric.ci_high)}"


def _glyph(metric: MetricEntry) -> str:
    """Verdict glyph for one metric (see legend constants above).

    Power adequacy is design-based: the MDE is compared to the metric's
    pre-registered ``MIN_EFFECT_OF_INTEREST`` floor, never to the
    observed estimate (post-hoc-power anti-pattern). Observational
    metrics get "obs." — an association is reported, not a verdict.
    """
    if metric.p_value is None or metric.null_value is None:
        return _GLYPH_DESCRIPTIVE
    if _is_observational(metric):
        return _GLYPH_OBSERVATIONAL
    if metric.q_value is not None and metric.q_value < FLAG_Q_THRESHOLD:
        return _GLYPH_REJECTED
    floor = MIN_EFFECT_OF_INTEREST.get(metric.name)
    if metric.mde is None or floor is None:
        return _GLYPH_UNDERPOWERED
    return _GLYPH_CLEAR if metric.mde <= floor else _GLYPH_UNDERPOWERED


def _judge_title(card: ReliabilityCard) -> str:
    judge = card.judge
    name = str(judge.get("model") or judge.get("name") or "unnamed judge")
    extras = [f"{k}={v}" for k, v in judge.items() if k not in ("model", "name")]
    return f"{name} ({', '.join(extras)})" if extras else name


def _ci_label(card: ReliabilityCard) -> str:
    alpha = card.config.get("alpha", 0.05)
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        alpha = 0.05
    return f"{(1 - alpha) * 100:g}% CI"


def _find(card: ReliabilityCard, probe: str, metric: str) -> MetricEntry | None:
    for entry in card.probes:
        if entry.probe == probe:
            for m in entry.metrics:
                if m.name == metric:
                    return m
    return None


def _find_anywhere(card: ReliabilityCard, metric: str) -> MetricEntry | None:
    for entry in card.probes:
        for m in entry.metrics:
            if m.name == metric:
                return m
    return None


# --------------------------------------------------------------------------
# Summary sentences (plain English, one per flag)
# --------------------------------------------------------------------------


def _summary_line(card: ReliabilityCard, flag: str, ci_label: str) -> str:
    """One plain-English bullet for a flag, with numbers pulled from the card."""
    if flag == "position_bias_detected":
        m = _find(card, "position", "first_pick_rate")
        if m is not None:
            return (
                f"**Position bias detected:** the judge picks the first-presented answer "
                f"{_fmt_pct(m.estimate)} of the time ({ci_label} {_ci_pct(m)}, "
                f"q = {_fmt_p(m.q_value)})."
            )
    if flag == "verbosity_bias_detected":
        m = _find(card, "verbosity", "pad_pick_rate")
        if m is not None:
            return (
                f"**Verbosity bias detected:** the judge picks the padded (longer) answer "
                f"{_fmt_pct(m.estimate)} of the time ({ci_label} {_ci_pct(m)}, "
                f"q = {_fmt_p(m.q_value)}) — the padded side is a rule-based restatement "
                f"of the same answer, with no added information."
            )
    if flag == "self_preference_detected":
        m = _find(card, "self_preference", "self_error_pick_excess")
        if m is not None:
            self_rate = m.detail.get("self_error_pick_rate")
            ctrl_rate = m.detail.get("control_error_pick_rate")
            n_self = m.detail.get("n_self_error")
            n_ctrl = m.detail.get("n_control")
            rates = ""
            if self_rate is not None and ctrl_rate is not None:
                rates = (
                    f" — raw rates: picks its own losing answer "
                    f"{_fmt_pct(float(self_rate))} of the time (n = {n_self}), versus "
                    f"{_fmt_pct(float(ctrl_rate))} for losing answers with no self side "
                    f"(n = {n_ctrl})"
                )
            return (
                f"**Self-preference detected:** when its own answer is the ground-truth "
                f"loser, the judge still picks it {m.estimate * 100:+.1f} percentage points "
                f"more often than the unadjusted observational control rate "
                f"({ci_label} {_ci_pct(m)}, q = {_fmt_p(m.q_value)}){rates}."
            )
    if flag == "template_sensitivity_high":
        kappa = _find(card, "template", "template_fleiss_kappa")
        flip = _find(card, "template", "template_max_flip")
        parts = []
        if kappa is not None:
            parts.append(f"Fleiss kappa = {_fmt(kappa.estimate, 2)}")
        if flip is not None:
            parts.append(f"worst template pair flips {_fmt_pct(flip.estimate)} of verdicts")
        detail = "; ".join(parts) if parts else "see template probe table"
        line = (
            f"**High template sensitivity:** semantically equivalent prompt paraphrases "
            f"change verdicts ({detail})."
        )
        if _find(card, "stability", "stability_fleiss_kappa") is None:
            line += (
                " Caveat: this may reflect verdict instability rather than template "
                "effects; run the stability probe to separate them."
            )
        return line
    if flag == "instability_high":
        m = _find(card, "stability", "unanimity_rate")
        if m is not None:
            return (
                f"**High instability:** only {_fmt_pct(m.estimate)} of items receive "
                f"identical verdicts across repeated identical runs "
                f"({ci_label} {_ci_pct(m)})."
            )
    if flag.startswith("underpowered:"):
        name = flag.split(":", 1)[1]
        m = _find_anywhere(card, name)
        if m is not None and m.null_value is not None:
            floor = MIN_EFFECT_OF_INTEREST.get(name)
            return (
                f"**Underpowered — `{name}`:** the smallest detectable effect at this "
                f"sample size is {_fmt(m.mde)}, above the {_fmt(floor)} "
                f"effect-size-of-interest floor — this audit could not have detected "
                f"effects as small as the floor. This null result is not evidence of "
                f"absence."
            )
    if flag == "high_invalid_rate":
        offenders = [
            f"{e.probe} ({_fmt_pct(e.invalid_rate)})"
            for e in card.probes
            if e.invalid_rate > HIGH_INVALID_RATE_THRESHOLD
        ]
        return (
            f"**High invalid rate:** unparseable judge responses in "
            f"{', '.join(offenders) or 'one or more probes'}; estimates exclude them, "
            f"which can bias results if invalidity is not random."
        )
    return f"**Flag raised:** `{flag}`."


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def render_markdown(card: ReliabilityCard) -> str:
    """Render a ReliabilityCard as a compact, readable Markdown report.

    Layout: header (judge, version, scale) → plain-English summary
    generated from the flags → one metrics table per probe (estimate
    [CI], n, p, q, MDE, verdict glyph) → notes → footer documenting the
    card-wide BH-FDR scope and the MDE discipline.

    Args:
        card: A card produced by :func:`judgecal.report.build_card`.

    Returns:
        The Markdown document as a single string.
    """
    ci_label = _ci_label(card)
    total_judgments = sum(e.n_judgments for e in card.probes)
    max_items = max((e.n_items for e in card.probes), default=0)

    lines: list[str] = []
    lines.append(f"# Judge Reliability Card — {_judge_title(card)}")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| **Judge** | {_judge_title(card)} |")
    lines.append(
        f"| **judgecal** | v{card.judgecal_version} "
        f"(card schema {card.card_schema_version}) |"
    )
    lines.append(f"| **Generated (UTC)** | {card.created_utc or _DASH} |")
    if card.datasets:
        names = ", ".join(str(d.get("name", d.get("hf_path", "?"))) for d in card.datasets)
        lines.append(f"| **Datasets** | {names} |")
    lines.append(
        f"| **Scale** | {max_items} items · {total_judgments} judgments · "
        f"{len(card.probes)} probes |"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if card.overall_flags:
        for flag in card.overall_flags:
            lines.append(f"- {_summary_line(card, flag, ci_label)}")
    else:
        lines.append(
            "- **No reliability flags raised** at the reporting thresholds. Before reading "
            "any null result as evidence of absence, check the MDE column: only effects "
            "at least that large were detectable at this sample size."
        )
    lines.append("")

    lines.append("## Probes")
    for entry in card.probes:
        lines.extend(_render_probe(entry, ci_label))

    if card.notes:
        lines.append("## Notes")
        lines.append("")
        for note in card.notes:
            lines.append(f"- {note}")
        lines.append("")

    n_family = sum(
        1
        for e in card.probes
        for m in e.metrics
        if m.p_value is not None and m.null_value is not None
    )
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Multiple comparisons:* q-values come from one Benjamini–Hochberg FDR correction "
        f"applied across all {n_family} null-tested metrics in this card (one family per "
        f"card — the pre-registered scope: conservative, and defined by the artifact you "
        f"are reading rather than post-hoc grouping)."
    )
    lines.append("")
    lines.append(
        "*Power:* every null-tested estimate carries an MDE — the smallest effect "
        "detectable at the realized sample size (two-sided, 80% power, from the "
        "realized clustered SE). Power adequacy compares the MDE to the metric's "
        "pre-registered effect-size-of-interest floor "
        "(`judgecal.report.thresholds.MIN_EFFECT_OF_INTEREST`), never to the observed "
        "estimate. Absence of evidence is quantified, not assumed."
    )
    lines.append("")
    lines.append(
        f"*Verdict key:* {_GLYPH_REJECTED} null rejected (q < {FLAG_Q_THRESHOLD:g}) · "
        f"{_GLYPH_CLEAR} no signal at adequate power (MDE ≤ effect-of-interest floor) · "
        f"{_GLYPH_UNDERPOWERED} no signal, underpowered for the floor · "
        f"{_GLYPH_DESCRIPTIVE} descriptive (no null) · "
        f"{_GLYPH_OBSERVATIONAL} observational association (no causal verdict). "
        f"Flag thresholds are reporting conventions "
        f"(see `judgecal.report.thresholds`), not truths."
    )
    lines.append("")
    return "\n".join(lines)


def _render_probe(entry: ProbeEntry, ci_label: str) -> list[str]:
    """Render one probe section: heading, counts, metric table, warnings."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"### {entry.probe}")
    lines.append("")
    lines.append(
        f"*items: {entry.n_items} · judgments: {entry.n_judgments} · "
        f"invalid: {_fmt_pct(entry.invalid_rate)}*"
    )
    lines.append("")
    if entry.metrics:
        lines.append(f"| Metric | Estimate [{ci_label}] | n | p | q | MDE | Verdict |")
        lines.append("|:--|:--|--:|--:|--:|--:|:-:|")
        for m in entry.metrics:
            lines.append(
                f"| `{m.name}` | {_ci_text(m)} | {m.n} | {_fmt_p(m.p_value)} | "
                f"{_fmt_p(m.q_value)} | {_fmt(m.mde)} | {_glyph(m)} |"
            )
        lines.append("")
        if any(_is_observational(m) for m in entry.metrics):
            lines.append(f"*{_OBSERVATIONAL_FOOTNOTE}*")
            lines.append("")
    if entry.flags:
        flag_text = ", ".join(f"`{flag}`" for flag in entry.flags)
        lines.append(f"**Flags:** {flag_text}")
        lines.append("")
    for warning in entry.warnings:
        lines.append(f"> Warning: {warning}")
    if entry.warnings:
        lines.append("")
    return lines


__all__ = ["render_markdown"]
