"""Reliability card: pydantic models, construction, flagging, persistence.

The card is the user-facing artifact of a judgecal audit. ``build_card``
turns a list of :class:`judgecal.core.ProbeResult` into a
:class:`ReliabilityCard`: it fills BH-FDR q-values across *all*
null-tested metrics in the card (one family per card — the pre-registered
scope decision), then applies the flag conventions from
:mod:`judgecal.report.thresholds`.

``created_utc`` is always caller-supplied. Library code never reads the
wall clock; the CLI layer passes the timestamp so cards built from the
same inputs are byte-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from judgecal import __version__ as _JUDGECAL_VERSION
from judgecal.core import Estimate, ProbeResult
from judgecal.report.thresholds import (
    FLAG_Q_THRESHOLD,
    HIGH_INVALID_RATE_THRESHOLD,
    MIN_EFFECT_OF_INTEREST,
    POSITION_BIAS_MIN_EFFECT,
    SELF_PREFERENCE_MIN_EFFECT,
    STABILITY_UNANIMITY_FLOOR,
    TEMPLATE_KAPPA_FLOOR,
    TEMPLATE_MAX_FLIP_CEILING,
    TEMPLATE_STABILITY_KAPPA_MARGIN,
    VERBOSITY_BIAS_MIN_EFFECT,
)
from judgecal.stats import bh_fdr

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class MetricEntry(BaseModel):
    """One statistical estimate on the card.

    Mirrors :class:`judgecal.core.Estimate`. ``q_value`` is filled by
    ``build_card`` via BH-FDR over the whole card; probes never set it.
    ``mde`` is the minimum detectable effect (two-sided, 80% power) at
    the realized n, on the same scale as ``estimate``.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    estimate: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    null_value: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    mde: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ProbeEntry(BaseModel):
    """All card content for one probe: counts, metrics, warnings, flags."""

    model_config = ConfigDict(extra="ignore")

    probe: str
    n_items: int
    n_judgments: int
    invalid_rate: float
    warnings: list[str] = Field(default_factory=list)
    metrics: list[MetricEntry] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ReliabilityCard(BaseModel):
    """The full reliability card (schema version 0.1).

    ``created_utc`` is caller-supplied (ISO-8601 string) and defaults to
    ``None`` — this library never calls ``datetime.now``. Unknown fields
    in serialized cards are ignored on load (forward compatibility).
    """

    model_config = ConfigDict(extra="ignore")

    card_schema_version: Literal["0.1"] = "0.1"
    judgecal_version: str
    judge: dict[str, Any]
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created_utc: str | None = None
    probes: list[ProbeEntry] = Field(default_factory=list)
    overall_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Coerce a value into plain JSON-serializable Python types.

    Numpy scalars (anything with a callable ``.item()``) are unboxed;
    containers are converted recursively; anything else unknown becomes
    its ``str()``. Keeps cards serializable no matter what probes stuff
    into ``detail``/``meta`` dicts.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _metric_from_estimate(est: Estimate) -> MetricEntry:
    """Convert a core Estimate into a card MetricEntry (q left as-is)."""
    return MetricEntry(
        name=est.name,
        estimate=float(est.estimate),
        ci_low=float(est.ci_low),
        ci_high=float(est.ci_high),
        n=int(est.n),
        method=est.method,
        null_value=None if est.null_value is None else float(est.null_value),
        p_value=None if est.p_value is None else float(est.p_value),
        q_value=None if est.q_value is None else float(est.q_value),
        mde=None if est.mde is None else float(est.mde),
        detail=_jsonable(dict(est.detail)),
    )


def _is_null_tested(metric: MetricEntry) -> bool:
    """A metric belongs to the FDR family iff it carries a p-value and null."""
    return metric.p_value is not None and metric.null_value is not None


def _is_significant(metric: MetricEntry) -> bool:
    return metric.q_value is not None and metric.q_value < FLAG_Q_THRESHOLD


def _is_underpowered(metric: MetricEntry) -> bool:
    """Non-significant metric whose MDE exceeds its effect size of interest.

    Design-based semantics: power adequacy is judged
    against the pre-registered ``MIN_EFFECT_OF_INTEREST`` floor for the
    metric, NEVER against the data-dependent observed effect (the
    post-hoc-power anti-pattern). Metrics without a registered floor, or
    without an MDE, are never flagged underpowered.
    """
    if not _is_null_tested(metric) or _is_significant(metric):
        return False
    if metric.mde is None:
        return False
    floor = MIN_EFFECT_OF_INTEREST.get(metric.name)
    if floor is None:
        return False
    return metric.mde > floor


#: (probe, metric, flag, minimum |estimate - null|) for the bias flags.
#: A bias flag fires when the metric is significant (q < FLAG_Q_THRESHOLD
#: after card-wide BH-FDR) AND the effect clears the practical floor.
_BIAS_FLAG_RULES: tuple[tuple[str, str, str, float], ...] = (
    ("position", "first_pick_rate", "position_bias_detected", POSITION_BIAS_MIN_EFFECT),
    ("verbosity", "pad_pick_rate", "verbosity_bias_detected", VERBOSITY_BIAS_MIN_EFFECT),
    (
        "self_preference",
        "self_error_pick_excess",
        "self_preference_detected",
        SELF_PREFERENCE_MIN_EFFECT,
    ),
)


def _find_metric(entry: ProbeEntry, name: str) -> MetricEntry | None:
    for metric in entry.metrics:
        if metric.name == name:
            return metric
    return None


def _find_metric_in(
    entries: Sequence[ProbeEntry], probe: str, name: str
) -> MetricEntry | None:
    for entry in entries:
        if entry.probe == probe:
            return _find_metric(entry, name)
    return None


def _probe_flags(entry: ProbeEntry, entries: Sequence[ProbeEntry]) -> list[str]:
    """Apply all flag conventions to one probe entry (q-values already filled).

    ``entries`` is the full card (all probes): the template flag uses the
    stability probe's kappa, when present, as the verdict-noise baseline.
    """
    flags: list[str] = []

    for probe_name, metric_name, flag, min_effect in _BIAS_FLAG_RULES:
        if entry.probe != probe_name:
            continue
        metric = _find_metric(entry, metric_name)
        if metric is None or metric.null_value is None:
            continue
        effect = abs(metric.estimate - metric.null_value)
        if not (_is_significant(metric) and effect > min_effect):
            continue
        if flag == "self_preference_detected" and metric.detail.get("composition_imbalance"):
            # The probe's composition diagnostic found the self/control
            # sets materially different (quality-gap/decisive-rate/label
            # imbalance): the excess may be confounded, so the detection
            # flag is suppressed. The metric, q-value, and the warning
            # remain on the card.
            continue
        flags.append(flag)

    if entry.probe == "template":
        # Template disagreement conflates template effects with verdict
        # stochasticity. Base rule: kappa CI *upper bound* below the
        # floor (not the noisy point estimate) OR max flip above the
        # ceiling. When the card carries stability data (same judge,
        # identical bodies repeated), additionally require template
        # kappa to sit TEMPLATE_STABILITY_KAPPA_MARGIN below stability
        # kappa — template disagreement must exceed repeat noise. With
        # no stability baseline the base rule stands alone and the
        # rendered summary carries an explicit caveat.
        kappa = _find_metric(entry, "template_fleiss_kappa")
        max_flip = _find_metric(entry, "template_max_flip")
        kappa_low = kappa is not None and kappa.ci_high < TEMPLATE_KAPPA_FLOOR
        flip_high = max_flip is not None and max_flip.estimate > TEMPLATE_MAX_FLIP_CEILING
        if kappa_low or flip_high:
            stability_kappa = _find_metric_in(entries, "stability", "stability_fleiss_kappa")
            exceeds_repeat_noise = (
                stability_kappa is None
                or kappa is None
                or kappa.estimate < stability_kappa.estimate - TEMPLATE_STABILITY_KAPPA_MARGIN
            )
            if exceeds_repeat_noise:
                flags.append("template_sensitivity_high")

    if entry.probe == "stability":
        unanimity = _find_metric(entry, "unanimity_rate")
        if unanimity is not None and unanimity.estimate < STABILITY_UNANIMITY_FLOOR:
            flags.append("instability_high")

    for metric in entry.metrics:
        if _is_underpowered(metric):
            flags.append(f"underpowered:{metric.name}")

    if entry.invalid_rate > HIGH_INVALID_RATE_THRESHOLD:
        flags.append("high_invalid_rate")

    return flags


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_card(
    results: Sequence[ProbeResult],
    judge: dict[str, Any],
    *,
    datasets: Sequence[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    created_utc: str | None = None,
    notes: Sequence[str] | None = None,
    judgecal_version: str | None = None,
) -> ReliabilityCard:
    """Assemble a ReliabilityCard from probe results: fill q-values, flag.

    **FDR scope (pre-registered decision):** Benjamini–Hochberg is applied
    once across *all* null-tested metrics in the card — every metric, from
    every probe, that carries both a ``p_value`` and a ``null_value`` forms
    a single family. One family per card is the conservative choice (more
    hypotheses in the family means stricter q-values than per-probe
    families) and the simplest to defend: the family is defined by the
    artifact the user reads, not by post-hoc grouping. Metrics without a
    p-value (descriptive metrics such as Fleiss kappa) keep ``q_value=None``
    and never enter the family.

    Flags are then applied per probe using the conventions in
    :mod:`judgecal.report.thresholds` (conventions, not truths);
    ``overall_flags`` is the deduplicated union in probe order.

    Args:
        results: Probe analysis results, in the order they should appear.
        judge: Description of the judge under audit (e.g. ``{"model":
            "qwen3.5-9b-awq", "quant": "awq"}``). Free-form; rendered in
            the card header.
        datasets: Optional dataset descriptors (name, split, license, ...).
        config: Optional audit configuration echo (alpha, n_boot, seed, ...).
            ``config["alpha"]`` is used by the renderer to label CIs.
        created_utc: Caller-supplied ISO-8601 UTC timestamp. Defaults to
            ``None``; this function never reads the clock (determinism).
        notes: Optional free-text notes appended to the card.
        judgecal_version: Override for the recorded library version
            (defaults to the installed ``judgecal.__version__``).

    Returns:
        A fully populated, JSON-serializable ReliabilityCard.
    """
    entries: list[ProbeEntry] = []
    for result in results:
        entries.append(
            ProbeEntry(
                probe=result.probe,
                n_items=int(result.n_items),
                n_judgments=int(result.n_judgments),
                invalid_rate=float(result.invalid_rate),
                warnings=list(result.warnings),
                metrics=[_metric_from_estimate(est) for est in result.estimates],
                flags=[],
            )
        )

    # One BH-FDR family across the whole card (see docstring).
    family: list[MetricEntry] = [
        metric for entry in entries for metric in entry.metrics if _is_null_tested(metric)
    ]
    if family:
        pvals = [metric.p_value for metric in family]
        qvals = bh_fdr(pvals)  # type: ignore[arg-type]  # _is_null_tested excludes None
        for metric, q in zip(family, qvals, strict=True):
            metric.q_value = float(q)

    overall_flags: list[str] = []
    for entry in entries:
        entry.flags = _probe_flags(entry, entries)
        for flag in entry.flags:
            if flag not in overall_flags:
                overall_flags.append(flag)

    return ReliabilityCard(
        card_schema_version="0.1",
        judgecal_version=judgecal_version if judgecal_version is not None else _JUDGECAL_VERSION,
        judge=_jsonable(dict(judge)),
        datasets=[_jsonable(dict(d)) for d in (datasets or [])],
        config=_jsonable(dict(config or {})),
        created_utc=created_utc,
        probes=entries,
        overall_flags=overall_flags,
        notes=list(notes or []),
    )


def save_card(card: ReliabilityCard, path: str | Path) -> None:
    """Write a card to ``path`` as pretty-printed JSON (UTF-8, trailing \\n)."""
    Path(path).write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_card(path: str | Path) -> ReliabilityCard:
    """Load a card from JSON. Tolerant: unknown fields are ignored.

    Raises:
        pydantic.ValidationError: If required fields are missing or the
            schema version is unsupported.
    """
    return ReliabilityCard.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "MetricEntry",
    "ProbeEntry",
    "ReliabilityCard",
    "build_card",
    "load_card",
    "save_card",
]
