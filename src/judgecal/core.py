"""Core shared types for judgecal.

Every module builds against these contracts. They are deliberately small,
frozen, and serialization-friendly: items flow in, judgment requests flow
out to executors (mock, fixture, claude-code, or SLURM batch manifests),
judgments flow back, and probe analyses turn judgments into estimates.

Design invariants (do not violate):

1. **Judgments are self-contained.** Every feature an analysis needs
   (presented lengths, authors, latent qualities, ground-truth mapping)
   is embedded in ``JudgmentRequest.meta`` at plan time and echoed into
   ``Judgment.meta`` by executors. Analyses never re-join against items.
2. **Execution payloads are content-hashed.** ``custom_id`` is a pure
   function of the rendered request body plus the repeat index, so
   identical work deduplicates across probes and manifests are resumable.
3. **Zero LLM in the dev loop.** Everything here runs against the
   deterministic mock judge or recorded fixtures on an 8GB laptop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------
# Verdicts and conditions
# --------------------------------------------------------------------------

#: A judge verdict over a presented pair, in *presented* coordinates.
#: "first"/"second" refer to presentation order, NOT to item.response_a/b.
#: "invalid" means the raw output could not be parsed into a verdict.
PresentedVerdict = Literal["first", "second", "tie", "invalid"]

#: A verdict mapped back to item coordinates (A = item.response_a).
MappedVerdict = Literal["A", "B", "tie", "invalid"]

#: Ground-truth label for a pairwise item, in item coordinates.
Label = Literal["A", "B", "tie"]


def map_verdict(presented: PresentedVerdict, first_is_a: bool) -> MappedVerdict:
    """Map a presented-coordinates verdict back to item coordinates."""
    if presented in ("tie", "invalid"):
        return presented  # type: ignore[return-value]
    if presented == "first":
        return "A" if first_is_a else "B"
    return "B" if first_is_a else "A"


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseItem:
    """A single pairwise-comparison item in canonical (A, B) coordinates.

    ``label`` is the ground-truth preference if known. ``author_a/b`` hold
    model-provenance strings (e.g. "qwen3.5-9b") used by the
    self-preference probe; ``None`` when unknown.

    ``meta`` may carry dataset-specific fields. The synthetic generator
    stores planted latent qualities under ``latent_quality_a`` /
    ``latent_quality_b`` (floats in [0, 1]); the mock judge reads them.
    """

    item_id: str
    prompt: str
    response_a: str
    response_b: str
    label: Label | None = None
    author_a: str | None = None
    author_b: str | None = None
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Requests and judgments
# --------------------------------------------------------------------------

#: Required keys in JudgmentRequest.meta / Judgment.meta. Probes MUST
#: populate all of these at plan time (None where not applicable):
#:
#:   probe          str   probe name, e.g. "position"
#:   condition      str   probe condition, e.g. "orig", "swap", "tpl:v2"
#:   item_id        str
#:   repeat         int   repeat index (0 for non-stability probes)
#:   first_is_a     bool  whether presented-first text is item.response_a
#:   first_len      int   len() of presented-first response text
#:   second_len     int
#:   first_author   str | None
#:   second_author  str | None
#:   first_latent_q  float | None   planted latent quality (synthetic only)
#:   second_latent_q float | None
#:   label_first    str | None  ground truth in presented coords:
#:                              "first" | "second" | "tie" | None
REQUIRED_META_KEYS = (
    "probe",
    "condition",
    "item_id",
    "repeat",
    "first_is_a",
    "first_len",
    "second_len",
    "first_author",
    "second_author",
    "first_latent_q",
    "second_latent_q",
    "label_first",
)

#: Reserved meta keys carrying the raw presented texts. Probes populate
#: them at plan time (``prompt_text``, ``first_text``, ``second_text``)
#: so manifest emission can build ``/v1/score`` request bodies for scalar
#: reward models. The sidecar writer strips them from stored usages —
#: analyses never need them; they exist only between plan and emit.
SCORE_TEXT_META_KEYS = (
    "prompt_text",
    "first_text",
    "second_text",
)


def canonical_json(obj: Any) -> str:
    """Canonical JSON encoding used for all content hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def body_hash(body: dict[str, Any]) -> str:
    """Content hash of an execution body (24 hex chars of sha256)."""
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:24]


def make_custom_id(body: dict[str, Any], repeat: int = 0) -> str:
    """Deterministic custom_id: 'jc-<hash24>-r<repeat>' (<= 64 chars).

    Identical bodies at the same repeat index collapse to one execution;
    the manifest sidecar fans results back out to every (probe, condition,
    item) usage.
    """
    return f"jc-{body_hash(body)}-r{repeat}"


@dataclass(frozen=True)
class JudgmentRequest:
    """One unit of judge work, fully rendered and self-describing.

    ``body`` is an OpenAI-format request body — for generative judges a
    chat-completions body ``{"messages": [...]}``; for scalar reward
    models a score body. Model name and sampling params are attached at
    manifest-emission time, not here (the same plan can target many
    judge arms).

    ``meta`` MUST contain all ``REQUIRED_META_KEYS``.
    """

    custom_id: str
    body: dict[str, Any]
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        missing = [k for k in REQUIRED_META_KEYS if k not in self.meta]
        if missing:
            raise ValueError(f"JudgmentRequest.meta missing required keys: {missing}")


@dataclass(frozen=True)
class Judgment:
    """A parsed judge response for one request usage.

    ``verdict`` is in presented coordinates; use ``mapped_verdict`` for
    item coordinates. ``raw_text`` preserves the unparsed judge output
    (or a score repr for reward models). ``meta`` echoes the request meta.
    """

    custom_id: str
    verdict: PresentedVerdict
    raw_text: str | None
    meta: dict[str, Any]

    @property
    def probe(self) -> str:
        return self.meta["probe"]

    @property
    def condition(self) -> str:
        return self.meta["condition"]

    @property
    def item_id(self) -> str:
        return self.meta["item_id"]

    @property
    def repeat(self) -> int:
        return self.meta["repeat"]

    @property
    def mapped_verdict(self) -> MappedVerdict:
        return map_verdict(self.verdict, self.meta["first_is_a"])

    @property
    def is_decisive(self) -> bool:
        return self.verdict in ("first", "second")


# --------------------------------------------------------------------------
# Estimates and probe results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """One statistical estimate with uncertainty, ready for the card.

    ``null_value`` is the no-bias reference (e.g. 0.5 for pick rates,
    0.0 for coefficients); ``p_value`` tests against it. ``q_value`` is
    filled in by BH-FDR at report time, not by the probe. ``mde`` is the
    minimum detectable effect (two-sided, 80% power) at the realized n,
    on the same scale as ``estimate``.
    """

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
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Everything a probe learned, ready for the reliability card."""

    probe: str
    estimates: list[Estimate]
    n_items: int
    n_judgments: int
    invalid_rate: float
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Estimate",
    "Judgment",
    "JudgmentRequest",
    "Label",
    "MappedVerdict",
    "PairwiseItem",
    "PresentedVerdict",
    "ProbeResult",
    "REQUIRED_META_KEYS",
    "SCORE_TEXT_META_KEYS",
    "body_hash",
    "canonical_json",
    "make_custom_id",
    "map_verdict",
]
