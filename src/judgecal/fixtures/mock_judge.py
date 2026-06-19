"""Deterministic mock judge with planted biases — THE validation engine.

The mock judge is a pure function of its config plus request *metadata*
(it never parses body text). It realizes a logistic model over planted
bias terms and emits a raw-text judge response containing reasoning
filler plus an ``[[A]]``/``[[B]]``/``[[C]]`` marker, so downstream code
exercises the REAL verdict parser.

Verdict mechanics (contract §3.2), for a request ``r``::

    logit_first = beta_quality * (q_first - q_second)
                + beta_position
                + beta_length * log(first_len / second_len)
                + beta_self * (1[first_author == self_name]
                               - 1[second_author == self_name])
                + template_offset(condition)   # N(0, template_sigma), keyed
                                               # by hash(seed, condition);
                                               # 0 for non-"tpl:*" conditions
                + eps                          # N(0, noise_sigma), keyed by
                                               # hash(seed, item_id,
                                               #      condition, repeat)

    verdict = "tie"              if |logit_first| < tie_band
            = "first"/"second"   else, via u < sigmoid(logit_first) with
                                 u = hash-uniform keyed by the request's
                                 custom_id *stem* (repeat suffix stripped)

All randomness is hash-keyed: sha256 over canonical JSON strings ->
64-bit ints -> floats in (0, 1); normal draws via ``scipy.stats.norm.ppf``
(inverse CDF). The stdlib ``random`` module and Python's salted ``hash()``
are never used, so outputs are stable across processes and platforms.

Deviation from a literal reading of §3.2 (deliberate, required for
consistency with §0 and §8): the verdict uniform and the invalid draw are
keyed by ``hash(seed, custom_id_stem)`` where the stem strips the
``-r<k>`` repeat suffix. Stability repeats have byte-identical bodies, and
§0 demands the judge be a pure function of config + request content while
§8 scenario 6 demands ``noise_sigma=0 -> unanimity == 1 exactly``. Keying
on the full custom_id would re-randomize the verdict per repeat and break
both. Repeat-to-repeat variation therefore enters ONLY through ``eps``
(whose key includes ``repeat``, exactly as §3.2 specifies).

Analytic truths: the ``expected_*`` helpers compute exact conditional
expectations that mirror the verdict mechanics. Decisiveness is
*deterministic* given config + request (tie iff ``|logit| < tie_band``;
the hash-keyed ``eps`` and template offsets are fixed numbers, not
randomness to integrate over). The only random element is the verdict
uniform ``u``, so among decisive requests the pick is exactly
``Bernoulli(sigmoid(logit))``, and

    E[first-pick rate over decisive judgments]
        = mean over requests with |logit| >= tie_band of sigmoid(logit).

Invalid corruption is keyed in an independent hash domain, hence
independent of verdict content: conditioning on "parseable" leaves these
expectations unchanged.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.special import expit
from scipy.stats import norm

from judgecal.core import JudgmentRequest, PresentedVerdict, canonical_json

#: Marker emitted at the end of each (parseable) raw response.
_MARKERS: dict[str, str] = {"first": "[[A]]", "second": "[[B]]", "tie": "[[C]]"}

_REPEAT_SUFFIX_RE = re.compile(r"^(?P<stem>.+)-r\d+$")

#: Reasoning-filler variants (deterministically chosen per request). None
#: may contain "[[" except via the appended marker.
_REASONING_TEMPLATES: tuple[str, ...] = (
    "I compared both responses on accuracy, completeness, and clarity. "
    "One response addresses the prompt with noticeably better support.",
    "Both answers engage with the question; weighing depth against "
    "concision, the stronger response stands out on balance.",
    "Considering factual grounding, structure, and relevance to the "
    "prompt, my evaluation favors the response indicated below.",
    "The two responses differ in coverage and precision. After checking "
    "each claim against the prompt requirements, I reach a verdict.",
)

#: Marker-free outputs used to simulate unparseable judge responses.
_INVALID_TEMPLATES: tuple[str, ...] = (
    "Both responses cover the prompt adequately and I find it genuinely "
    "difficult to express a preference in the requested format.",
    "The answers are close in quality; my assessment is that either could "
    "be acceptable depending on the reader's priorities.",
)


@dataclass(frozen=True)
class MockJudgeConfig:
    """Planted-bias configuration for the mock judge.

    Attributes:
        seed: Hash-domain seed; same config + request => same output.
        beta_quality: Weight on the latent quality gap (q_first - q_second).
        beta_position: Planted position bias (log-odds added for FIRST).
        beta_length: Planted verbosity bias per unit log-length-ratio.
        beta_self: Planted self-preference (log-odds when a side is
            authored by ``self_name``; antisymmetric in presentation).
        self_name: Author string this judge "is".
        template_sigma: SD of per-template log-odds offsets (conditions
            starting with ``"tpl:"`` each get one fixed hash-keyed draw;
            all other conditions get offset 0).
        noise_sigma: SD of per-(item, condition, repeat) Gaussian logit
            noise — models instability; 0 => perfectly stable.
        tie_band: ``|logit| < tie_band`` => verdict "tie".
        invalid_rate: Probability (hash-keyed per custom-id stem) of
            emitting marker-free, unparseable text.
    """

    seed: int = 0
    beta_quality: float = 3.0
    beta_position: float = 0.0
    beta_length: float = 0.0
    beta_self: float = 0.0
    self_name: str = "judge-self"
    template_sigma: float = 0.0
    noise_sigma: float = 0.0
    tie_band: float = 0.25
    invalid_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.tie_band < 0:
            raise ValueError(f"tie_band must be >= 0, got {self.tie_band}")
        if not 0.0 <= self.invalid_rate <= 1.0:
            raise ValueError(f"invalid_rate must be in [0, 1], got {self.invalid_rate}")
        if self.template_sigma < 0:
            raise ValueError(f"template_sigma must be >= 0, got {self.template_sigma}")
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma must be >= 0, got {self.noise_sigma}")


# ---------------------------------------------------------------------------
# Hash-keyed randomness (sha256 -> floats; never Python hash())
# ---------------------------------------------------------------------------


def _hash_unit(*parts: object) -> float:
    """Map key parts to a float in (0, 1) via sha256 of canonical JSON.

    Args:
        *parts: Key components; stringified then canonical-JSON encoded.

    Returns:
        ``(int(first 8 digest bytes) + 0.5) / 2**64`` — strictly inside
        (0, 1) so ``norm.ppf`` is always finite.
    """
    key = canonical_json([str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2.0**64


def _hash_normal(sigma: float, *parts: object) -> float:
    """A ``N(0, sigma)`` draw keyed by ``parts`` via inverse-CDF.

    Returns 0.0 when ``sigma == 0`` (no hashing performed).
    """
    if sigma == 0.0:
        return 0.0
    return float(sigma * norm.ppf(_hash_unit(*parts)))


def _custom_id_stem(custom_id: str) -> str:
    """Strip a trailing ``-r<digits>`` repeat suffix from a custom_id.

    Stability repeats share a body (and hence a stem); see the module
    docstring for why verdict/invalid draws key on the stem.
    """
    m = _REPEAT_SUFFIX_RE.match(custom_id)
    return m.group("stem") if m else custom_id


# ---------------------------------------------------------------------------
# Logit model
# ---------------------------------------------------------------------------


def _qualities(meta: dict[str, object]) -> tuple[float, float]:
    """Latent qualities (first, second) for a request's metadata.

    Uses planted ``first_latent_q``/``second_latent_q`` when both are
    present; otherwise derives from ``label_first`` per contract §3.2:
    labeled winner 0.8 / loser 0.2 / tie 0.5, 0.5. Unknown label (None)
    also maps to 0.5, 0.5 (no quality signal).
    """
    q_first = meta.get("first_latent_q")
    q_second = meta.get("second_latent_q")
    if q_first is not None and q_second is not None:
        return float(q_first), float(q_second)  # type: ignore[arg-type]
    label_first = meta.get("label_first")
    if label_first == "first":
        return 0.8, 0.2
    if label_first == "second":
        return 0.2, 0.8
    return 0.5, 0.5


def compute_logit(config: MockJudgeConfig, request: JudgmentRequest) -> float:
    """Exact ``logit_first`` for a request under this config.

    This is the single source of truth shared by the verdict generator
    and every analytic-expectation helper — they cannot disagree.

    Args:
        config: Mock judge configuration.
        request: A planned judgment request; only ``request.meta`` (and,
            for the hash keys, ids within it) is read — never body text.

    Returns:
        The log-odds that the judge picks the presented-first response
        (before tie-band censoring).
    """
    meta = request.meta
    q_first, q_second = _qualities(meta)
    logit = config.beta_quality * (q_first - q_second)
    logit += config.beta_position

    first_len = max(int(meta["first_len"]), 1)
    second_len = max(int(meta["second_len"]), 1)
    logit += config.beta_length * math.log(first_len / second_len)

    self_indicator = float(meta.get("first_author") == config.self_name) - float(
        meta.get("second_author") == config.self_name
    )
    logit += config.beta_self * self_indicator

    condition = str(meta["condition"])
    if condition.startswith("tpl:"):
        logit += _hash_normal(config.template_sigma, config.seed, "tpl", condition)
    logit += _hash_normal(
        config.noise_sigma, config.seed, "noise", meta["item_id"], condition, meta["repeat"]
    )
    return float(logit)


def request_logits(
    config: MockJudgeConfig, requests: Sequence[JudgmentRequest]
) -> list[float]:
    """Per-request ``logit_first`` values, for test introspection."""
    return [compute_logit(config, r) for r in requests]


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


class MockJudge:
    """Deterministic planted-bias judge emitting raw-text responses.

    Example:
        >>> judge = MockJudge(MockJudgeConfig(seed=0, beta_position=0.8))
        >>> raw = judge.judge(request)   # raw text ending in [[A]]/[[B]]/[[C]]
    """

    def __init__(self, config: MockJudgeConfig) -> None:
        """Store the (frozen) config."""
        self.config = config

    def logit_first(self, request: JudgmentRequest) -> float:
        """Exact logit for this request (see :func:`compute_logit`)."""
        return compute_logit(self.config, request)

    def decide(self, request: JudgmentRequest) -> PresentedVerdict:
        """Internal verdict for a request, ignoring invalid corruption.

        Mechanics: tie iff ``|logit| < tie_band``; otherwise "first" iff
        ``u < sigmoid(logit)`` with ``u`` hash-keyed by
        ``(seed, custom_id stem)``. Never returns "invalid" — use
        :meth:`is_invalid` / :meth:`judge` for the corruption channel.
        """
        logit = compute_logit(self.config, request)
        if abs(logit) < self.config.tie_band:
            return "tie"
        u = _hash_unit(self.config.seed, "verdict", _custom_id_stem(request.custom_id))
        return "first" if u < float(expit(logit)) else "second"

    def is_invalid(self, request: JudgmentRequest) -> bool:
        """Whether this request's output is emitted marker-free."""
        if self.config.invalid_rate <= 0.0:
            return False
        u = _hash_unit(self.config.seed, "invalid", _custom_id_stem(request.custom_id))
        return u < self.config.invalid_rate

    def judge(self, request: JudgmentRequest) -> str:
        """Produce the RAW TEXT judge response for a request.

        Returns reasoning filler plus the trailing ``[[A]]``/``[[B]]``/
        ``[[C]]`` marker; with hash-keyed probability
        ``config.invalid_rate``, marker-free filler instead (so the real
        parser maps it to "invalid"). Reads only ``request.meta`` and
        ``request.custom_id`` — never the body.

        Args:
            request: A planned judgment request with full required meta.

        Returns:
            The raw judge response text.
        """
        stem = _custom_id_stem(request.custom_id)
        if self.is_invalid(request):
            idx = int(_hash_unit(self.config.seed, "filler-inv", stem) * len(_INVALID_TEMPLATES))
            return _INVALID_TEMPLATES[idx]
        verdict = self.decide(request)
        idx = int(_hash_unit(self.config.seed, "filler", stem) * len(_REASONING_TEMPLATES))
        filler = _REASONING_TEMPLATES[idx]
        return f"{filler} My final verdict is: {_MARKERS[verdict]}"


# ---------------------------------------------------------------------------
# Analytic truths (the validation suite's reference values)
# ---------------------------------------------------------------------------


def _filtered(
    requests: Sequence[JudgmentRequest], conditions: tuple[str, ...] | None
) -> list[JudgmentRequest]:
    """Requests whose meta condition is in ``conditions`` (None => all)."""
    if conditions is None:
        return list(requests)
    allowed = set(conditions)
    return [r for r in requests if r.meta["condition"] in allowed]


def expected_first_pick_rate(
    config: MockJudgeConfig,
    requests: Sequence[JudgmentRequest],
    conditions: tuple[str, ...] | None = ("orig", "swap"),
) -> float:
    """Exact E[P(verdict == "first") | decisive] over a request set.

    Tie-band conditioning (mirrors the generator EXACTLY): a request is a
    tie iff ``|logit| < tie_band`` — this is *deterministic* given config
    and request, since template offsets and eps are fixed hash-keyed
    values. The verdict uniform ``u`` is drawn only AFTER the tie check,
    so among decisive requests (``|logit| >= tie_band``) the pick is
    exactly ``Bernoulli(sigmoid(logit))``. Hence the expected first-pick
    rate over decisive judgments is the plain (unconditional-looking)

        mean over requests with |logit| >= tie_band of sigmoid(logit).

    Invalid corruption lives in an independent hash domain, so further
    conditioning on "parseable" changes nothing.

    Args:
        config: Mock judge configuration (must match the executor's).
        requests: The planned requests the probe will analyze.
        conditions: Conditions to include; defaults to the position
            probe's pooled passes ("orig", "swap") per contract §5. Pass
            ``None`` to use every request.

    Returns:
        The exact expectation, or ``nan`` if no request is decisive.
    """
    reqs = _filtered(requests, conditions)
    probs = [
        float(expit(logit))
        for logit in (compute_logit(config, r) for r in reqs)
        if abs(logit) >= config.tie_band
    ]
    if not probs:
        return float("nan")
    return math.fsum(probs) / len(probs)


def expected_pad_pick_rate(
    config: MockJudgeConfig,
    requests: Sequence[JudgmentRequest],
    conditions: tuple[str, ...] | None = ("pad_first", "pad_second"),
) -> float:
    """Exact E[P(pick the PADDED side) | decisive] over pad requests.

    The padded side is identified by the condition: ``"pad_first"`` means
    the padded text is presented first (P(pick padded) = sigmoid(logit)),
    ``"pad_second"`` means it is presented second (P = 1 - sigmoid).
    Pooling both orders cancels position bias in the probe's estimate;
    this helper needs no such approximation — it averages the exact
    per-request probabilities over the decisive set (tie iff
    ``|logit| < tie_band``, deterministic; see
    :func:`expected_first_pick_rate` for the conditioning argument).

    Args:
        config: Mock judge configuration.
        requests: Planned requests (any superset; filtered by condition).
        conditions: Pad conditions to include (contract §5 names). Pass
            ``None`` only if every request is a pad request with one of
            the two canonical condition names.

    Returns:
        The exact expectation, or ``nan`` if no pad request is decisive.
    """
    reqs = _filtered(requests, conditions)
    probs: list[float] = []
    for r in reqs:
        logit = compute_logit(config, r)
        if abs(logit) < config.tie_band:
            continue
        p_first = float(expit(logit))
        probs.append(p_first if r.meta["condition"] == "pad_first" else 1.0 - p_first)
    if not probs:
        return float("nan")
    return math.fsum(probs) / len(probs)


def expected_self_error_pick_excess(
    config: MockJudgeConfig,
    requests: Sequence[JudgmentRequest],
    conditions: tuple[str, ...] | None = ("orig", "swap"),
) -> float:
    """Exact analytic value of the self-preference excess metric.

    Mirrors the probe metric ``self_error_pick_excess`` (contract §5):

    * Treatment set: decisive requests where EXACTLY one presented side is
      authored by ``config.self_name`` AND ground truth
      (``meta["label_first"]``) says the OTHER side wins. Per-request
      P(pick self side) = ``sigmoid(logit)`` if self is presented first,
      else ``1 - sigmoid(logit)``.
    * Control set: decisive requests with NO self-authored side where
      ground truth names a winner. Per-request P(pick the losing side) =
      ``1 - sigmoid(logit)`` if the winner is presented first, else
      ``sigmoid(logit)``.

    Excess = mean(treatment) - mean(control). Decisiveness is the same
    deterministic tie-band condition as everywhere else.

    Args:
        config: Mock judge configuration.
        requests: Planned requests (filtered to ``conditions``).
        conditions: Conditions to include (the probe reuses the position
            passes). ``None`` => all requests.

    Returns:
        The exact excess, or ``nan`` if either set is empty.
    """
    treatment: list[float] = []
    control: list[float] = []
    for r in _filtered(requests, conditions):
        meta = r.meta
        label_first = meta.get("label_first")
        if label_first not in ("first", "second"):
            continue
        logit = compute_logit(config, r)
        if abs(logit) < config.tie_band:
            continue
        p_first = float(expit(logit))
        self_first = meta.get("first_author") == config.self_name
        self_second = meta.get("second_author") == config.self_name
        if self_first ^ self_second:
            other_side = "second" if self_first else "first"
            if label_first == other_side:  # ground truth: the non-self side wins
                treatment.append(p_first if self_first else 1.0 - p_first)
        elif not self_first and not self_second:
            control.append(1.0 - p_first if label_first == "first" else p_first)
    if not treatment or not control:
        return float("nan")
    return math.fsum(treatment) / len(treatment) - math.fsum(control) / len(control)


__all__ = [
    "MockJudge",
    "MockJudgeConfig",
    "compute_logit",
    "expected_first_pick_rate",
    "expected_pad_pick_rate",
    "expected_self_error_pick_excess",
    "request_logits",
]
