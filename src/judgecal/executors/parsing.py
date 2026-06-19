"""Verdict parsing for raw judge outputs and reward-model scores.

The pinned output convention (contracts §0) is MT-Bench style: judge
templates instruct the model to end with ``[[A]]`` (presented-first wins),
``[[B]]`` (presented-second wins), or ``[[C]]`` (tie). The parser takes the
*last* occurrence, case-insensitively, tolerating whitespace inside the
brackets. Unparseable text maps to ``"invalid"`` — invalid judgments are
excluded from estimates but surfaced via ``invalid_rate``.

Score-endpoint results (scalar reward models via ``/v1/score``) bypass text
parsing entirely: two presented-side scores are compared directly with an
epsilon tie band via :func:`compare_scores`.
"""

from __future__ import annotations

import math
import re
import warnings

from judgecal.core import PresentedVerdict

#: Default verdict pattern: [[A]] / [[B]] / [[C]], case-insensitive,
#: whitespace tolerated inside the brackets. Named groups map the marker
#: to the presented verdict.
DEFAULT_VERDICT_PATTERN = (
    r"\[\[\s*(?P<first>A)\s*\]\]|\[\[\s*(?P<second>B)\s*\]\]|\[\[\s*(?P<tie>C)\s*\]\]"
)

#: Default epsilon band for score-pair comparison: |s1 - s2| <= eps → tie.
DEFAULT_SCORE_EPSILON = 1e-6

_VERDICT_GROUPS = ("first", "second", "tie")

_DEFAULT_RE = re.compile(DEFAULT_VERDICT_PATTERN, re.IGNORECASE)


def parse_verdict(raw_text: str | None, pattern: str | None = None) -> PresentedVerdict:
    """Parse a raw judge response into a presented-coordinates verdict.

    The **last** marker occurrence wins (judges often discuss "[[A]]" mid-
    reasoning before concluding); matching is case-insensitive and tolerates
    whitespace inside the brackets (``[[ b ]]`` parses as "second").

    Args:
        raw_text: The raw judge output text. ``None`` or empty → "invalid".
        pattern: Optional custom regex (compiled with ``re.IGNORECASE``). It
            must define at least one of the named groups ``first``,
            ``second``, ``tie``; the last match's matched group determines
            the verdict (checked in that precedence order). When ``None``,
            :data:`DEFAULT_VERDICT_PATTERN` is used.

    Returns:
        ``"first"``, ``"second"``, ``"tie"``, or ``"invalid"`` when no
        marker is found.

    Raises:
        ValueError: If a custom ``pattern`` defines none of the required
            named groups, or does not compile.
    """
    if not raw_text:
        return "invalid"
    if pattern is None:
        regex = _DEFAULT_RE
    else:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:  # pragma: no cover - message matters, not path
            raise ValueError(f"invalid verdict pattern {pattern!r}: {exc}") from exc
        if not set(regex.groupindex) & set(_VERDICT_GROUPS):
            raise ValueError(
                "custom verdict pattern must define at least one named group "
                f"out of {_VERDICT_GROUPS}; got groups {sorted(regex.groupindex)}"
            )
    matches = list(regex.finditer(raw_text))
    if not matches:
        return "invalid"
    last = matches[-1]
    for name in _VERDICT_GROUPS:
        if name in regex.groupindex and last.group(name) is not None:
            return name  # type: ignore[return-value]
    return "invalid"


def compare_scores(
    first_score: float,
    second_score: float,
    epsilon: float = DEFAULT_SCORE_EPSILON,
) -> PresentedVerdict:
    """Compare two presented-side reward-model scores with a tie band.

    Args:
        first_score: Score of the presented-first response.
        second_score: Score of the presented-second response.
        epsilon: Tie band: ``|first - second| <= epsilon`` → "tie". Must be
            non-negative.

    Returns:
        ``"first"`` / ``"second"`` / ``"tie"``; ``"invalid"`` (with a
        warning) if either score is non-finite (NaN or ±inf — a saturated
        or broken RM head must not yield a decisive verdict).

    Raises:
        ValueError: If ``epsilon`` is negative.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if not (math.isfinite(first_score) and math.isfinite(second_score)):
        warnings.warn(
            f"non-finite score pair ({first_score}, {second_score}); verdict is 'invalid'",
            UserWarning,
            stacklevel=2,
        )
        return "invalid"
    if abs(first_score - second_score) <= epsilon:
        return "tie"
    return "first" if first_score > second_score else "second"


__all__ = [
    "DEFAULT_SCORE_EPSILON",
    "DEFAULT_VERDICT_PATTERN",
    "compare_scores",
    "parse_verdict",
]
