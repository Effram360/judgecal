"""Executor protocol and shared helpers.

An *executor* turns :class:`~judgecal.core.JudgmentRequest` objects into
:class:`~judgecal.core.Judgment` objects. Local executors (mock judge,
fixture replay, Claude Code CLI) implement the protocol directly; batch
execution on a cluster goes through manifests instead (see
:mod:`judgecal.manifests` and :mod:`judgecal.executors.slurm`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from judgecal.core import Judgment, JudgmentRequest

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExecutorWarning(UserWarning):
    """Warning category for non-fatal executor problems (missing fixtures,
    per-request CLI failures). Filter with ``pytest.warns(ExecutorWarning)``."""


@runtime_checkable
class Executor(Protocol):
    """Anything that executes judgment requests synchronously."""

    def execute(self, requests: Sequence[JudgmentRequest]) -> list[Judgment]:
        """Execute requests in order; return exactly one judgment per request."""
        ...


def judgment_from_raw(
    request: JudgmentRequest,
    raw_text: str | None,
    pattern: str | None = None,
) -> Judgment:
    """Build a judgment by running raw judge text through the real parser.

    Args:
        request: The originating request; its ``meta`` is echoed (copied)
            into the judgment.
        raw_text: Raw judge output; ``None`` yields an "invalid" verdict.
        pattern: Optional custom verdict regex (see
            :func:`judgecal.executors.parsing.parse_verdict`).

    Returns:
        A :class:`~judgecal.core.Judgment` in presented coordinates.
    """
    from judgecal.executors.parsing import parse_verdict

    return Judgment(
        custom_id=request.custom_id,
        verdict=parse_verdict(raw_text, pattern),
        raw_text=raw_text,
        meta=dict(request.meta),
    )


def invalid_judgment(request: JudgmentRequest, raw_text: str | None = None) -> Judgment:
    """Build an "invalid" judgment for a request that could not be executed."""
    return Judgment(
        custom_id=request.custom_id,
        verdict="invalid",
        raw_text=raw_text,
        meta=dict(request.meta),
    )


__all__ = [
    "Executor",
    "ExecutorWarning",
    "invalid_judgment",
    "judgment_from_raw",
]
