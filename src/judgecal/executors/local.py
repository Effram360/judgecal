"""Local executors: deterministic mock judge and recorded-fixture replay.

These are the zero-LLM dev-loop workhorses (contracts §0): the mock judge
is a pure function of its config + request content, and fixture packs
replay raw judge texts recorded from real cluster runs. Both run their raw
text through the **real** verdict parser so the parsing path is always
exercised.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from judgecal.core import Judgment, JudgmentRequest
from judgecal.executors.base import ExecutorWarning, invalid_judgment, judgment_from_raw

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MISSING = object()

#: Candidate fixture-module entry points for the mock judge, tried in
#: order. The MOCK agent owns ``judgecal.fixtures``; this resolution is the
#: integration seam between the two modules.
_RESPONDER_CLASS_CANDIDATES = ("MockJudge",)
_RESPONDER_FN_CANDIDATES = ("mock_judge_response", "respond", "judge_response")


def _resolve_fixtures_responder(config: Any) -> Callable[[JudgmentRequest], str]:
    """Resolve a ``request -> raw_text`` callable from ``judgecal.fixtures``.

    Tries, in order: a ``MockJudge`` class instantiated with ``config``
    (using its ``respond`` method or ``__call__``), then module-level
    functions called as ``fn(config, request)``.

    Raises:
        ImportError: If ``judgecal.fixtures`` is not importable.
        AttributeError: If no known entry point is found (lists candidates).
    """
    import judgecal.fixtures as fixtures

    for cls_name in _RESPONDER_CLASS_CANDIDATES:
        cls = getattr(fixtures, cls_name, None)
        if cls is not None:
            judge = cls(config)
            for method_name in ("respond", "judge"):
                respond = getattr(judge, method_name, None)
                if callable(respond):
                    return respond
            if callable(judge):
                return judge
    for fn_name in _RESPONDER_FN_CANDIDATES:
        fn = getattr(fixtures, fn_name, None)
        if callable(fn):
            return lambda request: fn(config, request)
    raise AttributeError(
        "could not resolve a mock-judge responder from judgecal.fixtures; "
        f"tried classes {_RESPONDER_CLASS_CANDIDATES} and functions "
        f"{_RESPONDER_FN_CANDIDATES}. Pass responder= explicitly."
    )


class MockJudgeExecutor:
    """Executes requests against the deterministic mock judge.

    The mock judge (``judgecal.fixtures``) emits *raw text* containing an
    ``[[A]]``/``[[B]]``/``[[C]]`` marker (or deliberately unparseable text
    at its configured ``invalid_rate``); this executor runs that text
    through the real :func:`~judgecal.executors.parsing.parse_verdict`, so
    the full parse path is validated on every dev-loop run.

    Args:
        config: A ``judgecal.fixtures.MockJudgeConfig``.
        responder: Optional explicit ``request -> raw_text`` callable; when
            omitted it is resolved from ``judgecal.fixtures`` (see
            :func:`_resolve_fixtures_responder`).
        pattern: Optional custom verdict regex.
    """

    def __init__(
        self,
        config: Any,
        *,
        responder: Callable[[JudgmentRequest], str] | None = None,
        pattern: str | None = None,
    ) -> None:
        self.config = config
        if responder is None:
            responder = _resolve_fixtures_responder(config)
        self._responder = responder
        self._pattern = pattern

    def execute(self, requests: Sequence[JudgmentRequest]) -> list[Judgment]:
        """Judge every request deterministically; one judgment per request."""
        return [judgment_from_raw(req, self._responder(req), self._pattern) for req in requests]


class FixtureExecutor:
    """Replays recorded raw judge texts keyed by ``custom_id``.

    Args:
        pack: A ``judgecal.fixtures.packs.ResponsePack`` — or any object
            exposing a ``responses`` mapping attribute, or any plain
            mapping — from ``custom_id`` to raw response text.
        strict: When True (default), a request whose ``custom_id`` is not
            in the pack raises ``KeyError``. When False, it yields an
            "invalid" judgment and an :class:`ExecutorWarning`.
        pattern: Optional custom verdict regex.
    """

    def __init__(self, pack: Any, strict: bool = True, *, pattern: str | None = None) -> None:
        self.pack = pack
        self.strict = strict
        self._pattern = pattern

    def _lookup(self, custom_id: str) -> Any:
        source = getattr(self.pack, "responses", self.pack)
        getter = getattr(source, "get", None)
        if callable(getter):
            return getter(custom_id, _MISSING)
        try:
            return source[custom_id]
        except (KeyError, TypeError, IndexError):
            return _MISSING

    def execute(self, requests: Sequence[JudgmentRequest]) -> list[Judgment]:
        """Replay each request from the pack, parsing through the real parser."""
        judgments: list[Judgment] = []
        for req in requests:
            raw = self._lookup(req.custom_id)
            if raw is _MISSING:
                if self.strict:
                    raise KeyError(
                        f"custom_id {req.custom_id!r} not found in fixture pack "
                        "(strict=True); pass strict=False to emit invalid judgments"
                    )
                warnings.warn(
                    f"custom_id {req.custom_id!r} missing from fixture pack; "
                    "emitting an invalid judgment",
                    ExecutorWarning,
                    stacklevel=2,
                )
                judgments.append(invalid_judgment(req))
            else:
                judgments.append(judgment_from_raw(req, raw, self._pattern))
        return judgments


__all__ = ["FixtureExecutor", "MockJudgeExecutor"]
