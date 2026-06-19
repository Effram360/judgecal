"""Claude Code CLI executor — the zero-API real-LLM smoke path.

Runs the Claude Code CLI headlessly, one request at a time, so Max-plan
subscribers can sanity-check a probe suite against a real frontier model
without any API key. This is a *smoke* path, not a study path: it is
sequential and consumes the user's interactive subscription quota.

Rate-limit etiquette (document for users):

- Keep batches small (tens of requests, not hundreds) — every request is a
  full CLI session against your personal subscription limits.
- Never parallelize; this executor is deliberately sequential.
- No automatic retries: a failed request becomes an "invalid" judgment
  with a warning rather than burning more quota.

Flag verification (contracts §4.3 requires pinning what we verified):

.. code-block:: text

    # Verified locally 2026-06-10 against Claude Code CLI 2.1.172
    # (`claude --help` and `claude -p --help`):
    #   -p, --print              non-interactive; print response and exit
    #   --output-format json     single JSON result object on stdout
    #   --model <name>           alias ("sonnet") or full model name
    #   --tools ""               disable ALL built-in tools (pure text judging)
    #   --safe-mode              disable CLAUDE.md/skills/plugins/hooks/MCP;
    #                            auth, model selection still work normally
    #   --no-session-persistence don't write session files (print mode only)
    # NOT used: --bare — it restricts auth to ANTHROPIC_API_KEY only, which
    # would defeat the zero-API / subscription purpose of this executor.
    # JSON result shape observed/documented: {"type": "result",
    # "subtype": "success", "is_error": false, "result": "<assistant text>",
    # ...} — we read "result" with liberal fallbacks ("content",
    # "completion", "text").
"""

from __future__ import annotations

import json
import subprocess
import warnings
from typing import TYPE_CHECKING, Any

from judgecal.core import Judgment, JudgmentRequest
from judgecal.executors.base import ExecutorWarning, invalid_judgment, judgment_from_raw

if TYPE_CHECKING:
    from collections.abc import Sequence

_STDERR_TAIL_CHARS = 400


def _render_prompt(body: dict[str, Any]) -> str:
    """Render an OpenAI chat-completions body into one CLI prompt string.

    System messages are prefixed with ``[system]``, assistant turns with
    ``[assistant]``; user content is passed through. List-of-parts content
    is flattened to its text parts.

    Raises:
        ValueError: If the body has no ``messages`` list (e.g. a score
            body — scalar RMs cannot run through a chat CLI).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "ClaudeCodeExecutor requires chat-format bodies with a non-empty "
            f"'messages' list; got body keys {sorted(body)}"
        )
    chunks: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            chunks.append(content)
        else:
            chunks.append(f"[{role}]\n{content}")
    return "\n\n".join(chunks)


def _extract_result_text(payload: Any) -> str | None:
    """Pull the assistant text out of a `--output-format json` payload."""
    if not isinstance(payload, dict):
        return None
    if payload.get("is_error"):
        return None
    for key in ("result", "content", "completion", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [
                p["text"] for p in value if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            if parts:
                return "".join(parts)
    return None


class ClaudeCodeExecutor:
    """Sequential headless Claude Code CLI executor (zero-API smoke path).

    Args:
        model: Model alias or full name passed as ``--model``; ``None``
            uses the CLI default.
        claude_bin: Path or name of the Claude Code binary.
        timeout_s: Per-request wall timeout in seconds.
        pattern: Optional custom verdict regex.
        extra_args: Extra CLI arguments appended verbatim (escape hatch
            for flag drift across CLI versions).

    Raises:
        FileNotFoundError: At execute time, if ``claude_bin`` is missing —
            with install guidance.
    """

    def __init__(
        self,
        model: str | None = None,
        claude_bin: str = "claude",
        timeout_s: int = 120,
        *,
        pattern: str | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.claude_bin = claude_bin
        self.timeout_s = timeout_s
        self._pattern = pattern
        self.extra_args = tuple(extra_args)

    def _command(self) -> list[str]:
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--safe-mode",
            "--no-session-persistence",
        ]
        if self.model is not None:
            cmd.extend(["--model", self.model])
        cmd.extend(self.extra_args)
        return cmd

    def _run_one(self, request: JudgmentRequest) -> Judgment:
        prompt = _render_prompt(request.body)
        cmd = self._command()
        try:
            proc = subprocess.run(  # noqa: S603 - user-configured local binary
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Claude Code CLI not found at {self.claude_bin!r}. Install it "
                "(https://claude.com/claude-code) or pass claude_bin=<path>."
            ) from exc
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"claude CLI timed out after {self.timeout_s}s for "
                f"{request.custom_id!r}; emitting an invalid judgment",
                ExecutorWarning,
                stacklevel=2,
            )
            return invalid_judgment(request)

        if proc.returncode != 0:
            tail = (proc.stderr or "")[-_STDERR_TAIL_CHARS:]
            warnings.warn(
                f"claude CLI exited {proc.returncode} for {request.custom_id!r}: "
                f"{tail!r}; emitting an invalid judgment",
                ExecutorWarning,
                stacklevel=2,
            )
            return invalid_judgment(request)

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            warnings.warn(
                f"claude CLI emitted non-JSON output for {request.custom_id!r}; "
                "emitting an invalid judgment",
                ExecutorWarning,
                stacklevel=2,
            )
            return invalid_judgment(request, raw_text=proc.stdout or None)

        text = _extract_result_text(payload)
        if text is None:
            warnings.warn(
                f"could not extract result text for {request.custom_id!r} "
                "(error payload or unknown shape); emitting an invalid judgment",
                ExecutorWarning,
                stacklevel=2,
            )
            return invalid_judgment(request, raw_text=proc.stdout or None)
        return judgment_from_raw(request, text, self._pattern)

    def execute(self, requests: Sequence[JudgmentRequest]) -> list[Judgment]:
        """Run requests one at a time through the CLI; never parallel."""
        return [self._run_one(req) for req in requests]


__all__ = ["ClaudeCodeExecutor"]
