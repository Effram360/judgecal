"""Sidecar loading, result merging, and resume support.

Batch output lines follow the OpenAI batch output format::

    {"custom_id": ..., "response": {"status_code": 200, "body": {...}}, "error": null}

We parse them *liberally* (contracts §4.1) because vLLM ``run-batch``
versions differ slightly:

- chat completions: ``response.body.choices[0].message.content`` (also
  tolerating ``choices[0].text`` and list-of-parts content),
- score endpoint: ``response.body.data[*].score`` (vLLM list-style) or a
  bare ``response.body.score`` / ``response.body.scores``,
- bodies nested directly under ``response`` or at the top level.

Score results need **two** presented-side scores (first, second) to form a
pairwise verdict; they are compared via
:func:`judgecal.executors.parsing.compare_scores` with an epsilon tie band.
Error lines and unparseable bodies become "invalid" judgments for every
usage, with a :class:`ManifestWarning`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from judgecal.core import Judgment, PresentedVerdict
from judgecal.executors.parsing import DEFAULT_SCORE_EPSILON, compare_scores, parse_verdict


class ManifestWarning(UserWarning):
    """Warning category for non-fatal manifest/merge problems."""


@dataclass(frozen=True)
class SidecarEntry:
    """One sidecar line: an executed body and every usage that shares it."""

    custom_id: str
    body_hash: str
    usages: list[dict[str, Any]]


def load_sidecar(path: Path | str) -> dict[str, SidecarEntry]:
    """Load a ``<name>.meta.jsonl`` sidecar into an ordered mapping.

    Args:
        path: Sidecar file written by :func:`judgecal.manifests.emit_manifest`.

    Returns:
        Insertion-ordered ``{custom_id: SidecarEntry}``.

    Raises:
        ValueError: On malformed lines (missing keys, duplicate custom_id).
    """
    out: dict[str, SidecarEntry] = {}
    for lineno, raw in _iter_jsonl(Path(path)):
        try:
            entry = SidecarEntry(
                custom_id=raw["custom_id"],
                body_hash=raw["body_hash"],
                usages=list(raw["usages"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed sidecar line ({exc!r})") from exc
        if entry.custom_id in out:
            raise ValueError(f"{path}:{lineno}: duplicate custom_id {entry.custom_id!r}")
        out[entry.custom_id] = entry
    return out


def _iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a JSONL file, skipping blank lines.

    Malformed lines (e.g. a truncated trailing line from an interrupted
    batch run) are skipped with a :class:`ManifestWarning` naming the file
    and line number, so resume/ingest never traceback on the files
    interrupted runs actually produce.
    """
    lines: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"{path}:{lineno}: skipping malformed JSONL line "
                    f"(truncated batch output? {exc.msg})",
                    ManifestWarning,
                    stacklevel=3,
                )
                continue
            lines.append((lineno, obj))
    return lines


def _extract_response_body(line: Mapping[str, Any]) -> dict[str, Any] | None:
    """Find the response body in an output line, tolerating format variants."""
    resp = line.get("response")
    if isinstance(resp, dict):
        body = resp.get("body")
        if isinstance(body, dict):
            return body
        if any(k in resp for k in ("choices", "data", "score", "scores")):
            return dict(resp)
    if any(k in line for k in ("choices", "data", "score", "scores")):
        return dict(line)
    return None


def extract_raw_text(body: Mapping[str, Any]) -> str | None:
    """Extract chat-completion text from a response body, liberally.

    Looks for ``choices[0].message.content`` (string or list-of-parts),
    falling back to completions-style ``choices[0].text``.
    """
    choices = body.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        return None
    first = choices[0]
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            if parts:
                return "".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    return None


def extract_scores(body: Mapping[str, Any]) -> list[float]:
    """Extract score values from a score-endpoint response body, liberally.

    Checks ``data[*].score`` (vLLM list style), then ``score`` /
    ``scores`` as a scalar or list. Order is preserved: index 0 is the
    presented-first response.
    """
    data = body.get("data")
    if isinstance(data, list):
        scores = [
            float(d["score"])
            for d in data
            if isinstance(d, dict) and isinstance(d.get("score"), (int, float))
        ]
        if scores:
            return scores
    for key in ("score", "scores"):
        value = body.get(key)
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            scores = [float(v) for v in value if isinstance(v, (int, float))]
            if scores:
                return scores
    return []


def _classify_line(
    line: Mapping[str, Any],
    pattern: str | None,
    score_epsilon: float,
) -> tuple[PresentedVerdict, str | None, str | None]:
    """Classify one output line.

    Returns:
        ``(verdict, raw_text, problem)`` — ``problem`` is a warning message
        when the line was an error / unparseable, else ``None``. A verdict
        of "invalid" with ``problem is None`` means the body was fine but
        contained no marker (a judge failure, not a transport failure).
    """
    custom_id = line.get("custom_id", "<missing custom_id>")
    error = line.get("error")
    if error:
        return "invalid", None, f"error line for {custom_id!r}: {error!r}"
    body = _extract_response_body(line)
    if body is None:
        return "invalid", None, f"no response body found for {custom_id!r}"
    raw_text = extract_raw_text(body)
    if raw_text is not None:
        return parse_verdict(raw_text, pattern), raw_text, None
    scores = extract_scores(body)
    if len(scores) >= 2:
        raw = f"scores:{scores!r}"
        return compare_scores(scores[0], scores[1], score_epsilon), raw, None
    if len(scores) == 1:
        return (
            "invalid",
            f"scores:{scores!r}",
            f"single score for {custom_id!r}: a pairwise verdict needs two "
            "presented-side scores",
        )
    return "invalid", None, f"unrecognized response body for {custom_id!r}"


def _read_outputs(output_paths: Path | str | Sequence[Path | str]) -> list[dict[str, Any]]:
    """Read one or many batch-output JSONL files into a flat line list."""
    if isinstance(output_paths, (str, Path)):
        paths: list[Path] = [Path(output_paths)]
    else:
        paths = [Path(p) for p in output_paths]
    lines: list[dict[str, Any]] = []
    for p in paths:
        lines.extend(obj for _, obj in _iter_jsonl(p))
    return lines


def merge_results(
    sidecar: Path | str | Mapping[str, SidecarEntry],
    output_paths: Path | str | Sequence[Path | str],
    *,
    pattern: str | None = None,
    score_epsilon: float = DEFAULT_SCORE_EPSILON,
) -> list[Judgment]:
    """Merge batch outputs with the sidecar, fanning out to one judgment per usage.

    Each output line is parsed once; its verdict and raw text are then
    replicated to every usage recorded for that ``custom_id`` (probes that
    shared an execution body each get their own judgment with their own
    meta). Error lines, unknown custom_ids, single-score bodies, and
    conflicting duplicate lines raise :class:`ManifestWarning`;
    error/unparseable lines still produce "invalid" judgments so invalid
    rates are honest.

    Duplicate custom_ids resolve clean-line-first: a later line that
    classifies cleanly replaces an earlier error/unparseable line (the
    normal artifact of the documented retry/resume workflow, regardless of
    the order the results files are passed in); among multiple clean lines
    the first wins with a warning, and among multiple error lines the
    first wins. Problem warnings are emitted only for the line finally
    kept.

    Args:
        sidecar: Sidecar path or a preloaded :func:`load_sidecar` mapping.
        output_paths: One or many batch output JSONL files.
        pattern: Optional custom verdict regex.
        score_epsilon: Tie band for score-pair comparison.

    Returns:
        Judgments in output-line order of each custom_id's first
        occurrence (then usage order within a line).
    """
    entries = sidecar if isinstance(sidecar, Mapping) else load_sidecar(sidecar)
    # custom_id -> (verdict, raw_text, problem); insertion = first occurrence
    kept: dict[str, tuple[PresentedVerdict, str | None, str | None]] = {}
    for line in _read_outputs(output_paths):
        custom_id = line.get("custom_id")
        if not isinstance(custom_id, str):
            warnings.warn(
                f"output line without custom_id skipped: {line!r:.120}",
                ManifestWarning,
                stacklevel=2,
            )
            continue
        if custom_id not in entries:
            warnings.warn(
                f"output custom_id {custom_id!r} not in sidecar; skipped",
                ManifestWarning,
                stacklevel=2,
            )
            continue
        classified = _classify_line(line, pattern, score_epsilon)
        previous = kept.get(custom_id)
        if previous is None:
            kept[custom_id] = classified
        elif previous[2] is not None and classified[2] is None:
            # Later clean line replaces an earlier error/unparseable line
            # (retry/resume workflow); keep the first-occurrence position.
            kept[custom_id] = classified
        else:
            warnings.warn(
                f"duplicate output line for {custom_id!r}; keeping the first",
                ManifestWarning,
                stacklevel=2,
            )
    judgments: list[Judgment] = []
    for custom_id, (verdict, raw_text, problem) in kept.items():
        if problem is not None:
            warnings.warn(problem, ManifestWarning, stacklevel=2)
        for usage in entries[custom_id].usages:
            judgments.append(
                Judgment(
                    custom_id=custom_id,
                    verdict=verdict,
                    raw_text=raw_text,
                    meta=dict(usage),
                )
            )
    return judgments


def remaining_manifest(
    manifest: Path | str,
    sidecar: Path | str | Mapping[str, SidecarEntry],
    output_paths: Path | str | Sequence[Path | str],
) -> list[dict[str, Any]]:
    """Return manifest batch lines that still need to run (resume support).

    A custom_id counts as *completed* when an output line exists for it
    with no error and an extractable body (chat text or >= 2 scores) —
    even if the text contained no verdict marker (re-running a marker-free
    judgment at the same settings would not help). Error lines and
    unparseable bodies remain in the resume set.

    Args:
        manifest: The original ``<name>.jsonl`` manifest.
        sidecar: Matching sidecar (used for a consistency check).
        output_paths: Batch output file(s) collected so far. May be an
            empty list, in which case every line remains.

    Returns:
        The parsed manifest line dicts still to execute, in manifest order.
        Write them back out with ``json.dumps`` per line to get a runnable
        resume manifest.
    """
    entries = sidecar if isinstance(sidecar, Mapping) else load_sidecar(sidecar)
    manifest_lines = [obj for _, obj in _iter_jsonl(Path(manifest))]

    completed: set[str] = set()
    for line in _read_outputs(output_paths):
        custom_id = line.get("custom_id")
        if not isinstance(custom_id, str):
            continue
        _, _, problem = _classify_line(line, None, DEFAULT_SCORE_EPSILON)
        if problem is None:
            completed.add(custom_id)

    remaining: list[dict[str, Any]] = []
    for line in manifest_lines:
        custom_id = line.get("custom_id")
        if not isinstance(custom_id, str):
            warnings.warn(
                f"manifest line without custom_id kept verbatim: {line!r:.120}",
                ManifestWarning,
                stacklevel=2,
            )
            remaining.append(line)
            continue
        if custom_id not in entries:
            warnings.warn(
                f"manifest custom_id {custom_id!r} missing from sidecar",
                ManifestWarning,
                stacklevel=2,
            )
        if custom_id not in completed:
            remaining.append(line)
    return remaining


__all__ = [
    "ManifestWarning",
    "SidecarEntry",
    "extract_raw_text",
    "extract_scores",
    "load_sidecar",
    "merge_results",
    "remaining_manifest",
]
