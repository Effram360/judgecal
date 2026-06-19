"""Recorded response packs: replay real judge outputs as local fixtures.

A pack is a JSONL file whose FIRST line is a version header::

    {"pack_version": 1, "judge": "<judge name>", "created": "<iso8601 or empty>"}

followed by one line per recorded response::

    {"custom_id": "jc-...-r0", "raw_text": "...[[A]]"}

``pack_from_batch_output`` converts an OpenAI-format batch *output* file
(as produced by vLLM ``run-batch`` and compatible runners) into a pack,
extracting raw text liberally: ``choices[0].message.content`` for chat
completions, else a ``score`` field (stringified — note that scores
replayed through the text parser yield "invalid"; score handling proper
lives in the executors module).

Determinism note: ``created`` is always caller-supplied (default empty
string) — nothing in this module reads the wall clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Current pack file-format version (the header's ``pack_version``).
PACK_VERSION: int = 1


@dataclass
class ResponsePack:
    """An in-memory recorded-response pack.

    Attributes:
        responses: Mapping ``custom_id -> raw_text``.
        judge: Identifier of the judge that produced the responses.
        created: Caller-supplied creation timestamp string (may be empty;
            never auto-generated, to keep src/ free of wall-clock reads).
        pack_version: File-format version (``PACK_VERSION``).
    """

    responses: dict[str, str] = field(default_factory=dict)
    judge: str = "unknown"
    created: str = ""
    pack_version: int = PACK_VERSION

    def __len__(self) -> int:
        """Number of recorded responses."""
        return len(self.responses)

    def __contains__(self, custom_id: str) -> bool:
        """Whether ``custom_id`` has a recorded response."""
        return custom_id in self.responses

    def get(self, custom_id: str) -> str | None:
        """Raw text for ``custom_id``, or None if absent."""
        return self.responses.get(custom_id)


def save_pack(pack: ResponsePack, path: Path | str) -> None:
    """Write a pack to JSONL (header line first, entries sorted by id).

    Sorting makes the file content a pure function of the pack, so packs
    diff cleanly and tests can compare bytes.

    Args:
        pack: The pack to serialize.
        path: Destination file path (parent directory must exist).
    """
    p = Path(path)
    header = {
        "pack_version": pack.pack_version,
        "judge": pack.judge,
        "created": pack.created,
    }
    lines = [json.dumps(header, sort_keys=True, ensure_ascii=False)]
    for custom_id in sorted(pack.responses):
        lines.append(
            json.dumps(
                {"custom_id": custom_id, "raw_text": pack.responses[custom_id]},
                sort_keys=True,
                ensure_ascii=False,
            )
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pack(path: Path | str) -> ResponsePack:
    """Load a pack from JSONL, validating the version header.

    Args:
        path: Pack file written by :func:`save_pack` (or hand-rolled to
            the same format).

    Returns:
        The deserialized pack. Duplicate custom_ids: last line wins.

    Raises:
        ValueError: If the file is empty, the first line is not a valid
            header, or ``pack_version`` is unsupported.
    """
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty response pack: {p}")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pack header (not JSON) in {p}: {exc}") from exc
    if not isinstance(header, dict) or "pack_version" not in header:
        raise ValueError(f"first line of {p} is not a pack header (missing 'pack_version')")
    version = header["pack_version"]
    if version != PACK_VERSION:
        raise ValueError(f"unsupported pack_version {version!r} in {p} (expected {PACK_VERSION})")

    responses: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=2):
        obj = json.loads(line)
        if "custom_id" not in obj or "raw_text" not in obj:
            raise ValueError(f"{p}:{i}: pack entry missing 'custom_id' or 'raw_text'")
        responses[str(obj["custom_id"])] = str(obj["raw_text"])
    return ResponsePack(
        responses=responses,
        judge=str(header.get("judge", "unknown")),
        created=str(header.get("created", "")),
        pack_version=int(version),
    )


def _extract_raw_text(response: dict[str, Any]) -> str | None:
    """Liberally extract raw text from an OpenAI-format batch response.

    Tries, in order: ``body.choices[0].message.content`` (chat),
    ``body.score`` and ``body.data[0].score`` (score endpoints, vLLM
    variants) — stringified. Returns None when nothing usable is found.
    """
    body = response.get("body") or {}
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    if "score" in body:
        return str(body["score"])
    data = body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict) and "score" in data[0]:
        return str(data[0]["score"])
    return None


def pack_from_batch_output(
    path: Path | str, judge: str = "unknown", created: str = ""
) -> ResponsePack:
    """Build a pack from an OpenAI-format batch output JSONL file.

    Lines are tolerated liberally: blank lines and lines without a
    ``custom_id`` are skipped; lines with a non-null ``error`` are skipped
    (so strict fixture replay surfaces them as missing); duplicate
    custom_ids (e.g. runner retries): last line wins.

    Args:
        path: Batch output file (one JSON object per line, each with
            ``custom_id`` and ``response`` per the OpenAI batch format).
        judge: Judge identifier to record in the pack header.
        created: Caller-supplied timestamp string (never auto-generated).

    Returns:
        A pack containing every successfully extracted response.
    """
    p = Path(path)
    responses: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        custom_id = obj.get("custom_id")
        if not custom_id:
            continue
        if obj.get("error"):
            continue
        raw_text = _extract_raw_text(obj.get("response") or {})
        if raw_text is None:
            continue
        responses[str(custom_id)] = raw_text
    return ResponsePack(responses=responses, judge=judge, created=created)


__all__ = [
    "PACK_VERSION",
    "ResponsePack",
    "load_pack",
    "pack_from_batch_output",
    "save_pack",
]
