"""Manifest emission: judgment requests → OpenAI batch JSONL + sidecar.

The manifest (``<name>.jsonl``) is an OpenAI-batch-format file directly
runnable by ``vllm run-batch`` (or ``python -m
vllm.entrypoints.openai.run_batch``). The sidecar (``<name>.meta.jsonl``)
records, per ``custom_id``, the content hash and every *usage* — the
``JudgmentRequest.meta`` of each (probe, condition, item, repeat) that
shares that execution body, minus the reserved raw-text keys
(:data:`judgecal.core.SCORE_TEXT_META_KEYS`), which are stripped to keep
the sidecar lean. Identical bodies at the same repeat index are
deduplicated to a single batch line; :func:`judgecal.manifests.sidecar.merge_results`
fans results back out to one judgment per usage.

Score endpoint (``/v1/score``): chat-rendered request bodies are replaced
at emission time by vLLM score-API bodies in the 1xN form
``{"model": ..., "text_1": prompt_text, "text_2": [first_text,
second_text]}`` built from the reserved text meta keys; the ``custom_id``
is recomputed from the emitted score body (same ``-r<repeat>`` mechanism),
so requests whose chat bodies differ but whose presented texts coincide
deduplicate to one score line. Each line yields two presented-side scores
(index 0 = presented-first), matching the ingest convention in
:func:`judgecal.manifests.sidecar.merge_results`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from judgecal.core import (
    SCORE_TEXT_META_KEYS,
    JudgmentRequest,
    body_hash,
    canonical_json,
    make_custom_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Endpoints supported by the manifest format. ``/v1/score`` covers scalar
#: reward models served by vLLM run-batch (verified ground §4).
Endpoint = str  # Literal narrowing happens on ModelSpec below.


@dataclass(frozen=True)
class ModelSpec:
    """How to target a judge arm when emitting a manifest.

    Sampling parameters live here (not on requests) so the same plan can be
    emitted against many judge arms. ``extra`` is merged last into every
    batch body (e.g. ``{"seed": 0}`` or vLLM-specific knobs) and overrides
    colliding keys.

    Attributes:
        model: Served model name, e.g. ``"qwen3.5-9b-awq"``.
        endpoint: ``/v1/chat/completions`` for generative judges,
            ``/v1/score`` for scalar reward models.
        temperature: Sampling temperature (chat endpoint only).
        max_tokens: Generation cap (chat endpoint only).
        extra: Extra body fields merged into every batch body.
    """

    model: str
    endpoint: str = "/v1/chat/completions"
    temperature: float = 0.0
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = ("/v1/chat/completions", "/v1/score")
        if self.endpoint not in allowed:
            raise ValueError(f"endpoint must be one of {allowed}, got {self.endpoint!r}")


@dataclass(frozen=True)
class ManifestPaths:
    """Where a manifest landed, plus enough context to build job packs.

    Attributes:
        manifest: Path to ``<name>.jsonl`` (OpenAI batch lines).
        sidecar: Path to ``<name>.meta.jsonl`` (usage fan-out records).
        n_lines: Number of deduplicated batch lines written.
        n_usages: Total usages across all lines (= judgments after merge).
        model: Model name baked into the batch bodies.
        endpoint: Endpoint URL baked into the batch lines.
    """

    manifest: Path
    sidecar: Path
    n_lines: int
    n_usages: int
    model: str
    endpoint: str


def _batch_body(request_body: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    """Merge a request body with the model spec into a final batch body."""
    body: dict[str, Any] = {"model": spec.model, **request_body}
    if spec.endpoint == "/v1/chat/completions":
        body.setdefault("temperature", spec.temperature)
        body.setdefault("max_tokens", spec.max_tokens)
    body.update(spec.extra)
    return body


def _score_body(req: JudgmentRequest, spec: ModelSpec) -> dict[str, Any]:
    """Build a vLLM score-API body (1xN form) from a request's text meta.

    ``text_2`` order is (presented-first, presented-second), so the two
    returned scores align with the merge convention (index 0 = first).

    Raises:
        ValueError: If any reserved text meta key is missing (the request
            was not planned by a judgecal probe).
    """
    missing = [k for k in SCORE_TEXT_META_KEYS if not isinstance(req.meta.get(k), str)]
    if missing:
        raise ValueError(
            f"request {req.custom_id!r} cannot target /v1/score: meta lacks "
            f"raw-text key(s) {missing}; score manifests need requests planned "
            "with judgecal probes (which record prompt_text/first_text/second_text)"
        )
    body: dict[str, Any] = {
        "model": spec.model,
        "text_1": req.meta["prompt_text"],
        "text_2": [req.meta["first_text"], req.meta["second_text"]],
    }
    body.update(spec.extra)
    return body


def _strip_text_keys(meta: dict[str, Any]) -> dict[str, Any]:
    """Copy a usage meta without the reserved raw-text keys."""
    return {k: v for k, v in meta.items() if k not in SCORE_TEXT_META_KEYS}


def emit_manifest(
    requests: Sequence[JudgmentRequest],
    spec: ModelSpec,
    out_dir: Path | str,
    name: str,
) -> ManifestPaths:
    """Write an OpenAI batch manifest and its usage sidecar.

    Dedup: requests sharing a ``custom_id`` (identical bodies at the same
    repeat index, by construction of :func:`judgecal.core.make_custom_id`)
    collapse to a single batch line; their metas are merged into the
    sidecar line's ``usages`` list (reserved raw-text keys stripped).
    Byte-identical usages are recorded once. Stability repeats are *not*
    merged — their custom_ids differ via the ``-r<k>`` suffix even though
    bodies are identical.

    Score endpoint: bodies are rebuilt as vLLM score-API bodies from the
    reserved text meta keys and ``custom_id`` is recomputed from the
    emitted score body (see module docstring), so chat-level distinctions
    that do not change the presented texts (e.g. template variants)
    additionally collapse.

    Args:
        requests: Planned judgment requests (any probe mix).
        spec: Judge arm to target; model/sampling params are attached here.
        out_dir: Output directory (created if missing).
        name: Basename; writes ``<name>.jsonl`` + ``<name>.meta.jsonl``.

    Returns:
        :class:`ManifestPaths` describing what was written.

    Raises:
        ValueError: If two requests share a ``custom_id`` but have different
            bodies (a content-hash invariant violation), if ``requests`` is
            empty, or if ``spec.endpoint`` is ``/v1/score`` and a request
            lacks the reserved raw-text meta keys.
    """
    if not requests:
        raise ValueError("emit_manifest called with no requests")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    score = spec.endpoint == "/v1/score"

    # custom_id -> (emitted_body, body_hash, [usages], {usage_keys})
    entries: dict[str, tuple[dict[str, Any], str, list[dict[str, Any]], set[str]]] = {}
    for req in requests:
        if score:
            emitted = _score_body(req, spec)
            custom_id = make_custom_id(emitted, int(req.meta["repeat"]))
        else:
            emitted = _batch_body(req.body, spec)
            custom_id = req.custom_id
        h = body_hash(req.body) if not score else body_hash(emitted)
        usage = _strip_text_keys(req.meta)
        usage_key = canonical_json(usage)
        if custom_id in entries:
            prev_body, prev_hash, usages, seen = entries[custom_id]
            if prev_hash != h:
                raise ValueError(
                    f"custom_id {custom_id!r} maps to two different bodies "
                    f"(hashes {prev_hash} vs {h}); custom_ids must be content hashes"
                )
            if usage_key not in seen:
                usages.append(usage)
                seen.add(usage_key)
        else:
            entries[custom_id] = (emitted, h, [usage], {usage_key})

    manifest_path = out_path / f"{name}.jsonl"
    sidecar_path = out_path / f"{name}.meta.jsonl"
    n_usages = 0
    with manifest_path.open("w", encoding="utf-8") as mf, sidecar_path.open(
        "w", encoding="utf-8"
    ) as sf:
        for custom_id, (emitted_body, h, usages, _) in entries.items():
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": spec.endpoint,
                "body": emitted_body,
            }
            mf.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
            side = {"custom_id": custom_id, "body_hash": h, "usages": usages}
            sf.write(json.dumps(side, sort_keys=True, ensure_ascii=False) + "\n")
            n_usages += len(usages)

    return ManifestPaths(
        manifest=manifest_path,
        sidecar=sidecar_path,
        n_lines=len(entries),
        n_usages=n_usages,
        model=spec.model,
        endpoint=spec.endpoint,
    )


__all__ = ["ManifestPaths", "ModelSpec", "emit_manifest"]
