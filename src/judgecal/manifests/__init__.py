"""Batch manifest emission, sidecar fan-out, and resume support.

Public API (contracts §4.1): :class:`ModelSpec`, :func:`emit_manifest`,
:func:`load_sidecar`, :func:`merge_results`, :func:`remaining_manifest`.
"""

from judgecal.manifests.emit import ManifestPaths, ModelSpec, emit_manifest
from judgecal.manifests.sidecar import (
    ManifestWarning,
    SidecarEntry,
    extract_raw_text,
    extract_scores,
    load_sidecar,
    merge_results,
    remaining_manifest,
)

__all__ = [
    "ManifestPaths",
    "ManifestWarning",
    "ModelSpec",
    "SidecarEntry",
    "emit_manifest",
    "extract_raw_text",
    "extract_scores",
    "load_sidecar",
    "merge_results",
    "remaining_manifest",
]
