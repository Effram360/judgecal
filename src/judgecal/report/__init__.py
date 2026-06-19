"""judgecal.report — the reliability card: build, flag, render, persist.

Public API: pydantic card models, :func:`build_card` (fills BH-FDR
q-values card-wide and applies flag conventions), :func:`render_markdown`
(the user-facing report), and JSON persistence helpers.
"""

from __future__ import annotations

from judgecal.report.card import (
    MetricEntry,
    ProbeEntry,
    ReliabilityCard,
    build_card,
    load_card,
    save_card,
)
from judgecal.report.render import render_markdown

__all__ = [
    "MetricEntry",
    "ProbeEntry",
    "ReliabilityCard",
    "build_card",
    "load_card",
    "render_markdown",
    "save_card",
]
