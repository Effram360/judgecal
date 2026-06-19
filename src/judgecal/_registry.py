"""Inspect AI entry-point target: registers judgecal's scorer with Inspect.

Wired in ``pyproject.toml`` as::

    [project.entry-points.inspect_ai]
    judgecal = "judgecal._registry"

inspect-ai (verified 0.3.239) loads every entry point in the
``inspect_ai`` group at startup and imports its target module (see
``src/inspect_ai/_util/entrypoints.py`` in the inspect_ai source — the
mechanism, not the docs page, is the authority per the 2026-06-10 version
pins). Importing :mod:`judgecal.integrations.inspect_ai` runs its
``@scorer`` decorator, which registers :func:`judgecal_pairwise` in
Inspect's scorer registry (usable as ``--scorer judgecal/judgecal_pairwise``
with ``inspect score``).

This module must NEVER raise: environments without inspect-ai (or without
the ``judgecal[inspect]`` extra) import it too — e.g. any tool that scans
entry points — so the guarded import degrades silently and
``judgecal_pairwise`` is ``None``.
"""

from __future__ import annotations

try:
    from judgecal.integrations.inspect_ai import judgecal_pairwise
except ImportError:  # inspect-ai not installed — degrade silently.
    judgecal_pairwise = None  # type: ignore[assignment]

__all__ = ["judgecal_pairwise"]
