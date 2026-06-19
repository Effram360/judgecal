"""judgecal.probes — behavioral bias and reliability probes.

Five probes (the v1 cut of the CALM taxonomy, arXiv 2410.02736):
position, verbosity, self_preference, template, stability. Each probe
plans fully self-describing :class:`~judgecal.core.JudgmentRequest`
objects and analyzes executed :class:`~judgecal.core.Judgment` objects
into a :class:`~judgecal.core.ProbeResult`, with all inference routed
through :mod:`judgecal.stats`.

Importing this package registers every probe in ``PROBE_REGISTRY``.
"""

from __future__ import annotations

from judgecal.probes.base import (
    PROBE_REGISTRY,
    Probe,
    ProbeConfig,
    analyze_suite,
    get_probe,
    plan_suite,
    register_probe,
)
from judgecal.probes.padding import pad_text
from judgecal.probes.position import PositionProbe
from judgecal.probes.self_preference import SelfPreferenceProbe
from judgecal.probes.stability import StabilityProbe
from judgecal.probes.template import TemplateProbe
from judgecal.probes.templates import (
    DEFAULT_TEMPLATE_ID,
    TEMPLATE_IDS,
    JudgeTemplate,
    get_template,
    render,
    template_ids_for,
)
from judgecal.probes.verbosity import VerbosityProbe

__all__ = [
    "DEFAULT_TEMPLATE_ID",
    "PROBE_REGISTRY",
    "TEMPLATE_IDS",
    "JudgeTemplate",
    "PositionProbe",
    "Probe",
    "ProbeConfig",
    "SelfPreferenceProbe",
    "StabilityProbe",
    "TemplateProbe",
    "VerbosityProbe",
    "analyze_suite",
    "get_probe",
    "get_template",
    "pad_text",
    "plan_suite",
    "register_probe",
    "render",
    "template_ids_for",
]
