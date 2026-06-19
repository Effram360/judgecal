"""Probe ABC, registry, suite helpers, and shared plan/analysis utilities.

A probe turns items into fully self-describing :class:`JudgmentRequest`
objects (``plan``) and turns executed :class:`Judgment` objects back into
a :class:`ProbeResult` (``analyze``). Analyses read ``Judgment.meta``
exclusively — never the original items — and route all statistical
inference through :mod:`judgecal.stats`.

The registry maps probe names ("position", "verbosity",
"self_preference", "template", "stability") to classes; probe modules
self-register at import time and the package ``__init__`` imports them
all, so the registry is always populated once ``judgecal.probes`` is
importable.

``analyze_suite`` honors ``Probe.requires``: a probe that declares it
needs another probe's judgments (e.g. verbosity's observational GLM
re-uses the position probe's judgments) receives the union of its own
and the required probes' judgments.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from judgecal import stats
from judgecal.core import (
    Estimate,
    Judgment,
    JudgmentRequest,
    Label,
    PairwiseItem,
    ProbeResult,
    make_custom_id,
)
from judgecal.probes.templates import DEFAULT_TEMPLATE_ID, render
from judgecal.stats import mde_from_se

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Below this many distinct items the cluster bootstrap is unreliable
#: (and stats.cluster_bootstrap_ci requires >= 2 clusters); analyses fall
#: back to Wilson intervals (binary metrics) or skip with a warning.
MIN_BOOT_CLUSTERS = 3


# --------------------------------------------------------------------------
# Config, ABC, registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Shared configuration for planning and analyzing probes.

    Attributes:
        n_template_variants: Number of prompt templates the template
            probe compares (1..5; 5 ships in-package).
        stability_k: Number of identical-body repeats for the stability
            probe.
        pad_target_ratio: Target ``len(padded)/len(original)`` for the
            verbosity probe's constructed contrast.
        alpha: Two-sided level for all confidence intervals.
        n_boot: Bootstrap replicates for cluster-bootstrap CIs.
        seed: Seed for every stochastic analysis step (deterministic).
        judge_author: Author string identifying the judge itself; the
            self-preference probe compares sides authored by this string
            against others. Matches ``MockJudgeConfig.self_name`` and the
            synthetic generator's "judge-self" by default; set it to the
            judge's model name when auditing real datasets.
    """

    n_template_variants: int = 5
    stability_k: int = 5
    pad_target_ratio: float = 1.6
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    judge_author: str = "judge-self"


class Probe(ABC):
    """One bias/reliability probe: a planner plus an analyzer.

    Attributes:
        name: Registry name; also written into ``meta["probe"]``.
        requires: Names of other probes whose judgments this probe's
            ``analyze`` additionally consumes (``analyze_suite`` passes
            the union).
    """

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """Build fully-rendered judgment requests for these items."""

    @abstractmethod
    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult:
        """Turn executed judgments into estimates (meta-only; no items)."""


#: Probe name -> probe class. Populated by ``@register_probe`` when the
#: probe modules are imported (the package __init__ imports them all).
PROBE_REGISTRY: dict[str, type[Probe]] = {}


def register_probe(cls: type[Probe]) -> type[Probe]:
    """Class decorator adding a probe class to ``PROBE_REGISTRY``."""
    PROBE_REGISTRY[cls.name] = cls
    return cls


def get_probe(name: str) -> Probe:
    """Instantiate a registered probe by name.

    Args:
        name: Probe registry name (e.g. ``"position"``).

    Returns:
        A fresh probe instance.

    Raises:
        KeyError: If no probe with this name is registered.
    """
    try:
        cls = PROBE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown probe {name!r}; registered probes: {sorted(PROBE_REGISTRY)}"
        ) from None
    return cls()


def _resolve_probes(probes: Sequence[str | Probe]) -> list[Probe]:
    return [p if isinstance(p, Probe) else get_probe(p) for p in probes]


def plan_suite(
    items: Sequence[PairwiseItem],
    probes: Sequence[str | Probe],
    config: ProbeConfig,
) -> list[JudgmentRequest]:
    """Concatenated plans for several probes.

    Identical request bodies across probes (e.g. position's and
    self-preference's ``orig`` passes) intentionally share a
    ``custom_id``; deduplication happens at manifest-emission level,
    where the sidecar fans one execution back out to every usage.

    Args:
        items: Pairwise items to audit.
        probes: Probe names or instances.
        config: Shared probe configuration.

    Returns:
        All probes' requests, concatenated in probe order.
    """
    requests: list[JudgmentRequest] = []
    for probe in _resolve_probes(probes):
        requests.extend(probe.plan(items, config))
    return requests


def analyze_suite(
    judgments: Sequence[Judgment],
    probes: Sequence[str | Probe],
    config: ProbeConfig,
) -> list[ProbeResult]:
    """Run each probe's analysis over the relevant judgment subset.

    Each probe receives its own judgments plus those of any probe named
    in its ``requires`` tuple (e.g. verbosity receives position's
    judgments for the observational length GLM).

    Args:
        judgments: Executed judgments from any mix of probes.
        probes: Probe names or instances to analyze.
        config: Shared probe configuration.

    Returns:
        One :class:`ProbeResult` per requested probe, in order.
    """
    results: list[ProbeResult] = []
    for probe in _resolve_probes(probes):
        wanted = {probe.name, *probe.requires}
        subset = [j for j in judgments if j.meta.get("probe") in wanted]
        results.append(probe.analyze(subset, config))
    return results


# --------------------------------------------------------------------------
# Plan-time helpers
# --------------------------------------------------------------------------


def present_label(label: Label | None, first_is_a: bool) -> str | None:
    """Map an item-coordinates ground-truth label to presented coords."""
    if label is None or label == "tie":
        return label
    if label == "A":
        return "first" if first_is_a else "second"
    return "second" if first_is_a else "first"


def make_request(
    *,
    probe: str,
    condition: str,
    item_id: str,
    prompt: str,
    first_text: str,
    second_text: str,
    first_is_a: bool,
    first_author: str | None,
    second_author: str | None,
    first_latent_q: float | None,
    second_latent_q: float | None,
    label_first: str | None,
    template_id: str = DEFAULT_TEMPLATE_ID,
    repeat: int = 0,
) -> JudgmentRequest:
    """Build a fully self-describing request from presented-side values.

    The body is the rendered chat-completions messages list; the meta
    dict carries every ``REQUIRED_META_KEYS`` entry plus ``template_id``
    and the three reserved ``SCORE_TEXT_META_KEYS`` (``prompt_text``,
    ``first_text``, ``second_text`` — raw strings) so manifest emission
    can build ``/v1/score`` bodies for scalar reward models. ``custom_id``
    is content-hashed from the body and the repeat index, so identical
    work deduplicates across probes.
    """
    body = {"messages": render(template_id, prompt, first_text, second_text)}
    meta: dict[str, Any] = {
        "probe": probe,
        "condition": condition,
        "item_id": item_id,
        "repeat": repeat,
        "first_is_a": first_is_a,
        "first_len": len(first_text),
        "second_len": len(second_text),
        "first_author": first_author,
        "second_author": second_author,
        "first_latent_q": first_latent_q,
        "second_latent_q": second_latent_q,
        "label_first": label_first,
        "template_id": template_id,
        "prompt_text": prompt,
        "first_text": first_text,
        "second_text": second_text,
    }
    return JudgmentRequest(custom_id=make_custom_id(body, repeat), body=body, meta=meta)


def request_from_item(
    item: PairwiseItem,
    *,
    probe: str,
    condition: str,
    first_is_a: bool,
    template_id: str = DEFAULT_TEMPLATE_ID,
    repeat: int = 0,
) -> JudgmentRequest:
    """Build a request presenting the item's own two responses.

    ``first_is_a=True`` presents ``response_a`` first; ``False`` swaps.
    Authors, planted latent qualities (``item.meta["latent_quality_a"]``
    / ``"latent_quality_b"``), and the ground-truth label are mapped into
    presented coordinates.
    """
    qa = item.meta.get("latent_quality_a")
    qb = item.meta.get("latent_quality_b")
    if first_is_a:
        first_text, second_text = item.response_a, item.response_b
        first_author, second_author = item.author_a, item.author_b
        first_q, second_q = qa, qb
    else:
        first_text, second_text = item.response_b, item.response_a
        first_author, second_author = item.author_b, item.author_a
        first_q, second_q = qb, qa
    return make_request(
        probe=probe,
        condition=condition,
        item_id=item.item_id,
        prompt=item.prompt,
        first_text=first_text,
        second_text=second_text,
        first_is_a=first_is_a,
        first_author=first_author,
        second_author=second_author,
        first_latent_q=first_q,
        second_latent_q=second_q,
        label_first=present_label(item.label, first_is_a),
        template_id=template_id,
        repeat=repeat,
    )


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------


def own_judgments(judgments: Sequence[Judgment], probe_name: str) -> list[Judgment]:
    """Subset of judgments belonging to ``probe_name`` (meta-driven)."""
    return [j for j in judgments if j.meta.get("probe") == probe_name]


def decisive_judgments(judgments: Sequence[Judgment]) -> list[Judgment]:
    """Judgments whose verdict is "first" or "second"."""
    return [j for j in judgments if j.is_decisive]


def judgment_counts(judgments: Sequence[Judgment]) -> tuple[int, int, float]:
    """``(n_judgments, n_items, invalid_rate)`` for a probe's own set."""
    n = len(judgments)
    n_items = len({j.item_id for j in judgments})
    invalid = sum(1 for j in judgments if j.verdict == "invalid")
    return n, n_items, (invalid / n if n else 0.0)


def cluster_rate_estimate(
    name: str,
    rows: list[tuple[str, float]],
    *,
    null: float | None,
    config: ProbeConfig,
    warnings: list[str],
    binary: bool = True,
) -> Estimate | None:
    """Cluster-bootstrap estimate of a mean rate over (item, value) rows.

    Clusters are items; the statistic is the plain mean of ``value``.
    With a ``null``, the p-value comes from the bootstrap's
    percentile-rank CI inversion and the MDE from
    ``stats.mde_from_se`` on the realized bootstrap SE — consistent by
    construction with the reported CI (clustering and any negative
    within-item correlation are already in the SE; the design-effect
    formula ``stats.mde_proportion(n / deff)`` is kept for *planning*,
    where no data exist yet).

    Bootstrap caveats (e.g. the anti-conservative few-cluster regime,
    ``n_clusters < 15``) are surfaced into ``warnings`` with the metric
    name prefixed.

    Fallbacks: with fewer than ``MIN_BOOT_CLUSTERS`` items, binary
    metrics fall back to a Wilson interval (no p-value); non-binary
    metrics are skipped. Both paths append a warning. Returns ``None``
    when the metric is skipped.

    Args:
        name: Metric name for the :class:`Estimate` and warnings.
        rows: ``(item_id, value)`` pairs; values in {0, 1} when
            ``binary``.
        null: Null value (e.g. 0.5 for pick rates) or ``None``.
        config: Probe configuration (alpha, n_boot, seed).
        warnings: Warning sink (mutated in place).
        binary: Whether values are 0/1 (enables the Wilson fallback and
            proportion-scale MDE).
    """
    if not rows:
        warnings.append(f"{name}: no usable judgments; metric skipped")
        return None
    df = pd.DataFrame(rows, columns=["item_id", "value"])
    n = len(df)
    n_clusters = int(df["item_id"].nunique())

    if n_clusters < MIN_BOOT_CLUSTERS:
        if not binary:
            warnings.append(
                f"{name}: only {n_clusters} item(s); too few for a bootstrap CI; skipped"
            )
            return None
        warnings.append(
            f"{name}: only {n_clusters} item(s); Wilson fallback (no clustering, no p-value)"
        )
        k = int(round(float(df["value"].sum())))
        lo, hi = stats.wilson_ci(k, n, alpha=config.alpha)
        mde: float | None = (
            stats.mde_proportion(float(n), p0=null, alpha=config.alpha)
            if null is not None
            else None
        )
        return Estimate(
            name=name,
            estimate=k / n,
            ci_low=float(lo),
            ci_high=float(hi),
            n=n,
            method="wilson_small_n",
            null_value=null,
            mde=mde,
            detail={"n_clusters": n_clusters},
        )

    try:
        res = stats.cluster_bootstrap_ci(
            df,
            lambda d: float(d["value"].mean()),
            "item_id",
            n_boot=config.n_boot,
            alpha=config.alpha,
            seed=config.seed,
            null_value=null,
        )
    except ValueError as exc:
        warnings.append(f"{name}: bootstrap failed ({exc}); metric skipped")
        return None
    for caveat in res.warnings:
        warnings.append(f"{name}: {caveat}")

    detail: dict[str, Any] = {"n_clusters": n_clusters, "boot_se": float(res.boot_se)}
    mde = None
    if null is not None:
        mde = mde_from_se(float(res.boot_se), config.alpha)

    return Estimate(
        name=name,
        estimate=float(res.estimate),
        ci_low=float(res.ci_low),
        ci_high=float(res.ci_high),
        n=n,
        method="cluster_bootstrap",
        null_value=null,
        p_value=None if res.p_value is None else float(res.p_value),
        mde=mde,
        detail=detail,
    )


_KAPPA_CATEGORIES: tuple[str, ...] = ("first", "second", "tie")


def _kappa_tables(rows: Sequence[tuple[str, str]]) -> np.ndarray:
    """(item, verdict) rows -> items x {first, second, tie} count table.

    Item order follows first appearance (deterministic).
    """
    index = {c: k for k, c in enumerate(_KAPPA_CATEGORIES)}
    by_item: dict[str, list[int]] = {}
    for item_id, verdict in rows:
        counts = by_item.setdefault(item_id, [0] * len(_KAPPA_CATEGORIES))
        counts[index[verdict]] += 1
    return np.asarray(list(by_item.values()), dtype=float)


def _safe_kappa(table: np.ndarray) -> float:
    try:
        return float(stats.fleiss_kappa(table))
    except ValueError:
        return float("nan")


def kappa_estimate(
    name: str,
    rows: Sequence[tuple[str, str]],
    *,
    config: ProbeConfig,
    warnings: list[str],
) -> Estimate | None:
    """Fleiss' kappa with an item-level bootstrap CI (no null / p-value).

    ``rows`` are (item_id, verdict) ratings with verdict in
    {"first", "second", "tie"}; callers must supply the same number of
    ratings per item (a balanced table).

    The CI resamples *items* with a local bootstrap rather than
    ``stats.cluster_bootstrap_ci``: a contingency-table statistic cannot
    be recomputed from concatenated resampled rows, because two copies of
    a drawn item would merge into one doubled table row. Non-finite
    replicate kappas (degenerate resamples) are dropped; if every
    replicate is degenerate the CI collapses to the point estimate with a
    warning.

    Returns ``None`` (with a warning) when kappa is undefined or there
    are no rows.
    """
    if not rows:
        warnings.append(f"{name}: no usable judgments; metric skipped")
        return None
    tables = _kappa_tables(rows)
    n_items = tables.shape[0]
    n_raters = int(tables[0].sum()) if n_items else 0
    point = _safe_kappa(tables)
    if not math.isfinite(point):
        warnings.append(f"{name}: kappa undefined (degenerate agreement table); skipped")
        return None

    detail: dict[str, Any] = {"n_raters": n_raters}
    if n_items < MIN_BOOT_CLUSTERS:
        warnings.append(f"{name}: only {n_items} item(s); CI collapsed to point estimate")
        ci_low = ci_high = point
    else:
        rng = np.random.default_rng(config.seed)
        reps = np.empty(config.n_boot)
        for b in range(config.n_boot):
            idx = rng.integers(0, n_items, size=n_items)
            reps[b] = _safe_kappa(tables[idx])
        valid = reps[np.isfinite(reps)]
        detail["n_boot_valid"] = int(valid.size)
        if valid.size == 0:
            warnings.append(
                f"{name}: all bootstrap replicates degenerate; CI collapsed to point estimate"
            )
            ci_low = ci_high = point
        else:
            lo, hi = np.percentile(valid, [100 * config.alpha / 2, 100 * (1 - config.alpha / 2)])
            ci_low, ci_high = float(lo), float(hi)

    return Estimate(
        name=name,
        estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n_items,
        method="item_bootstrap",
        detail=detail,
    )


def first_judgment_per(
    judgments: Sequence[Judgment],
) -> dict[tuple[str, str], Judgment]:
    """First judgment per (item_id, condition), in input order."""
    seen: dict[tuple[str, str], Judgment] = {}
    for j in judgments:
        seen.setdefault((j.item_id, j.condition), j)
    return seen


__all__ = [
    "MIN_BOOT_CLUSTERS",
    "PROBE_REGISTRY",
    "Probe",
    "ProbeConfig",
    "analyze_suite",
    "cluster_rate_estimate",
    "decisive_judgments",
    "first_judgment_per",
    "get_probe",
    "judgment_counts",
    "kappa_estimate",
    "make_request",
    "mde_from_se",
    "own_judgments",
    "plan_suite",
    "present_label",
    "register_probe",
    "request_from_item",
]
