"""Stability probe: identical requests repeated k times.

Each item is judged ``config.stability_k`` times with byte-identical
bodies (default template, orig order, condition ``rep``); only the
``-r<k>`` suffix of the content-hashed ``custom_id`` distinguishes the
repeats, so manifests execute each repeat separately instead of
deduplicating them.

**What this measures** (the variance-component story from the plan): at
temperature 0 any disagreement across repeats is *serving*
nondeterminism — e.g. vLLM under continuous batching is not run-to-run
deterministic (Thinking Machines, Sep 2025) — while at temperature > 0
it additionally includes sampling noise. Flip rates from unstable
serving measure the infrastructure, not the judge; this probe makes
that variance component explicit.

Metrics (contract §5):

* ``unanimity_rate`` — fraction of items with identical verdicts across
  all repeats; Wilson CI. Items with any unparseable repeat are excluded
  (warned), as are items with fewer than 2 verdicts.
* ``mean_pairwise_flip`` — mean over items of the disagreement rate
  among decisive repeat pairs; cluster-bootstrap CI over items.
* ``stability_fleiss_kappa`` — Fleiss' kappa, repeats as raters, over
  items with the modal number of parseable repeats; item-bootstrap CI.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from judgecal import stats
from judgecal.core import Estimate, JudgmentRequest, ProbeResult
from judgecal.probes.base import (
    Probe,
    ProbeConfig,
    cluster_rate_estimate,
    judgment_counts,
    kappa_estimate,
    own_judgments,
    register_probe,
    request_from_item,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import Judgment, PairwiseItem


@register_probe
class StabilityProbe(Probe):
    """Repeat-identical-bodies probe for verdict stability."""

    name = "stability"

    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """``stability_k`` identical-body requests per item (repeats 0..k-1)."""
        requests: list[JudgmentRequest] = []
        for item in items:
            for repeat in range(config.stability_k):
                requests.append(
                    request_from_item(
                        item,
                        probe=self.name,
                        condition="rep",
                        first_is_a=True,
                        repeat=repeat,
                    )
                )
        return requests

    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult:
        """Unanimity, pairwise flip, and kappa across repeats (meta-only)."""
        warnings: list[str] = []
        own = own_judgments(judgments, self.name)
        n_judgments, n_items, invalid_rate = judgment_counts(own)
        estimates: list[Estimate] = []
        result = ProbeResult(
            probe=self.name,
            estimates=estimates,
            n_items=n_items,
            n_judgments=n_judgments,
            invalid_rate=invalid_rate,
            warnings=warnings,
            meta={"conditions": sorted({j.condition for j in own})},
        )
        if not own:
            warnings.append("no judgments for probe 'stability'")
            return result

        # Verdicts per item, one per (item, repeat), in repeat order.
        seen: dict[tuple[str, int], str] = {}
        for j in own:
            seen.setdefault((j.item_id, j.repeat), j.verdict)
        per_item: dict[str, list[str]] = {}
        for (item_id, _repeat), verdict in sorted(seen.items()):
            per_item.setdefault(item_id, []).append(verdict)

        # -- unanimity_rate -------------------------------------------------
        eligible: dict[str, list[str]] = {}
        n_invalid_items = 0
        n_single = 0
        for item_id, verdicts in per_item.items():
            if "invalid" in verdicts:
                n_invalid_items += 1
            elif len(verdicts) < 2:
                n_single += 1
            else:
                eligible[item_id] = verdicts
        if n_invalid_items:
            warnings.append(
                f"unanimity_rate: {n_invalid_items} item(s) with unparseable repeats excluded"
            )
        if n_single:
            warnings.append(
                f"unanimity_rate: {n_single} item(s) with fewer than 2 repeats excluded"
            )
        if not eligible:
            warnings.append("unanimity_rate: no items with >= 2 parseable repeats; skipped")
        else:
            n_eval = len(eligible)
            n_unanimous = sum(1 for v in eligible.values() if len(set(v)) == 1)
            lo, hi = stats.wilson_ci(n_unanimous, n_eval, alpha=config.alpha)
            estimates.append(
                Estimate(
                    name="unanimity_rate",
                    estimate=n_unanimous / n_eval,
                    ci_low=float(lo),
                    ci_high=float(hi),
                    n=n_eval,
                    method="wilson",
                    detail={"n_unanimous": n_unanimous},
                )
            )

        # -- mean_pairwise_flip ----------------------------------------------
        flip_rows: list[tuple[str, float]] = []
        for item_id, verdicts in per_item.items():
            n_first = sum(1 for v in verdicts if v == "first")
            n_second = sum(1 for v in verdicts if v == "second")
            n_decisive = n_first + n_second
            n_pairs = n_decisive * (n_decisive - 1) // 2
            if n_pairs == 0:
                continue
            flip_rows.append((item_id, (n_first * n_second) / n_pairs))
        est = cluster_rate_estimate(
            "mean_pairwise_flip",
            flip_rows,
            null=None,
            config=config,
            warnings=warnings,
            binary=False,
        )
        if est is not None:
            estimates.append(est)

        # -- stability_fleiss_kappa -------------------------------------------
        parseable = {
            item_id: verdicts
            for item_id, verdicts in per_item.items()
            if "invalid" not in verdicts and len(verdicts) >= 2
        }
        if parseable:
            counts = Counter(len(v) for v in parseable.values())
            # Modal repeat count; ties broken toward the larger k.
            k_modal = max(sorted(counts.items()), key=lambda kv: (kv[1], kv[0]))[0]
            balanced = {i: v for i, v in parseable.items() if len(v) == k_modal}
            n_unbalanced = len(parseable) - len(balanced)
            if n_unbalanced:
                warnings.append(
                    f"stability_fleiss_kappa: {n_unbalanced} item(s) without the modal "
                    f"{k_modal} parseable repeats excluded (balanced table required)"
                )
            kappa_rows = [(i, v) for i, verdicts in balanced.items() for v in verdicts]
            est = kappa_estimate(
                "stability_fleiss_kappa", kappa_rows, config=config, warnings=warnings
            )
            if est is not None:
                estimates.append(est)
        else:
            warnings.append("stability_fleiss_kappa: no items with >= 2 parseable repeats")

        return result


__all__ = ["StabilityProbe"]
