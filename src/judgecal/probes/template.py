"""Template-sensitivity probe: same items under paraphrased judge prompts.

Each item is judged once per prompt template (``tpl:default``,
``tpl:v1`` .. — semantically equivalent paraphrases shipped in
:mod:`judgecal.probes.templates`), original presentation order only.
A reliable judge should produce the same verdicts regardless of which
paraphrase carries the instructions.

Metrics (contract §5):

* ``template_fleiss_kappa`` — Fleiss' kappa over the items x
  {first, second, tie} table, templates as raters; item-bootstrap CI;
  no null / p-value (agreement is descriptive).
* ``template_max_flip`` — maximum over template pairs of the decisive
  disagreement rate; Wilson CI on the argmax pair, with the multiplicity
  caveat recorded in ``detail`` (the max over pairs is selection-biased;
  the CI is conditional on the selected pair).
* ``template_accuracy_range`` — when ground-truth labels exist,
  max minus min accuracy across templates, cluster-bootstrap CI.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

import pandas as pd

from judgecal import stats
from judgecal.core import Estimate, JudgmentRequest, ProbeResult
from judgecal.probes.base import (
    MIN_BOOT_CLUSTERS,
    Probe,
    ProbeConfig,
    first_judgment_per,
    judgment_counts,
    kappa_estimate,
    own_judgments,
    register_probe,
    request_from_item,
)
from judgecal.probes.templates import template_ids_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import Judgment, PairwiseItem


@register_probe
class TemplateProbe(Probe):
    """Prompt-paraphrase sensitivity probe."""

    name = "template"

    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """One request per (item, template), orig order only.

        The condition string *is* the template id (``tpl:...``), which is
        also what the mock judge keys its per-template offset on.

        Raises:
            ValueError: If ``config.n_template_variants`` exceeds the
                number of shipped templates.
        """
        template_ids = template_ids_for(config.n_template_variants)
        requests: list[JudgmentRequest] = []
        for item in items:
            for template_id in template_ids:
                requests.append(
                    request_from_item(
                        item,
                        probe=self.name,
                        condition=template_id,
                        first_is_a=True,
                        template_id=template_id,
                    )
                )
        return requests

    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult:
        """Agreement and accuracy-spread estimates across templates."""
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
            warnings.append("no judgments for probe 'template'")
            return result

        observed = sorted({j.condition for j in own})
        expected = template_ids_for(config.n_template_variants)
        for template_id in expected:
            if template_id not in observed:
                warnings.append(f"missing condition '{template_id}'")
        if len(observed) < 2:
            warnings.append("fewer than 2 template conditions; agreement metrics skipped")
            return result

        # Per-item parseable verdict per template (first judgment wins).
        by_key = first_judgment_per(own)
        verdicts: dict[str, dict[str, Judgment]] = {}
        for (item_id, condition), j in by_key.items():
            if condition in observed and j.verdict != "invalid":
                verdicts.setdefault(item_id, {})[condition] = j

        # -- template_fleiss_kappa (items rated by ALL observed templates) --
        complete = {item_id: per for item_id, per in verdicts.items() if len(per) == len(observed)}
        n_dropped = len(verdicts) - len(complete)
        if n_dropped:
            warnings.append(
                f"template_fleiss_kappa: {n_dropped} item(s) lacking parseable verdicts "
                "under every template excluded"
            )
        kappa_rows = [
            (item_id, j.verdict) for item_id, per in complete.items() for j in per.values()
        ]
        est = kappa_estimate("template_fleiss_kappa", kappa_rows, config=config, warnings=warnings)
        if est is not None:
            estimates.append(est)

        # -- template_max_flip (max decisive disagreement over pairs) ------
        best: tuple[float, int, int, tuple[str, str]] | None = None
        n_pairs_compared = 0
        for t1, t2 in combinations(observed, 2):
            n_both = 0
            n_flip = 0
            for per in verdicts.values():
                j1, j2 = per.get(t1), per.get(t2)
                if j1 is None or j2 is None or not (j1.is_decisive and j2.is_decisive):
                    continue
                n_both += 1
                if j1.mapped_verdict != j2.mapped_verdict:
                    n_flip += 1
            if n_both == 0:
                continue
            n_pairs_compared += 1
            rate = n_flip / n_both
            if best is None or rate > best[0]:
                best = (rate, n_flip, n_both, (t1, t2))
        if best is None:
            warnings.append("template_max_flip: no template pair with shared decisive items")
        else:
            rate, n_flip, n_both, pair = best
            lo, hi = stats.wilson_ci(n_flip, n_both, alpha=config.alpha)
            estimates.append(
                Estimate(
                    name="template_max_flip",
                    estimate=rate,
                    ci_low=float(lo),
                    ci_high=float(hi),
                    n=n_both,
                    method="wilson_max_pair",
                    detail={
                        "pair": list(pair),
                        "n_pairs_compared": n_pairs_compared,
                        "multiplicity_note": (
                            "maximum over template pairs; the Wilson CI is conditional "
                            "on the selected pair and not adjusted for selection"
                        ),
                    },
                )
            )

        # -- template_accuracy_range (only when labels exist) --------------
        acc = self._accuracy_range(complete, observed, config, warnings)
        if acc is not None:
            estimates.append(acc)
        return result

    @staticmethod
    def _accuracy_range(
        complete: dict[str, dict[str, Judgment]],
        observed: list[str],
        config: ProbeConfig,
        warnings: list[str],
    ) -> Estimate | None:
        """Max minus min accuracy across templates (labeled items only)."""
        records: list[dict[str, Any]] = []
        for item_id, per in complete.items():
            labels = {j.meta.get("label_first") for j in per.values()}
            label = labels.pop() if len(labels) == 1 else None
            if label is None:
                continue
            for condition, j in per.items():
                records.append(
                    {
                        "item_id": item_id,
                        "condition": condition,
                        "correct": 1.0 if j.verdict == label else 0.0,
                    }
                )
        if not records:
            return None  # no ground-truth labels: metric does not apply
        df = pd.DataFrame(records)
        if df["item_id"].nunique() < MIN_BOOT_CLUSTERS:
            warnings.append("template_accuracy_range: too few labeled items; skipped")
            return None

        def acc_range(d: pd.DataFrame) -> float:
            per_template = d.groupby("condition")["correct"].mean()
            return float(per_template.max() - per_template.min())

        try:
            res = stats.cluster_bootstrap_ci(
                df,
                acc_range,
                "item_id",
                n_boot=config.n_boot,
                alpha=config.alpha,
                seed=config.seed,
            )
        except ValueError as exc:
            warnings.append(f"template_accuracy_range: bootstrap failed ({exc}); skipped")
            return None
        for caveat in res.warnings:
            warnings.append(f"template_accuracy_range: {caveat}")
        per_template = df.groupby("condition")["correct"].mean()
        return Estimate(
            name="template_accuracy_range",
            estimate=float(res.estimate),
            ci_low=float(res.ci_low),
            ci_high=float(res.ci_high),
            n=len(df),
            method="cluster_bootstrap",
            detail={
                "n_labeled_items": int(df["item_id"].nunique()),
                "accuracy_by_template": {k: float(v) for k, v in per_template.items()},
            },
        )


__all__ = ["TemplateProbe"]
