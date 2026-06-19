"""Position-bias probe: symmetrized two-pass swap.

Every item is judged twice with the default template — once in original
order (``orig``: ``response_a`` presented first) and once swapped
(``swap``: ``response_b`` first). A judge free of position bias should
pick the presented-first response at rate 0.5 pooled over both passes,
and items decisive in both passes should rarely both land on the same
*presented* slot.

Metrics (contract §5):

* ``first_pick_rate`` — P(verdict == "first" | decisive), both passes
  pooled; null 0.5; cluster-bootstrap CI by item; p-value via bootstrap
  CI inversion; MDE via ``stats.mde_from_se`` on the realized bootstrap
  SE (consistent by construction with the reported CI).
* ``flip_rate_decisive`` — among items decisive in both passes, the
  fraction whose mapped (item-coordinates) winners disagree; Wilson CI.
* ``positional_mcnemar`` — b = items picking presented-first in both
  passes, c = presented-second in both; mid-p McNemar test; estimate
  ``b/(b+c)`` with null 0.5 and MDE via ``mde_mcnemar``.

Reference magnitudes (IJCNLP 2025 systematic study): median model flips
~44.8% of decisive swapped pairs; mean first-position pick rate 63.3%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from judgecal import stats
from judgecal.core import Estimate, JudgmentRequest, ProbeResult
from judgecal.probes.base import (
    Probe,
    ProbeConfig,
    cluster_rate_estimate,
    decisive_judgments,
    first_judgment_per,
    judgment_counts,
    own_judgments,
    register_probe,
    request_from_item,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import Judgment, PairwiseItem

_CONDITIONS = ("orig", "swap")


@register_probe
class PositionProbe(Probe):
    """Two-pass swap probe for position bias."""

    name = "position"

    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """Two requests per item: ``orig`` (A first) and ``swap`` (B first)."""
        requests: list[JudgmentRequest] = []
        for item in items:
            requests.append(
                request_from_item(item, probe=self.name, condition="orig", first_is_a=True)
            )
            requests.append(
                request_from_item(item, probe=self.name, condition="swap", first_is_a=False)
            )
        return requests

    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult:
        """Position-bias estimates from orig/swap judgments (meta-only)."""
        warnings: list[str] = []
        own = own_judgments(judgments, self.name)
        n_judgments, n_items, invalid_rate = judgment_counts(own)
        estimates: list[Estimate] = []

        if not own:
            return ProbeResult(
                probe=self.name,
                estimates=[],
                n_items=0,
                n_judgments=0,
                invalid_rate=0.0,
                warnings=["no judgments for probe 'position'"],
            )

        conditions = {j.condition for j in own}
        for cond in _CONDITIONS:
            if cond not in conditions:
                warnings.append(f"missing condition '{cond}'; paired metrics unavailable")

        decisive = decisive_judgments(own)

        # -- first_pick_rate (pooled over both passes) --------------------
        rows = [(j.item_id, 1.0 if j.verdict == "first" else 0.0) for j in decisive]
        est = cluster_rate_estimate(
            "first_pick_rate", rows, null=0.5, config=config, warnings=warnings
        )
        if est is not None:
            estimates.append(est)

        # -- paired metrics over items decisive in both passes ------------
        by_key = first_judgment_per(own)
        pairs: list[tuple[Judgment, Judgment]] = []
        for item_id in sorted({j.item_id for j in own}):
            orig = by_key.get((item_id, "orig"))
            swap = by_key.get((item_id, "swap"))
            if orig is not None and swap is not None and orig.is_decisive and swap.is_decisive:
                pairs.append((orig, swap))

        if not pairs:
            warnings.append("no items decisive in both passes; flip/McNemar metrics skipped")
        else:
            n_both = len(pairs)
            flips = sum(1 for o, s in pairs if o.mapped_verdict != s.mapped_verdict)
            lo, hi = stats.wilson_ci(flips, n_both, alpha=config.alpha)
            estimates.append(
                Estimate(
                    name="flip_rate_decisive",
                    estimate=flips / n_both,
                    ci_low=float(lo),
                    ci_high=float(hi),
                    n=n_both,
                    method="wilson",
                    detail={"n_items_both_decisive": n_both},
                )
            )

            b = sum(1 for o, s in pairs if o.verdict == "first" and s.verdict == "first")
            c = sum(1 for o, s in pairs if o.verdict == "second" and s.verdict == "second")
            if b + c == 0:
                warnings.append(
                    "no positionally inconsistent pairs (b + c == 0); McNemar metric skipped"
                )
            else:
                mc = stats.mcnemar_test(b, c)
                lo, hi = stats.wilson_ci(b, b + c, alpha=config.alpha)
                estimates.append(
                    Estimate(
                        name="positional_mcnemar",
                        estimate=b / (b + c),
                        ci_low=float(lo),
                        ci_high=float(hi),
                        n=b + c,
                        method=f"mcnemar_{mc.method}",
                        null_value=0.5,
                        p_value=float(mc.p_value),
                        mde=stats.mde_mcnemar(b + c, alpha=config.alpha),
                        detail={"b": b, "c": c, "n_items_both_decisive": n_both},
                    )
                )

        return ProbeResult(
            probe=self.name,
            estimates=estimates,
            n_items=n_items,
            n_judgments=n_judgments,
            invalid_rate=invalid_rate,
            warnings=warnings,
            meta={"conditions": sorted(conditions)},
        )


__all__ = ["PositionProbe"]
