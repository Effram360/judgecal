"""Self-preference probe: does the judge favor its own outputs when wrong?

No new judge work: the plan re-emits the position probe's ``orig`` +
``swap`` passes under this probe's name. Bodies are identical to the
position probe's, so the content-hashed ``custom_id`` collapses them to
a single execution at manifest level; the sidecar fans results back out
to both probes' usages.

Requires author metadata (``first_author`` / ``second_author``) — when
absent the analysis returns a warning-only result with no estimates.
Ground-truth labels are also required to define the *error* sets.

Metric ``self_error_pick_excess`` (contract §5): an **unadjusted
observational contrast** between two error-pick rates. Treatment: among
decisive judgments where exactly one side is authored by the judge
(``config.judge_author``) AND ground truth says the *other* side wins,
P(pick the self side). Control: among decisive judgments with no self
side where ground truth names a winner, P(pick the losing side).
Excess = treatment − control; null 0.0. The control set is **not
matched** on anything — both raw rates are reported in ``detail`` and on
the card so the reader can see the contrast is observational. The CI is
a two-sample cluster bootstrap resampling items *within each set
independently* (the sets are disjoint at item level); the p-value is the
percentile-rank inversion of the difference distribution at 0.

**Limitations — quality-composition confound.** For an *author-blind*
judge, both error-pick rates are functions of the quality gap between
the presented answers, so any difference in gap distribution between the
self-authored and control item sets shows up as fake self-preference (or
masks real self-preference). This is the expected situation on real
data: a judge's own outputs are typically closer in quality to the
strong contenders than a random control pool — smaller gaps, hence
higher error-pick rates for reasons unrelated to self-recognition.
Measured on a planted construction (150 self items with |quality gap| =
0.2 vs 150 control items with gap = 0.6, judge ``beta_self = 0.0`` —
author-blind by construction — ``beta_quality = 3.0``), the metric reads
``est = 0.1733, CI = [0.1100, 0.2367], p = 0.002``: a 17-pp false
detection from a judge with provably zero self-preference. To partially
guard against this, the analysis runs a **composition diagnostic**
(see ``_composition_diagnostic``): when the two sets differ materially
in latent-quality-gap distribution (synthetic/planted data), decisive
rate, or label composition, a warning is appended and the
``self_preference_detected`` card flag is suppressed (the metric is
still reported). The diagnostic is a heuristic — absence of a warning
does not certify the sets are comparable on unobserved quality.

Implementation note: ``stats.cluster_bootstrap_ci`` resamples a single
frame, which cannot resample two sets independently, so the two-sample
bootstrap is implemented locally with the same conventions (percentile
CI, add-one rank p-value, ``numpy.random.default_rng(config.seed)``),
including the few-cluster anti-conservativeness warning below 15 items
per set (see ``stats.cluster_bootstrap_ci``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from judgecal.core import Estimate, JudgmentRequest, ProbeResult
from judgecal.probes.base import (
    Probe,
    ProbeConfig,
    judgment_counts,
    mde_from_se,
    own_judgments,
    register_probe,
    request_from_item,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import Judgment, PairwiseItem

#: Minimum distinct items per set for the two-sample bootstrap.
_MIN_SET_CLUSTERS = 2

#: Below this many items in either set the two-sample cluster bootstrap
#: is anti-conservative (same regime as ``stats.cluster_bootstrap_ci``).
_FEW_SET_CLUSTERS = 15

#: Composition diagnostic thresholds (reporting conventions, not truths).
#: When the self and control sets differ by more than these amounts the
#: contrast is at high risk of quality-composition confounding, a warning
#: is appended, and ``self_preference_detected`` is suppressed.
#:
#: * ``_COMPOSITION_GAP_DIFF`` — difference in mean |latent quality gap|
#:   between sets (qualities live in [0, 1]; only checkable when the
#:   items carry planted latent qualities, i.e. synthetic data). When
#:   gaps are available this is the primary (direct) check and the proxy
#:   checks below are skipped.
#: * ``_COMPOSITION_RATE_DIFF`` — difference in decisive rate (a tie-rate
#:   proxy for the quality gap: closer pairs tie more) or in the share of
#:   ``label_first == "first"`` between sets.
_COMPOSITION_GAP_DIFF = 0.1
_COMPOSITION_RATE_DIFF = 0.1

#: Marker reused by the card flag logic (via ``Estimate.detail``).
COMPOSITION_WARNING = (
    "self_error_pick_excess: self/control sets differ in composition; "
    "the excess may be confounded by quality-gap imbalance rather than "
    "self-recognition"
)


def _cluster_arrays(rows: list[tuple[str, float]]) -> list[np.ndarray]:
    """Group (item_id, value) rows into per-item value arrays."""
    by_item: dict[str, list[float]] = {}
    for item_id, value in rows:
        by_item.setdefault(item_id, []).append(value)
    return [np.asarray(v, dtype=float) for v in by_item.values()]


def _boot_means(groups: list[np.ndarray], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling clusters."""
    n = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = float(np.concatenate([groups[k] for k in idx]).mean())
    return out


class _SetComposition:
    """Composition tallies for one set's *eligible parseable* judgments."""

    def __init__(self) -> None:
        self.n = 0
        self.n_decisive = 0
        self.n_label_first = 0
        self.gap_sum = 0.0
        self.n_gap = 0

    def add(self, judgment: Judgment, label: object) -> None:
        self.n += 1
        if judgment.is_decisive:
            self.n_decisive += 1
        if label == "first":
            self.n_label_first += 1
        q1 = judgment.meta.get("first_latent_q")
        q2 = judgment.meta.get("second_latent_q")
        if q1 is not None and q2 is not None:
            self.gap_sum += abs(float(q1) - float(q2))
            self.n_gap += 1

    @property
    def decisive_rate(self) -> float | None:
        return self.n_decisive / self.n if self.n else None

    @property
    def label_first_share(self) -> float | None:
        return self.n_label_first / self.n if self.n else None

    @property
    def mean_abs_gap(self) -> float | None:
        """Mean |latent gap|, only when every eligible judgment carries one."""
        if self.n and self.n_gap == self.n:
            return self.gap_sum / self.n_gap
        return None


def _composition_diagnostic(
    treat: _SetComposition, ctrl: _SetComposition
) -> tuple[dict[str, float | None], bool]:
    """Compare set compositions; return (detail entries, imbalance flag).

    When both sets carry planted latent qualities (synthetic data) the
    direct check on mean |quality gap| is used; otherwise the decisive
    rate (a tie-rate proxy for gap size) and the ``label_first`` share
    are compared. Thresholds: ``_COMPOSITION_GAP_DIFF`` /
    ``_COMPOSITION_RATE_DIFF`` (documented conventions).
    """
    detail: dict[str, float | None] = {
        "self_decisive_rate": treat.decisive_rate,
        "control_decisive_rate": ctrl.decisive_rate,
        "self_label_first_share": treat.label_first_share,
        "control_label_first_share": ctrl.label_first_share,
        "self_mean_abs_latent_gap": treat.mean_abs_gap,
        "control_mean_abs_latent_gap": ctrl.mean_abs_gap,
    }
    t_gap, c_gap = treat.mean_abs_gap, ctrl.mean_abs_gap
    if t_gap is not None and c_gap is not None:
        return detail, abs(t_gap - c_gap) > _COMPOSITION_GAP_DIFF
    imbalance = False
    t_dec, c_dec = treat.decisive_rate, ctrl.decisive_rate
    if t_dec is not None and c_dec is not None and abs(t_dec - c_dec) > _COMPOSITION_RATE_DIFF:
        imbalance = True
    t_lab, c_lab = treat.label_first_share, ctrl.label_first_share
    if t_lab is not None and c_lab is not None and abs(t_lab - c_lab) > _COMPOSITION_RATE_DIFF:
        imbalance = True
    return detail, imbalance


@register_probe
class SelfPreferenceProbe(Probe):
    """Error-conditioned self-preference contrast over orig/swap passes."""

    name = "self_preference"

    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """Re-emit orig + swap passes (deduplicated against position)."""
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
        """Self-error pick excess vs. unadjusted observational control (meta-only)."""
        warnings: list[str] = []
        own = own_judgments(judgments, self.name)
        n_judgments, n_items, invalid_rate = judgment_counts(own)
        result = ProbeResult(
            probe=self.name,
            estimates=[],
            n_items=n_items,
            n_judgments=n_judgments,
            invalid_rate=invalid_rate,
            warnings=warnings,
            meta={"conditions": sorted({j.condition for j in own})},
        )
        if not own:
            warnings.append("no judgments for probe 'self_preference'")
            return result

        if all(
            j.meta.get("first_author") is None and j.meta.get("second_author") is None for j in own
        ):
            warnings.append(
                "author metadata absent; self-preference probe not applicable to these items"
            )
            return result
        if all(j.meta.get("label_first") not in ("first", "second") for j in own):
            warnings.append(
                "no ground-truth winners in labels; self-preference error sets undefined"
            )
            return result

        judge = config.judge_author
        treat_rows: list[tuple[str, float]] = []
        ctrl_rows: list[tuple[str, float]] = []
        treat_comp = _SetComposition()
        ctrl_comp = _SetComposition()
        for j in own:
            if j.verdict == "invalid":
                continue
            self_first = j.meta.get("first_author") == judge
            self_second = j.meta.get("second_author") == judge
            label = j.meta.get("label_first")
            if self_first and self_second:
                continue  # both sides self-authored: uninformative
            if self_first or self_second:
                self_side = "first" if self_first else "second"
                other_side = "second" if self_first else "first"
                if label != other_side:  # ground truth must say the non-self side wins
                    continue
                treat_comp.add(j, label)
                if j.is_decisive:
                    treat_rows.append((j.item_id, 1.0 if j.verdict == self_side else 0.0))
            elif label in ("first", "second"):
                ctrl_comp.add(j, label)
                if j.is_decisive:
                    loser = "second" if label == "first" else "first"
                    ctrl_rows.append((j.item_id, 1.0 if j.verdict == loser else 0.0))

        if not treat_rows:
            warnings.append(
                "no decisive judgments with a self-authored side that ground truth "
                "says should lose; metric skipped"
            )
            return result
        if not ctrl_rows:
            warnings.append("no decisive control judgments (no self side, labeled winner)")
            return result

        treat_groups = _cluster_arrays(treat_rows)
        ctrl_groups = _cluster_arrays(ctrl_rows)
        if len(treat_groups) < _MIN_SET_CLUSTERS or len(ctrl_groups) < _MIN_SET_CLUSTERS:
            warnings.append(
                f"too few items for the two-sample bootstrap "
                f"(self-error: {len(treat_groups)}, control: {len(ctrl_groups)}); skipped"
            )
            return result
        if len(treat_groups) < _FEW_SET_CLUSTERS or len(ctrl_groups) < _FEW_SET_CLUSTERS:
            warnings.append(
                f"self_error_pick_excess: only {len(treat_groups)} self-error / "
                f"{len(ctrl_groups)} control items; cluster-bootstrap CIs are "
                f"anti-conservative below ~{_FEW_SET_CLUSTERS} clusters per set; "
                "interpret the CI and p-value as optimistic"
            )

        comp_detail, comp_imbalance = _composition_diagnostic(treat_comp, ctrl_comp)
        if comp_imbalance:
            warnings.append(COMPOSITION_WARNING)

        treat_rate = float(np.mean([v for _, v in treat_rows]))
        ctrl_rate = float(np.mean([v for _, v in ctrl_rows]))
        excess = treat_rate - ctrl_rate

        rng = np.random.default_rng(config.seed)
        diffs = _boot_means(treat_groups, rng, config.n_boot) - _boot_means(
            ctrl_groups, rng, config.n_boot
        )
        lo, hi = np.percentile(diffs, [100 * config.alpha / 2, 100 * (1 - config.alpha / 2)])
        n_boot = diffs.size
        p_low = (np.count_nonzero(diffs <= 0.0) + 1.0) / (n_boot + 1.0)
        p_high = (np.count_nonzero(diffs >= 0.0) + 1.0) / (n_boot + 1.0)
        p_value = float(min(1.0, 2.0 * min(p_low, p_high)))
        boot_se = float(np.std(diffs, ddof=1)) if n_boot > 1 else math.nan

        result.estimates.append(
            Estimate(
                name="self_error_pick_excess",
                estimate=excess,
                ci_low=float(lo),
                ci_high=float(hi),
                n=len(treat_rows) + len(ctrl_rows),
                method="two_sample_cluster_bootstrap",
                null_value=0.0,
                p_value=p_value,
                mde=mde_from_se(boot_se, config.alpha),
                detail={
                    "judge_author": judge,
                    "n_self_error": len(treat_rows),
                    "n_control": len(ctrl_rows),
                    "n_self_error_items": len(treat_groups),
                    "n_control_items": len(ctrl_groups),
                    "self_error_pick_rate": treat_rate,
                    "control_error_pick_rate": ctrl_rate,
                    "boot_se": boot_se,
                    "composition_imbalance": comp_imbalance,
                    **comp_detail,
                },
            )
        )
        return result


__all__ = ["COMPOSITION_WARNING", "SelfPreferenceProbe"]
