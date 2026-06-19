"""Verbosity-bias probe: constructed pad contrast + observational GLM.

**Constructed contrast.** From each item, ``response_a`` is paired
against a deterministically padded copy of itself
(:func:`judgecal.probes.padding.pad_text`, rule-based restatement +
recap reaching ``config.pad_target_ratio`` — NOT an LLM rewrite; see the
padding module for the documented limitation). Both presentation orders
are planned (``pad_second``: original first; ``pad_first``: padded
first) so position effects cancel in the pooled pick rate. Because both
sides carry identical content, ``label_first`` is planted as ``"tie"``
and both latent qualities equal the item's ``latent_quality_a`` — a
quality-driven judge has zero signal, leaving only length.

* ``pad_pick_rate`` — P(pick the padded side | decisive), pooled over
  both orders; null 0.5; cluster-bootstrap CI by item; p via bootstrap
  inversion; MDE via ``stats.mde_from_se`` on the realized bootstrap SE.

**Observational GLM.** Over the *position* probe's judgments (declared
via ``requires``; ``analyze_suite`` passes the union), a logistic
regression of pick-first on ``log(first_len / second_len)``, plus a
ground-truth-first control when every judgment carries a label. The CI
comes from cluster-bootstrap re-fitting (never Wald SEs).

* ``length_glm_coef`` — coefficient on the log length ratio; null 0.0.
  Method ``"cluster_bootstrap_logit (observational)"``: this is an
  **observational association, not a causal probe** — when length
  correlates with answer quality (the normal case in real data) the
  coarse label control (``gt_first ∈ {1, 0, 0.5}``) does not absorb the
  continuous quality gap and the coefficient picks up quality, not
  verbosity (measured: a provably length-blind judge, ``beta_length=0``,
  ``beta_quality=3``, with length-quality-correlated items reads
  ``est=1.99, CI=[1.07, 3.02], p=0.002`` against a truth of 0). The
  ``pad_pick_rate`` contrast is the experimental estimate; the bias flag
  gates only on it. The metric stays in the card's BH-FDR family because
  it carries a null and is reported with a p-value (excluding reported
  hypotheses from the family would understate multiplicity); the card
  renders its verdict cell as "obs." (never a rejected/clear glyph) with
  a footnote.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from judgecal import stats
from judgecal.core import Estimate, JudgmentRequest, ProbeResult
from judgecal.probes.base import (
    Probe,
    ProbeConfig,
    cluster_rate_estimate,
    decisive_judgments,
    judgment_counts,
    make_request,
    mde_from_se,
    own_judgments,
    register_probe,
)
from judgecal.probes.padding import pad_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from judgecal.core import Judgment, PairwiseItem

_CONDITIONS = ("pad_second", "pad_first")

#: label_first -> numeric ground-truth-first control value for the GLM.
_GT_VALUE = {"first": 1.0, "second": 0.0, "tie": 0.5}

#: Bootstrap SE at or below this is treated as degenerate (replicates
#: identical to numerical precision); a logit-scale SE this small is
#: never a real measure of uncertainty, so the MDE is suppressed.
_DEGENERATE_SE = 1e-8


@register_probe
class VerbosityProbe(Probe):
    """Pad contrast plus observational length GLM."""

    name = "verbosity"
    requires = ("position",)

    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]:
        """Two requests per item: original vs padded copy, both orders.

        ``first_is_a`` is ``True`` for both conditions by convention —
        both presented texts derive from ``response_a``, so item
        coordinates do not distinguish them; analyses identify the padded
        side from the condition name.
        """
        requests: list[JudgmentRequest] = []
        for item in items:
            base = item.response_a
            padded = pad_text(base, config.pad_target_ratio)
            quality = item.meta.get("latent_quality_a")
            for condition, first_text, second_text in (
                ("pad_second", base, padded),
                ("pad_first", padded, base),
            ):
                requests.append(
                    make_request(
                        probe=self.name,
                        condition=condition,
                        item_id=item.item_id,
                        prompt=item.prompt,
                        first_text=first_text,
                        second_text=second_text,
                        first_is_a=True,
                        first_author=item.author_a,
                        second_author=item.author_a,
                        first_latent_q=quality,
                        second_latent_q=quality,
                        label_first="tie",
                    )
                )
        return requests

    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult:
        """Pad pick rate from own judgments; length GLM from position's."""
        warnings: list[str] = []
        own = own_judgments(judgments, self.name)
        position = own_judgments(judgments, "position")
        n_judgments, n_items, invalid_rate = judgment_counts(own)
        estimates: list[Estimate] = []

        # -- pad_pick_rate -------------------------------------------------
        if not own:
            warnings.append("no judgments for probe 'verbosity'; pad_pick_rate skipped")
        else:
            conditions = {j.condition for j in own}
            for cond in _CONDITIONS:
                if cond not in conditions:
                    warnings.append(f"missing condition '{cond}'; pad rate covers one order only")
            rows: list[tuple[str, float]] = []
            for j in decisive_judgments(own):
                if j.condition not in _CONDITIONS:
                    continue
                padded_side = "first" if j.condition == "pad_first" else "second"
                rows.append((j.item_id, 1.0 if j.verdict == padded_side else 0.0))
            est = cluster_rate_estimate(
                "pad_pick_rate", rows, null=0.5, config=config, warnings=warnings
            )
            if est is not None:
                estimates.append(est)

        # -- length_glm_coef (observational, over position judgments) ------
        glm = self._length_glm(position, config, warnings)
        if glm is not None:
            estimates.append(glm)

        return ProbeResult(
            probe=self.name,
            estimates=estimates,
            n_items=n_items,
            n_judgments=n_judgments,
            invalid_rate=invalid_rate,
            warnings=warnings,
            meta={"conditions": sorted({j.condition for j in own})},
        )

    @staticmethod
    def _length_glm(
        position: list[Judgment], config: ProbeConfig, warnings: list[str]
    ) -> Estimate | None:
        """Logistic fit of pick-first on log length ratio (+ label control)."""
        if not position:
            warnings.append("no position judgments available; length_glm_coef skipped")
            return None

        records: list[dict[str, Any]] = []
        skipped_len = 0
        for j in decisive_judgments(position):
            first_len, second_len = j.meta["first_len"], j.meta["second_len"]
            if first_len <= 0 or second_len <= 0:
                skipped_len += 1
                continue
            records.append(
                {
                    "item_id": j.item_id,
                    "y": 1.0 if j.verdict == "first" else 0.0,
                    "x_loglen": math.log(first_len / second_len),
                    "label_first": j.meta.get("label_first"),
                }
            )
        if skipped_len:
            warnings.append(
                f"length_glm_coef: {skipped_len} judgment(s) with non-positive lengths excluded"
            )
        if not records:
            warnings.append("length_glm_coef: no decisive position judgments; skipped")
            return None

        df = pd.DataFrame(records)
        if df["y"].nunique() < 2:
            warnings.append("length_glm_coef: no variation in pick-first; GLM skipped")
            return None
        if df["x_loglen"].nunique() < 2:
            warnings.append("length_glm_coef: no variation in length ratio; GLM skipped")
            return None
        if df["item_id"].nunique() < 3:
            warnings.append("length_glm_coef: too few items for a bootstrap CI; skipped")
            return None

        labels = df["label_first"]
        feature_cols = ["const", "x_loglen"]
        if labels.notna().all():
            df["gt_first"] = labels.map(_GT_VALUE)
            if df["gt_first"].nunique() > 1:
                feature_cols.append("gt_first")
        elif labels.notna().any():
            warnings.append("length_glm_coef: partial ground-truth labels; control column omitted")
        df["const"] = 1.0
        fit_df = df[["item_id", "y", *feature_cols]]

        def coef_loglen(d: pd.DataFrame) -> float:
            x = d[feature_cols].to_numpy(dtype=float)
            y = d["y"].to_numpy(dtype=float)
            try:
                fit = stats.logistic_fit(x, y, add_intercept=False)
            except (ValueError, np.linalg.LinAlgError):
                return float("nan")
            coef = float(np.asarray(fit.coef, dtype=float)[feature_cols.index("x_loglen")])
            return coef if math.isfinite(coef) else float("nan")

        point_fit = stats.logistic_fit(
            fit_df[feature_cols].to_numpy(dtype=float),
            fit_df["y"].to_numpy(dtype=float),
            add_intercept=False,
        )
        if not point_fit.converged:
            warnings.append(
                "length_glm_coef: IRLS did not converge (possible separation); "
                "interpret with caution"
            )

        try:
            res = stats.cluster_bootstrap_ci(
                fit_df,
                coef_loglen,
                "item_id",
                n_boot=config.n_boot,
                alpha=config.alpha,
                seed=config.seed,
                null_value=0.0,
            )
        except ValueError as exc:
            warnings.append(f"length_glm_coef: bootstrap failed ({exc}); metric skipped")
            return None
        for caveat in res.warnings:
            warnings.append(f"length_glm_coef: {caveat}")

        # MDE is suppressed when the point fit did not converge or the
        # bootstrap SE is degenerate (all replicates ~identical, e.g. a
        # separated fit returning ~0 every time): a near-zero SE would
        # otherwise imply absurd power (MDE ~1e-12 observed on n=5).
        boot_se = float(res.boot_se)
        degenerate_se = not math.isfinite(boot_se) or boot_se <= _DEGENERATE_SE
        if degenerate_se:
            warnings.append(
                "length_glm_coef: bootstrap SE degenerate (replicates nearly "
                "identical); MDE suppressed"
            )
        mde = (
            None
            if (not point_fit.converged or degenerate_se)
            else mde_from_se(boot_se, config.alpha)
        )

        return Estimate(
            name="length_glm_coef",
            estimate=float(res.estimate),
            ci_low=float(res.ci_low),
            ci_high=float(res.ci_high),
            n=len(fit_df),
            method="cluster_bootstrap_logit (observational)",
            null_value=0.0,
            p_value=None if res.p_value is None else float(res.p_value),
            mde=mde,
            detail={
                "n_clusters": int(fit_df["item_id"].nunique()),
                "controls": feature_cols[2:],
                "converged": bool(point_fit.converged),
                "boot_se": boot_se,
            },
        )


__all__ = ["VerbosityProbe"]
