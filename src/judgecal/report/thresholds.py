"""Flagging thresholds for the reliability card.

Every constant here is a **reporting convention, not a truth**. The flags
exist so a reader can scan a card in ten seconds; they are deliberately
simple cutoffs on continuous quantities, and reasonable people could pick
different values. All raw estimates, CIs, p/q-values and MDEs are always
printed alongside the flags so nothing is hidden behind these choices.

Conventions were fixed before any study runs (pre-registered) and must not
be tuned after looking at results.
"""

from __future__ import annotations

#: q-value cutoff for calling a null-tested metric "significant".
#: Convention: the standard 5% level, applied to BH-FDR q-values (one
#: family per card — see ``build_card``). Not a truth about bias.
FLAG_Q_THRESHOLD: float = 0.05

#: Minimum |first_pick_rate - 0.5| for ``position_bias_detected``.
#: Convention: a 5-percentage-point lean is where position bias starts to
#: matter for eval rankings in practice; smaller significant effects are
#: reported but not flagged.
POSITION_BIAS_MIN_EFFECT: float = 0.05

#: Minimum |pad_pick_rate - 0.5| for ``verbosity_bias_detected``.
#: Convention: same 5-pp practical-relevance floor as position.
VERBOSITY_BIAS_MIN_EFFECT: float = 0.05

#: Minimum |self_error_pick_excess - 0.0| for ``self_preference_detected``.
#: Convention: a 5-pp excess of picking one's own *wrong* answer over the
#: control rate (an unadjusted observational contrast — the flag is also
#: suppressed when the probe's composition diagnostic warns that the
#: self/control sets differ; see ``probes/self_preference.py``).
SELF_PREFERENCE_MIN_EFFECT: float = 0.05

#: ``template_sensitivity_high`` fires when the *upper bound* of the
#: template_fleiss_kappa CI falls below this floor (the point estimate
#: alone conflates template effects with sampling noise). Convention:
#: 0.6 is the usual "substantial agreement" boundary (Landis & Koch);
#: below it, paraphrased prompts meaningfully change verdicts.
TEMPLATE_KAPPA_FLOOR: float = 0.6

#: ``template_sensitivity_high`` also fires when template_max_flip exceeds
#: this ceiling. Convention: if the worst template pair flips more than
#: one decisive verdict in five, the instrument is template-sensitive.
TEMPLATE_MAX_FLIP_CEILING: float = 0.20

#: Template disagreement = template effects + per-call verdict
#: stochasticity; the agreement metrics alone cannot separate them (a
#: temperature>0 or nondeterministically served judge depresses kappa
#: with zero template effect). When the same card carries a stability
#: probe (identical bodies repeated — the same judge's pure repeat
#: noise), ``template_sensitivity_high`` additionally requires the
#: template kappa point estimate to sit at least this margin *below* the
#: stability kappa point estimate: template disagreement must exceed
#: repeat noise before it is attributed to templates. Without stability
#: data the flag falls back to the kappa-CI/max-flip rule alone and the
#: rendered summary carries an explicit caveat. Convention, not a truth.
TEMPLATE_STABILITY_KAPPA_MARGIN: float = 0.05

#: Effect sizes of interest per null-tested metric — the *design-based*
#: anchor for power adequacy (reporting conventions fixed before any
#: study runs; never compare an MDE to the observed estimate, which is
#: the post-hoc-power anti-pattern, Hoenig & Heisey 2001).
#:
#: ``underpowered:<metric>`` fires iff the metric is not significant AND
#: its MDE exceeds this floor: the audit could not have detected the
#: smallest effect the tool itself considers practically relevant. The
#: "no signal at adequate power" verdict requires not-significant AND
#: MDE <= floor. Scales:
#:
#: * pick rates (null 0.5): 0.05 — the same 5-pp practical-relevance
#:   floor as the bias flags (POSITION/VERBOSITY_BIAS_MIN_EFFECT).
#: * discordant-pair proportion (positional_mcnemar, null 0.5): 0.05.
#: * excess rates (null 0.0): 0.05 — matches SELF_PREFERENCE_MIN_EFFECT.
#: * logistic coefficients (log-odds per unit log length ratio): 0.5 —
#:   a half-log-odds-per-doubling-scale slope is where length effects
#:   start to move verdicts materially.
#: * kappa-like / descriptive metrics: no entry — they carry no null, so
#:   power adequacy is not defined for them.
MIN_EFFECT_OF_INTEREST: dict[str, float] = {
    "first_pick_rate": 0.05,
    "positional_mcnemar": 0.05,
    "pad_pick_rate": 0.05,
    "self_error_pick_excess": 0.05,
    "length_glm_coef": 0.5,
}

#: ``instability_high`` fires when unanimity_rate falls below this floor.
#: Convention: if fewer than 80% of items get identical verdicts across
#: identical repeated runs, run-to-run noise is large enough to matter.
STABILITY_UNANIMITY_FLOOR: float = 0.80

#: ``high_invalid_rate`` fires when a probe's invalid (unparseable)
#: judgment fraction exceeds this. Convention: above 5%, exclusion of
#: invalid judgments could plausibly distort estimates.
HIGH_INVALID_RATE_THRESHOLD: float = 0.05

__all__ = [
    "FLAG_Q_THRESHOLD",
    "HIGH_INVALID_RATE_THRESHOLD",
    "MIN_EFFECT_OF_INTEREST",
    "POSITION_BIAS_MIN_EFFECT",
    "SELF_PREFERENCE_MIN_EFFECT",
    "STABILITY_UNANIMITY_FLOOR",
    "TEMPLATE_KAPPA_FLOOR",
    "TEMPLATE_MAX_FLIP_CEILING",
    "TEMPLATE_STABILITY_KAPPA_MARGIN",
    "VERBOSITY_BIAS_MIN_EFFECT",
]
