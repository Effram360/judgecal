"""Deterministic synthetic pairwise items with planted latent qualities.

The generator produces :class:`~judgecal.core.PairwiseItem` objects whose
*latent qualities* are known by construction (stored in ``item.meta``), so
the mock judge (`judgecal.fixtures.mock_judge`) can score them with planted
biases and the validation suite can compare probe estimates against exact
analytic truths.

Response bodies are template-generated filler text whose ``len()`` hits a
sampled target length exactly — content semantics never matter to the mock
judge (it reads only request metadata), but realistic length variation is
required for the verbosity machinery.

Determinism: everything is a pure function of the config (including the
seed); a single ``numpy.random.Generator`` seeded with ``config.seed``
drives all draws in a fixed order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judgecal.core import Label, PairwiseItem

#: Tie-label band on the *effective* latent-quality gap: |q_a - q_b| < 0.1
#: labels the item "tie" (contract §3.1).
TIE_GAP: float = 0.1

#: Median response length in characters (log-lengths vary around this).
_BASE_LEN: int = 400

#: Floor on response length so every response holds at least one sentence.
_MIN_LEN: int = 60

#: Sentence bank for filler responses. Must never contain "[[" so synthetic
#: bodies can never be mistaken for verdict markers.
_SENTENCES: tuple[str, ...] = (
    "The approach balances clarity with depth across each major consideration.",
    "Key trade-offs include cost, robustness, and long-term ease of maintenance.",
    "A staged rollout limits risk while preserving room for later optimization.",
    "Edge cases around concurrency deserve explicit tests before deployment.",
    "Resource usage stays modest because the hot path avoids redundant work.",
    "Documentation of the failure modes makes the design easier to audit.",
    "The alternative design trades simplicity for a marginal throughput gain.",
    "Monitoring hooks expose the relevant counters without extra overhead.",
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the synthetic item generator.

    Attributes:
        n_items: Number of pairwise items to generate.
        seed: Seed for the single ``numpy.random.Generator`` driving all
            draws. Same config => byte-identical items.
        quality_gap_sd: Standard deviation of latent quality gaps,
            drawn ``N(0, sd)`` (then mapped into [0, 1] qualities).
        tie_fraction: Fraction of items forced to have ~equal latent
            quality (|gap| < ``TIE_GAP``, hence labeled "tie"). Items
            outside this group may still land in the tie band by chance.
        length_log_sd: Standard deviation of response log-lengths around
            ``log(_BASE_LEN)``.
        authors: Author-name pool. The FIRST entry is treated as the
            judge-self author (matching ``MockJudgeConfig.self_name``'s
            default); remaining entries author all other responses.
        self_author_fraction: Fraction of items where exactly one side is
            authored by ``authors[0]``.
    """

    n_items: int
    seed: int
    quality_gap_sd: float = 1.0
    tie_fraction: float = 0.1
    length_log_sd: float = 0.4
    authors: tuple[str, ...] = ("judge-self", "other-model")
    self_author_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {self.n_items}")
        if not 0.0 <= self.tie_fraction <= 1.0:
            raise ValueError(f"tie_fraction must be in [0, 1], got {self.tie_fraction}")
        if not 0.0 <= self.self_author_fraction <= 1.0:
            raise ValueError(
                f"self_author_fraction must be in [0, 1], got {self.self_author_fraction}"
            )
        if self.quality_gap_sd < 0:
            raise ValueError(f"quality_gap_sd must be >= 0, got {self.quality_gap_sd}")
        if self.length_log_sd < 0:
            raise ValueError(f"length_log_sd must be >= 0, got {self.length_log_sd}")
        if not self.authors:
            raise ValueError("authors must be a non-empty tuple")


def _filler_text(start: int, target_len: int) -> str:
    """Build filler text of EXACTLY ``target_len`` characters.

    Cycles through the sentence bank starting at ``start`` and slices the
    joined text to the target. The final character may fall mid-word; the
    mock judge never reads body text, so only ``len()`` matters.

    Args:
        start: Index into the sentence bank to start cycling from.
        target_len: Exact character length of the returned string.

    Returns:
        A string with ``len() == target_len`` (empty if ``target_len <= 0``).
    """
    if target_len <= 0:
        return ""
    parts: list[str] = []
    total = 0
    i = start
    while total < target_len:
        sentence = _SENTENCES[i % len(_SENTENCES)]
        total += len(sentence) + (1 if parts else 0)
        parts.append(sentence)
        i += 1
    return " ".join(parts)[:target_len]


def generate_items(config: SyntheticConfig) -> list[PairwiseItem]:
    """Generate deterministic synthetic pairwise items.

    Mechanics (contract §3.1):

    * Latent quality gaps ``g ~ N(0, quality_gap_sd)``; a seeded subset of
      ``round(tie_fraction * n)`` items instead draws
      ``g ~ Uniform(-0.09, 0.09)`` (strictly inside the tie band).
    * Qualities are ``q_a = clip(0.5 + g/2, 0, 1)``,
      ``q_b = clip(0.5 - g/2, 0, 1)`` — floats in [0, 1] as required by
      :class:`~judgecal.core.PairwiseItem`. Clipping shrinks extreme gaps;
      the label uses the *effective* gap ``q_a - q_b``.
    * ``label = "tie"`` iff ``|q_a - q_b| < TIE_GAP``, else argmax.
    * Target response lengths are
      ``clip(round(_BASE_LEN * exp(N(0, length_log_sd))), _MIN_LEN, ∞)``;
      filler text hits each target length exactly (verifiable via
      ``meta["target_len_a"/"target_len_b"]``).
    * A seeded subset of ``round(self_author_fraction * n)`` items has
      exactly one side authored by ``authors[0]`` (the judge-self name);
      all other sides draw from ``authors[1:]`` (degenerate fallback: if
      the pool has a single entry, every side uses it).

    Args:
        config: Generator configuration (seed included).

    Returns:
        ``config.n_items`` items with ids ``"synthetic:{seed}:{i:05d}"``,
        latent qualities in ``meta["latent_quality_a"/"latent_quality_b"]``,
        and exact target lengths in ``meta["target_len_a"/"target_len_b"]``.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_items

    # --- latent quality gaps (fixed draw order for determinism) ---------
    gaps = rng.normal(0.0, config.quality_gap_sd, size=n)
    n_tie = int(round(config.tie_fraction * n))
    tie_idx = rng.permutation(n)[:n_tie]
    if n_tie:
        gaps[tie_idx] = rng.uniform(-0.09, 0.09, size=n_tie)
    q_a = np.clip(0.5 + gaps / 2.0, 0.0, 1.0)
    q_b = np.clip(0.5 - gaps / 2.0, 0.0, 1.0)

    # --- target lengths ---------------------------------------------------
    raw_len_a = _BASE_LEN * np.exp(rng.normal(0.0, config.length_log_sd, size=n))
    raw_len_b = _BASE_LEN * np.exp(rng.normal(0.0, config.length_log_sd, size=n))
    len_a = np.maximum(np.round(raw_len_a).astype(int), _MIN_LEN)
    len_b = np.maximum(np.round(raw_len_b).astype(int), _MIN_LEN)

    # --- authorship --------------------------------------------------------
    self_name = config.authors[0]
    others = config.authors[1:] if len(config.authors) > 1 else config.authors
    n_self = int(round(config.self_author_fraction * n))
    self_mask = np.zeros(n, dtype=bool)
    self_mask[rng.permutation(n)[:n_self]] = True
    self_side = rng.integers(0, 2, size=n)  # 0 => side A is self
    other_pick_a = rng.integers(0, len(others), size=n)
    other_pick_b = rng.integers(0, len(others), size=n)

    # --- filler-text start offsets ------------------------------------------
    start_a = rng.integers(0, len(_SENTENCES), size=n)
    start_b = rng.integers(0, len(_SENTENCES), size=n)

    items: list[PairwiseItem] = []
    for i in range(n):
        qa, qb = float(q_a[i]), float(q_b[i])
        gap_eff = qa - qb
        label: Label = "tie" if abs(gap_eff) < TIE_GAP else ("A" if gap_eff > 0 else "B")

        author_a = others[int(other_pick_a[i])]
        author_b = others[int(other_pick_b[i])]
        if self_mask[i]:
            if int(self_side[i]) == 0:
                author_a = self_name
            else:
                author_b = self_name

        ta, tb = int(len_a[i]), int(len_b[i])
        items.append(
            PairwiseItem(
                item_id=f"synthetic:{config.seed}:{i:05d}",
                prompt=(
                    f"Prompt {i}: Evaluate the proposed solution to task {i} "
                    "and explain the main trade-offs."
                ),
                response_a=_filler_text(int(start_a[i]), ta),
                response_b=_filler_text(int(start_b[i]), tb),
                label=label,
                author_a=author_a,
                author_b=author_b,
                source="synthetic",
                meta={
                    "latent_quality_a": qa,
                    "latent_quality_b": qb,
                    "target_len_a": ta,
                    "target_len_b": tb,
                },
            )
        )
    return items


__all__ = ["TIE_GAP", "SyntheticConfig", "generate_items"]
