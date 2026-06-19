"""RewardBench 2 adapter (``allenai/reward-bench-2``).

Expands best-of-4 sets into chosen-vs-rejected pairwise items: each raw
row contributes one pair per rejected completion (chosen always as
``response_a``, label ``"A"``). Probes present both orders, so the
fixed side assignment carries no position information.

Assumed raw HF schema (best-effort from public knowledge, 2026-06)::

    id            str | int       native example id (fallback: split-index)
                                  variants tried: id, example_id, prompt_id
    prompt        str             variants: prompt, text, instruction
    chosen        list[str] | str winning completion(s); first used if several
                                  variants: chosen, chosen_responses
    rejected      list[str] | str losing completions (typically 3)
                                  variants: rejected, rejected_responses
    subset        str             e.g. Factuality, Precise IF, Math, Safety,
                                  Focus, Ties (optional)
    chosen_model  str | list[str] provenance of chosen (optional → author_a)
    rejected_model(s) str | list[str] provenance of rejected (optional → author_b)

Rows whose chosen or rejected list is empty are skipped. ``split=None``
loads every available split (the hub layout's split naming is not
relied upon). Size/format per the verified ground doc: 1,865 prompts,
best-of-4; 6 subsets (Factuality 475, Precise IF 160, Math 183,
Safety 450, Focus 495, Ties 102). License: ODC-BY.
"""

from __future__ import annotations

from typing import Any, ClassVar

from judgecal.core import PairwiseItem
from judgecal.datasets import base

_HF_PATH = "allenai/reward-bench-2"

_CITATION = (
    "@misc{malik2025rewardbench2,\n"
    "  title  = {RewardBench 2: Advancing Reward Model Evaluation},\n"
    "  author = {Malik, Saumya and others},\n"
    "  year   = {2025},\n"
    "  note   = {arXiv:2506.01937},\n"
    "}"
)

_CAVEATS = (
    "Best-of-4 rows are expanded into chosen-vs-rejected pairs; response_a is "
    "always the chosen completion (label 'A'). Probes randomize/swap "
    "presentation order, so this carries no position signal.",
    "Pairs sharing one raw prompt are statistically dependent; cluster on "
    "meta['native_id'] (not item_id) when pooling across pairs.",
    "The Ties subset may carry multiple chosen completions; only the first "
    "chosen completion is used.",
    "Raw column names are resolved defensively against several known "
    "variants; schema assumptions are documented in the module docstring "
    "and not verified against the live hub.",
)


@base.register_adapter
class RewardBench2Adapter:
    """Best-of-4 → pairwise adapter for RewardBench 2."""

    name: ClassVar[str] = "rewardbench2"

    def info(self) -> base.DatasetInfo:
        """Static metadata for RewardBench 2."""
        return base.DatasetInfo(
            name=self.name,
            hf_path=_HF_PATH,
            license="ODC-BY",
            citation=_CITATION,
            caveats=_CAVEATS,
            size_hint="1,865 prompts, best-of-4; 6 subsets",
        )

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load RewardBench 2 and expand into pairwise items.

        Args:
            split: Concrete HF split name, or ``None`` for all splits.
            limit: Cap on expanded *pairs*; seeded without-replacement
                subsampling when binding (order-preserving).
            seed: Subsampling seed.

        Returns:
            Pairwise items with ids ``"rewardbench2:{native_id}#r{j}"``
            where ``j`` indexes the rejected completion.
        """
        rows_by_split = base.load_hf_rows(_HF_PATH, split)
        pairs: list[PairwiseItem] = []
        for split_name in sorted(rows_by_split):
            for idx, row in enumerate(rows_by_split[split_name]):
                pairs.extend(self._expand_row(row, split_name, idx))
        return base.sample_limit(pairs, limit, seed)

    def _expand_row(
        self, row: dict[str, Any], split_name: str, idx: int
    ) -> list[PairwiseItem]:
        """Expand one best-of-4 row into chosen-vs-rejected pairs."""
        prompt = str(
            base.pick_field(
                row,
                ("prompt", "text", "instruction"),
                dataset=self.name,
                fieldname="prompt",
            )
        )
        chosen = base.as_str_list(
            base.pick_field(
                row,
                ("chosen", "chosen_responses", "chosen_response"),
                dataset=self.name,
                fieldname="chosen",
            )
        )
        rejected = base.as_str_list(
            base.pick_field(
                row,
                ("rejected", "rejected_responses", "rejected_response"),
                dataset=self.name,
                fieldname="rejected",
            )
        )
        if not chosen or not rejected:
            return []
        native_id = _native_id(row, split_name, idx)
        subset = row.get("subset")
        author_a = base.first_or_none(row.get("chosen_model"))
        rejected_models = base.as_str_list(
            row.get("rejected_model", row.get("rejected_models"))
        )
        items: list[PairwiseItem] = []
        for j, rejected_text in enumerate(rejected):
            author_b = rejected_models[j] if j < len(rejected_models) else None
            items.append(
                PairwiseItem(
                    item_id=f"{self.name}:{native_id}#r{j}",
                    prompt=prompt,
                    response_a=chosen[0],
                    response_b=rejected_text,
                    label="A",
                    author_a=author_a,
                    author_b=author_b,
                    source=self.name,
                    meta={
                        "native_id": str(native_id),
                        "hf_split": split_name,
                        "subset": subset,
                        "rejected_index": j,
                        "n_chosen": len(chosen),
                        "n_rejected": len(rejected),
                    },
                )
            )
        return items


def _native_id(row: dict[str, Any], split_name: str, idx: int) -> str:
    """Stable native id: known id columns, else ``{split}-{index}``."""
    for key in ("id", "example_id", "prompt_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"{split_name}-{idx}"


__all__ = ["RewardBench2Adapter"]
