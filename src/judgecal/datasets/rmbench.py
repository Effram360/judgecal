"""RM-Bench adapter (``THU-KEG/RM-Bench``).

Style-robustness preference data: each prompt carries 3 chosen and 3
rejected completions across style variants. The adapter expands the
full 3x3 chosen-vs-rejected matrix into pairwise items (RM-Bench's own
accuracy matrix), labelling the chosen side ``"A"``. License **not
verified** as of 2026-06-10 — surfaced in the info caveats.

Assumed raw HF schema (best-effort from public knowledge, 2026-06)::

    id        str | int        native example id (fallback: split-index)
                               variants tried: id, example_id, prompt_id
    prompt    str              variants: prompt, text, instruction
    chosen    list[str] (3)    style variants of the correct response
                               variants: chosen, chosen_responses
    rejected  list[str] (3)    style variants of the incorrect response
                               variants: rejected, rejected_responses
    domain    str              e.g. chat, code, math, safety-refuse,
                               safety-response (optional → meta)

Assumed style-variant order (index meaning, NOT verified against the
live hub): 0 = concise, 1 = detailed plain-text, 2 = detailed markdown.
Indices are recorded in ``meta["chosen_style"]`` / ``meta["rejected_style"]``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from judgecal.core import PairwiseItem
from judgecal.datasets import base

_HF_PATH = "THU-KEG/RM-Bench"

_CITATION = (
    "@misc{liu2024rmbench,\n"
    "  title  = {RM-Bench: Benchmarking Reward Models of Language Models "
    "with Subtlety and Style},\n"
    "  author = {Liu, Yantao and others},\n"
    "  year   = {2024},\n"
    "  note   = {arXiv:2410.16184},\n"
    "}"
)

_CAVEATS = (
    "License not verified as of 2026-06-10 — check the upstream repository "
    "before redistributing any derived data.",
    "Each raw prompt expands into up to 9 chosen-vs-rejected pairs (3x3 "
    "style matrix); pairs from one prompt are strongly dependent — cluster "
    "on meta['native_id'] when pooling.",
    "response_a is always the chosen completion (label 'A'); probes "
    "randomize/swap presentation order.",
    "The style-variant index order (0 concise, 1 detailed, 2 markdown) is "
    "an unverified assumption; treat meta['*_style'] as opaque indices if "
    "in doubt.",
)


@base.register_adapter
class RMBenchAdapter:
    """Style-variant pairwise adapter for RM-Bench."""

    name: ClassVar[str] = "rmbench"

    def info(self) -> base.DatasetInfo:
        """Static metadata for RM-Bench."""
        return base.DatasetInfo(
            name=self.name,
            hf_path=_HF_PATH,
            license="not verified",
            citation=_CITATION,
            caveats=_CAVEATS,
            size_hint="~1.33k prompts; 3 chosen + 3 rejected style variants each",
        )

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load RM-Bench and expand the 3x3 style matrix into pairs.

        Args:
            split: Concrete HF split name, or ``None`` for all splits.
            limit: Cap on expanded *pairs*; seeded without-replacement
                subsampling when binding (order-preserving).
            seed: Subsampling seed.

        Returns:
            Pairwise items with ids ``"rmbench:{native_id}#c{i}r{j}"``
            where ``i``/``j`` index the chosen/rejected style variants.
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
        """Expand one raw row into the chosen-x-rejected style matrix."""
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
                ("chosen", "chosen_responses"),
                dataset=self.name,
                fieldname="chosen",
            )
        )
        rejected = base.as_str_list(
            base.pick_field(
                row,
                ("rejected", "rejected_responses"),
                dataset=self.name,
                fieldname="rejected",
            )
        )
        if not chosen or not rejected:
            return []
        native_id = _native_id(row, split_name, idx)
        domain = row.get("domain")
        items: list[PairwiseItem] = []
        for i, chosen_text in enumerate(chosen):
            for j, rejected_text in enumerate(rejected):
                items.append(
                    PairwiseItem(
                        item_id=f"{self.name}:{native_id}#c{i}r{j}",
                        prompt=prompt,
                        response_a=chosen_text,
                        response_b=rejected_text,
                        label="A",
                        author_a=None,
                        author_b=None,
                        source=self.name,
                        meta={
                            "native_id": str(native_id),
                            "hf_split": split_name,
                            "domain": domain,
                            "chosen_style": i,
                            "rejected_style": j,
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


__all__ = ["RMBenchAdapter"]
