"""JudgeBench adapter (``ScalerLab/JudgeBench``).

Objective-correctness pairwise comparisons over GPT-4o- and
Claude-generated response pairs (350 + 270 pairs per the verified
ground doc; SOTA judge accuracy ~64%). License **not verified** as of
2026-06-10 — surfaced in the info caveats.

Assumed raw HF schema (best-effort from public knowledge, 2026-06)::

    pair_id      str            native id; variants: pair_id, id, original_id
    question     str            variants: question, prompt, input
    response_A   str            variants: response_A, response_a, answer_A,
                                answer_a, output_A
    response_B   str            symmetric variants
    label        str            "A>B" | "B>A" (defensively also "A=B"/"B=A"
                                → tie); anything else raises
                                DatasetSchemaError naming the value
    source       str            originating benchmark (optional → meta)

Splits assumed: ``"gpt"`` (GPT-4o pairs) and ``"claude"`` (Claude
pairs); ``split=None`` loads and concatenates all available splits,
recording each row's split in ``meta["hf_split"]``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from judgecal.core import Label, PairwiseItem
from judgecal.datasets import base

_HF_PATH = "ScalerLab/JudgeBench"

_CITATION = (
    "@misc{tan2024judgebench,\n"
    "  title  = {JudgeBench: A Benchmark for Evaluating LLM-Based Judges},\n"
    "  author = {Tan, Sijun and others},\n"
    "  year   = {2024},\n"
    "  note   = {arXiv:2410.12784},\n"
    "}"
)

_CAVEATS = (
    "License not verified as of 2026-06-10 — check the upstream repository "
    "before redistributing any derived data.",
    "Split names 'gpt'/'claude' and column names are best-effort schema "
    "assumptions (see module docstring), resolved defensively at load time.",
    "Both responses in a pair come from the same generator model, so no "
    "author_a/author_b provenance is populated (not self-preference-capable).",
)

_LABEL_MAP: dict[str, Label] = {
    "A>B": "A",
    "B>A": "B",
    "A=B": "tie",
    "B=A": "tie",
}


@base.register_adapter
class JudgeBenchAdapter:
    """Pairwise adapter for JudgeBench."""

    name: ClassVar[str] = "judgebench"

    def info(self) -> base.DatasetInfo:
        """Static metadata for JudgeBench."""
        return base.DatasetInfo(
            name=self.name,
            hf_path=_HF_PATH,
            license="not verified",
            citation=_CITATION,
            caveats=_CAVEATS,
            size_hint="350 GPT-4o + 270 Claude pairs, objective-correctness",
        )

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load JudgeBench pairs.

        Args:
            split: HF split name (assumed ``"gpt"`` / ``"claude"``), or
                ``None`` for all splits concatenated.
            limit: Cap on items; seeded order-preserving subsample.
            seed: Subsampling seed.

        Returns:
            Items with ids ``"judgebench:{pair_id}"``.
        """
        rows_by_split = base.load_hf_rows(_HF_PATH, split)
        items: list[PairwiseItem] = []
        for split_name in sorted(rows_by_split):
            for idx, row in enumerate(rows_by_split[split_name]):
                items.append(self._map_row(row, split_name, idx))
        return base.sample_limit(items, limit, seed)

    def _map_row(self, row: dict[str, Any], split_name: str, idx: int) -> PairwiseItem:
        """Map one raw JudgeBench row to a PairwiseItem."""
        prompt = str(
            base.pick_field(
                row,
                ("question", "prompt", "input"),
                dataset=self.name,
                fieldname="question",
            )
        )
        response_a = str(
            base.pick_field(
                row,
                ("response_A", "response_a", "answer_A", "answer_a", "output_A"),
                dataset=self.name,
                fieldname="response_A",
            )
        )
        response_b = str(
            base.pick_field(
                row,
                ("response_B", "response_b", "answer_B", "answer_b", "output_B"),
                dataset=self.name,
                fieldname="response_B",
            )
        )
        raw_label = str(
            base.pick_field(
                row, ("label", "winner"), dataset=self.name, fieldname="label"
            )
        ).strip()
        if raw_label not in _LABEL_MAP:
            raise base.DatasetSchemaError(
                f"{self.name}: unrecognized label value {raw_label!r} "
                f"(expected one of {sorted(_LABEL_MAP)})"
            )
        native_id = _native_id(row, split_name, idx)
        return PairwiseItem(
            item_id=f"{self.name}:{native_id}",
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            label=_LABEL_MAP[raw_label],
            author_a=None,
            author_b=None,
            source=self.name,
            meta={
                "native_id": str(native_id),
                "hf_split": split_name,
                "origin": row.get("source"),
            },
        )


def _native_id(row: dict[str, Any], split_name: str, idx: int) -> str:
    """Stable native id: known id columns, else ``{split}-{index}``."""
    for key in ("pair_id", "id", "original_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"{split_name}-{idx}"


__all__ = ["JudgeBenchAdapter"]
