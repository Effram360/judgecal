"""MT-Bench human judgments adapter (``lmsys/mt_bench_human_judgments``).

~3.3k expert human pairwise preferences over MT-Bench model answers.
License: CC-BY-4.0. The human winner becomes the item label and the
generating model names become ``author_a``/``author_b``, making this the
self-preference-capable anchor dataset.

Assumed raw HF schema (best-effort from public knowledge, 2026-06)::

    question_id     int | str   MT-Bench question id
    model_a         str         generator of conversation_a → author_a
    model_b         str         generator of conversation_b → author_b
    winner          str         "model_a" | "model_b" | "tie" |
                                "tie (bothbad)" (any value starting with
                                "tie" maps to a tie label)
    judge           str         human judge id, e.g. "expert_22"
    conversation_a  list[{"role": str, "content": str}]
    conversation_b  list[{"role": str, "content": str}]
    turn            int         1 or 2 — which turn the judgment covers

Splits assumed: ``"human"`` (expert judgments; the default) and
``"gpt4_pair"`` (GPT-4 judgments, NOT human — only loaded if explicitly
requested via ``split="gpt4_pair"``).

Mapping notes:

* **Only turn-1 judgments are kept.** Turn-2 rows depend on the full
  multi-turn context, which a single prompt/response pair cannot carry;
  they are skipped (caveat in :meth:`MTBenchHumanAdapter.info`).
* ``prompt`` is the first user message of ``conversation_a``;
  ``response_a``/``response_b`` are the first assistant messages of the
  respective conversations.
* The same (question, model_a, model_b) pair may be judged by several
  humans; each row becomes its own item with the judge id in the item
  id, so cluster on ``meta["question_id"]`` when pooling.
"""

from __future__ import annotations

from typing import Any, ClassVar

from judgecal.core import Label, PairwiseItem
from judgecal.datasets import base

_HF_PATH = "lmsys/mt_bench_human_judgments"

_CITATION = (
    "@misc{mtbench_2023,\n"
    "  title  = {Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},\n"
    "  author = {Zheng, Lianmin and others},\n"
    "  year   = {2023},\n"
    "  note   = {arXiv:2306.05685},\n"
    "}"
)

_CAVEATS = (
    "Multi-turn judgments beyond turn 1 are skipped: a single "
    "prompt/response pair cannot represent the turn-2 context.",
    "Several human judges may rate the same model pair; items are one per "
    "(pair, judge) row — cluster on meta['question_id'] when pooling.",
    "The 'gpt4_pair' split contains GPT-4 (not human) judgments; it is "
    "only loaded when explicitly requested.",
    "Column names and split names are best-effort schema assumptions (see "
    "module docstring), resolved defensively at load time.",
)


@base.register_adapter
class MTBenchHumanAdapter:
    """Human-preference adapter for MT-Bench pairwise judgments."""

    name: ClassVar[str] = "mtbench_human"

    def info(self) -> base.DatasetInfo:
        """Static metadata for MT-Bench human judgments."""
        return base.DatasetInfo(
            name=self.name,
            hf_path=_HF_PATH,
            license="CC-BY-4.0",
            citation=_CITATION,
            caveats=_CAVEATS,
            size_hint="3.3k expert pairwise preferences (turn 1 only after mapping)",
        )

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load MT-Bench human judgments as pairwise items.

        Args:
            split: HF split; ``None`` defaults to ``"human"`` (expert
                judgments). Pass ``"gpt4_pair"`` explicitly for the
                GPT-4-judged split.
            limit: Cap on items; seeded order-preserving subsample.
            seed: Subsampling seed.

        Returns:
            Turn-1 items with ids
            ``"mtbench_human:{question_id}:{model_a}:{model_b}:{judge}"``.
        """
        effective_split = "human" if split is None else split
        rows_by_split = base.load_hf_rows(_HF_PATH, effective_split)
        items: list[PairwiseItem] = []
        for split_name in sorted(rows_by_split):
            for row in rows_by_split[split_name]:
                item = self._map_row(row, split_name)
                if item is not None:
                    items.append(item)
        return base.sample_limit(items, limit, seed)

    def _map_row(self, row: dict[str, Any], split_name: str) -> PairwiseItem | None:
        """Map one raw row; returns ``None`` for skipped turn>1 rows."""
        turn = int(
            base.pick_field(row, ("turn",), dataset=self.name, fieldname="turn")
        )
        if turn != 1:
            return None
        question_id = base.pick_field(
            row, ("question_id", "id"), dataset=self.name, fieldname="question_id"
        )
        model_a = str(
            base.pick_field(row, ("model_a",), dataset=self.name, fieldname="model_a")
        )
        model_b = str(
            base.pick_field(row, ("model_b",), dataset=self.name, fieldname="model_b")
        )
        judge = str(
            base.pick_field(
                row, ("judge", "annotator"), dataset=self.name, fieldname="judge"
            )
        )
        winner = str(
            base.pick_field(row, ("winner",), dataset=self.name, fieldname="winner")
        )
        conv_a = base.pick_field(
            row, ("conversation_a",), dataset=self.name, fieldname="conversation_a"
        )
        conv_b = base.pick_field(
            row, ("conversation_b",), dataset=self.name, fieldname="conversation_b"
        )
        prompt = _first_message(conv_a, "user", dataset=self.name, side="conversation_a")
        response_a = _first_message(
            conv_a, "assistant", dataset=self.name, side="conversation_a"
        )
        response_b = _first_message(
            conv_b, "assistant", dataset=self.name, side="conversation_b"
        )
        return PairwiseItem(
            item_id=f"{self.name}:{question_id}:{model_a}:{model_b}:{judge}",
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            label=_parse_winner(winner, dataset=self.name),
            author_a=model_a,
            author_b=model_b,
            source=self.name,
            meta={
                "native_id": f"{question_id}:{model_a}:{model_b}:{judge}",
                "hf_split": split_name,
                "question_id": str(question_id),
                "judge": judge,
                "turn": turn,
                "raw_winner": winner,
            },
        )


def _parse_winner(winner: str, *, dataset: str) -> Label:
    """Map a raw winner string to an item-coordinates label."""
    if winner == "model_a":
        return "A"
    if winner == "model_b":
        return "B"
    if winner.startswith("tie"):
        return "tie"
    raise base.DatasetSchemaError(
        f"{dataset}: unrecognized winner value {winner!r} "
        "(expected 'model_a', 'model_b', or 'tie*')"
    )


def _first_message(conversation: Any, role: str, *, dataset: str, side: str) -> str:
    """Extract the first message with ``role`` from a conversation list.

    Args:
        conversation: Raw conversation column: list of dicts with
            ``role`` and ``content`` keys.
        role: ``"user"`` or ``"assistant"``.
        dataset: Adapter name for error messages.
        side: Which conversation column, for error messages.

    Returns:
        The message content string.

    Raises:
        DatasetSchemaError: If the conversation structure is not a list
            of role/content dicts or contains no message with ``role``.
    """
    if not isinstance(conversation, list):
        raise base.DatasetSchemaError(
            f"{dataset}: expected {side} to be a list of messages, "
            f"got {type(conversation).__name__}"
        )
    for message in conversation:
        if not isinstance(message, dict):
            raise base.DatasetSchemaError(
                f"{dataset}: expected {side} entries to be dicts with "
                f"'role'/'content', got {type(message).__name__}"
            )
        if message.get("role") == role:
            content = message.get("content")
            if content is None:
                raise base.DatasetSchemaError(
                    f"{dataset}: {side} message with role {role!r} has no 'content' "
                    f"key; keys present: {sorted(message.keys())}"
                )
            return str(content)
    roles = [m.get("role") for m in conversation if isinstance(m, dict)]
    raise base.DatasetSchemaError(
        f"{dataset}: no message with role {role!r} in {side}; roles present: {roles}"
    )


__all__ = ["MTBenchHumanAdapter"]
