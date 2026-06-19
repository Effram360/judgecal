"""LLMBar adapter (``princeton-nlp/LLMBar``).

Instruction-following pairwise comparisons with objective gold
preferences: a Natural subset plus four Adversarial subsets designed to
fool LLM judges (419 pairs total). License: MIT (per the project's
GitHub repository, as recorded in the verified ground doc).

Data source (investigated live against the HF hub API, 2026-06-10):
the Hugging Face repo ``princeton-nlp/LLMBar`` is *script-based* — its
file tree contains only ``LLMBar.py``, a README stub, and
``.gitattributes``; there are **no raw data files on the hub**, so
neither ``datasets.load_dataset`` (``datasets>=5`` refuses loading
scripts) nor ``hf_hub_download`` can fetch the data. The loading script
itself downloads raw JSON from the project's GitHub repository, and this
adapter fetches those same files directly (stdlib ``urllib``; no
optional dependency needed)::

    https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/
        Natural/dataset.json                 (100 rows)
        Adversarial/Neighbor/dataset.json
        Adversarial/GPTInst/dataset.json
        Adversarial/GPTOut/dataset.json
        Adversarial/Manual/dataset.json

Raw JSON schema (verified live, 2026-06-10): each file is a JSON *array*
of objects::

    input     str   the instruction
    output_1  str   first response
    output_2  str   second response
    label     int   1 → output_1 preferred, 2 → output_2 preferred

Split selector semantics (split names match the HF script's split names):

* ``None`` — all five splits concatenated.
* ``"Adversarial"`` — pseudo-split: every split whose name starts with
  ``"Adversarial"``.
* one of ``Natural``, ``Adversarial_Neighbor``, ``Adversarial_GPTInst``,
  ``Adversarial_GPTOut``, ``Adversarial_Manual`` — that split only.

The single network seam is :func:`fetch_split_rows`; unit tests
monkeypatch it with in-memory tables (no network).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, ClassVar

from judgecal.core import Label, PairwiseItem
from judgecal.datasets import base

_HF_PATH = "princeton-nlp/LLMBar"

#: GitHub raw-file base for the LLMBar dataset (see module docstring for
#: why the data cannot be fetched from the HF hub).
_RAW_BASE = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/"

#: Split name -> dataset.json path under ``_RAW_BASE`` (verified layout).
_SPLIT_FILES: dict[str, str] = {
    "Natural": "Natural/dataset.json",
    "Adversarial_Neighbor": "Adversarial/Neighbor/dataset.json",
    "Adversarial_GPTInst": "Adversarial/GPTInst/dataset.json",
    "Adversarial_GPTOut": "Adversarial/GPTOut/dataset.json",
    "Adversarial_Manual": "Adversarial/Manual/dataset.json",
}

_CITATION = (
    "@article{zeng2023llmbar,\n"
    "  title   = {Evaluating Large Language Models at Evaluating "
    "Instruction Following},\n"
    "  author  = {Zeng, Zhiyuan and Yu, Jiatong and Gao, Tianyu and "
    "Meng, Yu and Goyal, Tanya and Chen, Danqi},\n"
    "  journal = {arXiv preprint arXiv:2310.07641},\n"
    "  year    = {2023},\n"
    "}  % published at ICLR 2024"
)

_CAVEATS = (
    "Adversarial subsets are constructed to fool LLM judges; accuracy on "
    "them is not comparable to natural-distribution accuracy — keep "
    "meta['subset'] in any pooled analysis.",
    "Data is fetched from the project's GitHub repository (raw JSON): the "
    "HF hub repo is a loading script only, which datasets>=5 refuses to "
    "execute and which itself downloads these same GitHub files.",
    "No response provenance is available; author_a/author_b are None "
    "(not self-preference-capable).",
)


def _download_json(url: str) -> Any:
    """Download and parse one JSON document (the only network call)."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https base
        return json.load(resp)


def fetch_split_rows(split_name: str) -> list[dict[str, Any]]:
    """Fetch one LLMBar split's raw rows from GitHub raw (network seam).

    Args:
        split_name: A key of ``_SPLIT_FILES``.

    Returns:
        The split's rows (list of raw dicts).

    Raises:
        KeyError: If ``split_name`` is not a known split.
        DatasetSchemaError: If the downloaded document is not a JSON array.
    """
    try:
        path = _SPLIT_FILES[split_name]
    except KeyError:
        raise KeyError(
            f"llmbar: unknown split {split_name!r}; available: "
            f"{sorted(_SPLIT_FILES)} (or the 'Adversarial' pseudo-split)"
        ) from None
    payload = _download_json(_RAW_BASE + path)
    if not isinstance(payload, list):
        raise base.DatasetSchemaError(
            f"llmbar: expected a JSON array at {_RAW_BASE + path}, "
            f"got {type(payload).__name__} — the upstream layout may have changed"
        )
    return [dict(row) for row in payload]


@base.register_adapter
class LLMBarAdapter:
    """Pairwise adapter for LLMBar (Natural + Adversarial subsets)."""

    name: ClassVar[str] = "llmbar"

    def info(self) -> base.DatasetInfo:
        """Static metadata for LLMBar."""
        return base.DatasetInfo(
            name=self.name,
            hf_path=_HF_PATH,
            license="MIT",
            citation=_CITATION,
            caveats=_CAVEATS,
            size_hint="419 pairs, Natural + 4 Adversarial subsets (GitHub raw JSON)",
        )

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load LLMBar pairs.

        Args:
            split: ``None`` (all subsets), ``"Adversarial"`` (all
                adversarial subsets), or a concrete split name (see
                module docstring).
            limit: Cap on items; seeded order-preserving subsample.
            seed: Subsampling seed.

        Returns:
            Items with ids ``"llmbar:{subset}-{index}"`` (the dataset
            exposes no native row id).

        Raises:
            KeyError: If ``split`` names an unknown split.
        """
        if split is None:
            split_names = sorted(_SPLIT_FILES)
        elif split == "Adversarial":
            split_names = sorted(s for s in _SPLIT_FILES if s.startswith("Adversarial"))
        else:
            split_names = [split]
        items: list[PairwiseItem] = []
        for split_name in split_names:
            for idx, row in enumerate(fetch_split_rows(split_name)):
                items.append(self._map_row(row, split_name, idx))
        return base.sample_limit(items, limit, seed)

    def _map_row(self, row: dict[str, Any], split_name: str, idx: int) -> PairwiseItem:
        """Map one raw LLMBar row to a PairwiseItem."""
        prompt = str(
            base.pick_field(
                row,
                ("input", "instruction", "prompt"),
                dataset=self.name,
                fieldname="input",
            )
        )
        response_a = str(
            base.pick_field(
                row,
                ("output_1", "response_1", "output1"),
                dataset=self.name,
                fieldname="output_1",
            )
        )
        response_b = str(
            base.pick_field(
                row,
                ("output_2", "response_2", "output2"),
                dataset=self.name,
                fieldname="output_2",
            )
        )
        raw_label = base.pick_field(
            row, ("label", "preference"), dataset=self.name, fieldname="label"
        )
        label = _parse_label(raw_label, dataset=self.name)
        return PairwiseItem(
            item_id=f"{self.name}:{split_name}-{idx}",
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            label=label,
            author_a=None,
            author_b=None,
            source=self.name,
            meta={
                "native_id": f"{split_name}-{idx}",
                "hf_split": split_name,
                "subset": split_name,
                "adversarial": split_name.startswith("Adversarial"),
            },
        )


def _parse_label(raw: Any, *, dataset: str) -> Label:
    """Parse an LLMBar preference label (1 → A, 2 → B)."""
    text = str(raw).strip()
    if text == "1":
        return "A"
    if text == "2":
        return "B"
    raise base.DatasetSchemaError(
        f"{dataset}: unrecognized label value {raw!r} (expected 1 or 2)"
    )


__all__ = ["LLMBarAdapter", "fetch_split_rows"]
