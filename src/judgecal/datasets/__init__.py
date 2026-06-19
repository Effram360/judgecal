"""judgecal.datasets — adapters from HF preference datasets to PairwiseItems.

Public API:

* :class:`DatasetAdapter` — the adapter protocol
  (``load(split, limit, seed) -> list[PairwiseItem]`` + ``info()``).
* :class:`DatasetInfo` — static metadata (hf_path, license, citation,
  caveats).
* :func:`get_adapter` / :func:`list_adapters` — the registry.
* :class:`DatasetSchemaError` — raised when raw rows match no known
  schema variant.
* The five adapter classes (also reachable via the registry):
  ``rewardbench2``, ``judgebench``, ``llmbar``, ``mtbench_human``,
  ``rmbench``.

The optional ``datasets`` package is imported lazily inside ``load()``
only; everything else (``info()``, the registry) works without it.
"""

from __future__ import annotations

from judgecal.datasets.base import (
    DatasetAdapter,
    DatasetInfo,
    DatasetSchemaError,
    get_adapter,
    list_adapters,
)
from judgecal.datasets.judgebench import JudgeBenchAdapter
from judgecal.datasets.llmbar import LLMBarAdapter
from judgecal.datasets.mtbench_human import MTBenchHumanAdapter
from judgecal.datasets.rewardbench2 import RewardBench2Adapter
from judgecal.datasets.rmbench import RMBenchAdapter

__all__ = [
    "DatasetAdapter",
    "DatasetInfo",
    "DatasetSchemaError",
    "JudgeBenchAdapter",
    "LLMBarAdapter",
    "MTBenchHumanAdapter",
    "RMBenchAdapter",
    "RewardBench2Adapter",
    "get_adapter",
    "list_adapters",
]
