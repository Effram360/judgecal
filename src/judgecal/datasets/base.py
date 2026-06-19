"""Adapter protocol, registry, and shared helpers for judgecal datasets.

A :class:`DatasetAdapter` turns a Hugging Face preference dataset into
canonical :class:`judgecal.core.PairwiseItem` lists. Adapters are pure
mapping code: the only I/O is a lazy ``import datasets`` +
``datasets.load_dataset`` call funneled through :func:`load_hf_rows`,
which tests monkeypatch with tiny in-memory tables (no network).

Conventions enforced here:

* **Lazy optional dependency.** The ``datasets`` package is imported
  inside :func:`load_hf_rows` only; without it adapters raise an
  ``ImportError`` pointing at ``pip install 'judgecal[hf]'``.
* **Stable item ids.** Every item id is ``"{dataset}:{native_id}"``
  (plus a deterministic suffix for adapters that expand one raw row
  into several pairs). Ids never depend on ``limit`` or ``seed``.
* **Deterministic subsampling.** When ``limit`` is below the number of
  available pairs, :func:`sample_limit` draws without replacement via a
  seeded ``numpy.random.Generator`` and preserves the original order.
* **Defensive schema resolution.** Raw HF column names are resolved via
  :func:`pick_field`, which tries several known variants and raises a
  :class:`DatasetSchemaError` listing the columns actually found.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

from judgecal.core import PairwiseItem

T = TypeVar("T")

#: Install hint used in every lazy-import error message (quoted so the
#: command survives zsh glob expansion when copy-pasted).
HF_INSTALL_HINT = "pip install 'judgecal[hf]'"


class DatasetSchemaError(ValueError):
    """Raised when a raw HF row does not match any known schema variant."""


# --------------------------------------------------------------------------
# Dataset info
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for a dataset adapter.

    Attributes:
        name: Registry name of the adapter (e.g. ``"llmbar"``).
        hf_path: Hugging Face hub path passed to ``load_dataset``.
        license: License string as recorded for the dataset;
            ``"not verified"`` when unverified (also surfaced in caveats).
        citation: BibTeX-ish citation string including the arXiv id
            where one is verified.
        caveats: Known limitations of the mapping or the dataset
            (license gaps, schema assumptions, expansion semantics).
        size_hint: Human-readable size/format note (informational only).
    """

    name: str
    hf_path: str
    license: str
    citation: str
    caveats: tuple[str, ...] = ()
    size_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form (caveats as a list) for cards and CLI output."""
        return {
            "name": self.name,
            "hf_path": self.hf_path,
            "license": self.license,
            "citation": self.citation,
            "caveats": list(self.caveats),
            "size_hint": self.size_hint,
        }


# --------------------------------------------------------------------------
# Adapter protocol
# --------------------------------------------------------------------------


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol every dataset adapter implements.

    Attributes:
        name: Registry name; also the ``source`` and item-id prefix.
    """

    name: ClassVar[str]

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
        seed: int = 0,
    ) -> list[PairwiseItem]:
        """Load and map the dataset into canonical pairwise items.

        Args:
            split: Dataset-specific split selector; ``None`` selects the
                adapter's documented default (see each adapter docstring).
            limit: Maximum number of items to return. When the mapped
                pairs exceed ``limit``, a seeded without-replacement
                subsample is taken (order-preserving).
            seed: Seed for the subsampling Generator. Ignored when
                ``limit`` is ``None`` or not binding.

        Returns:
            Mapped items with stable ids ``"{name}:{native_id}"``.

        Raises:
            ImportError: If the optional ``datasets`` package is missing.
            DatasetSchemaError: If raw rows match no known schema variant.
        """
        ...

    def info(self) -> DatasetInfo:
        """Static metadata: hf_path, license, citation, caveats."""
        ...


# --------------------------------------------------------------------------
# Lazy HF access (single seam; tests monkeypatch ``load_hf_rows``)
# --------------------------------------------------------------------------


def _import_datasets() -> Any:
    """Import the optional ``datasets`` package, with a helpful error.

    Returns:
        The imported ``datasets`` module.

    Raises:
        ImportError: With an install hint when the package is missing.
    """
    try:
        import datasets  # deliberate lazy import of the optional extra
    except ImportError as exc:
        raise ImportError(
            "judgecal dataset adapters need the optional 'datasets' package, "
            f"which is not installed. Install it with: {HF_INSTALL_HINT}"
        ) from exc
    return datasets


def load_hf_rows(hf_path: str, split: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load raw rows from the Hugging Face hub, grouped by split.

    This is the single network/IO seam of the datasets subpackage. Unit
    tests monkeypatch this function with in-memory fakes; only tests
    marked ``@pytest.mark.network`` ever hit the real hub.

    Args:
        hf_path: Hub path, e.g. ``"princeton-nlp/LLMBar"``.
        split: A concrete HF split name, or ``None`` to load every
            available split.

    Returns:
        Mapping of split name to materialized rows (list of dicts).
        Splits are ordered by sorted name for determinism.

    Raises:
        ImportError: If ``datasets`` is not installed.
    """
    datasets_mod = _import_datasets()
    if split is not None:
        ds = datasets_mod.load_dataset(hf_path, split=split)
        return {split: [dict(row) for row in ds]}
    dd = datasets_mod.load_dataset(hf_path)
    return {name: [dict(row) for row in dd[name]] for name in sorted(dd.keys())}


# --------------------------------------------------------------------------
# Mapping helpers
# --------------------------------------------------------------------------


def pick_field(
    row: Mapping[str, Any],
    candidates: Sequence[str],
    *,
    dataset: str,
    fieldname: str,
) -> Any:
    """Resolve a logical field from a raw row, trying known column variants.

    Args:
        row: Raw HF row.
        candidates: Column names to try, in priority order.
        dataset: Adapter name, for the error message.
        fieldname: Logical field being resolved, for the error message.

    Returns:
        The first present column's value (may be ``None`` if stored so).

    Raises:
        DatasetSchemaError: If none of the candidates is present; the
            message lists both the tried variants and the columns found.
    """
    for cand in candidates:
        if cand in row:
            return row[cand]
    raise DatasetSchemaError(
        f"{dataset}: could not resolve field '{fieldname}'. "
        f"Tried columns {list(candidates)}; row has columns {sorted(row.keys())}. "
        "The upstream HF schema may have changed — please open an issue."
    )


def as_str_list(value: Any) -> list[str]:
    """Normalize a column that may be a string or a sequence of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value]
    return [str(value)]


def first_or_none(value: Any) -> str | None:
    """First element of a maybe-list column, or ``None`` when empty."""
    values = as_str_list(value)
    return values[0] if values else None


def sample_limit(items: Sequence[T], limit: int | None, seed: int) -> list[T]:
    """Deterministically cap a sequence via seeded sampling.

    When ``limit`` binds, draws ``limit`` indices without replacement
    from a ``numpy.random.default_rng(seed)`` Generator and returns the
    selected elements in their original order (so ids stay sorted the
    way the dataset emitted them).

    Args:
        items: Mapped items (any sequence).
        limit: Cap, or ``None`` for no cap.
        seed: Generator seed.

    Returns:
        ``list(items)`` when the cap does not bind, else an
        order-preserving seeded subsample of size ``limit``.
    """
    if limit is None or len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=limit, replace=False)
    return [items[i] for i in sorted(int(i) for i in idx)]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}

#: Adapter modules imported lazily so ``base`` never imports adapters at
#: module load time (adapters import ``base`` — this avoids the cycle).
_ADAPTER_MODULES = (
    "judgecal.datasets.rewardbench2",
    "judgecal.datasets.judgebench",
    "judgecal.datasets.llmbar",
    "judgecal.datasets.mtbench_human",
    "judgecal.datasets.rmbench",
)


def register_adapter(cls: type) -> type:
    """Class decorator: register an adapter class under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"adapter class {cls!r} must define a non-empty 'name'")
    _REGISTRY[name] = cls
    return cls


def _ensure_adapters_loaded() -> None:
    """Import all adapter modules so their classes self-register."""
    for module in _ADAPTER_MODULES:
        importlib.import_module(module)


def list_adapters() -> list[str]:
    """Sorted names of all registered dataset adapters."""
    _ensure_adapters_loaded()
    return sorted(_REGISTRY)


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate the adapter registered under ``name``.

    Args:
        name: Registry name, e.g. ``"llmbar"``.

    Returns:
        A fresh adapter instance.

    Raises:
        KeyError: If no adapter is registered under ``name``; the
            message lists the available names.
    """
    _ensure_adapters_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset adapter '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


__all__ = [
    "HF_INSTALL_HINT",
    "DatasetAdapter",
    "DatasetInfo",
    "DatasetSchemaError",
    "as_str_list",
    "first_or_none",
    "get_adapter",
    "list_adapters",
    "load_hf_rows",
    "pick_field",
    "register_adapter",
    "sample_limit",
]
