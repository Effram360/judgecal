"""Tests for judgecal.datasets — no network; HF rows are in-memory fakes.

Each fake table mirrors the raw schema documented in the corresponding
adapter module docstring. ``base.load_hf_rows`` (the IO seam of the
HF-backed adapters) and ``llmbar.fetch_split_rows`` (LLMBar's GitHub-raw
seam) are monkeypatched; the ImportError test instead poisons
``sys.modules`` so the real lazy import path is exercised.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from judgecal.datasets import (
    DatasetInfo,
    DatasetSchemaError,
    base,
    get_adapter,
    list_adapters,
    llmbar,
)

# --------------------------------------------------------------------------
# Fake raw HF tables, mirroring each adapter's documented schema
# --------------------------------------------------------------------------

REWARDBENCH2_ROWS = {
    "test": [
        {
            "id": "rb2-001",
            "prompt": "What is 2+2?",
            "chosen": ["Four."],
            "rejected": ["Five.", "Twenty-two.", "Fish."],
            "subset": "Math",
            "chosen_model": "model-good",
            "rejected_model": ["model-bad-0", "model-bad-1", "model-bad-2"],
        },
        {
            "id": "rb2-002",
            "prompt": "Name a primary color.",
            "chosen": ["Red.", "Blue."],  # Ties-style multi-chosen
            "rejected": ["Purple.", "Brown."],
            "subset": "Ties",
        },
    ]
}

JUDGEBENCH_ROWS = {
    "gpt": [
        {
            "pair_id": "jb-gpt-1",
            "question": "Is the sky blue?",
            "response_A": "Yes, due to Rayleigh scattering.",
            "response_B": "No, it is green.",
            "label": "A>B",
            "source": "mmlu-pro",
        },
        {
            "pair_id": "jb-gpt-2",
            "question": "Compute 3*7.",
            "response_A": "20",
            "response_B": "21",
            "label": "B>A",
            "source": "livebench",
        },
    ],
    "claude": [
        {
            "pair_id": "jb-cl-1",
            "question": "Capital of France?",
            "response_A": "Paris.",
            "response_B": "Lyon.",
            "label": "A>B",
            "source": "mmlu-pro",
        },
    ],
}

LLMBAR_ROWS = {
    "Natural": [
        {
            "input": "Write a haiku about rain.",
            "output_1": "Rain falls on the roof / soft drumming in the gray dawn / puddles hold the sky",
            "output_2": "Rain is wet. The end.",
            "label": 1,
        },
    ],
    "Adversarial_Neighbor": [
        {
            "input": "Summarize the article in one sentence.",
            "output_1": "A long, fluent but off-topic essay about summaries.",
            "output_2": "The article argues X in one sentence.",
            "label": 2,
        },
    ],
    "Adversarial_GPTInst": [
        {
            "input": "List three fruits.",
            "output_1": "Apple, banana, cherry.",
            "output_2": "Apple, banana, cherry, and a great recipe for smoothies!",
            "label": 1,
        },
    ],
}

MTBENCH_ROWS = {
    "human": [
        {
            "question_id": 81,
            "model_a": "alpaca-13b",
            "model_b": "vicuna-13b",
            "winner": "model_b",
            "judge": "expert_0",
            "conversation_a": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Short Hawaii post."},
                {"role": "user", "content": "Rewrite it starting with A."},
                {"role": "assistant", "content": "Aloha-only attempt A."},
            ],
            "conversation_b": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Long vivid Hawaii post."},
                {"role": "user", "content": "Rewrite it starting with A."},
                {"role": "assistant", "content": "Aloha-only attempt B."},
            ],
            "turn": 1,
        },
        {
            # same pair, second human judge — must yield a distinct item id
            "question_id": 81,
            "model_a": "alpaca-13b",
            "model_b": "vicuna-13b",
            "winner": "tie (bothbad)",
            "judge": "expert_1",
            "conversation_a": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Short Hawaii post."},
            ],
            "conversation_b": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Long vivid Hawaii post."},
            ],
            "turn": 1,
        },
        {
            # turn-2 judgment — must be skipped
            "question_id": 81,
            "model_a": "alpaca-13b",
            "model_b": "vicuna-13b",
            "winner": "model_a",
            "judge": "expert_2",
            "conversation_a": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Short Hawaii post."},
            ],
            "conversation_b": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Long vivid Hawaii post."},
            ],
            "turn": 2,
        },
    ],
    "gpt4_pair": [
        {
            "question_id": 81,
            "model_a": "alpaca-13b",
            "model_b": "vicuna-13b",
            "winner": "model_a",
            "judge": "gpt4_pair",
            "conversation_a": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Short Hawaii post."},
            ],
            "conversation_b": [
                {"role": "user", "content": "Compose a travel blog post about Hawaii."},
                {"role": "assistant", "content": "Long vivid Hawaii post."},
            ],
            "turn": 1,
        },
    ],
}

RMBENCH_ROWS = {
    "train": [
        {
            "id": "rmb-1",
            "prompt": "Explain photosynthesis.",
            "chosen": ["Concise correct.", "Detailed correct.", "## Markdown correct."],
            "rejected": ["Concise wrong.", "Detailed wrong.", "## Markdown wrong."],
            "domain": "chat",
        },
        {
            "id": "rmb-2",
            "prompt": "Sort [3,1,2] in Python.",
            "chosen": ["sorted([3,1,2])", "Use sorted(); it returns a new list.", "## Sorting\nsorted([3,1,2])"],
            "rejected": ["[3,1,2].sort()[0]", "Use .sort() and return it (wrong).", "## Sorting\nlst.sort() returns the list (wrong)"],
            "domain": "code",
        },
    ]
}

FAKE_HUB: dict[str, dict[str, list[dict[str, Any]]]] = {
    "allenai/reward-bench-2": REWARDBENCH2_ROWS,
    "ScalerLab/JudgeBench": JUDGEBENCH_ROWS,
    "princeton-nlp/LLMBar": LLMBAR_ROWS,
    "lmsys/mt_bench_human_judgments": MTBENCH_ROWS,
    "THU-KEG/RM-Bench": RMBENCH_ROWS,
}


@pytest.fixture()
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[str, str | None]]]:
    """Monkeypatch base.load_hf_rows and llmbar.fetch_split_rows with fakes.

    Returns a call log: hf_path -> list of (hf_path, split) calls.
    """
    calls: dict[str, list[tuple[str, str | None]]] = {}

    def _fake_load(hf_path: str, split: str | None = None) -> dict[str, list[dict[str, Any]]]:
        calls.setdefault(hf_path, []).append((hf_path, split))
        table = FAKE_HUB[hf_path]
        if split is None:
            return {name: [dict(r) for r in rows] for name, rows in sorted(table.items())}
        if split not in table:
            raise KeyError(f"fake hub: unknown split {split!r} for {hf_path}")
        return {split: [dict(r) for r in table[split]]}

    def _fake_fetch_split(split_name: str) -> list[dict[str, Any]]:
        calls.setdefault("princeton-nlp/LLMBar", []).append(
            ("princeton-nlp/LLMBar", split_name)
        )
        # Splits absent from the fake table are simply empty (the real
        # seam would download them; tests only seed three splits).
        return [dict(r) for r in LLMBAR_ROWS.get(split_name, [])]

    monkeypatch.setattr(base, "load_hf_rows", _fake_load)
    monkeypatch.setattr(llmbar, "fetch_split_rows", _fake_fetch_split)
    return calls


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

EXPECTED_ADAPTERS = ["judgebench", "llmbar", "mtbench_human", "rewardbench2", "rmbench"]


def test_registry_lists_all_five_adapters() -> None:
    assert list_adapters() == EXPECTED_ADAPTERS


@pytest.mark.parametrize("name", EXPECTED_ADAPTERS)
def test_get_adapter_returns_conforming_instance(name: str) -> None:
    adapter = get_adapter(name)
    assert adapter.name == name
    info = adapter.info()
    assert isinstance(info, DatasetInfo)
    assert info.name == name
    assert info.hf_path
    assert info.license
    assert info.citation
    # as_dict gives list caveats for card embedding
    d = info.as_dict()
    assert isinstance(d["caveats"], list)


def test_get_adapter_unknown_name_lists_available() -> None:
    with pytest.raises(KeyError, match="rewardbench2"):
        get_adapter("nope")


def test_info_metadata_matches_verified_ground() -> None:
    expectations = {
        "rewardbench2": ("allenai/reward-bench-2", "ODC-BY", "2506.01937"),
        "judgebench": ("ScalerLab/JudgeBench", "not verified", "2410.12784"),
        "llmbar": ("princeton-nlp/LLMBar", "MIT", "ICLR"),
        "mtbench_human": ("lmsys/mt_bench_human_judgments", "CC-BY-4.0", "2306.05685"),
        "rmbench": ("THU-KEG/RM-Bench", "not verified", "2410.16184"),
    }
    for name, (hf_path, license_str, citation_token) in expectations.items():
        info = get_adapter(name).info()
        assert info.hf_path == hf_path, name
        assert info.license == license_str, name
        assert citation_token in info.citation, name


def test_unverified_licenses_are_surfaced_in_caveats() -> None:
    for name in ("judgebench", "rmbench"):
        caveats = " ".join(get_adapter(name).info().caveats).lower()
        assert "not verified" in caveats, name


# --------------------------------------------------------------------------
# RewardBench 2
# --------------------------------------------------------------------------


def test_rewardbench2_expands_best_of_4(fake_hub: dict) -> None:
    items = get_adapter("rewardbench2").load()
    # row 1: 3 rejected → 3 pairs; row 2: 2 rejected → 2 pairs
    assert len(items) == 5
    first = [i for i in items if i.meta["native_id"] == "rb2-001"]
    assert [i.item_id for i in first] == [
        "rewardbench2:rb2-001#r0",
        "rewardbench2:rb2-001#r1",
        "rewardbench2:rb2-001#r2",
    ]
    for j, item in enumerate(first):
        assert item.response_a == "Four."
        assert item.response_b == REWARDBENCH2_ROWS["test"][0]["rejected"][j]
        assert item.label == "A"
        assert item.author_a == "model-good"
        assert item.author_b == f"model-bad-{j}"
        assert item.source == "rewardbench2"
        assert item.meta["subset"] == "Math"
        assert item.meta["rejected_index"] == j


def test_rewardbench2_multi_chosen_uses_first(fake_hub: dict) -> None:
    items = [i for i in get_adapter("rewardbench2").load() if i.meta["native_id"] == "rb2-002"]
    assert len(items) == 2
    assert all(i.response_a == "Red." for i in items)
    assert all(i.meta["n_chosen"] == 2 for i in items)


def test_rewardbench2_limit_seed_determinism(fake_hub: dict) -> None:
    adapter = get_adapter("rewardbench2")
    full_ids = [i.item_id for i in adapter.load()]
    a = [i.item_id for i in adapter.load(limit=3, seed=7)]
    b = [i.item_id for i in adapter.load(limit=3, seed=7)]
    assert a == b
    assert len(a) == 3
    # subsample is order-preserving relative to the full id sequence
    assert sorted(a, key=full_ids.index) == a
    assert set(a) <= set(full_ids)


def test_rewardbench2_ids_do_not_depend_on_limit_or_seed(fake_hub: dict) -> None:
    adapter = get_adapter("rewardbench2")
    sampled = adapter.load(limit=4, seed=123)
    full = {i.item_id: i for i in adapter.load()}
    for item in sampled:
        assert item.item_id in full
        assert full[item.item_id].response_b == item.response_b


# --------------------------------------------------------------------------
# JudgeBench
# --------------------------------------------------------------------------


def test_judgebench_maps_labels_and_concatenates_splits(fake_hub: dict) -> None:
    items = get_adapter("judgebench").load()
    assert len(items) == 3
    by_id = {i.item_id: i for i in items}
    assert by_id["judgebench:jb-gpt-1"].label == "A"
    assert by_id["judgebench:jb-gpt-2"].label == "B"
    assert by_id["judgebench:jb-cl-1"].meta["hf_split"] == "claude"
    assert by_id["judgebench:jb-gpt-1"].meta["origin"] == "mmlu-pro"
    assert by_id["judgebench:jb-gpt-1"].author_a is None


def test_judgebench_single_split(fake_hub: dict) -> None:
    items = get_adapter("judgebench").load(split="gpt")
    assert {i.meta["hf_split"] for i in items} == {"gpt"}
    assert len(items) == 2


def test_judgebench_bad_label_raises(fake_hub: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"gpt": [dict(JUDGEBENCH_ROWS["gpt"][0], label="A>>B")]}
    monkeypatch.setattr(base, "load_hf_rows", lambda p, s=None: bad)
    with pytest.raises(DatasetSchemaError, match="A>>B"):
        get_adapter("judgebench").load()


def test_judgebench_lowercase_column_variant_resolves(
    fake_hub: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "id": "x1",
        "prompt": "Q?",
        "response_a": "left",
        "response_b": "right",
        "label": "B>A",
    }
    monkeypatch.setattr(base, "load_hf_rows", lambda p, s=None: {"gpt": [row]})
    (item,) = get_adapter("judgebench").load()
    assert item.item_id == "judgebench:x1"
    assert item.response_a == "left"
    assert item.label == "B"


# --------------------------------------------------------------------------
# LLMBar
# --------------------------------------------------------------------------


def test_llmbar_maps_labels_and_subsets(fake_hub: dict) -> None:
    items = get_adapter("llmbar").load()
    assert len(items) == 3
    natural = [i for i in items if i.meta["subset"] == "Natural"]
    assert len(natural) == 1
    assert natural[0].label == "A"  # label 1 → A
    assert natural[0].meta["adversarial"] is False
    neighbor = [i for i in items if i.meta["subset"] == "Adversarial_Neighbor"]
    assert neighbor[0].label == "B"  # label 2 → B
    assert neighbor[0].meta["adversarial"] is True


def test_llmbar_adversarial_pseudo_split(fake_hub: dict) -> None:
    items = get_adapter("llmbar").load(split="Adversarial")
    assert len(items) == 2
    assert all(i.meta["adversarial"] for i in items)


def test_llmbar_concrete_split_passthrough(fake_hub: dict) -> None:
    items = get_adapter("llmbar").load(split="Natural")
    assert [i.meta["subset"] for i in items] == ["Natural"]


def test_llmbar_bad_label_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"input": "x", "output_1": "a", "output_2": "b", "label": 3}
    monkeypatch.setattr(llmbar, "fetch_split_rows", lambda s: [dict(row)])
    with pytest.raises(DatasetSchemaError, match="expected 1 or 2"):
        get_adapter("llmbar").load(split="Natural")


def test_llmbar_unknown_split_raises_before_any_download() -> None:
    # The KeyError is raised by the seam itself before building a URL, so
    # this needs no mocking and never touches the network.
    with pytest.raises(KeyError, match="unknown split"):
        llmbar.fetch_split_rows("nope")


def test_llmbar_loads_without_datasets_package(
    fake_hub: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LLMBar is fetched from GitHub raw via stdlib urllib — the optional
    # `datasets` package must NOT be required (the HF repo is a loading
    # script only; see the adapter module docstring).
    monkeypatch.setitem(sys.modules, "datasets", None)
    items = get_adapter("llmbar").load(split="Natural")
    assert len(items) == 1


# --------------------------------------------------------------------------
# MT-Bench human judgments
# --------------------------------------------------------------------------


def test_mtbench_human_mapping(fake_hub: dict) -> None:
    items = get_adapter("mtbench_human").load()
    # 3 human rows, one is turn-2 → skipped; gpt4_pair split untouched
    assert len(items) == 2
    first = items[0]
    assert first.item_id == "mtbench_human:81:alpaca-13b:vicuna-13b:expert_0"
    assert first.author_a == "alpaca-13b"
    assert first.author_b == "vicuna-13b"
    assert first.label == "B"
    assert first.prompt == "Compose a travel blog post about Hawaii."
    assert first.response_a == "Short Hawaii post."
    assert first.response_b == "Long vivid Hawaii post."
    assert first.meta["question_id"] == "81"


def test_mtbench_human_tie_bothbad_maps_to_tie(fake_hub: dict) -> None:
    by_judge = {i.meta["judge"]: i for i in get_adapter("mtbench_human").load()}
    assert by_judge["expert_1"].label == "tie"
    assert by_judge["expert_1"].meta["raw_winner"] == "tie (bothbad)"


def test_mtbench_human_skips_turn_two(fake_hub: dict) -> None:
    items = get_adapter("mtbench_human").load()
    assert all(i.meta["turn"] == 1 for i in items)
    assert not any(i.meta["judge"] == "expert_2" for i in items)


def test_mtbench_human_default_split_is_human(fake_hub: dict) -> None:
    items = get_adapter("mtbench_human").load()
    assert all(i.meta["hf_split"] == "human" for i in items)
    # explicit gpt4_pair still reachable
    gpt4 = get_adapter("mtbench_human").load(split="gpt4_pair")
    assert len(gpt4) == 1
    assert gpt4[0].meta["judge"] == "gpt4_pair"


def test_mtbench_human_per_judge_ids_are_distinct(fake_hub: dict) -> None:
    ids = [i.item_id for i in get_adapter("mtbench_human").load()]
    assert len(ids) == len(set(ids))


def test_mtbench_human_bad_winner_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    row = dict(MTBENCH_ROWS["human"][0], winner="model_c")
    monkeypatch.setattr(base, "load_hf_rows", lambda p, s=None: {"human": [row]})
    with pytest.raises(DatasetSchemaError, match="model_c"):
        get_adapter("mtbench_human").load()


def test_mtbench_human_malformed_conversation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = dict(MTBENCH_ROWS["human"][0], conversation_a="not-a-list")
    monkeypatch.setattr(base, "load_hf_rows", lambda p, s=None: {"human": [row]})
    with pytest.raises(DatasetSchemaError, match="conversation_a"):
        get_adapter("mtbench_human").load()


# --------------------------------------------------------------------------
# RM-Bench
# --------------------------------------------------------------------------


def test_rmbench_expands_3x3_style_matrix(fake_hub: dict) -> None:
    items = get_adapter("rmbench").load()
    assert len(items) == 18  # 2 rows x 9 pairs
    row1 = [i for i in items if i.meta["native_id"] == "rmb-1"]
    assert len(row1) == 9
    ids = {i.item_id for i in row1}
    assert "rmbench:rmb-1#c0r0" in ids
    assert "rmbench:rmb-1#c2r1" in ids
    sample = next(i for i in row1 if i.item_id == "rmbench:rmb-1#c1r2")
    assert sample.response_a == "Detailed correct."
    assert sample.response_b == "## Markdown wrong."
    assert sample.label == "A"
    assert sample.meta["chosen_style"] == 1
    assert sample.meta["rejected_style"] == 2
    assert sample.meta["domain"] == "chat"


def test_rmbench_limit_seed_determinism(fake_hub: dict) -> None:
    adapter = get_adapter("rmbench")
    a = [i.item_id for i in adapter.load(limit=5, seed=11)]
    b = [i.item_id for i in adapter.load(limit=5, seed=11)]
    assert a == b and len(a) == 5
    c = [i.item_id for i in adapter.load(limit=5, seed=12)]
    # 18 choose 5 = 8568 subsets — different seeds virtually never collide
    assert a != c


# --------------------------------------------------------------------------
# Shared behavior: schema errors, missing-dependency message, helpers
# --------------------------------------------------------------------------


def test_missing_prompt_column_lists_found_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {"weird_col": "x", "chosen": ["a"], "rejected": ["b"]}
    monkeypatch.setattr(base, "load_hf_rows", lambda p, s=None: {"test": [row]})
    with pytest.raises(DatasetSchemaError) as exc_info:
        get_adapter("rewardbench2").load()
    message = str(exc_info.value)
    assert "prompt" in message
    assert "weird_col" in message  # found columns are listed


@pytest.mark.parametrize("name", [a for a in EXPECTED_ADAPTERS if a != "llmbar"])
def test_import_error_without_datasets_package(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Poison the import: None in sys.modules makes `import datasets` raise
    # ImportError regardless of whether the package is installed.
    # (llmbar is excluded: it fetches GitHub raw JSON via stdlib urllib
    # and needs no optional package — covered by
    # test_llmbar_loads_without_datasets_package.)
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"judgecal\[hf\]"):
        get_adapter(name).load()


def test_info_does_not_require_datasets_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    for name in EXPECTED_ADAPTERS:
        assert get_adapter(name).info().hf_path  # no ImportError


def test_sample_limit_no_op_when_not_binding() -> None:
    items = ["a", "b", "c"]
    assert base.sample_limit(items, None, 0) == items
    assert base.sample_limit(items, 3, 0) == items
    assert base.sample_limit(items, 10, 0) == items
    assert base.sample_limit(items, 0, 0) == []


def test_sample_limit_preserves_order() -> None:
    items = list(range(100))
    sampled = base.sample_limit(items, 10, seed=42)
    assert sampled == sorted(sampled)
    assert len(set(sampled)) == 10


def test_pick_field_priority_order() -> None:
    row = {"prompt": "p", "text": "t"}
    assert base.pick_field(row, ("prompt", "text"), dataset="x", fieldname="prompt") == "p"
    assert base.pick_field(row, ("text", "prompt"), dataset="x", fieldname="prompt") == "t"


def test_as_str_list_normalization() -> None:
    assert base.as_str_list(None) == []
    assert base.as_str_list("a") == ["a"]
    assert base.as_str_list(["a", "b"]) == ["a", "b"]
    assert base.first_or_none([]) is None
    assert base.first_or_none("x") == "x"


# --------------------------------------------------------------------------
# Real-download smoke test (opt-in: -m network AND JUDGECAL_NETWORK_TESTS=1)
# --------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("JUDGECAL_NETWORK_TESTS") != "1",
    reason="set JUDGECAL_NETWORK_TESTS=1 to enable real downloads",
)
def test_llmbar_real_download_smoke() -> None:
    # LLMBar downloads GitHub raw JSON via stdlib urllib — no extras needed.
    items = get_adapter("llmbar").load(split="Natural", limit=5, seed=0)
    assert 0 < len(items) <= 5
    for item in items:
        assert item.item_id.startswith("llmbar:")
        assert item.prompt and item.response_a and item.response_b
        assert item.label in ("A", "B")
