"""Tests for the Inspect AI integration (contracts §9).

Covers: the entry-point module degrading silently without inspect-ai
(simulated via sys.modules poisoning — no uninstall needed), the helpful
ImportError on the integration module itself, the ``judgecal_pairwise``
scorer factory against fabricated mockllm completions (zero network, zero
real models), and the ``samples_df_to_judgments`` audit adapter including
its strict column errors and the analyze/build_card pipeline.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import sys

import pandas as pd
import pytest

from judgecal.core import REQUIRED_META_KEYS
from judgecal.probes import ProbeConfig, analyze_suite
from judgecal.report import build_card

_HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None
requires_inspect = pytest.mark.skipif(
    not _HAS_INSPECT,
    reason="inspect-ai not installed (pip install 'judgecal[inspect]')",
)

try:
    importlib.metadata.version("judgecal")
    _JUDGECAL_INSTALLED = True
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    _JUDGECAL_INSTALLED = False

#: judgecal modules whose import outcome depends on inspect-ai availability;
#: purged around the poisoning tests so every test imports them fresh.
_GUARDED_MODULES = ("judgecal._registry", "judgecal.integrations.inspect_ai")


def _purge_guarded_modules() -> None:
    for name in _GUARDED_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture()
def no_inspect(monkeypatch: pytest.MonkeyPatch):
    """Simulate an environment without inspect-ai via sys.modules poisoning.

    ``None`` in ``sys.modules`` makes any ``import inspect_ai`` raise
    ImportError; cached ``inspect_ai.*`` submodules are removed too because
    ``from inspect_ai.model import X`` would otherwise hit the cache and
    bypass the poisoned parent. monkeypatch restores everything afterwards.
    """
    for name in [m for m in sys.modules if m == "inspect_ai" or m.startswith("inspect_ai.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "inspect_ai", None)  # type: ignore[arg-type]
    _purge_guarded_modules()
    yield
    _purge_guarded_modules()


# --------------------------------------------------------------------------
# Entry point / guarded imports
# --------------------------------------------------------------------------


def test_registry_imports_cleanly_without_inspect_ai(no_inspect: None) -> None:
    """The inspect_ai entry-point target must never crash an inspect-less env."""
    registry = importlib.import_module("judgecal._registry")
    assert registry.judgecal_pairwise is None


def test_integration_module_import_error_names_extra(no_inspect: None) -> None:
    with pytest.raises(ImportError, match=r"judgecal\[inspect\]"):
        importlib.import_module("judgecal.integrations.inspect_ai")


def test_integrations_namespace_imports_without_inspect_ai(no_inspect: None) -> None:
    importlib.import_module("judgecal.integrations")


@pytest.mark.skipif(not _JUDGECAL_INSTALLED, reason="judgecal distribution not installed")
def test_inspect_ai_entry_point_declared_and_loadable() -> None:
    """pyproject declares the inspect_ai-group entry point and it loads."""
    eps = importlib.metadata.entry_points(group="inspect_ai")
    ours = [ep for ep in eps if ep.value == "judgecal._registry"]
    assert ours, "expected an inspect_ai entry point targeting judgecal._registry"
    module = ours[0].load()
    assert hasattr(module, "judgecal_pairwise")


@requires_inspect
def test_registry_exposes_scorer_with_inspect_ai() -> None:
    _purge_guarded_modules()
    try:
        registry = importlib.import_module("judgecal._registry")
        assert registry.judgecal_pairwise is not None
        assert callable(registry.judgecal_pairwise)
    finally:
        _purge_guarded_modules()


# --------------------------------------------------------------------------
# Scorer factory (mockllm only — no real model, no network)
# --------------------------------------------------------------------------


def _mock_model(completion: str):
    from inspect_ai.model import ModelOutput, get_model

    return get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=completion)],
    )


def _state(metadata: dict | None):
    from inspect_ai.solver import TaskState

    return TaskState(
        model="mockllm/model",
        sample_id=1,
        epoch=1,
        input="What is 2 + 2?",
        messages=[],
        metadata=metadata,
    )


@requires_inspect
def test_scorer_parses_fabricated_completion() -> None:
    from inspect_ai.scorer import CORRECT, Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    model = _mock_model("A mentions [[A]] early, but on balance...\nFinal verdict: [[ b ]]")
    score_fn = judgecal_pairwise(model=model)
    score = asyncio.run(score_fn(_state({"first_text": "It is four.", "second_text": "4"}), Target("B")))
    # Last marker wins, case/whitespace tolerated → "second"; target "B" → CORRECT.
    assert score.value == CORRECT
    assert score.answer == "second"
    assert score.metadata is not None
    assert score.metadata["verdicts"] == ["second"]


@requires_inspect
def test_scorer_incorrect_and_no_target_values() -> None:
    from inspect_ai.scorer import INCORRECT, Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    score = asyncio.run(
        judgecal_pairwise(model=_mock_model("[[A]]"))(
            _state({"first_text": "x", "second_text": "y"}), Target("tie")
        )
    )
    assert score.value == INCORRECT
    assert score.answer == "first"

    # Without a target, the Score value is the verdict string itself.
    score = asyncio.run(
        judgecal_pairwise(model=_mock_model("[[C]]"))(
            _state({"first_text": "x", "second_text": "y"}), Target("")
        )
    )
    assert score.value == "tie"


@requires_inspect
def test_scorer_unparseable_completion_is_invalid() -> None:
    from inspect_ai.scorer import Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    score = asyncio.run(
        judgecal_pairwise(model=_mock_model("no verdict marker at all"))(
            _state({"first_text": "x", "second_text": "y"}), Target("")
        )
    )
    assert score.value == "invalid"
    assert score.answer == "invalid"


@requires_inspect
def test_scorer_majority_vote_over_model_list() -> None:
    from inspect_ai.scorer import Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    models = [_mock_model("[[A]]"), _mock_model("[[B]]"), _mock_model("[[A]]")]
    score = asyncio.run(
        judgecal_pairwise(model=models)(_state({"first_text": "x", "second_text": "y"}), Target(""))
    )
    assert score.value == "first"
    assert score.metadata is not None
    assert score.metadata["verdicts"] == ["first", "second", "first"]


@requires_inspect
def test_scorer_missing_metadata_raises() -> None:
    from inspect_ai.scorer import Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    score_fn = judgecal_pairwise(model=_mock_model("[[A]]"))
    with pytest.raises(ValueError, match="second_text"):
        asyncio.run(score_fn(_state({"first_text": "only one side"}), Target("")))
    with pytest.raises(ValueError, match="first_text"):
        asyncio.run(score_fn(_state(None), Target("")))


@requires_inspect
def test_scorer_rejects_unsupported_target() -> None:
    from inspect_ai.scorer import Target

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    score_fn = judgecal_pairwise(model=_mock_model("[[A]]"))
    with pytest.raises(ValueError, match="unsupported target"):
        asyncio.run(score_fn(_state({"first_text": "x", "second_text": "y"}), Target("D")))


# --------------------------------------------------------------------------
# samples_df adapter
# --------------------------------------------------------------------------


@requires_inspect
def test_adapter_maps_hand_built_dataframe() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame(
        {
            "item_id": ["it1", "it1", "it2", "it2"],
            "condition": ["orig", "swap", "orig", "swap"],
            "verdict": ["first", "Reasoning text...\n[[B]]", "tie", "no marker here"],
            "first_len": [10, 12, 8, 9],
            "second_len": [12, 10, 9, 8],
        }
    )
    judgments = samples_df_to_judgments(df)

    assert [j.verdict for j in judgments] == ["first", "second", "tie", "invalid"]
    # orig → presented-first is response A; swap → response B.
    assert [j.meta["first_is_a"] for j in judgments] == [True, False, True, False]
    # swap + "second" maps back to item coordinates "A".
    assert judgments[1].mapped_verdict == "A"
    # Literal verdicts carry no raw text; parsed text is preserved.
    assert judgments[0].raw_text is None
    assert judgments[1].raw_text == "Reasoning text...\n[[B]]"
    assert judgments[3].raw_text == "no marker here"
    # Every required meta key is populated; optional lengths picked up.
    for j in judgments:
        assert set(REQUIRED_META_KEYS) <= set(j.meta)
        assert j.meta["probe"] == "position"
        assert j.meta["first_latent_q"] is None
    assert judgments[0].meta["first_len"] == 10
    assert judgments[0].custom_id == "inspect-it1-orig-r0"


@requires_inspect
def test_adapter_pipes_into_position_analysis_and_card() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    rows = []
    for i in range(12):
        # A judge leaning to the presented-first slot in both passes.
        verdict_orig = "first" if i % 4 != 0 else "second"
        verdict_swap = "first" if i % 4 != 1 else "second"
        rows.append({"item_id": f"it{i}", "condition": "orig", "verdict": verdict_orig})
        rows.append({"item_id": f"it{i}", "condition": "swap", "verdict": verdict_swap})
    judgments = samples_df_to_judgments(pd.DataFrame(rows))

    config = ProbeConfig(n_boot=200, seed=0)
    results = analyze_suite(judgments, ["position"], config)
    assert len(results) == 1
    result = results[0]
    assert result.probe == "position"
    assert result.n_items == 12
    assert result.n_judgments == 24
    names = {e.name for e in result.estimates}
    assert "first_pick_rate" in names

    card = build_card(results, judge={"model": "inspect-rescored"})
    assert card.probes[0].probe == "position"
    assert card.probes[0].n_items == 12


@requires_inspect
def test_adapter_missing_columns_error_is_informative() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame({"item_id": ["it1"], "condition": ["orig"]})
    with pytest.raises(ValueError, match="verdict") as excinfo:
        samples_df_to_judgments(df)
    # Error lists the available columns to help the user re-map.
    assert "item_id" in str(excinfo.value)

    with pytest.raises(ValueError, match="my_verdict"):
        samples_df_to_judgments(df, verdict_col="my_verdict")


@requires_inspect
def test_adapter_rejects_bad_position_conditions() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame(
        {"item_id": ["it1"], "condition": ["reversed"], "verdict": ["first"]}
    )
    with pytest.raises(ValueError, match="orig"):
        samples_df_to_judgments(df)


@requires_inspect
def test_adapter_rejects_nulls_in_required_columns() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame(
        {"item_id": ["it1", None], "condition": ["orig", "swap"], "verdict": ["first", "second"]}
    )
    with pytest.raises(ValueError, match="null"):
        samples_df_to_judgments(df)


@requires_inspect
def test_adapter_non_position_probe_requires_first_is_a() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame(
        {"item_id": ["it1"], "condition": ["tpl:v1"], "verdict": ["first"]}
    )
    with pytest.raises(ValueError, match="first_is_a"):
        samples_df_to_judgments(df, probe="template")

    df["first_is_a"] = [True]
    judgments = samples_df_to_judgments(df, probe="template")
    assert judgments[0].meta["probe"] == "template"
    assert judgments[0].meta["first_is_a"] is True


@requires_inspect
def test_adapter_rejects_non_dataframe_and_non_string_verdicts() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    with pytest.raises(TypeError, match="DataFrame"):
        samples_df_to_judgments([{"item_id": "it1"}])  # type: ignore[arg-type]

    df = pd.DataFrame({"item_id": ["it1"], "condition": ["orig"], "verdict": [3.5]})
    with pytest.raises(ValueError, match="string"):
        samples_df_to_judgments(df)


@requires_inspect
def test_adapter_optional_columns_and_custom_ids() -> None:
    from judgecal.integrations.inspect_ai import samples_df_to_judgments

    df = pd.DataFrame(
        {
            "item_id": ["it1", "it2"],
            "condition": ["orig", "orig"],
            "verdict": ["first", "second"],
            "repeat": [0, 3],
            "first_author": ["judge-self", None],
            "label_first": ["first", None],
            "custom_id": ["my-id-1", None],
        }
    )
    judgments = samples_df_to_judgments(df)
    assert judgments[0].custom_id == "my-id-1"
    assert judgments[1].custom_id == "inspect-it2-orig-r3"
    assert judgments[0].meta["first_author"] == "judge-self"
    assert judgments[1].meta["first_author"] is None
    assert judgments[0].meta["label_first"] == "first"
    assert judgments[1].meta["repeat"] == 3
