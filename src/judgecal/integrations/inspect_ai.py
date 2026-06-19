"""Inspect AI integration: a pairwise judge scorer + a samples_df audit adapter.

Verified against inspect-ai 0.3.239 (2026-06-10 version pins,
``docs/research/03-version-pins-2026-06-10.md``):

* :func:`judgecal_pairwise` is an Inspect ``@scorer`` factory that judges a
  pair of candidate responses with judgecal's default pairwise template
  (``judgecal.probes.templates``, id ``tpl:default``) and parses the
  MT-Bench-style ``[[A]]``/``[[B]]``/``[[C]]`` marker with judgecal's real
  verdict parser. Inspect users therefore score with the exact instrument
  judgecal audits.
* :func:`samples_df_to_judgments` maps an Inspect ``samples_df()``-style
  dataframe (re-scored logs) into :class:`judgecal.core.Judgment` objects,
  so Inspect outputs can be piped into ``judgecal.probes.analyze_suite``
  and ``judgecal.report.build_card``.

Grader-resolution note (pins delta 2): like ``model_graded_qa``, when
``model=None`` the scorer grades with the model resolved by
``inspect_ai.model.get_model(None)`` — inside an eval that is the evaluated
model itself (self-grading). Pass an explicit ``model`` (or bind a grader
role) to control this; the resolved model name is recorded in
``Score.metadata["models"]``.

This module requires the optional ``inspect-ai`` dependency and raises a
helpful :class:`ImportError` otherwise; ``judgecal._registry`` (the
``inspect_ai`` entry point target) catches that error and degrades
silently.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import pandas as pd

from judgecal.core import Judgment, PresentedVerdict
from judgecal.executors.parsing import parse_verdict
from judgecal.probes.templates import DEFAULT_TEMPLATE_ID, render

try:
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
    from inspect_ai.scorer import (
        CORRECT,
        INCORRECT,
        Score,
        Scorer,
        Target,
        accuracy,
        scorer,
        stderr,
    )
except ImportError as _exc:  # pragma: no cover - exercised via sys.modules poisoning
    raise ImportError(
        "judgecal.integrations.inspect_ai requires the optional dependency "
        "'inspect-ai'. Install it with: pip install 'judgecal[inspect]'"
    ) from _exc

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage
    from inspect_ai.solver import TaskState

#: Literal presented-coordinates verdict strings accepted as-is by the
#: samples_df adapter (anything else is parsed as raw judge text).
_VERDICT_LITERALS: tuple[PresentedVerdict, ...] = ("first", "second", "tie", "invalid")

#: Deterministic tie-break precedence for multi-model majority voting.
_VOTE_PRECEDENCE: tuple[PresentedVerdict, ...] = ("first", "second", "tie", "invalid")

#: Target spellings accepted by the scorer, normalized to presented coords.
_TARGET_MAP: dict[str, PresentedVerdict] = {
    "a": "first",
    "b": "second",
    "c": "tie",
    "first": "first",
    "second": "second",
    "tie": "tie",
}

#: Conditions the position probe understands, with derived presentation order.
_POSITION_FIRST_IS_A: dict[str, bool] = {"orig": True, "swap": False}


def _majority(verdicts: list[PresentedVerdict]) -> PresentedVerdict:
    """Majority vote over verdicts; count ties broken by ``_VOTE_PRECEDENCE``."""
    counts = Counter(verdicts)
    best = max(counts.values())
    for verdict in _VOTE_PRECEDENCE:
        if counts.get(verdict, 0) == best:
            return verdict
    return "invalid"  # pragma: no cover - precedence covers every verdict


def _normalize_target(text: str) -> PresentedVerdict:
    """Normalize a target spelling ("A"/"B"/"C" or "first"/"second"/"tie")."""
    key = text.strip().lower()
    try:
        return _TARGET_MAP[key]
    except KeyError:
        raise ValueError(
            f"unsupported target {text!r}; expected one of "
            "'A'/'B'/'C' or 'first'/'second'/'tie' (presented coordinates)"
        ) from None


@scorer(metrics=[accuracy(), stderr()])
def judgecal_pairwise(model: str | Model | list[str | Model] | None = None) -> Scorer:
    """Pairwise judge scorer using judgecal's default template and parser.

    Expected Sample fields:

    * ``metadata["first_text"]`` (str, required) — the presented-first
      candidate response (the judge sees it as "Assistant A").
    * ``metadata["second_text"]`` (str, required) — the presented-second
      candidate response ("Assistant B").
    * ``metadata["prompt"]`` (str, optional) — the user question; when
      absent, ``state.input_text`` (the sample input) is used.
    * ``target`` (optional) — the expected verdict in *presented*
      coordinates: ``"A"``/``"B"``/``"C"`` or ``"first"``/``"second"``/
      ``"tie"``. With a non-empty target the Score value is
      CORRECT/INCORRECT (so ``accuracy()`` is meaningful); without one the
      Score value is the verdict string itself.

    The judge prompt is judgecal's ``tpl:default`` pairwise template
    (rendered via :func:`judgecal.probes.templates.render`) and the
    completion is parsed with :func:`judgecal.executors.parsing.parse_verdict`
    (last ``[[A]]``/``[[B]]``/``[[C]]`` marker; unparseable → "invalid").

    Args:
        model: Judge model — a model name, a ``Model`` instance, a list of
            either (independent grading + majority vote, count ties broken
            deterministically in the order first/second/tie/invalid), or
            ``None`` to use the model resolved by ``get_model(None)``
            (inside an eval: the evaluated model itself — self-grading).

    Returns:
        An async Inspect scorer. ``Score.answer`` is the parsed verdict;
        ``Score.metadata`` records per-model verdicts and model names.

    Raises:
        ValueError: At score time, if the sample metadata is missing
            ``first_text``/``second_text`` or the target spelling is
            unsupported.
    """

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        missing = [k for k in ("first_text", "second_text") if not isinstance(metadata.get(k), str)]
        if missing:
            raise ValueError(
                "judgecal_pairwise requires string sample metadata fields "
                f"{missing}; got metadata keys {sorted(metadata)}"
            )
        prompt = metadata.get("prompt") or state.input_text
        messages = render(
            DEFAULT_TEMPLATE_ID, prompt, metadata["first_text"], metadata["second_text"]
        )
        chat: list[ChatMessage] = [
            ChatMessageSystem(content=messages[0]["content"]),
            ChatMessageUser(content=messages[1]["content"]),
        ]

        graders = model if isinstance(model, list) else [model]
        verdicts: list[PresentedVerdict] = []
        completions: list[str] = []
        names: list[str] = []
        for grader in graders:
            resolved = get_model(grader)
            output = await resolved.generate(chat)
            verdicts.append(parse_verdict(output.completion))
            completions.append(output.completion)
            names.append(str(resolved.name))
        verdict = _majority(verdicts)

        value: str = verdict
        target_text = target.text.strip()
        if target_text:
            value = CORRECT if verdict == _normalize_target(target_text) else INCORRECT
        return Score(
            value=value,
            answer=verdict,
            explanation="\n\n---\n\n".join(completions),
            metadata={"verdicts": list(verdicts), "models": names},
        )

    return score


def _verdict_and_raw(value: Any, row: Any) -> tuple[PresentedVerdict, str | None]:
    """Map a verdict cell to (presented verdict, raw_text)."""
    if not isinstance(value, str):
        raise ValueError(
            f"verdict column value at row {row} must be a string; got {type(value).__name__} "
            f"({value!r}). Expected one of {list(_VERDICT_LITERALS)} or raw judge text "
            "containing a [[A]]/[[B]]/[[C]] marker."
        )
    if value in _VERDICT_LITERALS:
        return value, None  # type: ignore[return-value]
    return parse_verdict(value), value


def samples_df_to_judgments(
    df: pd.DataFrame,
    *,
    probe: str = "position",
    condition_col: str = "condition",
    verdict_col: str = "verdict",
    item_col: str = "item_id",
) -> list[Judgment]:
    """Map an Inspect ``samples_df()``-style dataframe to judgecal judgments.

    Pipe re-scored Inspect logs into judgecal's analyses::

        judgments = samples_df_to_judgments(df)
        results = analyze_suite(judgments, ["position"], ProbeConfig())
        card = build_card(results, judge={"model": "..."})

    Required columns (STRICT — a clear ``ValueError`` names anything
    missing, and null values in these columns are rejected):

    * ``item_col`` (default ``"item_id"``) — stable pair identifier; the
      same id must appear on every condition row of an item (it is the
      cluster id for the bootstrap).
    * ``condition_col`` (default ``"condition"``) — probe condition. For
      ``probe="position"`` values must be ``"orig"`` (presented-first is
      response A) or ``"swap"`` (presented-first is response B);
      ``first_is_a`` is derived from them. For any other probe the
      dataframe must additionally contain a boolean ``first_is_a`` column,
      because presentation order cannot be derived from arbitrary
      condition names.
    * ``verdict_col`` (default ``"verdict"``) — either a literal
      presented-coordinates verdict (``"first"``/``"second"``/``"tie"``/
      ``"invalid"``) used as-is, or raw judge text, which is parsed with
      :func:`judgecal.executors.parsing.parse_verdict` (no marker →
      ``"invalid"``) and preserved as ``Judgment.raw_text``.

    Optional columns, merged into ``Judgment.meta`` when present:
    ``repeat`` (int, default 0), ``first_len``/``second_len`` (int,
    default 0 — supply real presented lengths if you intend the verbosity
    probe's observational length GLM), ``first_author``/``second_author``
    (default ``None``), ``label_first`` (ground truth in presented
    coordinates, default ``None``), and ``custom_id`` (default: a
    deterministic ``"inspect-{item}-{condition}-r{repeat}"`` id).

    Args:
        df: The samples dataframe (one row per judged presentation).
        probe: Probe name written into ``meta["probe"]`` (default
            ``"position"``).
        condition_col: Name of the condition column.
        verdict_col: Name of the verdict column.
        item_col: Name of the item-id column.

    Returns:
        One :class:`~judgecal.core.Judgment` per row, in row order, with
        every ``judgecal.core.REQUIRED_META_KEYS`` entry populated
        (latent qualities are ``None``: Inspect logs carry no planted
        truths).

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If required columns are missing, contain nulls, hold
            invalid condition values for the position probe, hold
            non-string verdicts, or ``first_is_a`` is required but absent.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    required = {
        "item_col": item_col,
        "condition_col": condition_col,
        "verdict_col": verdict_col,
    }
    missing = [f"{role}={name!r}" for role, name in required.items() if name not in df.columns]
    if missing:
        raise ValueError(
            f"samples dataframe is missing required column(s) {missing}; "
            f"available columns: {list(df.columns)}"
        )
    for name in (item_col, condition_col, verdict_col):
        null_rows = df.index[df[name].isna()].tolist()
        if null_rows:
            raise ValueError(f"column {name!r} contains null values at rows {null_rows[:10]}")

    derive_first_is_a = probe == "position"
    if derive_first_is_a:
        bad = sorted(set(df[condition_col]) - set(_POSITION_FIRST_IS_A))
        if bad:
            raise ValueError(
                f"probe 'position' requires condition values in "
                f"{sorted(_POSITION_FIRST_IS_A)}; got unexpected values {bad}"
            )
    elif "first_is_a" not in df.columns:
        raise ValueError(
            f"probe {probe!r} requires an explicit boolean 'first_is_a' column "
            "(presentation order cannot be derived from its condition values); "
            f"available columns: {list(df.columns)}"
        )

    judgments: list[Judgment] = []
    for row_label, row in df.iterrows():
        item_id = str(row[item_col])
        condition = str(row[condition_col])
        repeat = int(row["repeat"]) if "repeat" in df.columns else 0
        first_is_a = (
            _POSITION_FIRST_IS_A[condition] if derive_first_is_a else bool(row["first_is_a"])
        )
        verdict, raw_text = _verdict_and_raw(row[verdict_col], row_label)
        meta: dict[str, Any] = {
            "probe": probe,
            "condition": condition,
            "item_id": item_id,
            "repeat": repeat,
            "first_is_a": first_is_a,
            "first_len": int(row["first_len"]) if "first_len" in df.columns else 0,
            "second_len": int(row["second_len"]) if "second_len" in df.columns else 0,
            "first_author": _opt_str(row, df, "first_author"),
            "second_author": _opt_str(row, df, "second_author"),
            "first_latent_q": None,
            "second_latent_q": None,
            "label_first": _opt_str(row, df, "label_first"),
        }
        custom_id = (
            str(row["custom_id"])
            if "custom_id" in df.columns and not pd.isna(row["custom_id"])
            else f"inspect-{item_id}-{condition}-r{repeat}"
        )
        judgments.append(
            Judgment(custom_id=custom_id, verdict=verdict, raw_text=raw_text, meta=meta)
        )
    return judgments


def _opt_str(row: pd.Series, df: pd.DataFrame, column: str) -> str | None:
    """Optional string cell: None when the column is absent or the cell null."""
    if column not in df.columns or pd.isna(row[column]):
        return None
    return str(row[column])


__all__ = ["judgecal_pairwise", "samples_df_to_judgments"]
