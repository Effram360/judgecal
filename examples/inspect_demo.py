"""judgecal x Inspect AI demo -- fully offline: no network, no real LLM.

Two things are demonstrated, both runnable on a laptop:

1. **Scoring with judgecal's instrument inside an Inspect eval.**
   ``judgecal_pairwise`` is an Inspect ``@scorer`` factory that renders
   judgecal's default pairwise judge template over a sample carrying two
   candidate responses and parses the ``[[A]]``/``[[B]]``/``[[C]]`` verdict
   marker with judgecal's real parser. Here the judge is Inspect's
   offline ``mockllm/model`` with scripted completions, so the demo never
   touches a network or a real model. With a real judge you would pass
   e.g. ``model="vllm/qwen3.5-9b-awq"`` instead -- the scorer is also
   registered via the ``inspect_ai`` entry-point group, so
   ``inspect score --scorer judgecal/judgecal_pairwise`` works on existing
   logs (note: ``inspect score`` writes a new ``*-scored.eval`` file by
   default and *appends* score sets; pass ``--overwrite`` to replace).

2. **Auditing re-scored Inspect logs with judgecal's probes.**
   ``samples_df_to_judgments`` maps a ``samples_df()``-style dataframe
   (one row per judged presentation, with ``item_id`` / ``condition`` /
   ``verdict`` columns) into judgecal judgments, which flow into
   ``analyze_suite`` and ``build_card`` exactly like judgments from
   judgecal's own executors. Here the dataframe is hand-built with a
   planted lean toward the presented-first slot, so the position probe
   has something to find.

Usage::

    python examples/inspect_demo.py

Requires the optional extra: ``pip install 'judgecal[inspect]'``.
"""

from __future__ import annotations

import sys
import tempfile

import pandas as pd


def run_inspect_eval_part() -> None:
    """Part 1: judgecal_pairwise scoring inside an offline Inspect eval."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model

    from judgecal.integrations.inspect_ai import judgecal_pairwise

    # Three pairwise samples. The scorer reads metadata["first_text"] /
    # metadata["second_text"] (the two candidates, in presentation order)
    # and an optional metadata["prompt"] (falls back to the sample input).
    # target is the expected verdict in presented coordinates ("A"/"B"/"C").
    samples = [
        Sample(
            input="What is 2 + 2?",
            target="A",
            metadata={
                "first_text": "2 + 2 = 4.",
                "second_text": "It might be 5, depending on the axioms.",
            },
        ),
        Sample(
            input="Name the capital of France.",
            target="A",
            metadata={
                "first_text": "Paris.",
                "second_text": "France has many large cities, such as Lyon.",
            },
        ),
        Sample(
            input="Is the Earth flat?",
            target="B",
            metadata={
                "first_text": "Yes, obviously.",
                "second_text": "No -- it is an oblate spheroid.",
            },
        ),
    ]

    # The judge: Inspect's offline mock model with scripted completions.
    # Every completion picks the presented-first response, so the third
    # sample (where B is correct) is scored INCORRECT -> accuracy 2/3.
    completion = "The first response answers directly and correctly.\n[[A]]"
    judge = get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content=completion)
            for _ in samples
        ],
    )

    task = Task(dataset=samples, scorer=judgecal_pairwise(model=judge))

    # The *evaluated* model is also mockllm (its outputs are irrelevant
    # here -- the scorer judges the metadata texts, not the model output).
    # Logs go to a temp dir so the demo leaves nothing behind.
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = inspect_eval(task, model="mockllm/model", log_dir=log_dir, display="none")

    print("== Part 1: judgecal_pairwise inside an Inspect eval (mockllm) ==")
    assert log.results is not None
    for metric_name, metric in log.results.scores[0].metrics.items():
        print(f"  {metric_name}: {metric.value:.3f}")
    print()


def run_audit_adapter_part() -> None:
    """Part 2: pipe re-scored Inspect logs into judgecal's position probe."""
    from judgecal.integrations.inspect_ai import samples_df_to_judgments
    from judgecal.probes import ProbeConfig, analyze_suite
    from judgecal.report import build_card, render_markdown

    # Hand-built stand-in for an Inspect samples_df() of re-scored logs:
    # every item judged twice (orig: response A presented first; swap:
    # response B first). The verdicts lean toward the presented-first
    # slot ~75% of the time in both passes -- a planted position bias.
    # In real use you would derive these columns from
    # ``inspect_ai.analysis.samples_df(<the *-scored.eval log>)``:
    # item_id from the sample id, condition from your task metadata, and
    # verdict from the judgecal_pairwise score's "answer" field.
    rows = []
    for i in range(40):
        rows.append(
            {
                "item_id": f"item-{i:03d}",
                "condition": "orig",
                "verdict": "first" if i % 4 != 0 else "second",
            }
        )
        rows.append(
            {
                "item_id": f"item-{i:03d}",
                "condition": "swap",
                "verdict": "first" if i % 4 != 1 else "second",
            }
        )
    df = pd.DataFrame(rows)

    judgments = samples_df_to_judgments(df)  # probe="position" is the default
    results = analyze_suite(judgments, ["position"], ProbeConfig(n_boot=500, seed=0))
    card = build_card(results, judge={"model": "mock-rescored-judge", "via": "inspect"})

    print("== Part 2: samples_df -> judgments -> position probe -> card ==")
    print(render_markdown(card))


def main() -> int:
    try:
        import judgecal.integrations.inspect_ai  # noqa: F401
    except ImportError as exc:
        print(exc, file=sys.stderr)
        print("This demo needs the extra: pip install 'judgecal[inspect]'", file=sys.stderr)
        return 1

    run_inspect_eval_part()
    run_audit_adapter_part()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
