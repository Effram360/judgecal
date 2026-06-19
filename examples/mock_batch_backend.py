"""Simulate a cluster batch run locally with the deterministic mock judge.

Plays the role of ``vllm run-batch`` for the batch workflow walkthrough:
reads a manifest JSONL (and its sidecar, for the request metadata the mock
judge needs) and writes an OpenAI-batch-format results JSONL — one
``{"custom_id", "response": {"body": ...}}`` line per manifest line — that
``judgecal ingest`` consumes exactly like real cluster output.

Optionally plants biases so the downstream analysis has something to find.
Zero LLM, zero network.

Usage::

    python examples/mock_batch_backend.py \
        --manifest out/manifest.jsonl --sidecar out/manifest.meta.jsonl \
        --beta-position 0.8 -o out/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from judgecal.core import JudgmentRequest
from judgecal.fixtures import MockJudge, MockJudgeConfig
from judgecal.manifests import load_sidecar


def main() -> None:
    """Execute manifest lines with the mock judge; write batch-format results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest JSONL from `plan`")
    parser.add_argument(
        "--sidecar", type=Path, required=True, help="matching sidecar (<name>.meta.jsonl)"
    )
    parser.add_argument("-o", "--out", type=Path, required=True, help="output results JSONL")
    parser.add_argument("--seed", type=int, default=7, help="mock judge seed (default: 7)")
    parser.add_argument(
        "--beta-position", type=float, default=0.0, help="planted position bias (log-odds)"
    )
    parser.add_argument(
        "--beta-length", type=float, default=0.0, help="planted verbosity bias (per log-ratio)"
    )
    args = parser.parse_args()

    judge = MockJudge(
        MockJudgeConfig(
            seed=args.seed, beta_position=args.beta_position, beta_length=args.beta_length
        )
    )
    entries = load_sidecar(args.sidecar)

    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8") as src, args.out.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            custom_id = line["custom_id"]
            entry = entries[custom_id]  # KeyError = manifest/sidecar mismatch; fail loudly
            request = JudgmentRequest(
                custom_id=custom_id, body=line["body"], meta=dict(entry.usages[0])
            )
            raw_text = judge.judge(request)
            dst.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "response": {"body": {"choices": [{"message": {"content": raw_text}}]}},
                        "error": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"executed {n} manifest lines (mock judge) -> {args.out}")


if __name__ == "__main__":
    main()
