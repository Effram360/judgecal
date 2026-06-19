"""Write a synthetic items JSONL file for the batch workflow walkthrough.

Generates deterministic pairwise items with judgecal's synthetic fixture
generator (zero LLM, zero network) and writes them in the canonical items
JSONL schema that ``judgecal plan`` consumes — the same schema
``judgecal datasets fetch`` produces from real preference datasets.

Usage::

    python examples/make_synthetic_items.py --n 150 --seed 7 -o items.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from judgecal.fixtures import SyntheticConfig, generate_items


def main() -> None:
    """Generate synthetic items and write them as items JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150, help="number of items (default: 150)")
    parser.add_argument("--seed", type=int, default=7, help="generator seed (default: 7)")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("items.jsonl"), help="output items JSONL path"
    )
    args = parser.parse_args()

    items = generate_items(SyntheticConfig(n_items=args.n, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "prompt": item.prompt,
                        "response_a": item.response_a,
                        "response_b": item.response_b,
                        "label": item.label,
                        "author_a": item.author_a,
                        "author_b": item.author_b,
                        "source": item.source,
                        "meta": item.meta,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"wrote {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
