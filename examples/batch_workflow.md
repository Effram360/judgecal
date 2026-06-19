# Batch workflow walkthrough — plan → run → ingest → analyze

This is the full judgecal batch loop, runnable **entirely offline on a laptop**: a
mock batch backend stands in for the GPU node, so you can rehearse every command
before spending a single GPU-hour. On a real run you swap exactly one step (step 3).

All commands were executed verbatim to produce the outputs shown (judgecal
0.1.0). Run them from the repo root with the package installed
(`pip install -e .` or `pip install judgecal`).

## 0. The shape of the loop

```text
items.jsonl ──> judgecal plan ──> manifest.jsonl ──────────────┐
                       │                                       ▼
                       │                          vllm run-batch  (GPU node)
                       │                          — or —
                       │                          judgecal slurm-pack → sbatch
                       │                                       │
                       └──> manifest.meta.jsonl ──┐            ▼
                            (sidecar)             ├──> judgecal ingest ──> judgments.jsonl
                            results.jsonl ────────┘            │
                                                               ▼
                                                    judgecal analyze ──> card.json + card.md
```

Key idea: judgecal itself never talks to a model. `plan` emits an OpenAI-batch-format
JSONL manifest; *anything* that can execute that format (vLLM `run-batch`, a SLURM
job, the mock backend below) produces results; `ingest` joins them back via the
sidecar; `analyze` does the statistics.

## 1. Items

Real audits start from a preference dataset (`judgecal datasets fetch llmbar
--limit 200 -o items.jsonl`, needs `judgecal[hf]` + network) or your own items
JSONL — one object per line with `item_id`, `prompt`, `response_a`, `response_b`
(plus optional `label`, `author_a`, `author_b`, `source`, `meta`).

For this offline walkthrough, generate synthetic items with known latent qualities:

```bash
python examples/make_synthetic_items.py --n 150 --seed 7 -o /tmp/jc-walk/items.jsonl
# wrote 150 items -> /tmp/jc-walk/items.jsonl
```

## 2. Plan: probe suite → manifest + sidecar

```bash
judgecal plan --items /tmp/jc-walk/items.jsonl \
    --probes position,verbosity,stability \
    --model qwen3.5-9b-awq -o /tmp/jc-walk/run
# manifest: /tmp/jc-walk/run/manifest.jsonl (1200 batch lines, 1350 usages)
# sidecar:  /tmp/jc-walk/run/manifest.meta.jsonl
```

Note the dedup: 1350 probe usages collapse to 1200 executed lines, because identical
request bodies (e.g. the position probe's `orig` pass and the stability probe's first
repeat) are content-hashed to the same `custom_id`. Each manifest line is a standard
OpenAI batch request:

```json
{"custom_id": "jc-<hash24>-r0", "method": "POST", "url": "/v1/chat/completions",
 "body": {"model": "qwen3.5-9b-awq", "messages": [...], "temperature": 0.0, ...}}
```

The sidecar records, per `custom_id`, every (probe, condition, item, repeat) usage
plus all the metadata the analyses need — results fan back out at ingest time.

For scalar reward models, plan with `--endpoint /v1/score` instead; the same
manifest format flows through vLLM's score endpoint and scores are compared
directly (no text parsing).

## 3. Execute the manifest

### 3a. Offline dry run (this walkthrough)

The mock backend executes manifest lines with judgecal's deterministic mock judge
and writes results in the same OpenAI-batch output format a real backend produces.
Plant a position bias so the analysis has something to find:

```bash
python examples/mock_batch_backend.py \
    --manifest /tmp/jc-walk/run/manifest.jsonl \
    --sidecar /tmp/jc-walk/run/manifest.meta.jsonl \
    --beta-position 0.8 -o /tmp/jc-walk/run/results.jsonl
# executed 1200 manifest lines (mock judge) -> /tmp/jc-walk/run/results.jsonl
```

### 3b. Real run, single GPU machine

Copy `manifest.jsonl` to any machine with vLLM (judgecal not needed there):

```bash
vllm run-batch -i manifest.jsonl -o results.jsonl --model qwen3.5-9b-awq
```

### 3c. Real run, SLURM cluster

```bash
judgecal slurm-pack --manifest /tmp/jc-walk/run/manifest.jsonl \
    --model qwen3.5-9b-awq --partition gpu --walltime 02:00:00 -o /tmp/jc-walk/pack
# wrote /tmp/jc-walk/pack/run_vllm.sbatch
# wrote /tmp/jc-walk/pack/run_llamacpp.sbatch
# wrote /tmp/jc-walk/pack/README_RUN.md
```

Copy the pack to the cluster and `sbatch run_vllm.sbatch` (or the llama.cpp/GGUF
variant). The scripts are array-friendly for sharded manifests, every parameter is
overridable at submit time, and `README_RUN.md` contains the exact submit + ingest
instructions. `--module`/`--container`/`--account` options adapt the pack to your
site.

## 4. Ingest: results + sidecar → judgments

```bash
judgecal ingest --sidecar /tmp/jc-walk/run/manifest.meta.jsonl \
    --results /tmp/jc-walk/run/results.jsonl \
    -o /tmp/jc-walk/run/judgments.jsonl
# wrote 1350 judgments -> /tmp/jc-walk/run/judgments.jsonl
```

1200 executed lines fan back out to 1350 judgments — one per recorded usage. The
verdict marker (`[[A]]`/`[[B]]`/`[[C]]`) is parsed here with judgecal's real parser;
unparseable outputs become `"invalid"` and are surfaced as `invalid_rate`, never
silently dropped. `--results` is repeatable for sharded outputs.

**Resume:** if the job died partway, re-plan with
`--resume /tmp/jc-walk/run/results.jsonl` — a `manifest.resume.jsonl` with only the
still-missing lines is written next to the manifest.

## 5. Analyze: judgments → reliability card

```bash
judgecal analyze --judgments /tmp/jc-walk/run/judgments.jsonl \
    --judge qwen3.5-9b-awq -o /tmp/jc-walk/card
```

This writes `card.json` + `card.md` and prints the card. With the planted 0.8
log-odds position bias, the real output begins:

```markdown
# Judge Reliability Card — qwen3.5-9b-awq

| **Scale** | 150 items · 1350 judgments · 3 probes |

## Summary

- **Position bias detected:** the judge picks the first-presented answer 64.7% of
  the time (95% CI 60.3%–69.3%, q = 0.002).
- **Underpowered — `pad_pick_rate`:** the smallest detectable effect at this sample
  size is 0.073, above the 0.050 effect-size-of-interest floor — this audit could
  not have detected effects as small as the floor. This null result is not evidence
  of absence.
- **Underpowered — `length_glm_coef`:** the smallest detectable effect at this
  sample size is 0.855, above the 0.500 effect-size-of-interest floor — this audit
  could not have detected effects as small as the floor. This null result is not
  evidence of absence.
```

Exactly right: the planted position bias is detected and FDR-significant, while the
*unplanted* verbosity metrics correctly show no signal — and because n=150 is too
small to detect effects at the pre-registered 5-pp / 0.5-coefficient floors, they
are reported as underpowered rather than as a clean bill of health.

## 6. Variations

- `judgecal demo` compresses steps 1–5 into one command (mock judge, no files).
- `judgecal claude-run` executes a small manifest through the Claude Code CLI —
  a zero-API real-LLM smoke path for templates and parsing. It uses your Claude
  subscription quota; keep `--limit` small.
- `judgecal validate` runs the planted-bias recovery suite that gates the whole
  instrument ("we test the tester" — see the README).
