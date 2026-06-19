# Quantized-judge study — pre-registration skeleton

**Status: DRAFT — to be frozen after an HPC feasibility check.**

Research question: *does quantization degrade a judge's reliability
(position/verbosity/self-preference bias, stability) before it degrades
its benchmark accuracy?* To our knowledge, neighboring work runs the
inverse direction (quantized models as evaluatees), not as judges.
Closest adjacent work: *Reliability Scaling Laws for Quantized Large
Language Models* (ICLR 2026 submission,
[OpenReview QhkW8xPH1v](https://openreview.net/forum?id=QhkW8xPH1v)) —
uncertainty/robustness of 2/3/4/8-bit quantized LLMs as evaluation
*subjects* (finds a 4-bit reliability peak); no judging task, no
agreement/bias metrics. No work we are aware of studies quantization of
the *judge*.

## Candidate matrix (finalized before the first run)

| Axis | Levels |
|---|---|
| Generative judges | Qwen3.5-9B, Qwen3.5-35B-A3B, Gemma 4 26B-A4B (incl. official QAT q4_0 arm), Qwen3.5-4B |
| Generative quant ladder | BF16, FP8, AWQ (vLLM `run-batch`); Q8/Q5/Q4/Q3 GGUF (llama.cpp in-job) |
| Scalar RMs | Skywork-Reward-V2-Llama-3.1-8B, ArmoRM-Llama3-8B |
| RM quant ladder | BF16, INT8 (bnb), NF4 (bnb), GPTQ, AWQ — via `/v1/score` |
| Ground truth | JudgeBench + LLMBar + MT-Bench-human (judges); RewardBench 2 (RMs) |
| Probes | position, verbosity, self_preference, template, stability |

Paired design (same items across all arms); per-item clustered SEs;
BH-FDR within the pre-registered family; serving noise isolated via the
stability probe at temperature 0 (vLLM batch-invariant mode and
llama.cpp single-slot arms).

## Pre-registered analysis plan (mandatory before any run)

To be written here before the first SLURM submission, containing:
1. Primary endpoints: per-probe effect deltas (quant arm − BF16 arm) with
   95% CIs; accuracy deltas on the same items.
2. MDEs at the planned n per cell (use `judgecal.stats.mde_proportion` /
   `mde_mcnemar` with design effects estimated from smoke manifests).
3. Decision rule for "reliability degrades before accuracy".
4. Null-result publication commitment (negative results are a publishable
   finding here, not a failure).

## Execution

30–50 single-GPU SLURM array jobs, ~120–180 GPU-hours total. Manifests
emitted by `judgecal plan`, job packs by `judgecal slurm-pack`. Debug-QoS
smoke manifests (10 samples/cell) first. A GPU-rental fallback is
pre-priced at $60–150 in case cluster access slips.

## Prerequisites

- HPC feasibility: GPU types, realistic allocation, scratch quota
  (~400 GB), debug QoS, container/module setup, expected queue waits.
- Hugging Face gated-model access: Gemma, Llama-3.1 (Skywork).

Raw judgments will be published to Hugging Face on completion.
