# judgecal v0.1 — Implementation Contracts

**Status:** binding for all implementation agents (2026-06-10).
This condenses the spec-phase decisions into buildable contracts. Source
context: `docs/plans/2026-06-09-judgecal-plan.md`,
`docs/research/01-verified-technical-ground.md`. Shared types live in
`src/judgecal/core.py` — read it before writing any code.

## 0. Ground rules (every agent)

- Python ≥3.10, `src/` layout, hatchling build (see `pyproject.toml` — do not edit it unless you own it).
- **Runtime deps are ONLY:** numpy, scipy, pandas, pydantic v2, click.
  `statsmodels` is a *dev/test-only* dependency used to cross-validate our
  estimators — never import it from `src/`. `inspect-ai`, `datasets`,
  `huggingface_hub` are optional extras with guarded imports.
- **Zero LLM at dev/test time.** Tests run on an 8GB Mac with no network.
  Anything touching HF or `claude` CLI is mocked in tests or marked
  `@pytest.mark.network`.
- **Determinism everywhere.** Every stochastic function takes an explicit
  `seed` (int) or `rng: np.random.Generator`. The mock judge is a pure
  function of its config + request content. No `random` module, no time-
  dependent behavior in `src/`.
- **File ownership is exclusive.** Touch only files assigned to you; the
  shared `core.py`, `__init__.py`, `pyproject.toml` are read-only for
  module agents. Create your own subpackage `__init__.py`.
- Style: ruff-clean (`ruff check src tests`), type hints throughout,
  Google-style docstrings, `from __future__ import annotations`.
- Tests: pytest, in `tests/`, named per ownership map. Fast by default
  (<30s per module); long statistical checks behind `@pytest.mark.slow`.
- Verdict conventions (pinned): judge templates instruct the judge to end
  with `[[first]]`-style markers — concretely `[[A]]` = presented-first,
  `[[B]]` = presented-second, `[[C]]` = tie (MT-Bench convention). The
  parser maps these to `PresentedVerdict` ("first"/"second"/"tie");
  unparseable → "invalid". Invalid judgments are excluded from estimates
  but surfaced via `invalid_rate` on every `ProbeResult`.

## 1. Module map and file ownership

| Agent | Owns (src) | Owns (tests) |
|---|---|---|
| STATS | `src/judgecal/stats/` | `tests/test_stats.py`, `tests/test_stats_crossval.py` |
| MOCK | `src/judgecal/fixtures/` (mock judge, synthetic items, recorded packs) | `tests/test_mock_judge.py`, `tests/test_synthetic.py` |
| EXEC | `src/judgecal/manifests/`, `src/judgecal/executors/` | `tests/test_manifests.py`, `tests/test_executors.py` |
| PROBES | `src/judgecal/probes/` | `tests/test_probes.py` |
| DATA | `src/judgecal/datasets/` | `tests/test_datasets.py` |
| CARD | `src/judgecal/report/` | `tests/test_report.py` |
| VALID (wave 2) | `src/judgecal/validate.py` | `tests/test_validation_recovery.py` |
| INSPECT (wave 2) | `src/judgecal/integrations/`, `src/judgecal/_registry.py` | `tests/test_inspect_integration.py` |
| CLI (wave 2) | `src/judgecal/cli.py` | `tests/test_cli.py` |
| PKG (wave 2) | `README.md`, `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `examples/` | — |

Cross-module imports allowed only via public APIs defined below.

## 2. STATS — `judgecal.stats`

Standalone-importable; no judgecal imports except `core.Estimate`
(allowed). Pure numpy/scipy. Public API (re-export in
`stats/__init__.py`):

```python
wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]

cluster_bootstrap_ci(
    df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float],
    cluster_col: str, n_boot: int = 2000, alpha: float = 0.05,
    seed: int = 0, method: Literal["percentile", "basic"] = "percentile",
) -> BootstrapResult  # .estimate .ci_low .ci_high .n_clusters .boot_se
# Resample CLUSTERS (items) with replacement; recompute stat_fn on the
# concatenated resampled rows. Also expose a bootstrap p-value against a
# null value via CI inversion: smallest alpha at which null is excluded
# (grid or percentile-rank implementation; document the approach).

mcnemar_test(b: int, c: int, method: Literal["exact", "midp"] = "midp") -> McNemarResult
# b, c = discordant counts. Exact: 2 * min(P(X<=min(b,c)), 0.5) with
# X ~ Binom(b+c, 0.5), capped at 1. Mid-p subtracts half the point mass.

bh_fdr(pvals: Sequence[float]) -> np.ndarray  # monotone BH q-values

mde_proportion(n_eff: float, p0: float = 0.5, alpha: float = 0.05,
               power: float = 0.8) -> float
# Two-sided normal-approx MDE as |p - p0| detectable at given power.
# n_eff = n / design_effect. Document formula in docstring.

design_effect(cluster_sizes: Sequence[int], icc: float) -> float
# 1 + (m_bar - 1) * icc, m_bar = mean cluster size. Also provide
# estimate_icc(df, value_col, cluster_col) via ANOVA estimator (clip >= 0).

mde_mcnemar(n_discordant: int, alpha: float = 0.05, power: float = 0.8) -> float
# Detectable |P(b) - 0.5| among discordant pairs.

fleiss_kappa(table: np.ndarray) -> float
# table: items x categories counts. Cross-validate against
# statsmodels.stats.inter_rater.fleiss_kappa in tests.

logistic_fit(X: np.ndarray, y: np.ndarray, add_intercept: bool = True,
             max_iter: int = 100, tol: float = 1e-8) -> LogisticFit
# Plain IRLS logistic regression (.coef, .se (sandwich-free Wald),
# .converged). Used by the verbosity GLM; CIs for probe use come from
# cluster_bootstrap_ci re-fitting, NOT from Wald SEs.
```

Result dataclasses live in `stats/types.py`. Every estimator gets a
cross-validation test against statsmodels (proportion_confint(method=
"wilson"), mcnemar, multipletests(method="fdr_bh"), fleiss_kappa,
Logit) with tolerance 1e-6 (1e-4 for IRLS coefficients), plus
behavioral tests (clustered data → wider CIs than naive; BH monotone).

## 3. MOCK — `judgecal.fixtures`

### 3.1 Synthetic item generator — `fixtures/synthetic.py`

```python
@dataclass(frozen=True)
class SyntheticConfig:
    n_items: int
    seed: int
    quality_gap_sd: float = 1.0      # latent quality gaps ~ N(0, sd)
    tie_fraction: float = 0.1        # items with ~equal latent quality
    length_log_sd: float = 0.4       # response log-lengths vary
    authors: tuple[str, ...] = ("judge-self", "other-model")
    self_author_fraction: float = 0.5  # fraction of items with one side
                                       # authored by "judge-self"

generate_items(config: SyntheticConfig) -> list[PairwiseItem]
```

Responses are template-generated filler text whose *length* hits the
sampled target (content semantics never matter to the mock judge).
Latent qualities stored in `item.meta["latent_quality_a"/"latent_quality_b"]`
(floats; label = argmax with tie when |gap| < 0.1). Deterministic per seed.

### 3.2 Mock judge — `fixtures/mock_judge.py` (THE validation engine)

```python
@dataclass(frozen=True)
class MockJudgeConfig:
    seed: int = 0
    beta_quality: float = 3.0     # weight on latent quality gap
    beta_position: float = 0.0    # planted position bias (log-odds for FIRST)
    beta_length: float = 0.0      # planted verbosity bias per unit log-len-ratio
    beta_self: float = 0.0        # planted self-preference (log-odds)
    self_name: str = "judge-self" # author string this judge "is"
    template_sigma: float = 0.0   # sd of per-template log-odds offsets
    noise_sigma: float = 0.0      # per-(item, repeat) Gaussian logit noise
                                  #  = instability; 0 → perfectly stable
    tie_band: float = 0.25        # |logit| < tie_band → "tie"
    invalid_rate: float = 0.0     # fraction of responses emitted unparseable
```

For a request (reads ONLY `request.meta`):

```
logit_first = beta_quality * (q_first - q_second)
            + beta_position
            + beta_length * log(first_len / second_len)
            + beta_self * (1[first_author == self_name] - 1[second_author == self_name])
            + template_offset(condition)   # ~N(0, template_sigma), keyed by
                                           # hash(seed, template_id), 0 for non-tpl
            + eps                          # ~N(0, noise_sigma), keyed by
                                           # hash(seed, item_id, condition, repeat)
verdict = "tie"                      if |logit_first| < tie_band
        = "first" / "second"         deterministic: u < sigmoid(logit_first),
                                     u = uniform from hash(seed, custom_id)
```

When latent qualities are absent (real-dataset items), q from label:
labeled winner 0.8 / loser 0.2 / tie 0.5,0.5. Output is a *raw text*
judge response containing reasoning filler + the `[[A]]`/`[[B]]`/`[[C]]`
marker (so the real parser is exercised); with prob `invalid_rate`
(hash-keyed) emit marker-free text.

**Analytic truths** (the validation suite's reference): expose
`expected_first_pick_rate(config, requests) -> float` and
`expected_pad_pick_rate(...)` etc. — exact expectations computed by
averaging P(first)/(P(first)+P(second)) over the request set, conditional
on decisiveness. Must mirror the verdict mechanics exactly (document the
tie-band conditioning).

### 3.3 Recorded packs — `fixtures/packs.py`

`ResponsePack`: JSONL of `{"custom_id": ..., "raw_text": ...}` with a
header line `{"pack_version": 1, "judge": ..., "created": ...}`.
`save_pack` / `load_pack` / `pack_from_batch_output(path)`. This is how
real cluster outputs are replayed locally as fixtures.

## 4. EXEC — `judgecal.manifests` + `judgecal.executors`

### 4.1 Manifests — `manifests/emit.py`, `manifests/sidecar.py`

```python
@dataclass(frozen=True)
class ModelSpec:
    model: str                    # e.g. "qwen3.5-9b-awq"
    endpoint: Literal["/v1/chat/completions", "/v1/score"] = "/v1/chat/completions"
    temperature: float = 0.0
    max_tokens: int = 1024
    extra: dict = field(default_factory=dict)

emit_manifest(requests: Sequence[JudgmentRequest], spec: ModelSpec,
              out_dir: Path, name: str) -> ManifestPaths
```

Writes `<name>.jsonl` (OpenAI batch lines: `{"custom_id", "method": "POST",
"url": spec.endpoint, "body": {model, messages|..., temperature, ...}}`)
plus sidecar `<name>.meta.jsonl`: one line per custom_id:
`{"custom_id", "body_hash", "usages": [<full request.meta>, ...]}`.
**Dedup:** identical `custom_id` → single batch line, usages merged.
Stability repeats differ via the `-r<k>` suffix (bodies identical).

`load_sidecar`, `merge_results(sidecar, output_paths) -> list[Judgment]`
(fan each parsed output back out to one Judgment per usage),
`remaining_manifest(manifest, sidecar, output_paths) -> list[batch lines]`
for resume. Output line format: OpenAI batch output
(`{"custom_id", "response": {"body": {...}}, "error": ...}`) — tolerate
both chat-completion and score bodies, and tolerate vLLM run-batch
variants (be liberal: look for `choices[0].message.content`, else
`response.body.score`/`data[0].score`).

### 4.2 Verdict parsing — `executors/parsing.py`

`parse_verdict(raw_text: str, pattern: str | None = None) -> PresentedVerdict`.
Default: last occurrence of `[[A]]`/`[[B]]`/`[[C]]` (case-insensitive,
tolerate whitespace). Configurable regex with named groups. Score
endpoint results bypass parsing (scores compared directly →
"first"/"second"/"tie" with an epsilon band; epsilon in config).

### 4.3 Executors — `executors/base.py`, `local.py`, `claude_code.py`, `slurm.py`

```python
class Executor(Protocol):
    def execute(self, requests: Sequence[JudgmentRequest]) -> list[Judgment]: ...
```

- `MockJudgeExecutor(config: MockJudgeConfig)` — calls the mock judge,
  parses its raw text through the REAL parser. (Import from
  `judgecal.fixtures` — allowed.)
- `FixtureExecutor(pack: ResponsePack, strict: bool = True)` — replays
  recorded raw texts; strict → KeyError on missing custom_id, else
  returns "invalid" judgments with a warning.
- `ClaudeCodeExecutor(model: str | None = None, claude_bin: str = "claude",
  timeout_s: int = 120)` — runs the **Claude Code CLI headlessly**
  (`claude -p <prompt> --output-format json` plus flags you verify
  against `claude --help` locally) one request at a time, sequential.
  Renders `body["messages"]` into a single prompt. This gives Max-plan
  users a zero-API real-LLM smoke path. Must: never run in tests except
  via a mocked `subprocess.run`; document rate-limit etiquette; raise a
  clear error if the binary is missing.
- `slurm.py` — NOT an executor: generators producing a runnable job pack:
  `write_slurm_pack(manifest_paths, out_dir, cluster: SlurmConfig)` emits
  (a) `run_vllm.sbatch` (array-friendly, `vllm run-batch -i manifest.jsonl
  -o results.jsonl --model ...`), (b) `run_llamacpp.sbatch` variant,
  (c) `README_RUN.md` with exact submit/ingest instructions.
  `SlurmConfig`: partition, gpus, walltime, modules/container lines,
  scratch dir. Golden-string tests; every parameter overridable.

## 5. PROBES — `judgecal.probes`

`probes/base.py`:

```python
class Probe(ABC):
    name: ClassVar[str]
    def plan(self, items: Sequence[PairwiseItem], config: ProbeConfig) -> list[JudgmentRequest]: ...
    def analyze(self, judgments: Sequence[Judgment], config: ProbeConfig) -> ProbeResult: ...

get_probe(name) / PROBE_REGISTRY  # "position", "verbosity",
                                  # "self_preference", "template", "stability"
plan_suite(items, probes, config) -> list[JudgmentRequest]   # concatenated;
                                  # dedup happens at manifest level
analyze_suite(judgments, probes, config) -> list[ProbeResult]
```

`ProbeConfig` (dataclass): `n_template_variants=5`, `stability_k=5`,
`pad_target_ratio=1.6`, `alpha=0.05`, `n_boot=2000`, `seed=0`.

Judge prompt templates: `probes/templates.py`. A default pairwise
template + 4 semantically equivalent paraphrases (hand-written, ship
in-package, ids "tpl:default", "tpl:v1".."tpl:v4"). All instruct the
`[[A]]`/`[[B]]`/`[[C]]` output convention. `render(template_id, prompt,
first_text, second_text) -> messages`.

Analyses consume ONLY `Judgment.meta` + verdicts (never items), use
`judgecal.stats` for all inference, and populate `Estimate.mde`.
Conditions/metrics pinned:

- **position** — conditions `orig` (first=A) and `swap` (first=B), template
  default, repeat 0. Metrics:
  - `first_pick_rate`: P(verdict=="first" | decisive), both passes pooled;
    null 0.5; cluster-bootstrap CI by item_id; p via bootstrap inversion;
    mde via `stats.mde_from_se` on the realized bootstrap SE (consistent
    by construction with the CI; `mde_proportion` remains the planning
    formula). [Amended 2026-06-10, adversarial review m7.]
  - `flip_rate_decisive`: among items decisive in both passes, fraction
    whose mapped winners disagree; Wilson CI.
  - `positional_mcnemar`: b = items picking presented-first in both
    passes, c = picking presented-second in both; `mcnemar_test(b, c)`;
    estimate = b/(b+c), null 0.5, mde via `mde_mcnemar`.
- **verbosity** — constructed contrast: from each item take `response_a`,
  build `pad(text)` (deterministic meaning-preserving padding in
  `probes/padding.py`: sentence restatement + enumerated recap reaching
  `pad_target_ratio`; document the limitation that this is rule-based,
  not an LLM rewrite). Conditions `pad_second` (orig, padded) and
  `pad_first` (padded, orig). Metric `pad_pick_rate`: P(pick padded |
  decisive), pooled over both orders (position cancels); null 0.5;
  cluster bootstrap; mde via `stats.mde_from_se` on the realized
  bootstrap SE. Plus observational `length_glm_coef`: logistic
  fit of pick-first on log(first_len/second_len) (+ ground-truth-first
  control when labels exist) over position-probe judgments — analysis
  may declare it needs `position` judgments via
  `requires: ClassVar[tuple[str, ...]]`; `analyze_suite` passes the
  union. Cluster-bootstrap CI on the coefficient; null 0.0; method
  tagged `"(observational)"` — rendered without a rejected/clear verdict
  glyph and footnoted on the card, because quality–length correlation
  inflates the coefficient (it is an association, not the experimental
  estimate; `pad_pick_rate` is). [Amended 2026-06-10, review M4.]
- **self_preference** — no new conditions: reuses `orig`+`swap`. Requires
  author metadata; if absent → ProbeResult with warning, no estimates.
  Metric `self_error_pick_excess`: an *unadjusted observational
  contrast* — among judgments where one side is self-authored AND ground
  truth says the *other* side wins: P(pick self-side), minus the control
  rate over judgments with no self side where ground truth names a
  winner: P(pick the losing side). The control set is NOT matched on
  anything; both raw rates are reported in `detail` and on the card, and
  a composition diagnostic (latent-gap / decisive-rate / label-share
  difference between sets, documented thresholds in the probe module)
  appends a warning and suppresses the `self_preference_detected` flag
  when the sets differ materially — quality-gap imbalance between sets
  otherwise reads as fake self-preference on an author-blind judge.
  Difference CI via paired-ish two-sample cluster bootstrap (resample
  items within each set independently); null 0.0.
  [Amended 2026-06-10, review M2: the original "matched control"
  wording was wrong — nothing is matched.]
- **template** — conditions `tpl:default`, `tpl:v1`..`tpl:v{k-1}`, orig
  order only. Metrics: `template_fleiss_kappa` (items × {first, second,
  tie} across templates; report kappa with bootstrap CI over items; no
  null/p), `template_max_flip` (max over template pairs of decisive
  disagreement rate, Wilson CI on the argmax pair, flag multiplicity in
  detail), and when labels exist `template_accuracy_range` (max−min
  accuracy across templates, cluster-bootstrap CI).
- **stability** — conditions `rep` with repeats 0..k−1 (identical bodies;
  custom_id differs by `-r<k>`), default template, orig order. Metrics:
  `unanimity_rate` (fraction of items with identical verdicts across
  k repeats; Wilson CI), `mean_pairwise_flip` (mean over item of
  disagreement rate among decisive repeat pairs; cluster bootstrap),
  `stability_fleiss_kappa`. Note in docstring: at temperature 0 this
  measures *serving* nondeterminism; at temperature>0, sampling noise —
  the variance-component story from the plan.

Every probe's `analyze` must tolerate: missing conditions (warn),
all-invalid judgments, n too small for bootstrap (fall back to Wilson
or warn). All warnings surface in `ProbeResult.warnings`.

## 6. DATA — `judgecal.datasets`

`datasets/base.py`: `DatasetAdapter` protocol → `load(split, limit, seed)
-> list[PairwiseItem]` + `info() -> DatasetInfo` (hf_path, license,
citation, caveats). Adapters (lazy `import datasets` inside methods,
clear ImportError message pointing at `pip install judgecal[hf]`):

- `rewardbench2` (allenai/reward-bench-2, ODC-BY) — best-of-4 →
  pairwise: chosen vs each rejected (cap via `limit`, seeded sampling).
- `judgebench` (ScalerLab/JudgeBench, license unverified → caveat field).
- `llmbar` (princeton-nlp/LLMBar, MIT) — Natural + Adversarial subsets.
- `mtbench_human` (lmsys/mt_bench_human_judgments, CC-BY-4.0) — human
  winner as label; model authors → author_a/b (self-preference-capable).
- `rmbench` (THU-KEG/RM-Bench) — style-variant pairs.

Registry `get_adapter(name)` / `list_adapters()`. Item ids:
`"{dataset}:{stable_native_id}"`. Tests mock the HF call with tiny
in-memory dicts mirroring real schemas (document the mirrored schema
per adapter); real-download tests behind `@pytest.mark.network`.

## 7. CARD — `judgecal.report`

Pydantic v2 models in `report/card.py`:

```python
class MetricEntry(BaseModel): name; estimate; ci_low; ci_high; n;
    method; null_value | None; p_value | None; q_value | None;
    mde | None; detail: dict = {}
class ProbeEntry(BaseModel): probe; n_items; n_judgments; invalid_rate;
    warnings: list[str]; metrics: list[MetricEntry]; flags: list[str]
class ReliabilityCard(BaseModel):
    card_schema_version: Literal["0.1"]; judgecal_version: str;
    judge: dict; datasets: list[dict]; config: dict;
    created_utc: str | None      # caller-supplied, NOT auto-now
    probes: list[ProbeEntry]; overall_flags: list[str]; notes: list[str]
```

`build_card(results: list[ProbeResult], judge: dict, ...) -> ReliabilityCard`:
applies **BH-FDR across all null-bearing metrics in the card** (the
pre-registered scope decision; document it), fills q_values, then flags
with documented conventions (e.g. `position_bias_detected` if
first_pick_rate q<0.05 and |est−0.5|>0.05; `underpowered:<metric>` if
mde > 2×|est−null| and not significant; `high_invalid_rate` if >5%).
Thresholds in `report/thresholds.py` as constants with comments — they
are conventions, not truths. `render_markdown(card) -> str`: compact
table per probe, a plain-English summary section, and a footer noting
n, MDE and the FDR scope. `save_card/load_card` (JSON round-trip test).

## 8. VALID (wave 2) — `judgecal.validate` + recovery tests

`run_validation(level: Literal["fast","full"], seed: int = 7) ->
ValidationReport` — the "we test the tester" suite:

scenarios (each: generate synthetic items → plan suite → MockJudgeExecutor
→ analyze → compare to the mock judge's *analytic* truths):
1. null judge (all betas 0, noise 0): every null-bearing metric's CI
   covers null; no flag fires (fixed seed).
2. position-biased (beta_position=0.8, n=400): `first_pick_rate` CI
   covers `expected_first_pick_rate`; q<0.05; flag fires.
3. verbose-biased (beta_length=1.0): pad_pick_rate detects; CI covers
   analytic truth; GLM coef positive and significant.
4. self-preferring (beta_self=1.0): excess detected.
5. template-sensitive (template_sigma=0.7): kappa < null-scenario kappa;
   max_flip elevated.
6. unstable (noise_sigma=1.0, k=5): unanimity well below 1; with
   noise_sigma=0 → unanimity == 1 exactly.
7. mixed (position+length together): both recovered without
   cross-contamination (pad test still centered at its truth).

`fast` = scenarios at n≈300, single seed — runs in CI (<60s, n_boot 500).
`full` = adds 200-seed coverage check (95% CIs cover ≥90%) — `slow` marker.
`ValidationReport` prints a pass/fail table; CLI `judgecal validate` wraps it.

## 9. INSPECT (wave 2) — `judgecal.integrations.inspect_ai`

**First verify current inspect-ai API via its docs (it releases weekly).**
Guarded imports (`judgecal.integrations.inspect_ai` raises a helpful
ImportError without the extra). Deliberately minimal (skeptic cut):
- `_registry.py` (root, already wired in pyproject entry point) —
  imports the decorated objects inside try/except ImportError (the entry
  point must not crash environments without inspect-ai).
- A `@scorer` factory `judgecal_pairwise(model=...)` that renders our
  default template and parses `[[A]]/[[B]]/[[C]]` — so Inspect users can
  score with the same instrument we audit.
- `audit_samples_df(df) -> list[Judgment]`-ish adapter: maps an Inspect
  `samples_df()` with paired/swap structure into judgecal judgments
  (document the required columns; raise clear errors otherwise).
- An example Task in `examples/inspect_demo.py` (PKG agent links it).
Tests: importable-without-inspect behavior + (if inspect-ai installs
cleanly on dev box) a mocked-model smoke test, else skip with reason.

## 10. CLI (wave 2) — `judgecal.cli` (click)

Commands (each thin; logic lives in modules):
- `judgecal demo [--n 200] [--bias position=0.8,...] [--seed 7] [-o DIR]`
  — synthetic end-to-end: items → suite plan → mock judge → analyze →
  card; prints the markdown card. THE readme money shot.
- `judgecal validate [--full]` — §8 suite, exit 1 on failure.
- `judgecal plan --items items.jsonl --probes position,verbosity
  --model NAME [--endpoint ...] -o outdir/` — emit manifest + sidecar
  (+ `--resume results.jsonl` to emit only missing).
- `judgecal ingest --sidecar X.meta.jsonl --results r.jsonl -o judgments.jsonl`
- `judgecal analyze --judgments judgments.jsonl --judge NAME -o card/`
  — writes card.json + card.md.
- `judgecal slurm-pack --manifest m.jsonl [cluster opts] -o pack/`
- `judgecal datasets list` / `judgecal datasets fetch NAME --limit N -o items.jsonl`
- `judgecal claude-run --manifest m.jsonl --sidecar m.meta.jsonl -o judgments.jsonl
  [--model sonnet] [--limit N]` — ClaudeCodeExecutor path, with a
  visible "uses your Claude subscription; keep N small" warning.
Items JSONL schema = PairwiseItem fields; loaders in `cli.py` or reuse
datasets/base helpers. `--version` flag. Tests via click's CliRunner
(mock executors; no network).

## 11. PKG (wave 2)

- `README.md`: positioning per the plan — **NEVER** claim "first
  judge-reliability toolkit" / "empty niche" / "contamination-proof"
  (banned-claims list). Structure: what/why (judges are unreliable;
  cite CALM taxonomy arXiv 2410.02736, Miller arXiv 2411.00640,
  position-bias magnitudes), 60-second demo (`uvx`/`pip install` +
  `judgecal demo`), the five probes table with metrics, the validation
  story ("we test the tester" — planted-bias recovery), batch-first
  workflow diagram (plan → manifest → vLLM run-batch/SLURM → ingest →
  card), Claude-Code smoke path, Inspect integration snippet, honest
  related-work section (RAND JRH arXiv 2603.05399, cje-eval, UW-Madison
  2511.21140 — what we add: statistical inference, validation suite,
  quantization study, offline manifests), roadmap (quantized-judge
  study), citation block.
- `.github/workflows/ci.yml`: ubuntu + macos, py 3.10/3.12/3.13, uv-based:
  ruff check, mypy (non-blocking ok), pytest (fast suite), `uv build`
  + twine check or pip-install-wheel smoke. No network in CI tests.
- `CONTRIBUTING.md` (short), `examples/` (demo notebook-free scripts).

## 12. Definition of done (whole package)

1. `uv pip install -e ".[dev]" && pytest` green on macOS/py3.13.
2. `judgecal demo` produces a sensible card; `judgecal validate` passes.
3. `uv build` wheel installs into a fresh venv; CLI works from it.
4. ruff clean; mypy no errors in `src/` (pragmatic ignores allowed).
5. README respects banned claims; all licenses/citations accurate.
