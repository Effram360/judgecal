"""judgecal command-line interface (contracts §10).

Thin click commands; all logic lives in the module public APIs
(:mod:`judgecal.fixtures`, :mod:`judgecal.probes`, :mod:`judgecal.manifests`,
:mod:`judgecal.executors`, :mod:`judgecal.report`, :mod:`judgecal.datasets`,
:mod:`judgecal.validate`). The CLI layer owns exactly two impurities the
library refuses to have: the wall clock (``created_utc`` on cards) and
process exit codes.

Items JSONL schema (``plan`` input, ``datasets fetch`` output): one JSON
object per line with :class:`judgecal.core.PairwiseItem` fields —
``item_id``, ``prompt``, ``response_a``, ``response_b`` required;
``label``, ``author_a``, ``author_b``, ``source``, ``meta`` optional.

Judgments JSONL schema (``ingest``/``claude-run`` output, ``analyze``
input): one flat JSON object per line with ``custom_id``, ``verdict``,
``raw_text``, ``meta`` (the full request meta).

Testability knob: ``demo``/``analyze`` accept ``--n-boot`` (hidden on
``demo``) so test suites can shrink the cluster bootstrap from the
default 2000 replicates; results are still deterministic per seed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from judgecal import __version__
from judgecal.core import REQUIRED_META_KEYS, Judgment, JudgmentRequest, PairwiseItem
from judgecal.datasets import get_adapter, list_adapters
from judgecal.executors import (
    VLLM_MODULE_COMMAND,
    VLLM_SUBCOMMAND,
    ClaudeCodeExecutor,
    MockJudgeExecutor,
    SlurmConfig,
    write_slurm_pack,
)
from judgecal.fixtures import MockJudgeConfig, SyntheticConfig, generate_items
from judgecal.manifests import (
    ModelSpec,
    emit_manifest,
    load_sidecar,
    merge_results,
    remaining_manifest,
)
from judgecal.probes import PROBE_REGISTRY, ProbeConfig, analyze_suite, plan_suite
from judgecal.report import ReliabilityCard, build_card, render_markdown, save_card

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Canonical probe order (contracts §5); also the default for demo/plan.
ALL_PROBES: tuple[str, ...] = ("position", "verbosity", "self_preference", "template", "stability")

#: ``--bias`` key -> MockJudgeConfig field.
_BIAS_KEY_MAP: dict[str, str] = {
    "position": "beta_position",
    "length": "beta_length",
    "self": "beta_self",
    "template_sigma": "template_sigma",
    "noise_sigma": "noise_sigma",
}

_VALID_VERDICTS = ("first", "second", "tie", "invalid")


# --------------------------------------------------------------------------
# Helpers (clock, parsing, JSONL IO)
# --------------------------------------------------------------------------


def _utc_now() -> str:
    """ISO-8601 UTC timestamp — the CLI layer's clock (library is clock-free)."""
    return datetime.now(timezone.utc).isoformat()


def _parse_bias(spec: str | None) -> dict[str, Any]:
    """Parse ``--bias position=0.8,length=1.0`` into MockJudgeConfig kwargs.

    Args:
        spec: Comma-separated ``KEY=VALUE`` pairs with keys from
            ``_BIAS_KEY_MAP``, or ``None``/empty for no planted bias.

    Returns:
        Keyword arguments for :class:`judgecal.fixtures.MockJudgeConfig`.

    Raises:
        click.UsageError: On unknown keys, missing ``=``, or non-numeric
            values.
    """
    if not spec:
        return {}
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, raw = part.partition("=")
        key = key.strip()
        if not sep:
            raise click.UsageError(f"--bias entries must be KEY=VALUE, got {part!r}")
        if key not in _BIAS_KEY_MAP:
            raise click.UsageError(
                f"unknown bias key {key!r}; valid keys: {', '.join(_BIAS_KEY_MAP)}"
            )
        try:
            out[_BIAS_KEY_MAP[key]] = float(raw)
        except ValueError:
            raise click.UsageError(
                f"bias value for {key!r} must be a number, got {raw.strip()!r}"
            ) from None
    return out


def _parse_probes(spec: str) -> list[str]:
    """Parse a comma-separated probe list, validating against the registry."""
    names = [p.strip() for p in spec.split(",") if p.strip()]
    if not names:
        raise click.UsageError("--probes must name at least one probe")
    unknown = [p for p in names if p not in PROBE_REGISTRY]
    if unknown:
        raise click.UsageError(
            f"unknown probe(s) {', '.join(unknown)}; registered: {', '.join(sorted(PROBE_REGISTRY))}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(lineno, obj)`` per non-blank JSONL line, with clean errors."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise click.ClickException(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise click.ClickException(f"{path}:{lineno}: expected one JSON object per line")
            yield lineno, obj


def _load_items(path: Path) -> list[PairwiseItem]:
    """Load PairwiseItems from an items JSONL file (schema in module docstring)."""
    items: list[PairwiseItem] = []
    for lineno, obj in _iter_jsonl(path):
        missing = [k for k in ("item_id", "prompt", "response_a", "response_b") if k not in obj]
        if missing:
            raise click.ClickException(
                f"{path}:{lineno}: item line missing required key(s): {', '.join(missing)}"
            )
        label = obj.get("label")
        if label not in (None, "A", "B", "tie"):
            raise click.ClickException(
                f"{path}:{lineno}: label must be 'A', 'B', 'tie', or null; got {label!r}"
            )
        items.append(
            PairwiseItem(
                item_id=str(obj["item_id"]),
                prompt=str(obj["prompt"]),
                response_a=str(obj["response_a"]),
                response_b=str(obj["response_b"]),
                label=label,
                author_a=obj.get("author_a"),
                author_b=obj.get("author_b"),
                source=obj.get("source", "unknown"),
                meta=obj.get("meta") or {},
            )
        )
    if not items:
        raise click.ClickException(f"{path}: no items found")
    return items


def _write_items(items: Sequence[PairwiseItem], path: Path) -> None:
    """Write PairwiseItems as items JSONL (the schema ``plan`` consumes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
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


def _load_judgments(path: Path) -> list[Judgment]:
    """Load Judgments from a judgments JSONL file, validating meta keys."""
    judgments: list[Judgment] = []
    for lineno, obj in _iter_jsonl(path):
        missing = [k for k in ("custom_id", "verdict", "meta") if k not in obj]
        if missing:
            raise click.ClickException(
                f"{path}:{lineno}: judgment line missing key(s): {', '.join(missing)}"
            )
        verdict = obj["verdict"]
        if verdict not in _VALID_VERDICTS:
            raise click.ClickException(
                f"{path}:{lineno}: verdict must be one of {_VALID_VERDICTS}, got {verdict!r}"
            )
        meta = obj["meta"]
        if not isinstance(meta, dict):
            raise click.ClickException(f"{path}:{lineno}: meta must be a JSON object")
        meta_missing = [k for k in REQUIRED_META_KEYS if k not in meta]
        if meta_missing:
            raise click.ClickException(
                f"{path}:{lineno}: meta missing required key(s): {', '.join(meta_missing)}"
            )
        judgments.append(
            Judgment(
                custom_id=str(obj["custom_id"]),
                verdict=verdict,
                raw_text=obj.get("raw_text"),
                meta=meta,
            )
        )
    if not judgments:
        raise click.ClickException(f"{path}: no judgments found")
    return judgments


def _write_judgments(judgments: Sequence[Judgment], path: Path) -> None:
    """Serialize Judgments as flat JSONL: custom_id, verdict, raw_text, meta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for j in judgments:
            fh.write(
                json.dumps(
                    {
                        "custom_id": j.custom_id,
                        "verdict": j.verdict,
                        "raw_text": j.raw_text,
                        "meta": j.meta,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_card_outputs(card: ReliabilityCard, markdown: str, out_dir: Path) -> None:
    """Write ``card.json`` + ``card.md`` into ``out_dir`` (created if missing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_card(card, out_dir / "card.json")
    (out_dir / "card.md").write_text(markdown, encoding="utf-8")
    click.echo(f"wrote {out_dir / 'card.json'} and {out_dir / 'card.md'}", err=True)


# --------------------------------------------------------------------------
# Root group
# --------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="judgecal")
def main() -> None:
    """judgecal — statistically rigorous, batch-first reliability auditing
    for LLM judges and reward models."""


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


@main.command()
@click.option(
    "--n",
    default=200,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of synthetic items (>= 1).",
)
@click.option(
    "--bias",
    default=None,
    metavar="KEY=VAL,...",
    help=(
        "Planted mock-judge biases; keys: position, length, self, "
        "template_sigma, noise_sigma (e.g. 'position=0.8,length=1.0')."
    ),
)
@click.option("--seed", default=7, show_default=True, help="Seed for items, judge, analysis.")
@click.option(
    "--probes",
    "probes_csv",
    default=",".join(ALL_PROBES),
    show_default=True,
    help="Comma-separated probes to run.",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional output directory for card.json + card.md.",
)
@click.option(
    "--n-boot",
    default=2000,
    show_default=True,
    hidden=True,
    help="Bootstrap replicates (hidden testability knob; deterministic per seed).",
)
def demo(
    n: int,
    bias: str | None,
    seed: int,
    probes_csv: str,
    out_dir: Path | None,
    n_boot: int,
) -> None:
    """Synthetic end-to-end audit against the deterministic mock judge.

    Generates items, plans the probe suite, executes it with a mock judge
    (optionally with planted biases), analyzes, and prints the reliability
    card as Markdown. Zero LLM, zero network — runs anywhere.
    """
    bias_kwargs = _parse_bias(bias)
    probe_names = _parse_probes(probes_csv)
    items = generate_items(SyntheticConfig(n_items=n, seed=seed))
    config = ProbeConfig(seed=seed, n_boot=n_boot)
    requests = plan_suite(items, probe_names, config)
    executor = MockJudgeExecutor(MockJudgeConfig(seed=seed, **bias_kwargs))
    judgments = executor.execute(requests)
    results = analyze_suite(judgments, probe_names, config)
    card = build_card(
        results,
        judge={"model": "mock-judge", "planted_bias": dict(bias_kwargs)},
        datasets=[{"name": "synthetic", "n_items": n, "seed": seed}],
        config={
            "alpha": config.alpha,
            "n_boot": n_boot,
            "seed": seed,
            "probes": probe_names,
        },
        created_utc=_utc_now(),
        notes=["Demo run against the deterministic mock judge (no LLM involved)."],
    )
    markdown = render_markdown(card)
    if out_dir is not None:
        _write_card_outputs(card, markdown, out_dir)
    click.echo(markdown)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


def _report_passed(report: Any) -> bool:
    """Duck-typed pass/fail extraction from a ValidationReport."""
    for attr in ("passed", "ok", "all_passed", "success"):
        value = getattr(report, attr, None)
        if value is not None:
            return bool(value() if callable(value) else value)
    return True


@main.command()
@click.option("--full", is_flag=True, help="Run the full (slow) validation suite.")
@click.option("--seed", default=7, show_default=True, help="Validation seed.")
@click.pass_context
def validate(ctx: click.Context, full: bool, seed: int) -> None:
    """Run the planted-bias recovery suite ("we test the tester").

    Exits 1 if any scenario fails.
    """
    # Imported inside the command: judgecal.validate is an optional-at-
    # import-time sibling (contracts §8) and tests mock it via sys.modules.
    from judgecal.validate import run_validation

    report = run_validation("full" if full else "fast", seed=seed)
    render = getattr(report, "render_table", None)
    click.echo(render() if callable(render) else str(report))
    if not _report_passed(report):
        ctx.exit(1)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@main.command()
@click.option(
    "--items",
    "items_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Items JSONL (PairwiseItem fields per line).",
)
@click.option(
    "--probes",
    "probes_csv",
    default=",".join(ALL_PROBES),
    show_default=True,
    help="Comma-separated probes to plan.",
)
@click.option("--model", required=True, help="Served model name baked into batch bodies.")
@click.option(
    "--endpoint",
    default="/v1/chat/completions",
    show_default=True,
    type=click.Choice(["/v1/chat/completions", "/v1/score"]),
    help=(
        "Batch endpoint: /v1/chat/completions for generative judges; "
        "/v1/score for scalar reward models (emits vLLM score-API bodies "
        "scoring both presented sides per line)."
    ),
)
@click.option(
    "--temperature",
    default=0.0,
    show_default=True,
    help="Sampling temperature (chat endpoint only).",
)
@click.option(
    "--max-tokens",
    default=1024,
    show_default=True,
    help="Generation cap (chat endpoint only).",
)
@click.option(
    "-o",
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for manifest + sidecar.",
)
@click.option("--name", default="manifest", show_default=True, help="Manifest basename.")
@click.option(
    "--resume",
    "resume_results",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Existing batch results JSONL; additionally writes "
        "<name>.resume.jsonl containing only still-missing lines."
    ),
)
def plan(
    items_path: Path,
    probes_csv: str,
    model: str,
    endpoint: str,
    temperature: float,
    max_tokens: int,
    out_dir: Path,
    name: str,
    resume_results: Path | None,
) -> None:
    """Plan a probe suite and emit an OpenAI batch manifest + sidecar.

    The manifest (``<name>.jsonl``) runs directly under ``vllm run-batch``;
    the sidecar (``<name>.meta.jsonl``) fans results back out at ingest
    time. With ``--resume``, a ``<name>.resume.jsonl`` containing only the
    not-yet-completed lines is written alongside.
    """
    items = _load_items(items_path)
    probe_names = _parse_probes(probes_csv)
    requests = plan_suite(items, probe_names, ProbeConfig())
    spec = ModelSpec(
        model=model, endpoint=endpoint, temperature=temperature, max_tokens=max_tokens
    )
    paths = emit_manifest(requests, spec, out_dir, name)
    click.echo(
        f"manifest: {paths.manifest} ({paths.n_lines} batch lines, {paths.n_usages} usages)"
    )
    click.echo(f"sidecar:  {paths.sidecar}")
    if resume_results is not None:
        try:
            remaining = remaining_manifest(paths.manifest, paths.sidecar, [resume_results])
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        resume_path = Path(out_dir) / f"{name}.resume.jsonl"
        with resume_path.open("w", encoding="utf-8") as fh:
            for line in remaining:
                fh.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
        click.echo(
            f"resume:   {resume_path} ({len(remaining)} of {paths.n_lines} lines still to run)"
        )


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@main.command()
@click.option(
    "--sidecar",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Sidecar (<name>.meta.jsonl) emitted by `judgecal plan`.",
)
@click.option(
    "--results",
    "results_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch output JSONL file(s); repeatable.",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output judgments JSONL.",
)
@click.option("--pattern", default=None, help="Optional custom verdict regex.")
def ingest(
    sidecar: Path,
    results_paths: tuple[Path, ...],
    out_path: Path,
    pattern: str | None,
) -> None:
    """Merge batch outputs with the sidecar into a judgments JSONL.

    Each executed line fans out to one judgment per recorded usage
    (probes that shared an execution body each get their own row).
    """
    try:
        judgments = merge_results(sidecar, list(results_paths), pattern=pattern)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not judgments:
        raise click.ClickException(
            "no judgments produced — do the results files match this sidecar?"
        )
    _write_judgments(judgments, out_path)
    click.echo(f"wrote {len(judgments)} judgments -> {out_path}")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


@main.command()
@click.option(
    "--judgments",
    "judgments_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Judgments JSONL from `judgecal ingest` or `judgecal claude-run`.",
)
@click.option("--judge", "judge_name", required=True, help="Judge name for the card header.")
@click.option(
    "-o",
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for card.json + card.md.",
)
@click.option("--alpha", default=0.05, show_default=True, help="Two-sided CI level.")
@click.option("--n-boot", default=2000, show_default=True, help="Bootstrap replicates.")
@click.option("--seed", default=0, show_default=True, help="Analysis seed (deterministic).")
@click.option(
    "--judge-author",
    default="judge-self",
    show_default=True,
    help="Author string identifying the judge (self-preference probe).",
)
def analyze(
    judgments_path: Path,
    judge_name: str,
    out_dir: Path,
    alpha: float,
    n_boot: int,
    seed: int,
    judge_author: str,
) -> None:
    """Analyze ingested judgments into a reliability card.

    Probes are detected from ``meta["probe"]``; each registered probe's
    analysis receives its own judgments plus those of any probe it
    declares in ``requires`` (``analyze_suite``'s union semantics).
    Writes ``card.json`` + ``card.md`` and prints the Markdown card.
    """
    judgments = _load_judgments(judgments_path)
    present: list[str] = []
    for j in judgments:
        p = j.meta.get("probe")
        if isinstance(p, str) and p not in present:
            present.append(p)
    probe_names = [p for p in ALL_PROBES if p in present]
    probe_names += sorted(p for p in present if p not in ALL_PROBES and p in PROBE_REGISTRY)
    skipped = sorted(p for p in present if p not in PROBE_REGISTRY)
    if skipped:
        click.echo(f"warning: skipping unregistered probe(s): {', '.join(skipped)}", err=True)
    if not probe_names:
        raise click.ClickException(
            f"{judgments_path}: no judgments from registered probes "
            f"(registered: {', '.join(sorted(PROBE_REGISTRY))})"
        )
    config = ProbeConfig(alpha=alpha, n_boot=n_boot, seed=seed, judge_author=judge_author)
    results = analyze_suite(judgments, probe_names, config)
    card = build_card(
        results,
        judge={"model": judge_name},
        config={
            "alpha": alpha,
            "n_boot": n_boot,
            "seed": seed,
            "judge_author": judge_author,
            "probes": probe_names,
        },
        created_utc=_utc_now(),
    )
    markdown = render_markdown(card)
    _write_card_outputs(card, markdown, out_dir)
    click.echo(markdown)


# --------------------------------------------------------------------------
# slurm-pack
# --------------------------------------------------------------------------


@main.command("slurm-pack")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Manifest JSONL emitted by `judgecal plan`.",
)
@click.option("--model", required=True, help="Served model name for the run scripts.")
@click.option(
    "-o",
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Pack output directory.",
)
@click.option("--partition", default="gpu", show_default=True)
@click.option("--gpus", default=1, show_default=True)
@click.option("--walltime", default="04:00:00", show_default=True)
@click.option("--cpus-per-task", default=8, show_default=True)
@click.option("--mem", default="32G", show_default=True)
@click.option("--job-name", default="judgecal", show_default=True)
@click.option("--account", default=None, help="Optional SLURM account.")
@click.option(
    "--module",
    "modules",
    multiple=True,
    help="Environment-setup line emitted verbatim (repeatable).",
)
@click.option("--container", default=None, help="Container command prefix (e.g. apptainer).")
@click.option("--scratch-dir", default="${SCRATCH:-/tmp}/judgecal", show_default=True)
@click.option(
    "--gguf-path",
    default="/path/to/model.gguf",
    show_default=True,
    help="GGUF path baked into the llama.cpp variant.",
)
@click.option(
    "--vllm-module-form",
    is_flag=True,
    help="Use 'python -m vllm.entrypoints.openai.run_batch' (older vLLM).",
)
def slurm_pack(
    manifest: Path,
    model: str,
    out_dir: Path,
    partition: str,
    gpus: int,
    walltime: str,
    cpus_per_task: int,
    mem: str,
    job_name: str,
    account: str | None,
    modules: tuple[str, ...],
    container: str | None,
    scratch_dir: str,
    gguf_path: str,
    vllm_module_form: bool,
) -> None:
    """Generate a runnable SLURM job pack for an emitted manifest.

    Writes ``run_vllm.sbatch``, ``run_llamacpp.sbatch``, and
    ``README_RUN.md`` with exact submit/ingest instructions. Nothing is
    executed; copy the pack to the cluster and submit.
    """
    cluster = SlurmConfig(
        partition=partition,
        gpus=gpus,
        walltime=walltime,
        cpus_per_task=cpus_per_task,
        mem=mem,
        job_name=job_name,
        account=account,
        modules=tuple(modules),
        container=container,
        scratch_dir=scratch_dir,
    )
    try:
        paths = write_slurm_pack(
            manifest,
            out_dir,
            cluster,
            model=model,
            gguf_path=gguf_path,
            vllm_command=VLLM_MODULE_COMMAND if vllm_module_form else VLLM_SUBCOMMAND,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for written in (paths.vllm_sbatch, paths.llamacpp_sbatch, paths.readme):
        click.echo(f"wrote {written}")


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


@main.group("datasets")
def datasets_group() -> None:
    """List and fetch preference-dataset adapters."""


@datasets_group.command("list")
def datasets_list() -> None:
    """List registered dataset adapters with HF path and license."""
    for name in list_adapters():
        info = get_adapter(name).info()
        click.echo(f"{name:16s} {info.hf_path:40s} license: {info.license}")


@datasets_group.command("fetch")
@click.argument("name")
@click.option("--split", default=None, help="Dataset split (adapter-specific default).")
@click.option("--limit", default=None, type=int, help="Max items (seeded sampling).")
@click.option("--seed", default=0, show_default=True, help="Sampling seed.")
@click.option(
    "-o",
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output items JSONL.",
)
def datasets_fetch(
    name: str,
    split: str | None,
    limit: int | None,
    seed: int,
    out_path: Path,
) -> None:
    """Fetch a dataset and write canonical items JSONL (requires network).

    Most adapters need the optional HF extra: ``pip install 'judgecal[hf]'``.
    """
    try:
        adapter = get_adapter(name)
    except KeyError as exc:
        raise click.UsageError(str(exc.args[0]) if exc.args else str(exc)) from exc
    info = adapter.info()
    try:
        items = adapter.load(split=split, limit=limit, seed=seed)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    _write_items(items, out_path)
    click.echo(f"wrote {len(items)} items -> {out_path} (license: {info.license})")
    for caveat in info.caveats:
        click.echo(f"caveat: {caveat}", err=True)


# --------------------------------------------------------------------------
# claude-run
# --------------------------------------------------------------------------


@main.command("claude-run")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Manifest JSONL emitted by `judgecal plan`.",
)
@click.option(
    "--sidecar",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Matching sidecar (<name>.meta.jsonl).",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output judgments JSONL.",
)
@click.option("--model", default=None, help="Model alias for the CLI (e.g. 'sonnet').")
@click.option(
    "--limit",
    default=25,
    show_default=True,
    help="Safety cap on manifest lines executed; pass 0 to run ALL lines.",
)
@click.option("--timeout-s", default=120, show_default=True, help="Per-request timeout.")
@click.option("--claude-bin", default="claude", show_default=True, help="Claude Code binary.")
def claude_run(
    manifest: Path,
    sidecar: Path,
    out_path: Path,
    model: str | None,
    limit: int,
    timeout_s: int,
    claude_bin: str,
) -> None:
    """Run manifest lines through the Claude Code CLI (zero-API smoke path).

    Sequential, one CLI session per request, against YOUR Claude
    subscription quota — keep batches small. ``--limit`` defaults to 25
    as a safety cap; ``--limit 0`` means "all lines".
    """
    if limit < 0:
        raise click.UsageError("--limit must be >= 0 (0 means run all lines)")
    try:
        entries = load_sidecar(sidecar)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    lines = [obj for _, obj in _iter_jsonl(manifest)]
    total = len(lines)
    if limit and total > limit:
        lines = lines[:limit]
        click.echo(
            f"note: running the first {limit} of {total} manifest lines "
            "(--limit 0 to run all)",
            err=True,
        )
    click.secho(
        f"WARNING: claude-run executes {len(lines)} request(s) sequentially through "
        "YOUR Claude subscription (Claude Code CLI). This is a smoke path, not a "
        "study path — keep N small.",
        fg="yellow",
        bold=True,
        err=True,
    )
    requests: list[JudgmentRequest] = []
    for obj in lines:
        custom_id = obj.get("custom_id")
        entry = entries.get(custom_id) if isinstance(custom_id, str) else None
        if entry is None or not entry.usages:
            raise click.ClickException(
                f"manifest custom_id {custom_id!r} has no sidecar entry; "
                "manifest/sidecar mismatch"
            )
        assert isinstance(custom_id, str)  # entry is not None ⇒ custom_id was a str
        body = obj.get("body")
        if not isinstance(body, dict):
            raise click.ClickException(f"manifest line for {custom_id!r} has no body object")
        try:
            requests.append(
                JudgmentRequest(custom_id=custom_id, body=body, meta=dict(entry.usages[0]))
            )
        except ValueError as exc:
            raise click.ClickException(f"sidecar usage for {custom_id!r}: {exc}") from exc
    executor = ClaudeCodeExecutor(model=model, claude_bin=claude_bin, timeout_s=timeout_s)
    try:
        executed = executor.execute(requests)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    # Fan each executed judgment back out to every recorded usage, exactly
    # as merge_results does for batch outputs.
    fanned: list[Judgment] = []
    for judgment in executed:
        for usage in entries[judgment.custom_id].usages:
            fanned.append(
                Judgment(
                    custom_id=judgment.custom_id,
                    verdict=judgment.verdict,
                    raw_text=judgment.raw_text,
                    meta=dict(usage),
                )
            )
    _write_judgments(fanned, out_path)
    click.echo(
        f"wrote {len(fanned)} judgments ({len(executed)} executions) -> {out_path}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
