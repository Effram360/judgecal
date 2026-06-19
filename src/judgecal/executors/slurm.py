"""SLURM job-pack generators (NOT an executor).

:func:`write_slurm_pack` turns an emitted manifest into a self-contained,
runnable batch pack: a copy of the manifest, a vLLM ``run-batch`` sbatch
script, a llama.cpp (GGUF) variant, and a ``README_RUN.md`` with exact
submit/ingest instructions. All paths inside the scripts are relative to
the pack directory (the scripts ``cd`` to ``$SLURM_SUBMIT_DIR``), so the
pack works as-is after copying it to the cluster and submitting from
inside it. Nothing here executes anything — the user copies the pack to
the cluster, submits, and brings ``results.jsonl`` back for
:func:`judgecal.manifests.merge_results`.

vLLM invocation note (verified ground §4): newer vLLM exposes a
``vllm run-batch`` subcommand; older versions only ship the module form
``python -m vllm.entrypoints.openai.run_batch``. Cluster installs vary, so
the generated scripts default to the subcommand but the module form is
included as a documented one-line swap (and is parameterizable via
``vllm_command``).
"""

from __future__ import annotations

import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from judgecal.manifests.emit import ManifestPaths

#: Module-form invocation for older vLLM versions (documented in every pack).
VLLM_MODULE_COMMAND = "python -m vllm.entrypoints.openai.run_batch"

#: Subcommand-form invocation for newer vLLM versions (the default).
VLLM_SUBCOMMAND = "vllm run-batch"


@dataclass(frozen=True)
class SlurmConfig:
    """Cluster-side knobs for generated sbatch scripts. Every field is
    overridable; defaults are deliberately conservative.

    Attributes:
        partition: SLURM partition name.
        gpus: GPUs per job (``--gres=gpu:N``).
        walltime: ``--time`` limit, ``HH:MM:SS``.
        cpus_per_task: ``--cpus-per-task``.
        mem: ``--mem`` (e.g. ``"32G"``).
        job_name: Base ``--job-name`` (suffixed per script).
        account: Optional ``--account`` line; omitted when ``None``.
        modules: Raw environment-setup lines emitted verbatim before the
            run command (e.g. ``("module load cuda/12.4",)``).
        container: Optional command prefix for containerized runs (e.g.
            ``"apptainer exec --nv vllm.sif"``); prepended to the run
            command when set.
        scratch_dir: Scratch directory created before running.
        extra_sbatch: Extra raw ``#SBATCH ...`` lines emitted verbatim.
    """

    partition: str = "gpu"
    gpus: int = 1
    walltime: str = "04:00:00"
    cpus_per_task: int = 8
    mem: str = "32G"
    job_name: str = "judgecal"
    account: str | None = None
    modules: tuple[str, ...] = ()
    container: str | None = None
    scratch_dir: str = "${SCRATCH:-/tmp}/judgecal"
    extra_sbatch: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlurmPackPaths:
    """Files written by :func:`write_slurm_pack`."""

    out_dir: Path
    vllm_sbatch: Path
    llamacpp_sbatch: Path
    readme: Path
    manifest: Path


def _sbatch_header(cluster: SlurmConfig, suffix: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={cluster.job_name}-{suffix}",
        f"#SBATCH --partition={cluster.partition}",
        f"#SBATCH --gres=gpu:{cluster.gpus}",
        f"#SBATCH --time={cluster.walltime}",
        f"#SBATCH --cpus-per-task={cluster.cpus_per_task}",
        f"#SBATCH --mem={cluster.mem}",
        f"#SBATCH --output={cluster.job_name}-{suffix}-%j.out",
    ]
    if cluster.account is not None:
        lines.append(f"#SBATCH --account={cluster.account}")
    lines.extend(cluster.extra_sbatch)
    return "\n".join(lines)


def _common_setup(
    cluster: SlurmConfig,
    manifest: str,
    results: str,
    model_var: str,
    model_value: str,
    script_name: str,
) -> str:
    module_block = "\n".join(cluster.modules) if cluster.modules else "# (no module lines)"
    return f"""
set -euo pipefail

# Manifest/results paths are relative to the pack directory: under sbatch
# we start from $SLURM_SUBMIT_DIR (submit from inside the pack); when run
# directly with bash, fall back to the script's own directory.
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$0")}}"
mkdir -p "{cluster.scratch_dir}"

{module_block}

# Override any of these at submit time, e.g.:
#   sbatch --export=ALL,{model_var}=other-model {script_name}
MANIFEST="${{MANIFEST:-{manifest}}}"
RESULTS="${{RESULTS:-{results}}}"
{model_var}="${{{model_var}:-{model_value}}}"

# Array-friendly: with --array, each task runs its own shard
# (produce shards by splitting the manifest into <name>.shard<K>.jsonl).
if [[ -n "${{SLURM_ARRAY_TASK_ID:-}}" ]]; then
  MANIFEST="${{MANIFEST%.jsonl}}.shard${{SLURM_ARRAY_TASK_ID}}.jsonl"
  RESULTS="${{RESULTS%.jsonl}}.shard${{SLURM_ARRAY_TASK_ID}}.jsonl"
fi
"""


def _vllm_script(
    cluster: SlurmConfig,
    manifest: str,
    results: str,
    model: str,
    vllm_command: str,
) -> str:
    prefix = f"{cluster.container} " if cluster.container else ""
    header = _sbatch_header(cluster, "vllm")
    setup = _common_setup(cluster, manifest, results, "MODEL", model, "run_vllm.sbatch")
    return f"""{header}
{setup}
# Primary invocation (newer vLLM ships a `run-batch` subcommand):
{prefix}{vllm_command} -i "$MANIFEST" -o "$RESULTS" --model "$MODEL"

# If your cluster's vLLM predates the subcommand, comment the line above
# and use the module form instead:
#   {prefix}{VLLM_MODULE_COMMAND} -i "$MANIFEST" -o "$RESULTS" --model "$MODEL"

echo "wrote $RESULTS"
"""


def _llamacpp_script(
    cluster: SlurmConfig,
    manifest: str,
    results: str,
    gguf_path: str,
    port: int = 8080,
) -> str:
    prefix = f"{cluster.container} " if cluster.container else ""
    header = _sbatch_header(cluster, "llamacpp")
    setup = _common_setup(cluster, manifest, results, "GGUF", gguf_path, "run_llamacpp.sbatch")
    return f"""{header}
{setup}
# Single-slot serial serving: --parallel 1 keeps llama.cpp deterministic
# (working assumption per the verified ground doc — verify empirically).
{prefix}llama-server -m "$GGUF" --port {port} --parallel 1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:{port}/health" > /dev/null; then break; fi
  sleep 2
done

python3 - "$MANIFEST" "$RESULTS" <<'PY'
import json, sys, urllib.request

manifest, results = sys.argv[1], sys.argv[2]
base = "http://127.0.0.1:{port}"
with open(manifest, encoding="utf-8") as fin, open(results, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        url = base + req["url"]
        payload = json.dumps(req["body"]).encode("utf-8")
        http_req = urllib.request.Request(
            url, data=payload, headers={{"Content-Type": "application/json"}}
        )
        try:
            with urllib.request.urlopen(http_req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            out = {{
                "custom_id": req["custom_id"],
                "response": {{"status_code": 200, "body": body}},
                "error": None,
            }}
        except Exception as exc:  # noqa: BLE001 - record-and-continue by design
            out = {{
                "custom_id": req["custom_id"],
                "response": None,
                "error": {{"message": str(exc)}},
            }}
        fout.write(json.dumps(out) + "\\n")
PY

echo "wrote $RESULTS"
"""


def _readme(
    manifest: str,
    sidecar: str,
    results: str,
    model: str,
    vllm_command: str,
) -> str:
    return f"""# judgecal SLURM run pack

Generated by `judgecal.executors.slurm.write_slurm_pack`. Contains:

- `{manifest}` — the batch manifest (copied into the pack; the scripts
  read it from the pack directory).
- `run_vllm.sbatch` — vLLM batch execution of the manifest (chat or score
  endpoint; one manifest format covers generative judges AND scalar RMs).
- `run_llamacpp.sbatch` — GGUF variant via `llama-server` (single-slot,
  `--parallel 1`, for serial determinism). Set `GGUF` to your model file.
- This README.

Manifest (in this pack): `{manifest}`
Sidecar (stays with you; NOT needed on the cluster): `{sidecar}`
Default model: `{model}`

## 1. Submit

Copy this pack directory to the cluster, then submit **from inside it**
(the scripts resolve the manifest and results relative to the submit
directory):

```bash
cd <pack-dir>
sbatch run_vllm.sbatch
# override the model or paths without editing the script:
sbatch --export=ALL,MODEL=some-other-model run_vllm.sbatch
# sharded array run (split the manifest into <name>.shard<K>.jsonl first):
sbatch --array=0-3 run_vllm.sbatch
```

## 2. vLLM invocation — version note

Cluster vLLM versions vary. The script defaults to:

```bash
{vllm_command} -i manifest.jsonl -o results.jsonl --model MODEL
```

If your vLLM predates the subcommand, swap in the module form (also left
as a comment inside the script):

```bash
{VLLM_MODULE_COMMAND} -i manifest.jsonl -o results.jsonl --model MODEL
```

## 3. Determinism caveat

vLLM at temperature 0 is NOT run-to-run deterministic under batching.
If you need batch invariance, enable vLLM's batch-invariance mode (at a
latency cost) — otherwise treat serving noise as a variance component
(judgecal's stability probe measures it). llama.cpp with `--parallel 1`
is serially deterministic (verify on your build).

## 4. Bring results home and ingest

```bash
scp cluster:<pack-dir>/{Path(results).name} .
judgecal ingest --sidecar {Path(sidecar).name} --results {Path(results).name} \\
    -o judgments.jsonl
```

## 5. Resume an interrupted run

```bash
judgecal plan ... --resume {Path(results).name}
```

or in Python: `judgecal.manifests.remaining_manifest(manifest, sidecar,
results)` returns only the batch lines that still need to run (error
lines are retried; completed lines are excluded).
"""


def write_slurm_pack(
    manifest_paths: ManifestPaths | Path | str,
    out_dir: Path | str,
    cluster: SlurmConfig | None = None,
    *,
    model: str | None = None,
    gguf_path: str = "/path/to/model.gguf",
    vllm_command: str = VLLM_SUBCOMMAND,
    results_name: str = "results.jsonl",
) -> SlurmPackPaths:
    """Write a runnable SLURM job pack for an emitted manifest.

    The manifest is **copied into the pack directory** and the generated
    scripts reference it (and the results file) relative to the pack, so
    the pack is genuinely self-contained: copy it to the cluster and
    submit from inside it.

    Args:
        manifest_paths: A :class:`~judgecal.manifests.ManifestPaths` (model
            and sidecar are taken from it) or a bare manifest path (then
            pass ``model``; the sidecar name is inferred as
            ``<name>.meta.jsonl``).
        out_dir: Pack directory (created if missing).
        cluster: Cluster knobs; defaults to :class:`SlurmConfig()`.
        model: Model name override (required when ``manifest_paths`` is a
            bare path and there is no ManifestPaths to read it from).
        gguf_path: Default GGUF file path baked into the llama.cpp script
            (overridable at submit time via the ``GGUF`` env var).
        vllm_command: vLLM invocation; default ``"vllm run-batch"``. Pass
            :data:`VLLM_MODULE_COMMAND` for older clusters.
        results_name: Output filename the scripts write to (inside the
            pack directory).

    Returns:
        :class:`SlurmPackPaths` with the written files (including the
        manifest copy).

    Raises:
        ValueError: If no model name is available.
    """
    cluster = cluster if cluster is not None else SlurmConfig()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(manifest_paths, ManifestPaths):
        manifest_src = Path(manifest_paths.manifest)
        sidecar = str(manifest_paths.sidecar)
        resolved_model = model if model is not None else manifest_paths.model
    else:
        manifest_src = Path(manifest_paths)
        sidecar = re.sub(r"\.jsonl$", "", str(manifest_paths)) + ".meta.jsonl"
        if model is None:
            raise ValueError(
                "model is required when manifest_paths is a bare path "
                "(pass model=... or a ManifestPaths)"
            )
        resolved_model = model

    # Copy the manifest into the pack; everything in the scripts is
    # pack-relative from here on.
    manifest_copy = out_path / manifest_src.name
    if manifest_src.resolve() != manifest_copy.resolve():
        shutil.copyfile(manifest_src, manifest_copy)
    manifest = manifest_src.name
    results = results_name

    vllm_sbatch = out_path / "run_vllm.sbatch"
    llamacpp_sbatch = out_path / "run_llamacpp.sbatch"
    readme = out_path / "README_RUN.md"

    vllm_sbatch.write_text(
        _vllm_script(cluster, manifest, results, resolved_model, vllm_command),
        encoding="utf-8",
    )
    llamacpp_sbatch.write_text(
        _llamacpp_script(cluster, manifest, results, gguf_path),
        encoding="utf-8",
    )
    readme.write_text(
        _readme(manifest, sidecar, results, resolved_model, vllm_command),
        encoding="utf-8",
    )
    for script in (vllm_sbatch, llamacpp_sbatch):
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return SlurmPackPaths(
        out_dir=out_path,
        vllm_sbatch=vllm_sbatch,
        llamacpp_sbatch=llamacpp_sbatch,
        readme=readme,
        manifest=manifest_copy,
    )


__all__ = [
    "VLLM_MODULE_COMMAND",
    "VLLM_SUBCOMMAND",
    "SlurmConfig",
    "SlurmPackPaths",
    "write_slurm_pack",
]
