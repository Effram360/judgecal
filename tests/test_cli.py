"""Tests for the judgecal CLI (contracts §10).

All tests run through click's ``CliRunner`` with zero network and zero
real LLMs: the demo uses the deterministic mock judge, ``validate`` is
mocked via ``sys.modules`` (the validate module is owned by another
agent), and ``claude-run`` uses a fake executor patched into the CLI
namespace. ``demo``/``analyze`` use the hidden/visible ``--n-boot``
testability knob to keep bootstraps small.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from judgecal import __version__
from judgecal.cli import main
from judgecal.core import Judgment, JudgmentRequest, PairwiseItem
from judgecal.fixtures import MockJudgeConfig, SyntheticConfig, generate_items
from judgecal.manifests import ModelSpec, emit_manifest
from judgecal.probes import ProbeConfig, plan_suite
from judgecal.report import load_card

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def combined_output(result: Result) -> str:
    """stdout + stderr regardless of click version's stream handling."""
    out = result.output
    try:
        err = result.stderr
    except (ValueError, AttributeError):
        err = ""
    return out + err


def write_items_jsonl(path: Path, items: list[PairwiseItem]) -> None:
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
                    }
                )
                + "\n"
            )


def make_meta(item_id: str, condition: str = "orig", probe: str = "position") -> dict[str, Any]:
    return {
        "probe": probe,
        "condition": condition,
        "item_id": item_id,
        "repeat": 0,
        "first_is_a": True,
        "first_len": 50,
        "second_len": 60,
        "first_author": None,
        "second_author": None,
        "first_latent_q": None,
        "second_latent_q": None,
        "label_first": None,
    }


def make_requests(n: int) -> list[JudgmentRequest]:
    """Distinct-body requests with full metas (for manifest fixtures)."""
    from judgecal.core import make_custom_id

    requests = []
    for k in range(n):
        body = {"messages": [{"role": "user", "content": f"Judge pair {k}. End with [[A]]."}]}
        requests.append(
            JudgmentRequest(
                custom_id=make_custom_id(body, 0),
                body=body,
                meta=make_meta(f"it-{k}"),
            )
        )
    return requests


# ---------------------------------------------------------------------------
# version / help
# ---------------------------------------------------------------------------


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_all_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("demo", "validate", "plan", "ingest", "analyze", "slurm-pack", "datasets",
                "claude-run"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestDemo:
    def test_end_to_end_with_output_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "card"
        result = runner.invoke(
            main,
            [
                "demo",
                "--n",
                "40",
                "--seed",
                "7",
                "--n-boot",
                "50",
                "--bias",
                "position=0.8",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        # markdown card always printed to stdout
        assert "Judge Reliability Card" in result.output
        assert "position" in result.output
        # card files written
        assert (out / "card.md").exists()
        card = load_card(out / "card.json")
        assert card.created_utc is not None  # CLI supplies the clock
        assert {e.probe for e in card.probes} == {
            "position",
            "verbosity",
            "self_preference",
            "template",
            "stability",
        }

    def test_stdout_only_without_output_dir(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            ["demo", "--n", "12", "--n-boot", "20", "--probes", "position"],
        )
        assert result.exit_code == 0, combined_output(result)
        assert "Judge Reliability Card" in result.output

    def test_unknown_bias_key_is_usage_error(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["demo", "--bias", "speed=1.0"])
        assert result.exit_code == 2
        assert "unknown bias key" in combined_output(result)
        assert "Traceback" not in combined_output(result)

    def test_malformed_bias_value(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["demo", "--bias", "position=fast"])
        assert result.exit_code == 2
        assert "must be a number" in combined_output(result)

    def test_unknown_probe_is_usage_error(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["demo", "--probes", "position,vibes"])
        assert result.exit_code == 2
        assert "unknown probe" in combined_output(result)

    @pytest.mark.parametrize("n", ["0", "-3"])
    def test_n_below_one_is_usage_error(self, runner: CliRunner, n: str) -> None:
        result = runner.invoke(main, ["demo", "--n", n])
        assert result.exit_code == 2
        assert "Traceback" not in combined_output(result)
        assert "--n" in combined_output(result)


# ---------------------------------------------------------------------------
# plan -> ingest -> analyze round trip
# ---------------------------------------------------------------------------


class TestPipeline:
    PROBES = "position,verbosity"

    @pytest.fixture()
    def items(self) -> list[PairwiseItem]:
        return generate_items(SyntheticConfig(n_items=10, seed=3))

    @pytest.fixture()
    def planned_dir(self, runner: CliRunner, tmp_path: Path, items: list[PairwiseItem]) -> Path:
        items_path = tmp_path / "items.jsonl"
        write_items_jsonl(items_path, items)
        out_dir = tmp_path / "plan"
        result = runner.invoke(
            main,
            [
                "plan",
                "--items",
                str(items_path),
                "--probes",
                self.PROBES,
                "--model",
                "test-model",
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        return out_dir

    def fake_batch_results(
        self, items: list[PairwiseItem], results_path: Path
    ) -> dict[str, str]:
        """MockJudgeExecutor outputs written in OpenAI batch-output format.

        Replans the same suite the CLI planned (same items, same default
        ProbeConfig => identical content-hashed custom_ids), executes it
        with a planted-bias mock judge, and writes one output line per
        unique custom_id.
        """
        requests = plan_suite(items, ["position", "verbosity"], ProbeConfig())
        from judgecal.executors import MockJudgeExecutor

        executor = MockJudgeExecutor(MockJudgeConfig(seed=5, beta_position=1.2))
        judgments = executor.execute(requests)
        raw_by_id: dict[str, str] = {}
        for j in judgments:
            assert j.raw_text is not None
            raw_by_id.setdefault(j.custom_id, j.raw_text)
        with results_path.open("w", encoding="utf-8") as fh:
            for custom_id, raw in raw_by_id.items():
                line = {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": raw}}]},
                    },
                    "error": None,
                }
                fh.write(json.dumps(line) + "\n")
        return raw_by_id

    def test_plan_emits_manifest_and_sidecar(self, planned_dir: Path) -> None:
        manifest = planned_dir / "manifest.jsonl"
        sidecar = planned_dir / "manifest.meta.jsonl"
        assert manifest.exists() and sidecar.exists()
        lines = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        assert lines, "manifest should not be empty"
        for line in lines:
            assert line["method"] == "POST"
            assert line["url"] == "/v1/chat/completions"
            assert line["body"]["model"] == "test-model"

    def test_ingest_then_analyze(
        self,
        runner: CliRunner,
        tmp_path: Path,
        items: list[PairwiseItem],
        planned_dir: Path,
    ) -> None:
        results_path = tmp_path / "results.jsonl"
        raw_by_id = self.fake_batch_results(items, results_path)
        sidecar = planned_dir / "manifest.meta.jsonl"

        judgments_path = tmp_path / "judgments.jsonl"
        result = runner.invoke(
            main,
            [
                "ingest",
                "--sidecar",
                str(sidecar),
                "--results",
                str(results_path),
                "-o",
                str(judgments_path),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        rows = [json.loads(s) for s in judgments_path.read_text().splitlines() if s.strip()]
        # fan-out: one judgment per sidecar usage
        n_usages = sum(
            len(json.loads(s)["usages"])
            for s in sidecar.read_text().splitlines()
            if s.strip()
        )
        assert len(rows) == n_usages
        for row in rows:
            assert set(row) == {"custom_id", "verdict", "raw_text", "meta"}
            assert row["verdict"] in ("first", "second", "tie", "invalid")
            assert row["custom_id"] in raw_by_id
            assert row["meta"]["probe"] in ("position", "verbosity")

        card_dir = tmp_path / "card"
        result = runner.invoke(
            main,
            [
                "analyze",
                "--judgments",
                str(judgments_path),
                "--judge",
                "test-model",
                "-o",
                str(card_dir),
                "--n-boot",
                "60",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        assert "Judge Reliability Card" in result.output
        assert (card_dir / "card.md").exists()
        card = load_card(card_dir / "card.json")
        assert [e.probe for e in card.probes] == ["position", "verbosity"]
        assert card.judge["model"] == "test-model"
        assert card.created_utc is not None

    def test_plan_resume_writes_remaining(
        self,
        runner: CliRunner,
        tmp_path: Path,
        items: list[PairwiseItem],
        planned_dir: Path,
    ) -> None:
        # Partial results: only the first 10 output lines exist.
        full_results = tmp_path / "full_results.jsonl"
        self.fake_batch_results(items, full_results)
        partial = tmp_path / "partial.jsonl"
        partial.write_text(
            "\n".join(full_results.read_text().splitlines()[:10]) + "\n", encoding="utf-8"
        )

        items_path = tmp_path / "items2.jsonl"
        write_items_jsonl(items_path, items)
        out_dir = tmp_path / "plan2"
        result = runner.invoke(
            main,
            [
                "plan",
                "--items",
                str(items_path),
                "--probes",
                self.PROBES,
                "--model",
                "test-model",
                "-o",
                str(out_dir),
                "--resume",
                str(partial),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        resume_path = out_dir / "manifest.resume.jsonl"
        assert resume_path.exists()
        n_total = len(
            [s for s in (out_dir / "manifest.jsonl").read_text().splitlines() if s.strip()]
        )
        n_remaining = len([s for s in resume_path.read_text().splitlines() if s.strip()])
        assert n_remaining == n_total - 10

    def test_plan_missing_item_key_errors_cleanly(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"item_id": "x", "prompt": "p", "response_a": "a"}) + "\n")
        result = runner.invoke(
            main,
            ["plan", "--items", str(bad), "--model", "m", "-o", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "missing required key" in combined_output(result)
        assert "Traceback" not in combined_output(result)

    def test_plan_score_endpoint_emits_score_bodies(
        self, runner: CliRunner, tmp_path: Path, items: list[PairwiseItem]
    ) -> None:
        """`plan --endpoint /v1/score` produces runnable vLLM score lines."""
        items_path = tmp_path / "items.jsonl"
        write_items_jsonl(items_path, items)
        out_dir = tmp_path / "score_plan"
        result = runner.invoke(
            main,
            [
                "plan",
                "--items",
                str(items_path),
                "--probes",
                "position",
                "--model",
                "rm-model",
                "--endpoint",
                "/v1/score",
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        lines = [
            json.loads(s)
            for s in (out_dir / "manifest.jsonl").read_text().splitlines()
            if s.strip()
        ]
        assert lines
        for line in lines:
            assert line["url"] == "/v1/score"
            assert set(line["body"]) == {"model", "text_1", "text_2"}
            assert line["body"]["model"] == "rm-model"
            assert isinstance(line["body"]["text_2"], list)
            assert len(line["body"]["text_2"]) == 2
        # sidecar usages stay lean: no raw-text keys stored
        for s in (out_dir / "manifest.meta.jsonl").read_text().splitlines():
            for usage in json.loads(s)["usages"]:
                assert "prompt_text" not in usage

    def test_plan_resume_tolerates_truncated_results(
        self,
        runner: CliRunner,
        tmp_path: Path,
        items: list[PairwiseItem],
        planned_dir: Path,
    ) -> None:
        """A byte-truncated results file (interrupted run) must not traceback."""
        results = tmp_path / "results.jsonl"
        self.fake_batch_results(items, results)
        raw = results.read_bytes()
        results.write_bytes(raw[:-40])  # truncate inside the final line

        items_path = tmp_path / "items3.jsonl"
        write_items_jsonl(items_path, items)
        out_dir = tmp_path / "plan3"
        result = runner.invoke(
            main,
            [
                "plan",
                "--items",
                str(items_path),
                "--probes",
                self.PROBES,
                "--model",
                "test-model",
                "-o",
                str(out_dir),
                "--resume",
                str(results),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        assert "Traceback" not in combined_output(result)
        resume_path = out_dir / "manifest.resume.jsonl"
        assert resume_path.exists()
        # the truncated line's request is back in the resume set
        n_total = len(
            [s for s in (out_dir / "manifest.jsonl").read_text().splitlines() if s.strip()]
        )
        n_results = len([s for s in results.read_text().splitlines() if s.strip()])
        n_remaining = len([s for s in resume_path.read_text().splitlines() if s.strip()])
        assert n_remaining == n_total - n_results + 1

    def test_ingest_tolerates_truncated_results(
        self,
        runner: CliRunner,
        tmp_path: Path,
        items: list[PairwiseItem],
        planned_dir: Path,
    ) -> None:
        results = tmp_path / "results.jsonl"
        self.fake_batch_results(items, results)
        raw = results.read_bytes()
        results.write_bytes(raw[:-40])
        out_path = tmp_path / "judgments.jsonl"
        result = runner.invoke(
            main,
            [
                "ingest",
                "--sidecar",
                str(planned_dir / "manifest.meta.jsonl"),
                "--results",
                str(results),
                "-o",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        assert "Traceback" not in combined_output(result)
        assert out_path.exists()
        rows = [json.loads(s) for s in out_path.read_text().splitlines() if s.strip()]
        assert rows  # intact lines still ingested


# ---------------------------------------------------------------------------
# validate (mocked: the validate module is owned by a concurrent agent)
# ---------------------------------------------------------------------------


class TestValidate:
    def install_fake(
        self, monkeypatch: pytest.MonkeyPatch, *, passed: bool, calls: dict[str, Any]
    ) -> None:
        fake = types.ModuleType("judgecal.validate")

        class FakeReport:
            def render_table(self) -> str:
                return "scenario      status\nnull-judge    " + ("PASS" if passed else "FAIL")

        FakeReport.passed = passed  # type: ignore[attr-defined]

        def run_validation(level: str, seed: int = 7) -> Any:
            calls["level"] = level
            calls["seed"] = seed
            return FakeReport()

        fake.run_validation = run_validation  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "judgecal.validate", fake)

    def test_pass_exits_zero(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, Any] = {}
        self.install_fake(monkeypatch, passed=True, calls=calls)
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0, combined_output(result)
        assert "PASS" in result.output
        assert calls == {"level": "fast", "seed": 7}

    def test_full_flag_and_failure_exits_one(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}
        self.install_fake(monkeypatch, passed=False, calls=calls)
        result = runner.invoke(main, ["validate", "--full", "--seed", "11"])
        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert calls == {"level": "full", "seed": 11}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_list_names_all_adapters(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["datasets", "list"])
        assert result.exit_code == 0, combined_output(result)
        for name in ("rewardbench2", "judgebench", "llmbar", "mtbench_human", "rmbench"):
            assert name in result.output
        assert "license" in result.output

    def test_fetch_unknown_adapter_is_usage_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            main, ["datasets", "fetch", "nope", "-o", str(tmp_path / "x.jsonl")]
        )
        assert result.exit_code == 2
        assert "nope" in combined_output(result)
        assert "Traceback" not in combined_output(result)


# ---------------------------------------------------------------------------
# slurm-pack
# ---------------------------------------------------------------------------


class TestSlurmPack:
    @pytest.fixture()
    def manifest_path(self, tmp_path: Path) -> Path:
        paths = emit_manifest(
            make_requests(3), ModelSpec(model="test-model"), tmp_path, "mani"
        )
        return paths.manifest

    def test_writes_pack_files(
        self, runner: CliRunner, tmp_path: Path, manifest_path: Path
    ) -> None:
        pack = tmp_path / "pack"
        result = runner.invoke(
            main,
            [
                "slurm-pack",
                "--manifest",
                str(manifest_path),
                "--model",
                "test-model",
                "-o",
                str(pack),
                "--partition",
                "a100",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        for fname in ("run_vllm.sbatch", "run_llamacpp.sbatch", "README_RUN.md", "mani.jsonl"):
            assert (pack / fname).exists()  # manifest is copied into the pack
        script = (pack / "run_vllm.sbatch").read_text()
        assert "test-model" in script
        assert "--partition=a100" in script
        assert "vllm run-batch" in script
        assert 'MANIFEST="${MANIFEST:-mani.jsonl}"' in script  # pack-relative

    def test_module_form_flag(
        self, runner: CliRunner, tmp_path: Path, manifest_path: Path
    ) -> None:
        pack = tmp_path / "pack_module"
        result = runner.invoke(
            main,
            [
                "slurm-pack",
                "--manifest",
                str(manifest_path),
                "--model",
                "test-model",
                "-o",
                str(pack),
                "--vllm-module-form",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        assert "vllm.entrypoints.openai.run_batch" in (pack / "run_vllm.sbatch").read_text()


# ---------------------------------------------------------------------------
# claude-run (mocked executor; never invokes the real CLI)
# ---------------------------------------------------------------------------


class FakeClaudeExecutor:
    """Stands in for ClaudeCodeExecutor; records construction and requests."""

    last: FakeClaudeExecutor | None = None

    def __init__(
        self,
        model: str | None = None,
        claude_bin: str = "claude",
        timeout_s: int = 120,
        **_: Any,
    ) -> None:
        self.model = model
        self.claude_bin = claude_bin
        self.timeout_s = timeout_s
        self.requests: list[JudgmentRequest] = []
        FakeClaudeExecutor.last = self

    def execute(self, requests: Any) -> list[Judgment]:
        self.requests = list(requests)
        return [
            Judgment(
                custom_id=r.custom_id,
                verdict="first",
                raw_text="The first answer is better. [[A]]",
                meta=dict(r.meta),
            )
            for r in self.requests
        ]


class TestClaudeRun:
    @pytest.fixture()
    def pack(self, tmp_path: Path) -> tuple[Path, Path]:
        paths = emit_manifest(
            make_requests(4), ModelSpec(model="test-model"), tmp_path, "mani"
        )
        return paths.manifest, paths.sidecar

    @pytest.fixture(autouse=True)
    def patch_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeClaudeExecutor.last = None
        monkeypatch.setattr("judgecal.cli.ClaudeCodeExecutor", FakeClaudeExecutor)

    def test_runs_and_writes_judgments(
        self, runner: CliRunner, tmp_path: Path, pack: tuple[Path, Path]
    ) -> None:
        manifest, sidecar = pack
        out = tmp_path / "judgments.jsonl"
        result = runner.invoke(
            main,
            [
                "claude-run",
                "--manifest",
                str(manifest),
                "--sidecar",
                str(sidecar),
                "-o",
                str(out),
                "--model",
                "sonnet",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        # loud subscription warning
        assert "subscription" in combined_output(result)
        fake = FakeClaudeExecutor.last
        assert fake is not None
        assert fake.model == "sonnet"
        assert len(fake.requests) == 4  # 4 lines < default limit 25
        rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
        assert len(rows) == 4  # one usage per custom_id here
        assert all(r["verdict"] == "first" for r in rows)
        assert all(set(r) == {"custom_id", "verdict", "raw_text", "meta"} for r in rows)

    def test_limit_caps_executed_lines(
        self, runner: CliRunner, tmp_path: Path, pack: tuple[Path, Path]
    ) -> None:
        manifest, sidecar = pack
        out = tmp_path / "j.jsonl"
        result = runner.invoke(
            main,
            [
                "claude-run",
                "--manifest",
                str(manifest),
                "--sidecar",
                str(sidecar),
                "-o",
                str(out),
                "--limit",
                "2",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        fake = FakeClaudeExecutor.last
        assert fake is not None and len(fake.requests) == 2
        assert "--limit 0" in combined_output(result)

    def test_limit_zero_means_all(
        self, runner: CliRunner, tmp_path: Path, pack: tuple[Path, Path]
    ) -> None:
        manifest, sidecar = pack
        result = runner.invoke(
            main,
            [
                "claude-run",
                "--manifest",
                str(manifest),
                "--sidecar",
                str(sidecar),
                "-o",
                str(tmp_path / "j.jsonl"),
                "--limit",
                "0",
            ],
        )
        assert result.exit_code == 0, combined_output(result)
        fake = FakeClaudeExecutor.last
        assert fake is not None and len(fake.requests) == 4

    def test_negative_limit_is_usage_error(
        self, runner: CliRunner, tmp_path: Path, pack: tuple[Path, Path]
    ) -> None:
        manifest, sidecar = pack
        result = runner.invoke(
            main,
            [
                "claude-run",
                "--manifest",
                str(manifest),
                "--sidecar",
                str(sidecar),
                "-o",
                str(tmp_path / "j.jsonl"),
                "--limit",
                "-1",
            ],
        )
        assert result.exit_code == 2
        assert "--limit" in combined_output(result)

    def test_sidecar_mismatch_errors_cleanly(
        self, runner: CliRunner, tmp_path: Path, pack: tuple[Path, Path]
    ) -> None:
        manifest, _ = pack
        # Sidecar from a DIFFERENT set of requests: no shared custom_ids.
        other = emit_manifest(
            [
                JudgmentRequest(
                    custom_id="jc-deadbeefdeadbeefdeadbeef-r0",
                    body={"messages": [{"role": "user", "content": "other"}]},
                    meta=make_meta("zz-1"),
                )
            ],
            ModelSpec(model="m"),
            tmp_path / "other",
            "other",
        )
        result = runner.invoke(
            main,
            [
                "claude-run",
                "--manifest",
                str(manifest),
                "--sidecar",
                str(other.sidecar),
                "-o",
                str(tmp_path / "j.jsonl"),
            ],
        )
        assert result.exit_code == 1
        assert "no sidecar entry" in combined_output(result)
        assert "Traceback" not in combined_output(result)
