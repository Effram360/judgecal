"""Tests for judgecal.executors: parsing, local executors, CLI executor, SLURM packs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from judgecal.core import JudgmentRequest, make_custom_id
from judgecal.executors import (
    ClaudeCodeExecutor,
    Executor,
    ExecutorWarning,
    FixtureExecutor,
    MockJudgeExecutor,
    SlurmConfig,
    compare_scores,
    parse_verdict,
    write_slurm_pack,
)
from judgecal.executors.base import judgment_from_raw
from judgecal.executors.slurm import VLLM_MODULE_COMMAND
from judgecal.manifests import ModelSpec, emit_manifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_meta(
    item_id: str = "it-1",
    probe: str = "position",
    condition: str = "orig",
    repeat: int = 0,
    **over: Any,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "probe": probe,
        "condition": condition,
        "item_id": item_id,
        "repeat": repeat,
        "first_is_a": True,
        "first_len": 100,
        "second_len": 100,
        "first_author": None,
        "second_author": None,
        "first_latent_q": None,
        "second_latent_q": None,
        "label_first": None,
    }
    meta.update(over)
    return meta


def make_request(
    content: str = "Judge this.",
    meta: dict[str, Any] | None = None,
    repeat: int = 0,
) -> JudgmentRequest:
    body = {"messages": [{"role": "system", "content": "You are a judge."},
                         {"role": "user", "content": content}]}
    return JudgmentRequest(
        custom_id=make_custom_id(body, repeat),
        body=body,
        meta=meta if meta is not None else make_meta(repeat=repeat),
    )


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The winner is [[A]]", "first"),
            ("The winner is [[B]]", "second"),
            ("It's a tie: [[C]]", "tie"),
            ("[[a]]", "first"),  # lowercase
            ("[[b]]", "second"),
            ("[[c]]", "tie"),
            ("[[ A ]]", "first"),  # whitespace inside brackets
            ("verdict: [[\tB ]]", "second"),
        ],
    )
    def test_basic_markers(self, text: str, expected: str) -> None:
        assert parse_verdict(text) == expected

    def test_last_marker_wins(self) -> None:
        text = "At first I thought [[A]], but reconsidering... my verdict is [[B]]"
        assert parse_verdict(text) == "second"

    def test_last_marker_wins_three_markers(self) -> None:
        assert parse_verdict("[[A]] then [[C]] finally [[a]]") == "first"

    def test_no_marker_is_invalid(self) -> None:
        assert parse_verdict("I cannot decide between these responses.") == "invalid"

    def test_single_brackets_do_not_match(self) -> None:
        assert parse_verdict("[A] is better") == "invalid"

    def test_empty_and_none_are_invalid(self) -> None:
        assert parse_verdict("") == "invalid"
        assert parse_verdict(None) == "invalid"

    def test_custom_pattern_named_groups(self) -> None:
        pattern = r"GRADE:\s*(?P<first>WIN)|GRADE:\s*(?P<second>LOSE)|GRADE:\s*(?P<tie>DRAW)"
        assert parse_verdict("blah GRADE: WIN", pattern) == "first"
        assert parse_verdict("grade: lose", pattern) == "second"  # case-insensitive
        assert parse_verdict("GRADE: WIN ... GRADE: DRAW", pattern) == "tie"  # last wins
        assert parse_verdict("no grade here", pattern) == "invalid"

    def test_custom_pattern_without_named_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="named group"):
            parse_verdict("[[A]]", pattern=r"\[\[([ABC])\]\]")

    def test_custom_pattern_partial_groups_ok(self) -> None:
        pattern = r"(?P<first>FIRST WINS)"
        assert parse_verdict("FIRST WINS", pattern) == "first"
        assert parse_verdict("SECOND WINS", pattern) == "invalid"


# ---------------------------------------------------------------------------
# compare_scores
# ---------------------------------------------------------------------------


class TestCompareScores:
    def test_clear_winner(self) -> None:
        assert compare_scores(0.9, 0.1) == "first"
        assert compare_scores(0.1, 0.9) == "second"

    def test_tie_within_epsilon(self) -> None:
        assert compare_scores(0.5, 0.5) == "tie"
        assert compare_scores(0.500001, 0.5, epsilon=1e-5) == "tie"
        assert compare_scores(0.51, 0.5, epsilon=0.02) == "tie"

    def test_just_outside_epsilon_is_decisive(self) -> None:
        assert compare_scores(0.52, 0.5, epsilon=0.01) == "first"

    def test_nan_is_invalid(self) -> None:
        with pytest.warns(UserWarning, match="non-finite"):
            assert compare_scores(float("nan"), 0.5) == "invalid"
        with pytest.warns(UserWarning, match="non-finite"):
            assert compare_scores(0.5, float("nan")) == "invalid"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (float("inf"), float("inf")),
            (float("-inf"), float("-inf")),
            (float("inf"), 0.5),
            (0.5, float("-inf")),
        ],
    )
    def test_non_finite_is_invalid_with_warning(self, first: float, second: float) -> None:
        # equal infinite scores must NOT come out as a decisive "second"
        with pytest.warns(UserWarning, match="non-finite"):
            assert compare_scores(first, second) == "invalid"

    def test_negative_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            compare_scores(0.5, 0.5, epsilon=-0.1)


# ---------------------------------------------------------------------------
# MockJudgeExecutor (with injected responder; real fixtures path is soft)
# ---------------------------------------------------------------------------


class TestMockJudgeExecutor:
    def test_runs_raw_text_through_real_parser(self) -> None:
        def responder(request: JudgmentRequest) -> str:
            # deterministic function of the request meta
            return "thinking... [[A]]" if request.meta["condition"] == "orig" else "[[B]]"

        executor = MockJudgeExecutor(config=None, responder=responder)
        reqs = [
            make_request("x", make_meta(condition="orig")),
            make_request("y", make_meta(condition="swap", first_is_a=False)),
        ]
        j_orig, j_swap = executor.execute(reqs)
        assert j_orig.verdict == "first"
        assert j_swap.verdict == "second"
        assert j_swap.mapped_verdict == "A"  # swap: presented-second is response_a
        assert j_orig.raw_text == "thinking... [[A]]"

    def test_unparseable_responder_output_is_invalid(self) -> None:
        executor = MockJudgeExecutor(config=None, responder=lambda _: "no marker")
        (j,) = executor.execute([make_request()])
        assert j.verdict == "invalid"

    def test_satisfies_executor_protocol(self) -> None:
        executor = MockJudgeExecutor(config=None, responder=lambda _: "[[C]]")
        assert isinstance(executor, Executor)

    def test_real_fixtures_resolution(self) -> None:
        """Soft integration check against judgecal.fixtures (MOCK agent's module)."""
        fixtures = pytest.importorskip("judgecal.fixtures")
        config_cls = getattr(fixtures, "MockJudgeConfig", None)
        if config_cls is None:
            pytest.skip("judgecal.fixtures has no MockJudgeConfig yet")
        try:
            executor = MockJudgeExecutor(config_cls(seed=0))
        except (AttributeError, TypeError) as exc:
            pytest.skip(
                f"fixtures responder API not resolvable by MockJudgeExecutor: {exc} "
                "— integration phase must align the entry-point names"
            )
        judgments = executor.execute([make_request()])
        assert len(judgments) == 1
        assert judgments[0].verdict in ("first", "second", "tie", "invalid")


# ---------------------------------------------------------------------------
# FixtureExecutor
# ---------------------------------------------------------------------------


class TestFixtureExecutor:
    def test_replays_dict_pack(self) -> None:
        req = make_request()
        executor = FixtureExecutor({req.custom_id: "verdict: [[B]]"})
        (j,) = executor.execute([req])
        assert j.verdict == "second"
        assert j.raw_text == "verdict: [[B]]"
        assert j.meta == req.meta

    def test_pack_object_with_responses_attr(self) -> None:
        req = make_request()

        class Pack:
            responses = {req.custom_id: "[[C]]"}

        (j,) = FixtureExecutor(Pack()).execute([req])
        assert j.verdict == "tie"

    def test_strict_missing_raises_keyerror(self) -> None:
        executor = FixtureExecutor({}, strict=True)
        with pytest.raises(KeyError, match="not found in fixture pack"):
            executor.execute([make_request()])

    def test_lenient_missing_warns_and_is_invalid(self) -> None:
        executor = FixtureExecutor({}, strict=False)
        req = make_request()
        with pytest.warns(ExecutorWarning, match="missing from fixture pack"):
            (j,) = executor.execute([req])
        assert j.verdict == "invalid"
        assert j.raw_text is None
        assert j.custom_id == req.custom_id

    def test_satisfies_executor_protocol(self) -> None:
        assert isinstance(FixtureExecutor({}), Executor)


# ---------------------------------------------------------------------------
# ClaudeCodeExecutor — subprocess.run is ALWAYS mocked here
# ---------------------------------------------------------------------------


def _cli_json(text: str, is_error: bool = False) -> str:
    return json.dumps(
        {"type": "result", "subtype": "success", "is_error": is_error, "result": text}
    )


class TestClaudeCodeExecutor:
    def test_success_parses_result_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"cmd": cmd, **kwargs})
            return subprocess.CompletedProcess(cmd, 0, stdout=_cli_json("I pick [[B]]"), stderr="")

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        executor = ClaudeCodeExecutor(model="sonnet", timeout_s=42)
        req = make_request("Compare these.")
        (j,) = executor.execute([req])

        assert j.verdict == "second"
        assert j.raw_text == "I pick [[B]]"
        assert len(calls) == 1
        cmd = calls[0]["cmd"]
        # verified headless flags (Claude Code CLI 2.1.172)
        assert "-p" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert "--tools" in cmd
        assert "--no-session-persistence" in cmd
        # prompt is passed via stdin, rendered from the messages
        assert "Compare these." in calls[0]["input"]
        assert "[system]" in calls[0]["input"]
        assert calls[0]["timeout"] == 42

    def test_sequential_one_subprocess_per_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        count = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal count
            count += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=_cli_json("[[A]]"), stderr="")

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        reqs = [make_request(f"item {i}", make_meta(item_id=f"it-{i}")) for i in range(3)]
        judgments = ClaudeCodeExecutor().execute(reqs)
        assert count == 3
        assert [j.verdict for j in judgments] == ["first"] * 3

    def test_timeout_yields_invalid_with_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd, timeout=kwargs.get("timeout", 0))

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        with pytest.warns(ExecutorWarning, match="timed out"):
            (j,) = ClaudeCodeExecutor(timeout_s=1).execute([make_request()])
        assert j.verdict == "invalid"

    def test_nonzero_exit_yields_invalid_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rate limited")

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        with pytest.warns(ExecutorWarning, match="exited 1"):
            (j,) = ClaudeCodeExecutor().execute([make_request()])
        assert j.verdict == "invalid"

    def test_bad_json_yields_invalid_with_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 0, stdout="not json at all", stderr="")

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        with pytest.warns(ExecutorWarning, match="non-JSON"):
            (j,) = ClaudeCodeExecutor().execute([make_request()])
        assert j.verdict == "invalid"
        assert j.raw_text == "not json at all"

    def test_error_payload_yields_invalid_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_cli_json("ignored", is_error=True), stderr=""
            )

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        with pytest.warns(ExecutorWarning, match="could not extract"):
            (j,) = ClaudeCodeExecutor().execute([make_request()])
        assert j.verdict == "invalid"

    def test_missing_binary_raises_helpful_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr("judgecal.executors.claude_code.subprocess.run", fake_run)
        with pytest.raises(FileNotFoundError, match="claude.com/claude-code"):
            ClaudeCodeExecutor(claude_bin="claude-nonexistent").execute([make_request()])

    def test_score_body_raises_value_error(self) -> None:
        req = JudgmentRequest(
            custom_id=make_custom_id({"text_1": "q"}),
            body={"text_1": "q", "text_2": ["a", "b"]},
            meta=make_meta(),
        )
        with pytest.raises(ValueError, match="messages"):
            ClaudeCodeExecutor().execute([req])

    def test_satisfies_executor_protocol(self) -> None:
        assert isinstance(ClaudeCodeExecutor(), Executor)


# ---------------------------------------------------------------------------
# judgment_from_raw
# ---------------------------------------------------------------------------


def test_judgment_from_raw_copies_meta() -> None:
    req = make_request()
    j = judgment_from_raw(req, "[[A]]")
    assert j.meta == req.meta
    assert j.meta is not req.meta  # defensive copy
    assert j.verdict == "first"


# ---------------------------------------------------------------------------
# SLURM pack generation (golden strings)
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_paths(tmp_path: Path):  # type: ignore[no-untyped-def]
    spec = ModelSpec(model="qwen3.5-9b-awq")
    req = make_request()
    return emit_manifest([req], spec, tmp_path / "manifests", "suite")


class TestWriteSlurmPack:
    def test_pack_files_written_and_executable(self, manifest_paths, tmp_path: Path) -> None:
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack")
        assert pack.vllm_sbatch.exists()
        assert pack.llamacpp_sbatch.exists()
        assert pack.readme.exists()
        assert pack.vllm_sbatch.stat().st_mode & 0o111  # executable
        assert pack.llamacpp_sbatch.stat().st_mode & 0o111

    def test_manifest_copied_into_pack(self, manifest_paths, tmp_path: Path) -> None:
        """The pack is self-contained: the manifest travels with it."""
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack")
        assert pack.manifest == tmp_path / "pack" / "suite.jsonl"
        assert pack.manifest.exists()
        assert pack.manifest.read_bytes() == Path(manifest_paths.manifest).read_bytes()

    def test_scripts_use_pack_relative_paths(self, manifest_paths, tmp_path: Path) -> None:
        """No laptop-local paths are baked in; the scripts cd to the pack."""
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack")
        for script in (pack.vllm_sbatch, pack.llamacpp_sbatch):
            text = script.read_text()
            assert 'MANIFEST="${MANIFEST:-suite.jsonl}"' in text
            assert 'RESULTS="${RESULTS:-results.jsonl}"' in text
            assert 'cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"' in text
            # the generation-time absolute path must NOT leak into scripts
            assert str(tmp_path) not in text

    def test_vllm_sbatch_golden_strings(self, manifest_paths, tmp_path: Path) -> None:
        cluster = SlurmConfig(
            partition="batch-gpu",
            gpus=2,
            walltime="08:00:00",
            account="proj-judges",
            modules=("module load cuda/12.4", "module load vllm/0.9"),
        )
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack", cluster)
        text = pack.vllm_sbatch.read_text()
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "#SBATCH --partition=batch-gpu" in text
        assert "#SBATCH --gres=gpu:2" in text
        assert "#SBATCH --time=08:00:00" in text
        assert "#SBATCH --account=proj-judges" in text
        assert "module load cuda/12.4" in text
        assert "module load vllm/0.9" in text
        assert 'vllm run-batch -i "$MANIFEST" -o "$RESULTS" --model "$MODEL"' in text
        # the submit-time override comment names this script and variable
        assert "sbatch --export=ALL,MODEL=other-model run_vllm.sbatch" in text
        # module-form alternative documented inline for older vLLM versions
        assert "python -m vllm.entrypoints.openai.run_batch" in text
        # model from ManifestPaths baked in as the default
        assert "qwen3.5-9b-awq" in text
        # array-friendliness
        assert "SLURM_ARRAY_TASK_ID" in text
        assert "set -euo pipefail" in text

    def test_account_omitted_when_none(self, manifest_paths, tmp_path: Path) -> None:
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack", SlurmConfig(account=None))
        assert "--account" not in pack.vllm_sbatch.read_text()

    def test_llamacpp_sbatch_golden_strings(self, manifest_paths, tmp_path: Path) -> None:
        pack = write_slurm_pack(
            manifest_paths, tmp_path / "pack", gguf_path="/models/q4/judge.gguf"
        )
        text = pack.llamacpp_sbatch.read_text()
        assert "llama-server" in text
        assert "--parallel 1" in text  # single-slot determinism
        assert "/models/q4/judge.gguf" in text
        assert "custom_id" in text  # embedded driver writes batch-format output
        assert "/health" in text

    def test_llamacpp_override_comment_names_gguf_and_own_script(
        self, manifest_paths, tmp_path: Path
    ) -> None:
        """The submit-time override hint must reference GGUF and the
        llama.cpp script, not the vLLM script's MODEL variable."""
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack")
        text = pack.llamacpp_sbatch.read_text()
        assert "sbatch --export=ALL,GGUF=other-model run_llamacpp.sbatch" in text
        assert "MODEL=other-model run_vllm.sbatch" not in text

    def test_readme_documents_both_vllm_invocations(
        self, manifest_paths, tmp_path: Path
    ) -> None:
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack")
        text = pack.readme.read_text()
        assert "vllm run-batch" in text
        assert VLLM_MODULE_COMMAND in text
        assert "sbatch run_vllm.sbatch" in text
        assert "judgecal ingest" in text
        assert "suite.meta.jsonl" in text
        assert "--array" in text
        # instructions are runnable as written from inside the pack dir
        assert "cd <pack-dir>" in text
        assert "suite.jsonl" in text

    def test_vllm_command_parameterized(self, manifest_paths, tmp_path: Path) -> None:
        pack = write_slurm_pack(
            manifest_paths, tmp_path / "pack", vllm_command=VLLM_MODULE_COMMAND
        )
        text = pack.vllm_sbatch.read_text()
        assert (
            f'{VLLM_MODULE_COMMAND} -i "$MANIFEST" -o "$RESULTS" --model "$MODEL"' in text
        )

    def test_container_prefix(self, manifest_paths, tmp_path: Path) -> None:
        cluster = SlurmConfig(container="apptainer exec --nv vllm.sif")
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack", cluster)
        assert "apptainer exec --nv vllm.sif vllm run-batch" in pack.vllm_sbatch.read_text()

    def test_bare_path_requires_model(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("")
        with pytest.raises(ValueError, match="model"):
            write_slurm_pack(manifest, tmp_path / "pack")
        pack = write_slurm_pack(manifest, tmp_path / "pack", model="my-model")
        assert "my-model" in pack.vllm_sbatch.read_text()
        # inferred sidecar name in the README
        assert "m.meta.jsonl" in pack.readme.read_text()

    def test_extra_sbatch_lines(self, manifest_paths, tmp_path: Path) -> None:
        cluster = SlurmConfig(extra_sbatch=("#SBATCH --qos=high",))
        pack = write_slurm_pack(manifest_paths, tmp_path / "pack", cluster)
        assert "#SBATCH --qos=high" in pack.vllm_sbatch.read_text()
