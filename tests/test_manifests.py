"""Tests for judgecal.manifests: emission, dedup, sidecar merge, resume."""

from __future__ import annotations

import json
import warnings as warnings_module
from pathlib import Path
from typing import Any

import pytest

from judgecal.core import (
    SCORE_TEXT_META_KEYS,
    Judgment,
    JudgmentRequest,
    PairwiseItem,
    make_custom_id,
)
from judgecal.manifests import (
    ManifestWarning,
    ModelSpec,
    emit_manifest,
    load_sidecar,
    merge_results,
    remaining_manifest,
)
from judgecal.probes import ProbeConfig, plan_suite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_meta(
    item_id: str = "it-1",
    probe: str = "position",
    condition: str = "orig",
    repeat: int = 0,
    first_is_a: bool = True,
    **over: Any,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "probe": probe,
        "condition": condition,
        "item_id": item_id,
        "repeat": repeat,
        "first_is_a": first_is_a,
        "first_len": 120,
        "second_len": 80,
        "first_author": None,
        "second_author": None,
        "first_latent_q": None,
        "second_latent_q": None,
        "label_first": None,
    }
    meta.update(over)
    return meta


def make_request(
    body: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    repeat: int = 0,
) -> JudgmentRequest:
    body = body if body is not None else {
        "messages": [{"role": "user", "content": "Judge this pair."}]
    }
    return JudgmentRequest(
        custom_id=make_custom_id(body, repeat),
        body=body,
        meta=meta if meta is not None else make_meta(repeat=repeat),
    )


def make_text_meta(
    item_id: str = "it-1",
    prompt_text: str = "Judge this.",
    first_text: str = "resp first",
    second_text: str = "resp second",
    **over: Any,
) -> dict[str, Any]:
    """Meta including the reserved raw-text keys probes plan with."""
    meta = make_meta(item_id=item_id, **over)
    meta.update(
        {
            "prompt_text": prompt_text,
            "first_text": first_text,
            "second_text": second_text,
        }
    )
    return meta


def chat_output_line(custom_id: str, content: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"role": "assistant", "content": content}}]},
        },
        "error": None,
    }


def score_output_line(custom_id: str, scores: list[float]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {"data": [{"index": i, "score": s} for i, s in enumerate(scores)]},
        },
        "error": None,
    }


def write_jsonl(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


SPEC = ModelSpec(model="qwen3.5-9b-awq")


# ---------------------------------------------------------------------------
# emit_manifest
# ---------------------------------------------------------------------------


class TestEmitManifest:
    def test_identical_bodies_dedup_to_one_line(self, tmp_path: Path) -> None:
        body = {"messages": [{"role": "user", "content": "same body"}]}
        r1 = make_request(body, make_meta(probe="position", condition="orig"))
        r2 = make_request(body, make_meta(probe="verbosity", condition="pad_second"))
        assert r1.custom_id == r2.custom_id

        paths = emit_manifest([r1, r2], SPEC, tmp_path, "m")
        manifest_lines = [json.loads(s) for s in paths.manifest.read_text().splitlines()]
        assert len(manifest_lines) == 1
        assert paths.n_lines == 1
        assert paths.n_usages == 2

        sidecar = load_sidecar(paths.sidecar)
        assert len(sidecar) == 1
        entry = sidecar[r1.custom_id]
        assert {u["probe"] for u in entry.usages} == {"position", "verbosity"}

    def test_repeats_are_not_deduped(self, tmp_path: Path) -> None:
        body = {"messages": [{"role": "user", "content": "same body"}]}
        r0 = make_request(body, make_meta(probe="stability", condition="rep", repeat=0), repeat=0)
        r1 = make_request(body, make_meta(probe="stability", condition="rep", repeat=1), repeat=1)
        assert r0.custom_id != r1.custom_id
        assert r0.custom_id.endswith("-r0")
        assert r1.custom_id.endswith("-r1")

        paths = emit_manifest([r0, r1], SPEC, tmp_path, "m")
        assert paths.n_lines == 2

    def test_byte_identical_usages_recorded_once(self, tmp_path: Path) -> None:
        body = {"messages": [{"role": "user", "content": "x"}]}
        meta = make_meta()
        r1 = make_request(body, dict(meta))
        r2 = make_request(body, dict(meta))
        paths = emit_manifest([r1, r2], SPEC, tmp_path, "m")
        assert paths.n_usages == 1

    def test_batch_line_format(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        line = json.loads(paths.manifest.read_text().splitlines()[0])
        assert line["custom_id"] == req.custom_id
        assert line["method"] == "POST"
        assert line["url"] == "/v1/chat/completions"
        assert line["body"]["model"] == "qwen3.5-9b-awq"
        assert line["body"]["temperature"] == 0.0
        assert line["body"]["max_tokens"] == 1024
        assert line["body"]["messages"] == req.body["messages"]

    def test_score_endpoint_builds_score_bodies(self, tmp_path: Path) -> None:
        """A score manifest line is a vLLM score-API body, NOT a chat body."""
        spec = ModelSpec(model="skywork-rm-8b", endpoint="/v1/score")
        req = make_request(meta=make_text_meta(prompt_text="q", first_text="a", second_text="b"))
        paths = emit_manifest([req], spec, tmp_path, "m")
        line = json.loads(paths.manifest.read_text().splitlines()[0])
        assert line["url"] == "/v1/score"
        assert line["body"] == {
            "model": "skywork-rm-8b",
            "text_1": "q",
            "text_2": ["a", "b"],  # presented order: (first, second)
        }
        assert "messages" not in line["body"]
        assert "temperature" not in line["body"]
        assert "max_tokens" not in line["body"]
        # custom_id recomputed from the EMITTED score body, -r<repeat> kept
        assert line["custom_id"] == make_custom_id(line["body"], 0)
        assert line["custom_id"].endswith("-r0")
        # sidecar keyed by the recomputed id; hash matches the score body
        entry = load_sidecar(paths.sidecar)[line["custom_id"]]
        assert f"jc-{entry.body_hash}-r0" == line["custom_id"]

    def test_score_endpoint_requires_text_meta_keys(self, tmp_path: Path) -> None:
        spec = ModelSpec(model="rm", endpoint="/v1/score")
        req = make_request()  # plain meta without the reserved text keys
        with pytest.raises(ValueError, match="prompt_text"):
            emit_manifest([req], spec, tmp_path, "m")

    def test_score_endpoint_dedups_same_texts_across_chat_bodies(
        self, tmp_path: Path
    ) -> None:
        """Different chat renderings of the same texts collapse to one score line."""
        spec = ModelSpec(model="rm", endpoint="/v1/score")
        r1 = make_request(
            {"messages": [{"role": "user", "content": "template one"}]},
            make_text_meta(probe="position", condition="orig"),
        )
        r2 = make_request(
            {"messages": [{"role": "user", "content": "template two"}]},
            make_text_meta(probe="template", condition="tpl:v2"),
        )
        assert r1.custom_id != r2.custom_id  # chat ids differ...
        paths = emit_manifest([r1, r2], spec, tmp_path, "m")
        assert paths.n_lines == 1  # ...but the score bodies coincide
        assert paths.n_usages == 2

    def test_sidecar_strips_raw_text_keys(self, tmp_path: Path) -> None:
        """Stored usages never carry prompt_text/first_text/second_text."""
        for spec in (SPEC, ModelSpec(model="rm", endpoint="/v1/score")):
            req = make_request(meta=make_text_meta())
            paths = emit_manifest([req], spec, tmp_path, f"m-{spec.endpoint.strip('/').replace('/', '-')}")
            (entry,) = load_sidecar(paths.sidecar).values()
            for usage in entry.usages:
                for key in SCORE_TEXT_META_KEYS:
                    assert key not in usage

    def test_extra_fields_merged_last(self, tmp_path: Path) -> None:
        spec = ModelSpec(model="m", extra={"seed": 7, "temperature": 0.3})
        paths = emit_manifest([make_request()], spec, tmp_path, "m")
        line = json.loads(paths.manifest.read_text().splitlines()[0])
        assert line["body"]["seed"] == 7
        assert line["body"]["temperature"] == 0.3  # extra overrides default

    def test_conflicting_custom_id_raises(self, tmp_path: Path) -> None:
        r1 = make_request({"messages": [{"role": "user", "content": "one"}]})
        bad = JudgmentRequest(
            custom_id=r1.custom_id,  # forged: same id, different body
            body={"messages": [{"role": "user", "content": "two"}]},
            meta=make_meta(),
        )
        with pytest.raises(ValueError, match="different bodies"):
            emit_manifest([r1, bad], SPEC, tmp_path, "m")

    def test_empty_requests_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no requests"):
            emit_manifest([], SPEC, tmp_path, "m")

    def test_invalid_endpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            ModelSpec(model="m", endpoint="/v1/embeddings")

    def test_sidecar_body_hash_matches_custom_id(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        entry = load_sidecar(paths.sidecar)[req.custom_id]
        assert f"jc-{entry.body_hash}-r0" == req.custom_id


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_round_trip_fan_out(self, tmp_path: Path) -> None:
        """manifest -> sidecar -> fake output -> merge yields one judgment per usage."""
        shared_body = {"messages": [{"role": "user", "content": "shared"}]}
        r_pos = make_request(shared_body, make_meta(probe="position", item_id="it-1"))
        r_verb = make_request(shared_body, make_meta(probe="verbosity", item_id="it-1"))
        other_body = {"messages": [{"role": "user", "content": "other"}]}
        r_other = make_request(other_body, make_meta(probe="position", item_id="it-2"))

        paths = emit_manifest([r_pos, r_verb, r_other], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "results.jsonl",
            [
                chat_output_line(r_pos.custom_id, "Reasoning... final: [[A]]"),
                chat_output_line(r_other.custom_id, "I conclude [[B]]"),
            ],
        )

        judgments = merge_results(paths.sidecar, out)
        assert len(judgments) == 3  # 2 usages on the shared line + 1
        assert all(isinstance(j, Judgment) for j in judgments)

        shared = [j for j in judgments if j.custom_id == r_pos.custom_id]
        assert {j.probe for j in shared} == {"position", "verbosity"}
        assert all(j.verdict == "first" for j in shared)
        assert all(j.raw_text == "Reasoning... final: [[A]]" for j in shared)

        other = [j for j in judgments if j.custom_id == r_other.custom_id]
        assert len(other) == 1
        assert other[0].verdict == "second"
        assert other[0].item_id == "it-2"

    def test_meta_echoed_into_judgments(self, tmp_path: Path) -> None:
        meta = make_meta(first_is_a=False, label_first="second")
        req = make_request(meta=meta)
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl", [chat_output_line(req.custom_id, "[[A]]")]
        )
        (j,) = merge_results(paths.sidecar, out)
        assert j.meta == meta
        assert j.mapped_verdict == "B"  # first wins, first is response_b

    def test_score_pair_outputs(self, tmp_path: Path) -> None:
        spec = ModelSpec(model="rm", endpoint="/v1/score")
        reqs = [
            make_request(
                {"messages": [{"role": "user", "content": f"chat {i}"}]},
                make_text_meta(item_id=f"it-{i}", first_text=f"a{i}", second_text=f"b{i}"),
            )
            for i in range(3)
        ]
        paths = emit_manifest(reqs, spec, tmp_path, "m")
        emitted_ids = [
            json.loads(s)["custom_id"] for s in paths.manifest.read_text().splitlines()
        ]
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                score_output_line(emitted_ids[0], [0.9, 0.1]),
                score_output_line(emitted_ids[1], [0.1, 0.9]),
                score_output_line(emitted_ids[2], [0.5, 0.5 + 1e-9]),
            ],
        )
        judgments = {j.item_id: j for j in merge_results(paths.sidecar, out)}
        assert judgments["it-0"].verdict == "first"
        assert judgments["it-1"].verdict == "second"
        assert judgments["it-2"].verdict == "tie"  # within default epsilon
        assert judgments["it-0"].raw_text is not None
        assert judgments["it-0"].raw_text.startswith("scores:")

    def test_score_epsilon_configurable(self, tmp_path: Path) -> None:
        spec = ModelSpec(model="rm", endpoint="/v1/score")
        req = make_request(meta=make_text_meta())
        paths = emit_manifest([req], spec, tmp_path, "m")
        emitted_id = json.loads(paths.manifest.read_text().splitlines()[0])["custom_id"]
        out = write_jsonl(
            tmp_path / "r.jsonl", [score_output_line(emitted_id, [0.52, 0.50])]
        )
        (tight,) = merge_results(paths.sidecar, out, score_epsilon=1e-6)
        assert tight.verdict == "first"
        (wide,) = merge_results(paths.sidecar, out, score_epsilon=0.05)
        assert wide.verdict == "tie"

    def test_single_score_is_invalid_with_warning(self, tmp_path: Path) -> None:
        spec = ModelSpec(model="rm", endpoint="/v1/score")
        req = make_request(meta=make_text_meta())
        paths = emit_manifest([req], spec, tmp_path, "m")
        emitted_id = json.loads(paths.manifest.read_text().splitlines()[0])["custom_id"]
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [{"custom_id": emitted_id, "response": {"body": {"score": 0.7}}, "error": None}],
        )
        with pytest.warns(ManifestWarning, match="single score"):
            (j,) = merge_results(paths.sidecar, out)
        assert j.verdict == "invalid"

    def test_score_round_trip_from_probe_plan(self, tmp_path: Path) -> None:
        """plan -> score manifest -> fake vLLM score outputs -> merge.

        The full scalar-RM path: probes plan against real items, the
        emitted bodies are valid vLLM score-API bodies whose text_2 order
        is (presented-first, presented-second), and merged verdicts map
        back to item coordinates correctly for both presentation orders.
        """
        item = PairwiseItem(
            item_id="rt-1",
            prompt="Which answer is better?",
            response_a="the good answer",
            response_b="bad",
        )
        requests = plan_suite([item], ["position"], ProbeConfig())
        spec = ModelSpec(model="rm-8b", endpoint="/v1/score")
        paths = emit_manifest(requests, spec, tmp_path, "m")

        lines = [json.loads(s) for s in paths.manifest.read_text().splitlines()]
        assert len(lines) == 2  # orig + swap
        by_first_text = {}
        for line in lines:
            assert line["url"] == "/v1/score"
            body = line["body"]
            assert set(body) == {"model", "text_1", "text_2"}
            assert body["model"] == "rm-8b"
            assert body["text_1"] == "Which answer is better?"
            assert sorted(body["text_2"]) == ["bad", "the good answer"]
            by_first_text[body["text_2"][0]] = line["custom_id"]
        # orig presents response_a first; swap presents response_b first
        assert set(by_first_text) == {"the good answer", "bad"}

        # RM scores response_a higher regardless of presentation order:
        # scores align with text_2 order (first, second).
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                score_output_line(by_first_text["the good answer"], [0.9, 0.2]),
                score_output_line(by_first_text["bad"], [0.2, 0.9]),
            ],
        )
        judgments = merge_results(paths.sidecar, out)
        assert len(judgments) == 2
        by_condition = {j.condition: j for j in judgments}
        assert by_condition["orig"].verdict == "first"
        assert by_condition["swap"].verdict == "second"
        # both map back to response_a in item coordinates
        assert by_condition["orig"].mapped_verdict == "A"
        assert by_condition["swap"].mapped_verdict == "A"

    def test_error_lines_become_invalid_judgments(self, tmp_path: Path) -> None:
        body = {"messages": [{"role": "user", "content": "x"}]}
        r1 = make_request(body, make_meta(probe="position"))
        r2 = make_request(body, make_meta(probe="verbosity"))
        paths = emit_manifest([r1, r2], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                {
                    "custom_id": r1.custom_id,
                    "response": None,
                    "error": {"message": "CUDA out of memory"},
                }
            ],
        )
        with pytest.warns(ManifestWarning, match="error line"):
            judgments = merge_results(paths.sidecar, out)
        assert len(judgments) == 2  # fanned out to both usages
        assert all(j.verdict == "invalid" for j in judgments)
        assert all(j.raw_text is None for j in judgments)

    def test_unknown_custom_id_warns_and_skips(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line("jc-ffffffffffffffffffffffff-r0", "[[A]]"),
                chat_output_line(req.custom_id, "[[C]]"),
            ],
        )
        with pytest.warns(ManifestWarning, match="not in sidecar"):
            judgments = merge_results(paths.sidecar, out)
        assert len(judgments) == 1
        assert judgments[0].verdict == "tie"

    def test_duplicate_output_lines_keep_first(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line(req.custom_id, "[[A]]"),
                chat_output_line(req.custom_id, "[[B]]"),
            ],
        )
        with pytest.warns(ManifestWarning, match="duplicate"):
            (j,) = merge_results(paths.sidecar, out)
        assert j.verdict == "first"

    def test_retried_success_replaces_earlier_error_line(self, tmp_path: Path) -> None:
        """Resume-retry regression: error in run 1, success in run 2.

        Ingesting results files in chronological order must keep the
        retried success, not the stale error line (last-non-error-wins).
        """
        req = make_request()
        other = make_request({"messages": [{"role": "user", "content": "other"}]})
        paths = emit_manifest([req, other], SPEC, tmp_path, "m")
        run1 = write_jsonl(
            tmp_path / "results.run1.jsonl",
            [
                {
                    "custom_id": req.custom_id,
                    "response": None,
                    "error": {"message": "CUDA OOM"},
                },
                chat_output_line(other.custom_id, "[[A]]"),
            ],
        )
        run2 = write_jsonl(
            tmp_path / "results.run2.jsonl",
            [chat_output_line(req.custom_id, "retried fine [[B]]")],
        )
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")  # no warning expected
            judgments = {j.custom_id: j for j in merge_results(paths.sidecar, [run1, run2])}
        assert judgments[req.custom_id].verdict == "second"  # retried [[B]] kept
        assert judgments[req.custom_id].raw_text == "retried fine [[B]]"
        assert judgments[other.custom_id].verdict == "first"

    def test_retried_success_replaces_error_in_same_appended_file(
        self, tmp_path: Path
    ) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                {
                    "custom_id": req.custom_id,
                    "response": None,
                    "error": {"message": "boom"},
                },
                chat_output_line(req.custom_id, "[[A]]"),
            ],
        )
        (j,) = merge_results(paths.sidecar, out)
        assert j.verdict == "first"

    def test_error_after_success_keeps_success_with_warning(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line(req.custom_id, "[[C]]"),
                {
                    "custom_id": req.custom_id,
                    "response": None,
                    "error": {"message": "late duplicate"},
                },
            ],
        )
        with pytest.warns(ManifestWarning, match="duplicate"):
            (j,) = merge_results(paths.sidecar, out)
        assert j.verdict == "tie"

    def test_truncated_trailing_line_skipped_with_warning(self, tmp_path: Path) -> None:
        """A byte-truncated final line (interrupted run) must not raise."""
        r1 = make_request({"messages": [{"role": "user", "content": "one"}]})
        r2 = make_request({"messages": [{"role": "user", "content": "two"}]})
        paths = emit_manifest([r1, r2], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line(r1.custom_id, "[[A]]"),
                chat_output_line(r2.custom_id, "[[B]]"),
            ],
        )
        raw = out.read_bytes()
        out.write_bytes(raw[:-40])  # truncate inside the final JSON line
        with pytest.warns(ManifestWarning, match=r"r\.jsonl:2.*malformed"):
            judgments = merge_results(paths.sidecar, out)
        assert len(judgments) == 1
        assert judgments[0].custom_id == r1.custom_id
        assert judgments[0].verdict == "first"

    def test_liberal_body_variants(self, tmp_path: Path) -> None:
        """Tolerate body nested under response directly and completions-style text."""
        body = {"messages": [{"role": "user", "content": "x"}]}
        r1 = make_request(body, make_meta(item_id="it-1"))
        body2 = {"messages": [{"role": "user", "content": "y"}]}
        r2 = make_request(body2, make_meta(item_id="it-2"))
        paths = emit_manifest([r1, r2], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                {  # body fields directly under "response"
                    "custom_id": r1.custom_id,
                    "response": {"choices": [{"message": {"content": "[[B]]"}}]},
                },
                {  # completions-style "text", list-of-parts tolerated elsewhere
                    "custom_id": r2.custom_id,
                    "response": {"body": {"choices": [{"text": "verdict [[C]]"}]}},
                },
            ],
        )
        by_item = {j.item_id: j for j in merge_results(paths.sidecar, out)}
        assert by_item["it-1"].verdict == "second"
        assert by_item["it-2"].verdict == "tie"

    def test_accepts_preloaded_sidecar_and_many_output_files(self, tmp_path: Path) -> None:
        req = make_request()
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        sidecar = load_sidecar(paths.sidecar)
        out1 = write_jsonl(tmp_path / "r1.jsonl", [chat_output_line(req.custom_id, "[[A]]")])
        out2 = write_jsonl(tmp_path / "r2.jsonl", [])
        judgments = merge_results(sidecar, [out1, out2])
        assert len(judgments) == 1


# ---------------------------------------------------------------------------
# remaining_manifest (resume)
# ---------------------------------------------------------------------------


class TestRemainingManifest:
    def _three_requests(self) -> list[JudgmentRequest]:
        return [
            make_request(
                {"messages": [{"role": "user", "content": f"item {i}"}]},
                make_meta(item_id=f"it-{i}"),
            )
            for i in range(3)
        ]

    def test_excludes_completed_keeps_errors_and_missing(self, tmp_path: Path) -> None:
        reqs = self._three_requests()
        paths = emit_manifest(reqs, SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line(reqs[0].custom_id, "done [[A]]"),  # completed
                {
                    "custom_id": reqs[1].custom_id,
                    "response": None,
                    "error": {"message": "boom"},
                },  # errored -> retry
                # reqs[2] missing entirely -> retry
            ],
        )
        remaining = remaining_manifest(paths.manifest, paths.sidecar, out)
        remaining_ids = [line["custom_id"] for line in remaining]
        assert remaining_ids == [reqs[1].custom_id, reqs[2].custom_id]
        # lines are runnable batch lines
        assert all(line["method"] == "POST" for line in remaining)
        assert all("body" in line for line in remaining)

    def test_markerless_but_parsed_output_counts_completed(self, tmp_path: Path) -> None:
        (req,) = [self._three_requests()[0]]
        paths = emit_manifest([req], SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [chat_output_line(req.custom_id, "no marker here at all")],
        )
        assert remaining_manifest(paths.manifest, paths.sidecar, out) == []

    def test_no_outputs_means_everything_remains(self, tmp_path: Path) -> None:
        reqs = self._three_requests()
        paths = emit_manifest(reqs, SPEC, tmp_path, "m")
        remaining = remaining_manifest(paths.manifest, paths.sidecar, [])
        assert len(remaining) == 3

    def test_truncated_results_file_resumes_cleanly(self, tmp_path: Path) -> None:
        """The normal artifact of an interrupted run must not traceback."""
        reqs = self._three_requests()
        paths = emit_manifest(reqs, SPEC, tmp_path, "m")
        out = write_jsonl(
            tmp_path / "r.jsonl",
            [
                chat_output_line(reqs[0].custom_id, "done [[A]]"),
                chat_output_line(reqs[1].custom_id, "done [[B]]"),
            ],
        )
        raw = out.read_bytes()
        out.write_bytes(raw[:-40])  # interrupt mid-write of the second line
        with pytest.warns(ManifestWarning, match="malformed"):
            remaining = remaining_manifest(paths.manifest, paths.sidecar, out)
        # the truncated line's request stays in the resume set
        assert [line["custom_id"] for line in remaining] == [
            reqs[1].custom_id,
            reqs[2].custom_id,
        ]

    def test_manifest_id_missing_from_sidecar_warns(self, tmp_path: Path) -> None:
        reqs = self._three_requests()
        paths = emit_manifest(reqs, SPEC, tmp_path, "m")
        # sidecar from a different (subset) emission
        other = emit_manifest(reqs[:1], SPEC, tmp_path, "other")
        with pytest.warns(ManifestWarning, match="missing from sidecar"):
            remaining = remaining_manifest(paths.manifest, other.sidecar, [])
        assert len(remaining) == 3


# ---------------------------------------------------------------------------
# load_sidecar
# ---------------------------------------------------------------------------


class TestLoadSidecar:
    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        p = write_jsonl(tmp_path / "s.meta.jsonl", [{"custom_id": "x"}])
        with pytest.raises(ValueError, match="malformed"):
            load_sidecar(p)

    def test_duplicate_custom_id_raises(self, tmp_path: Path) -> None:
        line = {"custom_id": "x", "body_hash": "h", "usages": [make_meta()]}
        p = write_jsonl(tmp_path / "s.meta.jsonl", [line, line])
        with pytest.raises(ValueError, match="duplicate"):
            load_sidecar(p)

    def test_preserves_order(self, tmp_path: Path) -> None:
        reqs = [
            make_request({"messages": [{"role": "user", "content": f"i{i}"}]})
            for i in range(5)
        ]
        paths = emit_manifest(reqs, SPEC, tmp_path, "m")
        assert list(load_sidecar(paths.sidecar)) == [r.custom_id for r in reqs]
