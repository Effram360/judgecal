"""Tests for judgecal.fixtures.mock_judge and judgecal.fixtures.packs.

The property tests prove the CRITICAL contract: empirical rates realized
by the verdict generator match the analytic ``expected_*`` truths within
3 standard errors (the validation suite depends on this consistency).
"""

from __future__ import annotations

import math
import re

import pytest

from judgecal.core import JudgmentRequest, PairwiseItem, make_custom_id
from judgecal.fixtures import (
    MockJudge,
    MockJudgeConfig,
    ResponsePack,
    SyntheticConfig,
    compute_logit,
    expected_first_pick_rate,
    expected_pad_pick_rate,
    expected_self_error_pick_excess,
    generate_items,
    load_pack,
    pack_from_batch_output,
    request_logits,
    save_pack,
)

# ---------------------------------------------------------------------------
# Helpers (local mini-planner + mini-parser; probes/executors are other
# agents' modules and may not exist yet)
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[\[\s*([ABC])\s*\]\]", re.IGNORECASE)
_MARKER_TO_VERDICT = {"A": "first", "B": "second", "C": "tie"}


def _parse(raw: str) -> str:
    found = _MARKER_RE.findall(raw)
    return _MARKER_TO_VERDICT[found[-1].upper()] if found else "invalid"


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _request_from_item(
    item: PairwiseItem,
    *,
    condition: str,
    first_is_a: bool = True,
    repeat: int = 0,
    probe: str = "position",
) -> JudgmentRequest:
    first_text = item.response_a if first_is_a else item.response_b
    second_text = item.response_b if first_is_a else item.response_a
    if item.label is None:
        label_first = None
    elif item.label == "tie":
        label_first = "tie"
    elif (item.label == "A") == first_is_a:
        label_first = "first"
    else:
        label_first = "second"
    qa = item.meta.get("latent_quality_a")
    qb = item.meta.get("latent_quality_b")
    meta = {
        "probe": probe,
        "condition": condition,
        "item_id": item.item_id,
        "repeat": repeat,
        "first_is_a": first_is_a,
        "first_len": len(first_text),
        "second_len": len(second_text),
        "first_author": item.author_a if first_is_a else item.author_b,
        "second_author": item.author_b if first_is_a else item.author_a,
        "first_latent_q": qa if first_is_a else qb,
        "second_latent_q": qb if first_is_a else qa,
        "label_first": label_first,
    }
    body = {
        "messages": [
            {
                "role": "user",
                "content": f"{item.prompt}\n\nFIRST:\n{first_text}\n\nSECOND:\n{second_text}",
            }
        ]
    }
    return JudgmentRequest(custom_id=make_custom_id(body, repeat), body=body, meta=meta)


def _position_requests(items: list[PairwiseItem]) -> list[JudgmentRequest]:
    reqs: list[JudgmentRequest] = []
    for item in items:
        reqs.append(_request_from_item(item, condition="orig", first_is_a=True))
        reqs.append(_request_from_item(item, condition="swap", first_is_a=False))
    return reqs


def _pad(text: str, ratio: float = 1.6) -> str:
    target = int(len(text) * ratio)
    padded = text
    while len(padded) < target:
        padded += " " + text
    return padded[:target]


def _pad_requests(items: list[PairwiseItem]) -> list[JudgmentRequest]:
    """Verbosity-style constructed contrast: response_a vs padded(response_a)."""
    reqs: list[JudgmentRequest] = []
    for item in items:
        orig = item.response_a
        padded = _pad(orig)
        qa = item.meta.get("latent_quality_a")
        for condition, first_text, second_text in (
            ("pad_second", orig, padded),
            ("pad_first", padded, orig),
        ):
            meta = {
                "probe": "verbosity",
                "condition": condition,
                "item_id": item.item_id,
                "repeat": 0,
                "first_is_a": True,
                "first_len": len(first_text),
                "second_len": len(second_text),
                "first_author": item.author_a,
                "second_author": item.author_a,
                "first_latent_q": qa,
                "second_latent_q": qa,  # same content => same quality
                "label_first": None,
            }
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{item.prompt}\n\nFIRST:\n{first_text}\n\nSECOND:\n{second_text}"
                        ),
                    }
                ]
            }
            reqs.append(JudgmentRequest(custom_id=make_custom_id(body, 0), body=body, meta=meta))
    return reqs


def _simple_request(
    *,
    q_first: float | None = 0.5,
    q_second: float | None = 0.5,
    first_len: int = 400,
    second_len: int = 400,
    first_author: str | None = None,
    second_author: str | None = None,
    label_first: str | None = None,
    condition: str = "orig",
    repeat: int = 0,
    item_id: str = "it-0",
    tag: str = "x",
) -> JudgmentRequest:
    meta = {
        "probe": "position",
        "condition": condition,
        "item_id": item_id,
        "repeat": repeat,
        "first_is_a": True,
        "first_len": first_len,
        "second_len": second_len,
        "first_author": first_author,
        "second_author": second_author,
        "first_latent_q": q_first,
        "second_latent_q": q_second,
        "label_first": label_first,
    }
    body = {"messages": [{"role": "user", "content": f"{item_id}:{condition}:{tag}"}]}
    return JudgmentRequest(custom_id=make_custom_id(body, repeat), body=body, meta=meta)


# ---------------------------------------------------------------------------
# Determinism and raw-text format
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_across_calls_and_instances(self) -> None:
        cfg = MockJudgeConfig(seed=5, beta_position=0.4, noise_sigma=0.5, template_sigma=0.3)
        items = generate_items(SyntheticConfig(n_items=20, seed=1))
        reqs = _position_requests(items)
        out1 = [MockJudge(cfg).judge(r) for r in reqs]
        out2 = [MockJudge(cfg).judge(r) for r in reqs]
        assert out1 == out2

    def test_seed_changes_output(self) -> None:
        items = generate_items(SyntheticConfig(n_items=40, seed=1))
        reqs = _position_requests(items)
        a = [MockJudge(MockJudgeConfig(seed=1)).judge(r) for r in reqs]
        b = [MockJudge(MockJudgeConfig(seed=2)).judge(r) for r in reqs]
        assert a != b


class TestRawTextFormat:
    def test_contains_exactly_one_marker_and_filler(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_position=1.0)
        judge = MockJudge(cfg)
        items = generate_items(SyntheticConfig(n_items=30, seed=2))
        for r in _position_requests(items):
            raw = judge.judge(r)
            markers = _MARKER_RE.findall(raw)
            assert len(markers) == 1
            assert markers[0] in ("A", "B", "C")
            assert len(raw) > 40  # reasoning filler present, not bare marker

    def test_parsed_verdict_matches_decide(self) -> None:
        cfg = MockJudgeConfig(seed=3, beta_position=0.5)
        judge = MockJudge(cfg)
        items = generate_items(SyntheticConfig(n_items=30, seed=4))
        for r in _position_requests(items):
            assert _parse(judge.judge(r)) == judge.decide(r)


# ---------------------------------------------------------------------------
# Logit model mechanics
# ---------------------------------------------------------------------------


class TestLogitModel:
    def test_pure_position_logit(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_quality=3.0, beta_position=0.8)
        r = _simple_request()  # equal qualities, equal lengths, no authors
        assert compute_logit(cfg, r) == pytest.approx(0.8)

    def test_quality_term(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_quality=3.0)
        r = _simple_request(q_first=0.9, q_second=0.3)
        assert compute_logit(cfg, r) == pytest.approx(3.0 * 0.6)

    def test_length_term(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_length=1.0)
        r = _simple_request(first_len=640, second_len=400)
        assert compute_logit(cfg, r) == pytest.approx(math.log(1.6))

    def test_self_term_antisymmetric(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_self=1.0)
        r1 = _simple_request(first_author="judge-self", second_author="other")
        r2 = _simple_request(first_author="other", second_author="judge-self")
        assert compute_logit(cfg, r1) == pytest.approx(1.0)
        assert compute_logit(cfg, r2) == pytest.approx(-1.0)

    def test_q_fallback_from_label(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_quality=3.0)
        winner_first = _simple_request(q_first=None, q_second=None, label_first="first")
        winner_second = _simple_request(q_first=None, q_second=None, label_first="second")
        tie = _simple_request(q_first=None, q_second=None, label_first="tie")
        unknown = _simple_request(q_first=None, q_second=None, label_first=None)
        assert compute_logit(cfg, winner_first) == pytest.approx(3.0 * 0.6)
        assert compute_logit(cfg, winner_second) == pytest.approx(-3.0 * 0.6)
        assert compute_logit(cfg, tie) == pytest.approx(0.0)
        assert compute_logit(cfg, unknown) == pytest.approx(0.0)

    def test_template_offset_only_for_tpl_conditions(self) -> None:
        cfg = MockJudgeConfig(seed=0, template_sigma=0.7)
        base = compute_logit(cfg, _simple_request(condition="orig"))
        v1 = compute_logit(cfg, _simple_request(condition="tpl:v1"))
        v2 = compute_logit(cfg, _simple_request(condition="tpl:v2"))
        assert base == pytest.approx(0.0)
        assert v1 != 0.0
        assert v2 != 0.0
        assert v1 != v2
        # offset is fixed per (seed, condition)
        assert v1 == compute_logit(cfg, _simple_request(condition="tpl:v1", item_id="it-9"))

    def test_noise_keyed_by_item_condition_repeat(self) -> None:
        cfg = MockJudgeConfig(seed=0, noise_sigma=1.0)
        r0 = _simple_request(condition="rep", repeat=0)
        r1 = _simple_request(condition="rep", repeat=1)
        other_item = _simple_request(condition="rep", repeat=0, item_id="it-1")
        assert compute_logit(cfg, r0) != compute_logit(cfg, r1)
        assert compute_logit(cfg, r0) != compute_logit(cfg, other_item)
        # reproducible
        assert compute_logit(cfg, r0) == compute_logit(cfg, r0)

    def test_logit_first_method_and_request_logits(self) -> None:
        cfg = MockJudgeConfig(seed=2, beta_position=0.3, noise_sigma=0.4)
        items = generate_items(SyntheticConfig(n_items=10, seed=6))
        reqs = _position_requests(items)
        judge = MockJudge(cfg)
        assert request_logits(cfg, reqs) == [judge.logit_first(r) for r in reqs]

    def test_zero_length_guarded(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_length=1.0)
        r = _simple_request(first_len=0, second_len=400)
        assert math.isfinite(compute_logit(cfg, r))


# ---------------------------------------------------------------------------
# Verdict mechanics: tie band, repeats, invalid channel
# ---------------------------------------------------------------------------


class TestVerdictMechanics:
    def test_tie_band(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_position=0.2, tie_band=0.25)
        judge = MockJudge(cfg)
        r = _simple_request()  # logit = 0.2 < 0.25
        assert judge.decide(r) == "tie"
        assert _parse(judge.judge(r)) == "tie"

    def test_decisive_outside_band(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_position=5.0, tie_band=0.25)
        judge = MockJudge(cfg)
        # sigmoid(5) ~ 0.993: across many requests nearly all picks = first
        items = generate_items(SyntheticConfig(n_items=100, seed=8, quality_gap_sd=0.0))
        verdicts = [judge.decide(r) for r in _position_requests(items)]
        assert "tie" not in verdicts
        assert verdicts.count("first") / len(verdicts) > 0.9

    def test_zero_noise_repeats_identical(self) -> None:
        """noise_sigma=0 => byte-identical output across repeats (unanimity == 1)."""
        cfg = MockJudgeConfig(seed=0, beta_position=0.3, noise_sigma=0.0)
        judge = MockJudge(cfg)
        items = generate_items(SyntheticConfig(n_items=50, seed=10))
        for item in items:
            outs = {
                judge.judge(
                    _request_from_item(item, condition="rep", repeat=k, probe="stability")
                )
                for k in range(5)
            }
            assert len(outs) == 1

    def test_noise_creates_repeat_instability(self) -> None:
        cfg = MockJudgeConfig(seed=0, noise_sigma=2.0, tie_band=0.0)
        judge = MockJudge(cfg)
        items = generate_items(SyntheticConfig(n_items=60, seed=12, quality_gap_sd=0.0))
        n_nonunanimous = 0
        for item in items:
            verdicts = {
                judge.decide(_request_from_item(item, condition="rep", repeat=k, probe="stability"))
                for k in range(5)
            }
            if len(verdicts) > 1:
                n_nonunanimous += 1
        assert n_nonunanimous > 10  # far from perfectly stable

    def test_invalid_rate_zero_and_one(self) -> None:
        items = generate_items(SyntheticConfig(n_items=40, seed=14))
        reqs = _position_requests(items)
        clean = MockJudge(MockJudgeConfig(seed=0, invalid_rate=0.0))
        broken = MockJudge(MockJudgeConfig(seed=0, invalid_rate=1.0))
        assert all(_parse(clean.judge(r)) != "invalid" for r in reqs)
        assert all(_parse(broken.judge(r)) == "invalid" for r in reqs)

    def test_invalid_rate_calibrated(self) -> None:
        items = generate_items(SyntheticConfig(n_items=300, seed=16))
        reqs = _position_requests(items)  # 600 requests
        judge = MockJudge(MockJudgeConfig(seed=1, invalid_rate=0.3))
        rate = sum(_parse(judge.judge(r)) == "invalid" for r in reqs) / len(reqs)
        se = math.sqrt(0.3 * 0.7 / len(reqs))
        assert abs(rate - 0.3) <= 3 * se

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="invalid_rate"):
            MockJudgeConfig(invalid_rate=1.5)
        with pytest.raises(ValueError, match="tie_band"):
            MockJudgeConfig(tie_band=-0.1)
        with pytest.raises(ValueError, match="noise_sigma"):
            MockJudgeConfig(noise_sigma=-1.0)


# ---------------------------------------------------------------------------
# CRITICAL property tests: empirical rates vs analytic truths
# ---------------------------------------------------------------------------


class TestAnalyticConsistency:
    def test_first_pick_rate_matches_analytic(self) -> None:
        """Empirical first-pick rate ~ analytic expectation within 3 SE."""
        items = generate_items(SyntheticConfig(n_items=400, seed=11))
        reqs = _position_requests(items)
        cfg = MockJudgeConfig(seed=3, beta_quality=3.0, beta_position=0.8)
        judge = MockJudge(cfg)

        logits = request_logits(cfg, reqs)
        decisive_probs = [_sigmoid(lg) for lg in logits if abs(lg) >= cfg.tie_band]
        analytic = expected_first_pick_rate(cfg, reqs)
        assert analytic == pytest.approx(sum(decisive_probs) / len(decisive_probs))

        verdicts = [_parse(judge.judge(r)) for r in reqs]
        decisive = [v for v in verdicts if v in ("first", "second")]
        # tie sets must coincide exactly (deterministic tie band)
        assert len(decisive) == len(decisive_probs)
        empirical = decisive.count("first") / len(decisive)

        se = math.sqrt(sum(p * (1 - p) for p in decisive_probs)) / len(decisive_probs)
        assert se > 0
        assert abs(empirical - analytic) <= 3 * se

    def test_first_pick_rate_null_judge_exactly_half(self) -> None:
        # Nonzero quality gaps so decisive judgments exist (zero gaps put
        # every logit inside the tie band -> no decisive judgments -> nan).
        # With orig+swap symmetry and no planted biases, the quality terms
        # are equal-and-opposite across passes and sigmoid(x)+sigmoid(-x)=1,
        # so the decisive-conditional first-pick expectation is exactly 0.5.
        items = generate_items(SyntheticConfig(n_items=400, seed=17, quality_gap_sd=1.0))
        reqs = _position_requests(items)
        cfg = MockJudgeConfig(seed=5, beta_quality=3.0)  # no planted biases
        analytic = expected_first_pick_rate(cfg, reqs)
        assert analytic == pytest.approx(0.5, abs=1e-9)

    def test_pad_pick_rate_matches_analytic(self) -> None:
        items = generate_items(SyntheticConfig(n_items=300, seed=19))
        reqs = _pad_requests(items)
        cfg = MockJudgeConfig(seed=7, beta_quality=3.0, beta_length=1.0)
        judge = MockJudge(cfg)

        analytic = expected_pad_pick_rate(cfg, reqs)
        assert 0.55 < analytic < 0.75  # planted verbosity bias is detectable

        picked_padded: list[bool] = []
        probs: list[float] = []
        for r in reqs:
            lg = compute_logit(cfg, r)
            if abs(lg) < cfg.tie_band:
                continue
            v = _parse(judge.judge(r))
            assert v in ("first", "second")
            padded_first = r.meta["condition"] == "pad_first"
            picked_padded.append((v == "first") == padded_first)
            p_first = _sigmoid(lg)
            probs.append(p_first if padded_first else 1.0 - p_first)

        empirical = sum(picked_padded) / len(picked_padded)
        se = math.sqrt(sum(p * (1 - p) for p in probs)) / len(probs)
        assert abs(empirical - analytic) <= 3 * se

    def test_self_excess_matches_analytic(self) -> None:
        items = generate_items(SyntheticConfig(n_items=400, seed=23))
        reqs = _position_requests(items)
        cfg = MockJudgeConfig(seed=9, beta_quality=3.0, beta_self=1.0)
        judge = MockJudge(cfg)

        analytic = expected_self_error_pick_excess(cfg, reqs)
        assert math.isfinite(analytic)
        assert analytic > 0.05  # planted self-preference shows up

        treat: list[bool] = []
        treat_p: list[float] = []
        ctrl: list[bool] = []
        ctrl_p: list[float] = []
        for r in reqs:
            meta = r.meta
            label_first = meta["label_first"]
            if label_first not in ("first", "second"):
                continue
            lg = compute_logit(cfg, r)
            if abs(lg) < cfg.tie_band:
                continue
            v = _parse(judge.judge(r))
            assert v in ("first", "second")
            p_first = _sigmoid(lg)
            self_first = meta["first_author"] == cfg.self_name
            self_second = meta["second_author"] == cfg.self_name
            if self_first ^ self_second:
                other = "second" if self_first else "first"
                if label_first == other:
                    treat.append((v == "first") == self_first)
                    treat_p.append(p_first if self_first else 1.0 - p_first)
            elif not self_first and not self_second:
                ctrl.append(v != label_first)
                ctrl_p.append(1.0 - p_first if label_first == "first" else p_first)

        assert len(treat) > 30
        assert len(ctrl) > 30
        empirical = sum(treat) / len(treat) - sum(ctrl) / len(ctrl)
        se_t = math.sqrt(sum(p * (1 - p) for p in treat_p)) / len(treat_p)
        se_c = math.sqrt(sum(p * (1 - p) for p in ctrl_p)) / len(ctrl_p)
        se = math.sqrt(se_t**2 + se_c**2)
        assert abs(empirical - analytic) <= 3 * se

    def test_expected_rates_nan_when_no_decisive(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_quality=0.0, tie_band=0.25)
        reqs = [_simple_request(item_id=f"it-{i}", tag=str(i)) for i in range(10)]  # all logit 0
        assert math.isnan(expected_first_pick_rate(cfg, reqs))
        assert math.isnan(expected_pad_pick_rate(cfg, reqs, conditions=None))

    def test_self_excess_nan_without_authors(self) -> None:
        cfg = MockJudgeConfig(seed=0, beta_self=1.0)
        items = generate_items(SyntheticConfig(n_items=20, seed=25, self_author_fraction=1.0))
        # every item has a self side => control set empty => nan
        reqs = _position_requests(items)
        assert math.isnan(expected_self_error_pick_excess(cfg, reqs))

    def test_condition_filter_defaults(self) -> None:
        """Pad requests are invisible to the first-pick truth and vice versa."""
        items = generate_items(SyntheticConfig(n_items=50, seed=27))
        cfg = MockJudgeConfig(seed=1, beta_position=0.8, beta_length=1.0)
        pos = _position_requests(items)
        pad = _pad_requests(items)
        mixed = pos + pad
        assert expected_first_pick_rate(cfg, mixed) == pytest.approx(
            expected_first_pick_rate(cfg, pos)
        )
        assert expected_pad_pick_rate(cfg, mixed) == pytest.approx(
            expected_pad_pick_rate(cfg, pad)
        )


# ---------------------------------------------------------------------------
# Response packs
# ---------------------------------------------------------------------------


class TestResponsePack:
    def test_round_trip(self, tmp_path) -> None:
        pack = ResponsePack(
            responses={"jc-aaa-r0": "filler [[A]]", "jc-bbb-r0": "filler [[C]]"},
            judge="mock-judge",
            created="2026-06-10T00:00:00Z",
        )
        path = tmp_path / "pack.jsonl"
        save_pack(pack, path)
        loaded = load_pack(path)
        assert loaded.responses == pack.responses
        assert loaded.judge == "mock-judge"
        assert loaded.created == "2026-06-10T00:00:00Z"
        assert loaded.pack_version == 1
        assert len(loaded) == 2
        assert "jc-aaa-r0" in loaded
        assert loaded.get("jc-zzz-r0") is None

    def test_save_is_deterministic(self, tmp_path) -> None:
        pack = ResponsePack(responses={"b": "2", "a": "1"}, judge="j")
        p1, p2 = tmp_path / "p1.jsonl", tmp_path / "p2.jsonl"
        save_pack(pack, p1)
        save_pack(pack, p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_header_first_line(self, tmp_path) -> None:
        path = tmp_path / "pack.jsonl"
        save_pack(ResponsePack(responses={"x": "y"}, judge="j"), path)
        first = path.read_text().splitlines()[0]
        assert '"pack_version": 1' in first

    def test_load_rejects_missing_header(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"custom_id": "x", "raw_text": "y"}\n')
        with pytest.raises(ValueError, match="pack_version"):
            load_pack(path)

    def test_load_rejects_wrong_version(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"pack_version": 99, "judge": "j", "created": ""}\n')
        with pytest.raises(ValueError, match="unsupported pack_version"):
            load_pack(path)

    def test_load_rejects_empty_file(self, tmp_path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_pack(path)

    def test_pack_from_batch_output(self, tmp_path) -> None:
        lines = [
            # chat-completion line
            '{"custom_id": "jc-1-r0", "response": {"body": {"choices": '
            '[{"message": {"content": "reasoning [[B]]"}}]}}, "error": null}',
            # score line (vLLM /v1/score variant)
            '{"custom_id": "jc-2-r0", "response": {"body": {"score": 0.91}}}',
            # data[0].score variant
            '{"custom_id": "jc-3-r0", "response": {"body": {"data": [{"score": 0.2}]}}}',
            # errored line: skipped
            '{"custom_id": "jc-4-r0", "response": null, "error": {"message": "boom"}}',
            # junk: skipped
            "not json at all",
            '{"no_custom_id": true}',
        ]
        path = tmp_path / "results.jsonl"
        path.write_text("\n".join(lines) + "\n")
        pack = pack_from_batch_output(path, judge="real-judge", created="2026-06-10")
        assert pack.judge == "real-judge"
        assert pack.responses == {
            "jc-1-r0": "reasoning [[B]]",
            "jc-2-r0": "0.91",
            "jc-3-r0": "0.2",
        }

    def test_batch_output_round_trips_through_save(self, tmp_path) -> None:
        src = tmp_path / "results.jsonl"
        src.write_text(
            '{"custom_id": "jc-1-r0", "response": {"body": {"choices": '
            '[{"message": {"content": "ok [[A]]"}}]}}}\n'
        )
        pack = pack_from_batch_output(src, judge="j")
        dest = tmp_path / "pack.jsonl"
        save_pack(pack, dest)
        assert load_pack(dest).responses == {"jc-1-r0": "ok [[A]]"}

    def test_mock_judge_outputs_replayable(self, tmp_path) -> None:
        """End-to-end: record mock outputs into a pack and reload them."""
        items = generate_items(SyntheticConfig(n_items=10, seed=29))
        reqs = _position_requests(items)
        judge = MockJudge(MockJudgeConfig(seed=0, beta_position=0.8))
        pack = ResponsePack(
            responses={r.custom_id: judge.judge(r) for r in reqs}, judge="mock"
        )
        path = tmp_path / "mock.jsonl"
        save_pack(pack, path)
        loaded = load_pack(path)
        for r in reqs:
            assert _parse(loaded.responses[r.custom_id]) == judge.decide(r)
