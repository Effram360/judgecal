"""Tests for judgecal.fixtures.synthetic — deterministic item generation."""

from __future__ import annotations

import math

import pytest

from judgecal.core import PairwiseItem
from judgecal.fixtures import TIE_GAP, SyntheticConfig, generate_items


def _fingerprint(items: list[PairwiseItem]) -> list[tuple]:
    return [
        (
            it.item_id,
            it.prompt,
            it.response_a,
            it.response_b,
            it.label,
            it.author_a,
            it.author_b,
            it.meta["latent_quality_a"],
            it.meta["latent_quality_b"],
        )
        for it in items
    ]


class TestDeterminism:
    def test_same_config_identical_items(self) -> None:
        cfg = SyntheticConfig(n_items=50, seed=42)
        assert _fingerprint(generate_items(cfg)) == _fingerprint(generate_items(cfg))

    def test_different_seed_differs(self) -> None:
        a = generate_items(SyntheticConfig(n_items=50, seed=1))
        b = generate_items(SyntheticConfig(n_items=50, seed=2))
        assert _fingerprint(a) != _fingerprint(b)


class TestShape:
    def test_count_types_and_ids(self) -> None:
        cfg = SyntheticConfig(n_items=120, seed=7)
        items = generate_items(cfg)
        assert len(items) == 120
        assert all(isinstance(it, PairwiseItem) for it in items)
        ids = [it.item_id for it in items]
        assert len(set(ids)) == 120
        assert all(i.startswith("synthetic:7:") for i in ids)
        assert all(it.source == "synthetic" for it in items)
        assert all(it.prompt for it in items)
        assert all(it.response_a and it.response_b for it in items)

    def test_no_marker_collision_in_bodies(self) -> None:
        items = generate_items(SyntheticConfig(n_items=40, seed=3))
        for it in items:
            assert "[[" not in it.response_a
            assert "[[" not in it.response_b


class TestQualities:
    def test_qualities_in_unit_interval_and_in_meta(self) -> None:
        items = generate_items(SyntheticConfig(n_items=200, seed=11))
        for it in items:
            qa = it.meta["latent_quality_a"]
            qb = it.meta["latent_quality_b"]
            assert 0.0 <= qa <= 1.0
            assert 0.0 <= qb <= 1.0

    def test_label_matches_tie_band_rule(self) -> None:
        items = generate_items(SyntheticConfig(n_items=300, seed=5))
        for it in items:
            gap = it.meta["latent_quality_a"] - it.meta["latent_quality_b"]
            if abs(gap) < TIE_GAP:
                assert it.label == "tie"
            elif gap > 0:
                assert it.label == "A"
            else:
                assert it.label == "B"

    def test_tie_fraction_at_least_requested(self) -> None:
        cfg = SyntheticConfig(n_items=400, seed=9, tie_fraction=0.2)
        items = generate_items(cfg)
        n_tie = sum(1 for it in items if it.label == "tie")
        # 80 forced ties; normal-gap items may add a few accidental ones.
        assert n_tie >= 80

    def test_zero_tie_fraction_forces_none(self) -> None:
        # No forced ties; accidental ones still possible (|N(0,1)| < 0.1
        # has ~8% mass), so only check the forced mechanism is off.
        items = generate_items(SyntheticConfig(n_items=200, seed=13, tie_fraction=0.0))
        n_tie = sum(1 for it in items if it.label == "tie")
        assert n_tie < 50  # would be >= ~190 if everything were forced tie


class TestLengths:
    def test_lengths_hit_targets_exactly(self) -> None:
        items = generate_items(SyntheticConfig(n_items=150, seed=21))
        for it in items:
            assert len(it.response_a) == it.meta["target_len_a"]
            assert len(it.response_b) == it.meta["target_len_b"]

    def test_lengths_vary_with_log_sd(self) -> None:
        items = generate_items(SyntheticConfig(n_items=400, seed=23, length_log_sd=0.4))
        logs = [math.log(len(it.response_a)) for it in items]
        mean = sum(logs) / len(logs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
        assert 0.25 < sd < 0.55  # loose band around the configured 0.4

    def test_min_length_respected(self) -> None:
        items = generate_items(SyntheticConfig(n_items=200, seed=29, length_log_sd=1.5))
        for it in items:
            assert len(it.response_a) >= 60
            assert len(it.response_b) >= 60


class TestAuthors:
    def test_self_fraction_exact_one_side(self) -> None:
        cfg = SyntheticConfig(n_items=200, seed=31, self_author_fraction=0.5)
        items = generate_items(cfg)
        n_self = 0
        for it in items:
            a_self = it.author_a == "judge-self"
            b_self = it.author_b == "judge-self"
            assert not (a_self and b_self)  # never both sides
            if a_self or b_self:
                n_self += 1
        assert n_self == 100

    def test_zero_self_fraction(self) -> None:
        items = generate_items(SyntheticConfig(n_items=80, seed=37, self_author_fraction=0.0))
        for it in items:
            assert it.author_a != "judge-self"
            assert it.author_b != "judge-self"

    def test_full_self_fraction(self) -> None:
        items = generate_items(SyntheticConfig(n_items=80, seed=41, self_author_fraction=1.0))
        for it in items:
            assert (it.author_a == "judge-self") ^ (it.author_b == "judge-self")

    def test_both_self_sides_drawn_from_pool(self) -> None:
        cfg = SyntheticConfig(
            n_items=100, seed=43, authors=("me", "x", "y"), self_author_fraction=0.3
        )
        items = generate_items(cfg)
        pool = {"me", "x", "y"}
        for it in items:
            assert it.author_a in pool
            assert it.author_b in pool


class TestValidation:
    def test_bad_n_items(self) -> None:
        with pytest.raises(ValueError, match="n_items"):
            SyntheticConfig(n_items=0, seed=0)

    def test_bad_tie_fraction(self) -> None:
        with pytest.raises(ValueError, match="tie_fraction"):
            SyntheticConfig(n_items=10, seed=0, tie_fraction=1.5)

    def test_bad_self_fraction(self) -> None:
        with pytest.raises(ValueError, match="self_author_fraction"):
            SyntheticConfig(n_items=10, seed=0, self_author_fraction=-0.1)

    def test_empty_authors(self) -> None:
        with pytest.raises(ValueError, match="authors"):
            SyntheticConfig(n_items=10, seed=0, authors=())
