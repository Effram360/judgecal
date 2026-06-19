"""judgecal.fixtures — synthetic items, the deterministic mock judge, and
recorded response packs (the zero-LLM dev loop).

Public API (contract §3): everything re-exported below.
"""

from __future__ import annotations

from judgecal.fixtures.mock_judge import (
    MockJudge,
    MockJudgeConfig,
    compute_logit,
    expected_first_pick_rate,
    expected_pad_pick_rate,
    expected_self_error_pick_excess,
    request_logits,
)
from judgecal.fixtures.packs import (
    PACK_VERSION,
    ResponsePack,
    load_pack,
    pack_from_batch_output,
    save_pack,
)
from judgecal.fixtures.synthetic import TIE_GAP, SyntheticConfig, generate_items

__all__ = [
    "PACK_VERSION",
    "TIE_GAP",
    "MockJudge",
    "MockJudgeConfig",
    "ResponsePack",
    "SyntheticConfig",
    "compute_logit",
    "expected_first_pick_rate",
    "expected_pad_pick_rate",
    "expected_self_error_pick_excess",
    "generate_items",
    "load_pack",
    "pack_from_batch_output",
    "request_logits",
    "save_pack",
]
