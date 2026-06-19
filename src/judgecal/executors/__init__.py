"""Executors: mock judge, fixture replay, Claude Code CLI, SLURM packs.

Public API (contracts §4.2–4.3): :class:`Executor` protocol,
:func:`parse_verdict` / :func:`compare_scores`, :class:`MockJudgeExecutor`,
:class:`FixtureExecutor`, :class:`ClaudeCodeExecutor`, and the SLURM pack
generators (:class:`SlurmConfig`, :func:`write_slurm_pack`).
"""

from judgecal.executors.base import (
    Executor,
    ExecutorWarning,
    invalid_judgment,
    judgment_from_raw,
)
from judgecal.executors.claude_code import ClaudeCodeExecutor
from judgecal.executors.local import FixtureExecutor, MockJudgeExecutor
from judgecal.executors.parsing import (
    DEFAULT_SCORE_EPSILON,
    DEFAULT_VERDICT_PATTERN,
    compare_scores,
    parse_verdict,
)
from judgecal.executors.slurm import (
    VLLM_MODULE_COMMAND,
    VLLM_SUBCOMMAND,
    SlurmConfig,
    SlurmPackPaths,
    write_slurm_pack,
)

__all__ = [
    "DEFAULT_SCORE_EPSILON",
    "DEFAULT_VERDICT_PATTERN",
    "VLLM_MODULE_COMMAND",
    "VLLM_SUBCOMMAND",
    "ClaudeCodeExecutor",
    "Executor",
    "ExecutorWarning",
    "FixtureExecutor",
    "MockJudgeExecutor",
    "SlurmConfig",
    "SlurmPackPaths",
    "compare_scores",
    "invalid_judgment",
    "judgment_from_raw",
    "parse_verdict",
    "write_slurm_pack",
]
