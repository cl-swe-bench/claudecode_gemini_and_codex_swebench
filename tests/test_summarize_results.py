"""Tests for ``ClaudeCodeInterface._summarize_results`` — the aggregator
that turns one-or-many ``type=result`` events into a single canonical
summary dict.

Background: CC may emit multiple result events per CLI invocation
(observed on a real prod run, root cause appears to be context
compaction / session-resume; treated as model-agnostic). The original
parser kept only the last event, dropping per-batch fields from prior
batches and producing dramatically wrong durations + token counts +
costs. See ``cl-benchmark/docs/cost-accounting-fix-spec.md``.

The fixture fixture-cc47-multi-result.jsonl is a 3-result NDJSON
distilled from the real ``instance_protonmail__webclients-09fcf0db…``
log; preserves the per-batch / cumulative semantics so the test
exercises actual CC output shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.claude_interface import ClaudeCodeInterface  # noqa: E402


def _iface(output_format: str = "stream-json") -> ClaudeCodeInterface:
    """Return a barely-initialized interface — only ``output_format``
    is read by ``_summarize_results`` / ``_collect_result_events``."""
    iface = ClaudeCodeInterface.__new__(ClaudeCodeInterface)
    iface.output_format = output_format
    return iface


# ── stream-json: multi-result aggregation (the bug case) ────────────


_MULTI_RESULT_FIXTURE = [
    # result[0] — primary agent loop, 49 turns over 6m 15s
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 375104,
        "duration_api_ms": 193084,
        "num_turns": 49,
        "session_id": "5f0190ca-e0f1-49fb-9f29-77dfe79e3055",
        "total_cost_usd": 2.0333385,
        "usage": {
            "input_tokens": 53,
            "output_tokens": 12305,
            "cache_creation_input_tokens": 74056,
            "cache_read_input_tokens": 2522921,
        },
        "modelUsage": {
            "claude-opus-4-7": {
                "inputTokens": 53,
                "outputTokens": 12305,
                "cacheCreationInputTokens": 74056,
                "cacheReadInputTokens": 2522921,
                "costUSD": 2.0322005,
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 1043,
                "outputTokens": 19,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": 0.001138,
            },
        },
    },
    # result[1] — small follow-up batch
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 2954,
        "duration_api_ms": 196028,
        "num_turns": 1,
        "session_id": "5f0190ca-e0f1-49fb-9f29-77dfe79e3055",
        "total_cost_usd": 2.079248,
        "usage": {
            "input_tokens": 6,
            "output_tokens": 111,
            "cache_creation_input_tokens": 642,
            "cache_read_input_tokens": 78184,
        },
        "modelUsage": {
            "claude-opus-4-7": {
                "inputTokens": 59,
                "outputTokens": 12416,
                "cacheCreationInputTokens": 74698,
                "cacheReadInputTokens": 2601105,
                "costUSD": 2.07811,
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 1043,
                "outputTokens": 19,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": 0.001138,
            },
        },
    },
    # result[2] — final small batch (the one the old parser captured
    # exclusively; values here look "tiny" in isolation)
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1586,
        "duration_api_ms": 197606,
        "num_turns": 1,
        "session_id": "5f0190ca-e0f1-49fb-9f29-77dfe79e3055",
        "total_cost_usd": 2.1217347500000003,
        "usage": {
            "input_tokens": 6,
            "output_tokens": 35,
            "cache_creation_input_tokens": 347,
            "cache_read_input_tokens": 78826,
        },
        "modelUsage": {
            "claude-opus-4-7": {
                "inputTokens": 65,
                "outputTokens": 12451,
                "cacheCreationInputTokens": 75045,
                "cacheReadInputTokens": 2679931,
                "costUSD": 2.1205967500000003,
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 1043,
                "outputTokens": 19,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": 0.001138,
            },
        },
    },
]


def _ndjson(events: list[dict]) -> str:
    """Serialize to NDJSON, mixing in some non-result events to
    simulate the real stream's noise (``user`` / ``assistant`` /
    ``system`` interleaved)."""
    lines: list[str] = []
    for i, e in enumerate(events):
        # Sprinkle a non-result event before each result so the
        # filter isn't trivially the only-line case.
        lines.append(json.dumps({"type": "assistant", "n": i}))
        lines.append(json.dumps(e))
    return "\n".join(lines) + "\n"


def test_three_results_sum_per_batch_take_last_cumulative() -> None:
    iface = _iface("stream-json")
    summary = iface._summarize_results(_ndjson(_MULTI_RESULT_FIXTURE))

    # Per-batch: SUM
    assert summary["duration_ms"] == 375104 + 2954 + 1586  # 379_644 ms ≈ 6m 20s
    assert summary["num_turns"] == 49 + 1 + 1  # 51
    assert summary["usage"]["input_tokens"] == 53 + 6 + 6  # 65
    assert summary["usage"]["output_tokens"] == 12305 + 111 + 35  # 12_451
    assert summary["usage"]["cache_creation_input_tokens"] == 74056 + 642 + 347  # 75_045
    assert summary["usage"]["cache_read_input_tokens"] == 2522921 + 78184 + 78826  # 2_679_931

    # Cumulative: take LAST event's value
    assert summary["total_cost_usd"] == 2.1217347500000003
    assert summary["duration_api_ms"] == 197606
    assert summary["session_id"] == "5f0190ca-e0f1-49fb-9f29-77dfe79e3055"
    # modelUsage is from the last event — already cumulative per CC's contract
    mu = summary["modelUsage"]
    assert mu["claude-opus-4-7"]["costUSD"] == 2.1205967500000003
    assert mu["claude-opus-4-7"]["inputTokens"] == 65
    assert mu["claude-haiku-4-5-20251001"]["costUSD"] == 0.001138


def test_single_result_4_6_path_unchanged() -> None:
    """Regression target: the 4.6 happy path (single result event)
    must produce the same numbers as the old single-event parser."""
    iface = _iface("stream-json")
    one = _MULTI_RESULT_FIXTURE[0]
    stdout = json.dumps({"type": "system", "subtype": "init"}) + "\n" + json.dumps(one) + "\n"

    summary = iface._summarize_results(stdout)
    assert summary["duration_ms"] == 375104
    assert summary["num_turns"] == 49
    assert summary["usage"]["input_tokens"] == 53
    assert summary["usage"]["output_tokens"] == 12305
    assert summary["total_cost_usd"] == 2.0333385
    assert summary["session_id"] == one["session_id"]


def test_zero_results_returns_empty() -> None:
    """CLI crashed mid-run, killed by OOM, or otherwise produced no
    result event. Aggregator returns ``{}`` so callers fall back to
    wall-clock timing without raising."""
    iface = _iface("stream-json")
    stdout = "\n".join(
        json.dumps(e)
        for e in [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "n": 0},
            {"type": "user", "n": 1},
        ]
    )
    assert iface._summarize_results(stdout) == {}


def test_malformed_lines_are_skipped() -> None:
    """A single bad NDJSON line shouldn't kill aggregation. Mid-stream
    truncations (subprocess timeout, signal mid-write) are common
    enough to handle gracefully."""
    iface = _iface("stream-json")
    parts = [
        json.dumps({"type": "system", "subtype": "init"}),
        "{this is not json}",
        json.dumps(_MULTI_RESULT_FIXTURE[0]),
        "incomplete json {",
        json.dumps(_MULTI_RESULT_FIXTURE[1]),
    ]
    summary = iface._summarize_results("\n".join(parts) + "\n")
    assert summary["duration_ms"] == 375104 + 2954
    assert summary["num_turns"] == 50


def test_missing_modelUsage_returns_empty_dict() -> None:
    """Older CLIs don't emit ``modelUsage``. Aggregator surfaces an
    empty dict (not None) so downstream code can iterate without a
    None check."""
    iface = _iface("stream-json")
    one = dict(_MULTI_RESULT_FIXTURE[0])
    one.pop("modelUsage")
    summary = iface._summarize_results(json.dumps(one) + "\n")
    assert summary["modelUsage"] == {}


# ── json mode: single blob ──────────────────────────────────────────


def test_json_mode_single_blob_round_trips() -> None:
    """In ``--output-format json`` the CLI emits the final result as
    a single JSON blob (no NDJSON wrapping). Aggregator handles it
    via the same code path so json + stream-json runs produce
    identical surface contracts."""
    iface = _iface("json")
    stdout = json.dumps(_MULTI_RESULT_FIXTURE[0])
    summary = iface._summarize_results(stdout)
    assert summary["duration_ms"] == 375104
    assert summary["num_turns"] == 49
    assert summary["total_cost_usd"] == 2.0333385


def test_json_mode_non_result_blob_returns_empty() -> None:
    """Defensive: if json-mode stdout is a JSON object that ISN'T a
    result event, treat as no-result. Empirically this happens on
    the CLI's error paths where it dumps a different envelope."""
    iface = _iface("json")
    stdout = json.dumps({"type": "error", "message": "boom"})
    assert iface._summarize_results(stdout) == {}


def test_json_mode_unparseable_returns_empty() -> None:
    iface = _iface("json")
    assert iface._summarize_results("not even json") == {}


def test_empty_stdout_returns_empty() -> None:
    """Whitespace-only / completely empty stdout → empty summary
    rather than raise. CLIs killed before any output is the trigger."""
    for stdout in ("", "   ", "\n\n\n"):
        assert _iface("stream-json")._summarize_results(stdout) == {}
        assert _iface("json")._summarize_results(stdout) == {}


# ── _to_token_usage normalization ───────────────────────────────────


def test_to_token_usage_passes_through_modelUsage() -> None:
    """Per-model breakdown re-keys camelCase → snake_case + stringifies
    cost_usd for downstream JSONB serialization without float drift."""
    raw = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_creation_tokens": 300,
        "cache_read_tokens": 400,
        "model_usage": {
            "claude-opus-4-7": {
                "inputTokens": 100,
                "outputTokens": 200,
                "cacheCreationInputTokens": 300,
                "cacheReadInputTokens": 400,
                "costUSD": 1.2345,
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 50,
                "outputTokens": 5,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": 0.001,
            },
        },
    }
    out = ClaudeCodeInterface._to_token_usage(raw)
    assert out["input"] == 100
    assert out["output"] == 200
    assert out["cache_creation"] == 300
    assert out["cache_read"] == 400
    mu = out["model_usage"]
    assert set(mu.keys()) == {"claude-opus-4-7", "claude-haiku-4-5-20251001"}
    opus = mu["claude-opus-4-7"]
    assert opus["input_tokens"] == 100
    assert opus["output_tokens"] == 200
    assert opus["cache_creation_input_tokens"] == 300
    assert opus["cache_read_input_tokens"] == 400
    # cost_usd serialized as string to dodge float-drift.
    assert isinstance(opus["cost_usd"], str)
    assert opus["cost_usd"] == "1.2345"
    haiku = mu["claude-haiku-4-5-20251001"]
    assert haiku["cost_usd"] == "0.001"


def test_to_token_usage_omits_modelUsage_when_absent() -> None:
    """Older CLIs / older runs without modelUsage → output dict
    doesn't include ``model_usage`` at all (so downstream pydantic
    leaves the field as None)."""
    raw = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_creation_tokens": 3,
        "cache_read_tokens": 4,
    }
    out = ClaudeCodeInterface._to_token_usage(raw)
    assert "model_usage" not in out


def test_to_token_usage_handles_none_costUSD() -> None:
    """Pre-pricing CC versions emitted ``costUSD: null`` for unpriced
    models. The normalizer must coerce to ``"0"`` rather than raising."""
    raw = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "model_usage": {
            "some-model": {
                "inputTokens": 1,
                "outputTokens": 2,
                "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0,
                "costUSD": None,
            },
        },
    }
    out = ClaudeCodeInterface._to_token_usage(raw)
    assert out["model_usage"]["some-model"]["cost_usd"] == "0"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"input_tokens": 0},
        None,
    ],
)
def test_to_token_usage_handles_empty_input(raw: dict | None) -> None:
    out = ClaudeCodeInterface._to_token_usage(raw)
    assert out["input"] == 0
    assert out["output"] == 0
    assert out["cache_creation"] == 0
    assert out["cache_read"] == 0
    assert "model_usage" not in out
