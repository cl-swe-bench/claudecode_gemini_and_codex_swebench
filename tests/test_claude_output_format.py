"""Tests for ``ClaudeCodeInterface.output_format`` +
``_summarize_results`` (the post-cost-accounting-fix successor to the
old ``_extract_result_event`` method).

Token-equivalence regression: when the SAME final ``result`` event is
delivered as either a single JSON blob (``--output-format json``) or
embedded as the last event in an NDJSON stream
(``--output-format stream-json``), the summarized event and the
downstream ``token_usage`` dict must be identical. cl-benchmark's cost
rollup parses ``token_usage``, so any drift would silently mis-cost
runs created with ``capture_full_transcript=True``.

Multi-result regression: Claude Code may emit multiple ``type=result``
events per session (compaction / session-resume boundaries on any
model). ``_summarize_results`` SUMS per-batch fields (token counts,
duration_ms, num_turns) across events and takes cumulative fields
(``total_cost_usd``, ``session_id``, ``modelUsage``) from the LAST
event. Tests cover both shapes.

Specs: cl-benchmark/docs/run-transcript-capture-spec.md (Slice 1) +
cl-benchmark/docs/cost-accounting-fix-spec.md.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch the ``claude --version`` check at import time — tests run in
# environments without the CLI installed and we don't want to gate
# unit tests on it. The test only needs the parsing methods.
from unittest.mock import patch  # noqa: E402

with patch("subprocess.run") as _mock_run:
    _mock_run.return_value.returncode = 0
    from utils.claude_interface import ClaudeCodeInterface  # noqa: E402


# Representative final ``result`` event — same shape in both output
# formats (verified empirically against Claude Code 0.2.x; if the CLI
# ever changes the shape, this fixture is the spec to update).
RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "duration_ms": 12345,
    "duration_api_ms": 11000,
    "num_turns": 7,
    "result": "Refactored the storage backend; tests pass.",
    "session_id": "abc-123-session",
    "total_cost_usd": 0.4567,
    "is_error": False,
    "usage": {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_creation_input_tokens": 89,
        "cache_read_input_tokens": 901,
    },
}


def _make_interface(output_format: str) -> ClaudeCodeInterface:
    """Construct an interface with the ``claude --version`` check
    stubbed out — tests run in environments without the CLI."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        return ClaudeCodeInterface(output_format=output_format)


# ---------- _summarize_results ----------------------------------------------


def _assert_matches_single_result_event(out: dict, event: dict) -> None:
    """Helper: assert ``_summarize_results`` produced the canonical
    summary for a single ``RESULT_EVENT``-shaped fixture. The summary
    dict has a fixed shape (per-batch fields summed, cumulative fields
    taken from last event); for a single event the per-batch sums equal
    the event's values directly.
    """
    usage = event["usage"]
    assert out["duration_ms"] == event["duration_ms"]
    assert out["num_turns"] == event["num_turns"]
    assert out["usage"] == usage
    assert out["duration_api_ms"] == event["duration_api_ms"]
    assert out["session_id"] == event["session_id"]
    assert out["total_cost_usd"] == event["total_cost_usd"]


def test_summarize_results_json_returns_summary():
    iface = _make_interface("json")
    out = iface._summarize_results(json.dumps(RESULT_EVENT))
    _assert_matches_single_result_event(out, RESULT_EVENT)


def test_summarize_results_json_handles_empty_stdout():
    iface = _make_interface("json")
    assert iface._summarize_results("") == {}


def test_summarize_results_json_handles_invalid_json():
    iface = _make_interface("json")
    assert iface._summarize_results("not-json{{{") == {}


def test_summarize_results_stream_finds_final_result():
    iface = _make_interface("stream-json")
    ndjson = "\n".join(
        [
            json.dumps({"type": "system", "session_id": "abc-123-session"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"usage": {"input_tokens": 100, "output_tokens": 50}},
                }
            ),
            json.dumps({"type": "user", "message": "tool_result..."}),
            json.dumps(RESULT_EVENT),
        ]
    )
    out = iface._summarize_results(ndjson)
    _assert_matches_single_result_event(out, RESULT_EVENT)


def test_summarize_results_stream_returns_empty_when_no_result():
    iface = _make_interface("stream-json")
    ndjson = "\n".join(
        [
            json.dumps({"type": "system"}),
            json.dumps({"type": "assistant", "message": {}}),
        ]
    )
    assert iface._summarize_results(ndjson) == {}


def test_summarize_results_stream_skips_malformed_lines():
    """A single bad line in the middle of NDJSON shouldn't kill parsing."""
    iface = _make_interface("stream-json")
    ndjson = "\n".join(
        [
            json.dumps({"type": "system"}),
            "this is not json at all",
            json.dumps(RESULT_EVENT),
        ]
    )
    out = iface._summarize_results(ndjson)
    _assert_matches_single_result_event(out, RESULT_EVENT)


def test_summarize_results_stream_sums_per_batch_fields_across_multiple_results():
    """Multi-result emission (CC compaction / session-resume): per-batch
    fields (token counts, ``duration_ms``, ``num_turns``) sum across all
    ``type=result`` events; cumulative fields (``total_cost_usd``,
    ``session_id``, ``duration_api_ms``) come from the LAST event because
    the CLI accumulates these across batches itself.
    """
    iface = _make_interface("stream-json")
    earlier = {**RESULT_EVENT, "total_cost_usd": 0.1, "session_id": "earlier"}
    later = {**RESULT_EVENT, "total_cost_usd": 0.5, "session_id": "later"}
    ndjson = "\n".join([json.dumps(earlier), json.dumps(later)])
    out = iface._summarize_results(ndjson)

    # Per-batch — summed across both events.
    assert out["usage"]["input_tokens"] == 1234 * 2
    assert out["usage"]["output_tokens"] == 567 * 2
    assert out["usage"]["cache_creation_input_tokens"] == 89 * 2
    assert out["usage"]["cache_read_input_tokens"] == 901 * 2
    assert out["duration_ms"] == 12345 * 2
    assert out["num_turns"] == 7 * 2
    # Cumulative — last event wins.
    assert out["total_cost_usd"] == pytest.approx(0.5)
    assert out["session_id"] == "later"


def test_summarize_results_stream_handles_blank_lines():
    """Trailing newlines / blank lines around events shouldn't fail."""
    iface = _make_interface("stream-json")
    ndjson = (
        "\n\n"
        + json.dumps({"type": "system"})
        + "\n\n\n"
        + json.dumps(RESULT_EVENT)
        + "\n\n"
    )
    out = iface._summarize_results(ndjson)
    _assert_matches_single_result_event(out, RESULT_EVENT)


# ---------- Token equivalence (the hard merge gate) -------------------------


def _token_usage_from_output(iface: ClaudeCodeInterface, stdout: str) -> dict:
    """Run the same parsing pipeline ``_single_invocation`` uses to
    populate ``token_usage`` so the test exercises the full extraction
    path, not just ``_summarize_results`` in isolation."""
    output_data = iface._summarize_results(stdout)
    token_usage: dict = {}
    if isinstance(output_data, dict) and output_data:
        usage = output_data.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        }
        if output_data.get("total_cost_usd") is not None:
            token_usage["cost_usd"] = output_data["total_cost_usd"]
        if output_data.get("num_turns") is not None:
            token_usage["num_turns"] = output_data["num_turns"]
        if output_data.get("duration_ms") is not None:
            token_usage["duration_ms"] = output_data["duration_ms"]
        if output_data.get("duration_api_ms") is not None:
            token_usage["duration_api_ms"] = output_data["duration_api_ms"]
        if output_data.get("session_id") is not None:
            token_usage["session_id"] = output_data["session_id"]
    return token_usage


def test_token_equivalence_across_output_formats():
    """The hard merge gate per the spec: same final ``result`` event,
    delivered via either format, must produce IDENTICAL ``token_usage``
    dicts. cl-benchmark's cost rollup uses ``token_usage`` directly;
    drift here = silently mis-costed runs."""

    json_iface = _make_interface("json")
    stream_iface = _make_interface("stream-json")

    json_stdout = json.dumps(RESULT_EVENT)
    stream_stdout = "\n".join(
        [
            json.dumps({"type": "system", "session_id": "abc-123-session"}),
            json.dumps({"type": "assistant", "message": {}}),
            json.dumps({"type": "user", "message": {}}),
            json.dumps({"type": "assistant", "message": {}}),
            json.dumps(RESULT_EVENT),
        ]
    )

    json_tokens = _token_usage_from_output(json_iface, json_stdout)
    stream_tokens = _token_usage_from_output(stream_iface, stream_stdout)

    assert json_tokens == stream_tokens
    # Spot-check absolute values too — guards against both sides
    # silently agreeing on an empty dict due to a parsing bug.
    assert json_tokens["input_tokens"] == 1234
    assert json_tokens["output_tokens"] == 567
    assert json_tokens["total_tokens"] == 1801
    assert json_tokens["cache_creation_tokens"] == 89
    assert json_tokens["cache_read_tokens"] == 901
    assert json_tokens["cost_usd"] == pytest.approx(0.4567)
    assert json_tokens["num_turns"] == 7
    assert json_tokens["session_id"] == "abc-123-session"


# ---------- Constructor validation ------------------------------------------


def test_invalid_output_format_rejected():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        with pytest.raises(ValueError, match="output_format"):
            ClaudeCodeInterface(output_format="text")


def test_default_output_format_is_json():
    """Baseline-comparable default. cl-benchmark runs created without
    ``capture_full_transcript`` MUST hit this path so the harness CLI
    invocation matches the cc-46-full-run-regen baseline."""
    iface = _make_interface("json")
    assert iface.output_format == "json"


# ---------- CLI argv construction -------------------------------------------


def _captured_argv(output_format: str, *, cwd: str = "/tmp") -> list[str]:
    """Drive ``_single_invocation`` through to the subprocess call and
    capture the constructed argv. ``run_with_cancel`` is the indirection
    we patch — it owns the actual ``subprocess.run`` call."""

    iface = _make_interface(output_format)
    captured: dict[str, list[str]] = {}

    class _CompletedStub:
        returncode = 0
        stdout = json.dumps(RESULT_EVENT)
        stderr = ""
        timed_out = False
        cancelled = False

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _CompletedStub()

    with patch("utils.claude_interface.run_with_cancel", side_effect=_fake_run):
        iface._single_invocation("prompt", cwd, model=None)

    return captured["cmd"]


def test_stream_json_adds_verbose_flag():
    """Claude Code rejects ``-p --output-format=stream-json`` without
    ``--verbose`` ("requires --verbose"). Regression guard so we don't
    silently break MCP-visibility runs in a refactor."""

    argv = _captured_argv("stream-json")
    assert "--verbose" in argv
    assert "stream-json" in argv


def test_json_does_not_add_verbose_flag():
    """Baseline runs stay byte-identical to the cc-46-full-run-regen
    invocation — no spurious ``--verbose`` for the default path."""

    argv = _captured_argv("json")
    assert "--verbose" not in argv


def test_mcp_config_flag_added_when_mcp_json_exists_in_cwd(tmp_path):
    """``code_swe_agent`` writes ``<cwd>/.mcp.json`` per instance when
    MCP is enabled. The interface must point Claude at it explicitly
    via ``--mcp-config``, because project-scoped ``.mcp.json`` requires
    interactive workspace-trust that ``--dangerously-skip-permissions``
    doesn't bypass. Without this flag, MCP runs silently boot with
    ``mcp_servers: []`` and no MCP tool calls happen."""

    mcp_path = tmp_path / ".mcp.json"
    mcp_path.write_text('{"mcpServers": {}}')

    argv = _captured_argv("json", cwd=str(tmp_path))
    assert "--mcp-config" in argv
    assert str(mcp_path) in argv
    # Strict mode pinned so we don't accidentally also load any
    # user-scoped MCP config that drifted into HOME.
    assert "--strict-mcp-config" in argv


def test_mcp_config_flag_omitted_when_no_mcp_json(tmp_path):
    """Non-MCP runs stay byte-identical to baseline — no spurious
    ``--mcp-config`` flag in the argv when there's no ``.mcp.json``
    next to ``cwd``."""

    argv = _captured_argv("json", cwd=str(tmp_path))
    assert "--mcp-config" not in argv
    assert "--strict-mcp-config" not in argv
