"""Tests for ``ClaudeCodeInterface.mcp_debug`` — captures the CLI's
``--debug-file`` output for MCP forensics.

When ``mcp_debug=True``, the interface:

1. Appends ``--debug-file <cwd>/mcp-debug.log`` to the ``claude`` argv.
2. After the subprocess returns (success, timeout, cancel, or
   exception), reads the log file from disk, deletes it, and
   surfaces contents via ``result["mcp_debug_log"]``. ``None`` on
   missing / empty / unreadable files.

When ``mcp_debug=False`` (default), the flag is absent and the result
dict's ``mcp_debug_log`` is ``None``. Baseline-comparable.

Spec: cl-benchmark/docs/mcp-debug-capture-spec.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch  # noqa: E402

with patch("subprocess.run") as _mock_run:
    _mock_run.return_value.returncode = 0
    from utils.claude_interface import ClaudeCodeInterface  # noqa: E402


RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "duration_ms": 12345,
    "duration_api_ms": 11000,
    "num_turns": 7,
    "result": "ok",
    "session_id": "abc-123",
    "total_cost_usd": 0.4567,
    "is_error": False,
    "usage": {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_creation_input_tokens": 89,
        "cache_read_input_tokens": 901,
    },
}


def _make_interface(**kwargs) -> ClaudeCodeInterface:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        return ClaudeCodeInterface(**kwargs)


# ---------- argv construction -----------------------------------------------


def _captured_argv(*, mcp_debug: bool, cwd: str) -> list[str]:
    """Drive ``_single_invocation`` and capture the constructed argv."""
    iface = _make_interface(mcp_debug=mcp_debug)
    captured: dict[str, list[str]] = {}

    class _Stub:
        returncode = 0
        stdout = json.dumps(RESULT_EVENT)
        stderr = ""
        timed_out = False
        cancelled = False

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Stub()

    with patch("utils.claude_interface.run_with_cancel", side_effect=_fake_run):
        iface._single_invocation("prompt", cwd, model=None)
    return captured["cmd"]


def test_debug_file_flag_added_when_mcp_debug_true(tmp_path):
    """``--debug-file <cwd>/mcp-debug.log`` lands in the argv when the
    interface was constructed with ``mcp_debug=True``."""
    argv = _captured_argv(mcp_debug=True, cwd=str(tmp_path))
    assert "--debug-file" in argv
    expected = str(tmp_path / "mcp-debug.log")
    assert expected in argv
    # Flag + value adjacent so caller can read it cleanly.
    idx = argv.index("--debug-file")
    assert argv[idx + 1] == expected


def test_debug_file_flag_omitted_by_default(tmp_path):
    """Default ``mcp_debug=False`` keeps the argv baseline-identical
    to the cc-46-full-run-regen invocation — no spurious flag."""
    argv = _captured_argv(mcp_debug=False, cwd=str(tmp_path))
    assert "--debug-file" not in argv


def test_mcp_debug_default_is_false():
    """Regression guard: future signature changes can't quietly flip
    the default and start writing debug logs on every run."""
    iface = _make_interface()
    assert iface.mcp_debug is False


# ---------- _read_and_clean_mcp_debug ---------------------------------------


def test_read_returns_none_when_path_is_none():
    """Caller passed ``mcp_debug=False`` → no path to read."""
    assert ClaudeCodeInterface._read_and_clean_mcp_debug(None) is None


def test_read_returns_none_when_file_missing(tmp_path):
    """CLI crashed before debug-mode init → file never written.
    Return None silently; the absence is meaningful but not an
    error condition."""
    missing = tmp_path / "mcp-debug.log"
    assert not missing.exists()
    assert ClaudeCodeInterface._read_and_clean_mcp_debug(str(missing)) is None


def test_read_returns_none_when_file_empty(tmp_path):
    """An empty file shouldn't surface as an empty string —
    semantically the same as "no debug captured"."""
    empty = tmp_path / "mcp-debug.log"
    empty.write_text("")
    assert ClaudeCodeInterface._read_and_clean_mcp_debug(str(empty)) is None
    # Cleanup happened — we don't leave stray empty files around.
    assert not empty.exists()


def test_read_returns_contents_and_deletes_file(tmp_path):
    """The happy path: file has real debug data, return it as a
    string, then unlink so re-runs in the same cwd don't accumulate
    stale logs."""
    log = tmp_path / "mcp-debug.log"
    body = (
        "2026-05-14T00:57:36.398Z [DEBUG] MDM settings load completed in 0ms\n"
        "2026-05-14T00:57:39.527Z [DEBUG] MCP server \"code-lexica\": "
        "HTTP transport options: {\"timeoutMs\":60000}\n"
    )
    log.write_text(body)
    out = ClaudeCodeInterface._read_and_clean_mcp_debug(str(log))
    assert out == body
    assert not log.exists()


def test_read_handles_non_utf8_bytes(tmp_path):
    """``errors='replace'`` keeps garbage from breaking the read.
    The log is for humans, not strict JSON — partial bytes are
    better than a hard fail."""
    log = tmp_path / "mcp-debug.log"
    log.write_bytes(b"valid prefix " + b"\xff\xfe" + b" valid suffix\n")
    out = ClaudeCodeInterface._read_and_clean_mcp_debug(str(log))
    assert out is not None
    assert "valid prefix" in out
    assert "valid suffix" in out
    assert not log.exists()


def test_read_returns_none_on_unreadable_file(tmp_path, monkeypatch):
    """``open()`` raising OSError → return None, don't propagate.
    Captures the edge case where the file exists but the worker
    can't read it (e.g., wrong owner inside a docker mount)."""
    log = tmp_path / "mcp-debug.log"
    log.write_text("body")

    real_open = open

    def _raise_open(path, *args, **kwargs):
        if str(path) == str(log):
            raise OSError("simulated read failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _raise_open)
    assert ClaudeCodeInterface._read_and_clean_mcp_debug(str(log)) is None


def test_read_tolerates_unlink_failure(tmp_path, monkeypatch):
    """If ``os.unlink`` throws (race with another reader, exotic
    filesystem), we should still return the contents — the unlink
    is best-effort cleanup, not a hard requirement."""
    log = tmp_path / "mcp-debug.log"
    log.write_text("real content")

    def _raise_unlink(path):
        raise OSError("simulated unlink failure")

    monkeypatch.setattr("os.unlink", _raise_unlink)
    out = ClaudeCodeInterface._read_and_clean_mcp_debug(str(log))
    assert out == "real content"


# ---------- mcp_debug_log on each return path -------------------------------


def _drive_invocation(*, mcp_debug: bool, cwd: str, stdout: str,
                      timed_out: bool = False, cancelled: bool = False,
                      returncode: int = 0,
                      debug_log_body: str | None = "debug body\n") -> dict:
    """Run ``_single_invocation`` with a stubbed run_with_cancel and
    a pre-populated debug log file in cwd."""
    iface = _make_interface(mcp_debug=mcp_debug)

    if debug_log_body is not None:
        Path(cwd, "mcp-debug.log").write_text(debug_log_body)

    class _Stub:
        pass

    stub = _Stub()
    stub.returncode = returncode
    stub.stdout = stdout
    stub.stderr = ""
    stub.timed_out = timed_out
    stub.cancelled = cancelled

    with patch("utils.claude_interface.run_with_cancel", return_value=stub):
        return iface._single_invocation("prompt", cwd, model=None)


def test_mcp_debug_log_surfaced_on_success(tmp_path):
    out = _drive_invocation(
        mcp_debug=True, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        debug_log_body="DEBUG line A\nDEBUG line B\n",
    )
    assert out["success"] is True
    assert out["mcp_debug_log"] == "DEBUG line A\nDEBUG line B\n"


def test_mcp_debug_log_surfaced_on_timeout(tmp_path):
    """Real-world case: most MCP debug captures happen on timed-out
    samples. The debug log is the primary forensic artifact here —
    don't drop it just because the CLI timed out."""
    ndjson = "\n".join([json.dumps({"type": "system"}), json.dumps(RESULT_EVENT)])
    out = _drive_invocation(
        mcp_debug=True, cwd=str(tmp_path), stdout=ndjson,
        timed_out=True, returncode=-1,
        debug_log_body="MCP timeout detail",
    )
    assert out["timed_out"] is True
    assert out["mcp_debug_log"] == "MCP timeout detail"


def test_mcp_debug_log_surfaced_on_cancel(tmp_path):
    out = _drive_invocation(
        mcp_debug=True, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        cancelled=True, returncode=-15,
        debug_log_body="cancel mid-run",
    )
    assert out["cancelled"] is True
    assert out["mcp_debug_log"] == "cancel mid-run"


def test_mcp_debug_log_surfaced_on_subprocess_exception(tmp_path):
    """Even if ``run_with_cancel`` raises (e.g. ``FileNotFoundError``
    on ``cmd[0]``) the file might still exist from a previous run in
    this cwd — surface it. Honest reporting beats silent loss."""
    iface = _make_interface(mcp_debug=True)
    Path(tmp_path, "mcp-debug.log").write_text("pre-spawn body")
    with patch(
        "utils.claude_interface.run_with_cancel",
        side_effect=FileNotFoundError("claude not found"),
    ):
        out = iface._single_invocation("prompt", str(tmp_path), model=None)
    assert out["success"] is False
    assert out["mcp_debug_log"] == "pre-spawn body"


def test_mcp_debug_log_none_when_flag_off(tmp_path):
    """``mcp_debug=False`` → never read the file, return None even
    if one happens to exist (e.g. left over from a prior run)."""
    Path(tmp_path, "mcp-debug.log").write_text("not ours")
    out = _drive_invocation(
        mcp_debug=False, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        debug_log_body=None,
    )
    assert out["success"] is True
    assert out["mcp_debug_log"] is None


def test_mcp_debug_log_none_when_cli_did_not_write(tmp_path):
    """``mcp_debug=True`` but the CLI crashed too early to write the
    file → None. No spurious empty strings."""
    out = _drive_invocation(
        mcp_debug=True, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        debug_log_body=None,
    )
    assert out["mcp_debug_log"] is None


# ---------- validation that no other paths break ----------------------------


def test_mcp_debug_does_not_affect_token_usage(tmp_path):
    """Token extraction must remain identical regardless of mcp_debug
    state — the debug log lives in a side file and never feeds back
    into ``token_usage``."""
    out_with = _drive_invocation(
        mcp_debug=True, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        debug_log_body="x",
    )
    # Different tmp dir to avoid stale file leakage.
    other = tmp_path / "other"
    other.mkdir()
    out_without = _drive_invocation(
        mcp_debug=False, cwd=str(other), stdout=json.dumps(RESULT_EVENT),
        debug_log_body=None,
    )
    assert out_with["token_usage"] == out_without["token_usage"]


def test_mcp_debug_log_in_dict_even_on_success_when_off(tmp_path):
    """``mcp_debug_log`` is ALWAYS in the result dict (None or str),
    so worker callers can use ``result.get("mcp_debug_log")``
    uniformly without worrying about the key being missing."""
    out = _drive_invocation(
        mcp_debug=False, cwd=str(tmp_path), stdout=json.dumps(RESULT_EVENT),
        debug_log_body=None,
    )
    assert "mcp_debug_log" in out
    assert out["mcp_debug_log"] is None
