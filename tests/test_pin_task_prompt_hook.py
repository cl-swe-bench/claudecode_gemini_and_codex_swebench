"""Tests for the Code Lexica taskPrompt-pin mechanism (Option A).

Three pieces, per cl-benchmark/docs/mcp-priming-spec.md (Amendment 2026-05-27):

  * ``code_lexica_pin_hook.py`` — the PreToolUse hook Claude Code invokes.
    Rewrites taskPrompt / repoIdentifier / commitHash from
    ``.code_lexica/pinned_params.json`` via ``updatedInput``; fails open.
  * ``inject_pin_hook`` / ``remove_pin_hook`` — write the pinned params +
    register/strip the hook in ``.claude/settings.json``, idempotently and
    without clobbering unrelated settings.
  * ``CodeSWEAgent`` gating — ``mcp_pin_task_prompt`` only takes effect on
    the claude backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch  # noqa: E402

from utils.mcp_config import (  # noqa: E402
    PINNED_PARAMS_RELPATH,
    _PIN_HOOK_MATCHER,
    inject_pin_hook,
    remove_pin_hook,
)

HOOK_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "utils", "code_lexica_pin_hook.py")
)
REPO_ID = "https://github.com/acme/widget.git"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TASK = "Fix the bug.\n\nRequirements:\nR1\n\nNew interfaces introduced:\nI1"


def _run_hook(project_dir, event, *, set_project_env=True):
    env = dict(os.environ)
    if set_project_env:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    else:
        env.pop("CLAUDE_PROJECT_DIR", None)
    proc = subprocess.run(
        [sys.executable, HOOK_SCRIPT],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        cwd=str(project_dir),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# -------------------- hook script --------------------


def test_hook_overrides_pinned_keys_and_preserves_others(tmp_path):
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    event = {
        "tool_name": _PIN_HOOK_MATCHER,
        "tool_input": {
            "repoIdentifier": "short-form-wrong",
            "commitHash": "deadbeef",
            "taskPrompt": "AGENT SUMMARY (wrong)",
            "someOtherArg": "keep-me",
        },
    }
    out = json.loads(_run_hook(tmp_path, event))
    upd = out["hookSpecificOutput"]["updatedInput"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert upd["taskPrompt"] == TASK
    assert upd["repoIdentifier"] == REPO_ID
    assert upd["commitHash"] == COMMIT
    # Unrelated args survive.
    assert upd["someOtherArg"] == "keep-me"


def test_hook_reads_via_cwd_when_no_project_dir_env(tmp_path):
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    event = {"tool_name": _PIN_HOOK_MATCHER, "tool_input": {"taskPrompt": "x"}}
    out = json.loads(_run_hook(tmp_path, event, set_project_env=False))
    assert out["hookSpecificOutput"]["updatedInput"]["taskPrompt"] == TASK


def test_hook_fails_open_when_no_params_file(tmp_path):
    event = {"tool_name": _PIN_HOOK_MATCHER, "tool_input": {"taskPrompt": "x"}}
    # No pinned_params.json written -> no decision emitted (tool unchanged).
    assert _run_hook(tmp_path, event) == ""


def test_hook_fails_open_on_corrupt_params(tmp_path):
    params = tmp_path / PINNED_PARAMS_RELPATH
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text("{not valid json")
    event = {"tool_name": _PIN_HOOK_MATCHER, "tool_input": {"taskPrompt": "x"}}
    assert _run_hook(tmp_path, event) == ""


# -------------------- inject_pin_hook / remove_pin_hook --------------------


def _pretooluse(settings_path):
    return json.loads(settings_path.read_text())["hooks"]["PreToolUse"]


def _our_entries(pretooluse):
    return [
        e for e in pretooluse
        if e.get("matcher") == _PIN_HOOK_MATCHER
        and any("code_lexica_pin_hook.py" in h.get("command", "") for h in e["hooks"])
    ]


def test_inject_writes_params_and_settings(tmp_path):
    settings_path = inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    params = json.loads((tmp_path / PINNED_PARAMS_RELPATH).read_text())
    assert params == {"repoIdentifier": REPO_ID, "commitHash": COMMIT, "taskPrompt": TASK}
    assert settings_path == tmp_path / ".claude" / "settings.json"
    assert len(_our_entries(_pretooluse(settings_path))) == 1


def test_inject_is_idempotent(tmp_path):
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    settings_path = inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    assert len(_our_entries(_pretooluse(settings_path))) == 1


def test_inject_refreshes_task_prompt_on_reinject(tmp_path):
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, "OLD TASK")
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, "NEW TASK")
    params = json.loads((tmp_path / PINNED_PARAMS_RELPATH).read_text())
    assert params["taskPrompt"] == "NEW TASK"


def test_inject_preserves_unrelated_settings(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps({
        "model": "opus",
        "hooks": {"PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "other.sh"}]}
        ]},
    }))
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    settings = json.loads((claude_dir / "settings.json").read_text())
    assert settings["model"] == "opus"
    pretooluse = settings["hooks"]["PreToolUse"]
    assert any(e.get("matcher") == "Bash" for e in pretooluse)  # unrelated kept
    assert len(_our_entries(pretooluse)) == 1                   # ours added


def test_remove_strips_only_our_entry_and_params(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps({
        "hooks": {"PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "other.sh"}]}
        ]},
    }))
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    assert (tmp_path / PINNED_PARAMS_RELPATH).exists()

    remove_pin_hook(str(tmp_path))
    settings = json.loads((claude_dir / "settings.json").read_text())
    pretooluse = settings["hooks"]["PreToolUse"]
    assert any(e.get("matcher") == "Bash" for e in pretooluse)  # unrelated preserved
    assert _our_entries(pretooluse) == []                       # ours gone
    assert not (tmp_path / PINNED_PARAMS_RELPATH).exists()      # params removed


def test_remove_deletes_settings_when_only_ours(tmp_path):
    inject_pin_hook(str(tmp_path), REPO_ID, COMMIT, TASK)
    assert (tmp_path / ".claude" / "settings.json").exists()
    remove_pin_hook(str(tmp_path))
    assert not (tmp_path / ".claude" / "settings.json").exists()
    assert not (tmp_path / PINNED_PARAMS_RELPATH).exists()


def test_remove_is_noop_when_absent(tmp_path):
    remove_pin_hook(str(tmp_path))  # must not raise
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_remove_preserves_unrelated_only_settings(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    original = json.dumps({"model": "opus"}, indent=2)
    (claude_dir / "settings.json").write_text(original)
    remove_pin_hook(str(tmp_path))
    assert json.loads((claude_dir / "settings.json").read_text()) == {"model": "opus"}


# -------------------- CodeSWEAgent backend gate --------------------


def test_pin_gate_enabled_on_claude_backend():
    import code_swe_agent
    agent = code_swe_agent.CodeSWEAgent(backend="claude", mcp_pin_task_prompt=True)
    assert agent.mcp_pin_task_prompt is True
    agent_off = code_swe_agent.CodeSWEAgent(backend="claude", mcp_pin_task_prompt=False)
    assert agent_off.mcp_pin_task_prompt is False


def test_pin_gate_disabled_on_non_claude_backend():
    import code_swe_agent
    # Neutralize the codex interface construction; we only care about the gate.
    with patch.object(code_swe_agent, "CodexCodeInterface"):
        agent = code_swe_agent.CodeSWEAgent(backend="codex", mcp_pin_task_prompt=True)
    assert agent.mcp_pin_task_prompt is False
