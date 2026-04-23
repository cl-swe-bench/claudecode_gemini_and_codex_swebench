"""Regression: ``setup_repository`` captures git subprocess stdout/stderr.

Before this fix a failed ``git clone`` printed to the worker's own
stdout and the agent returned ``raw_stderr=""``; the MinIO gen-log blob
for that instance was 0 bytes and the UI had no way to see the git
error. ``setup_repository`` now returns ``(path, stdout, stderr)`` and
``process_instance``'s early-return threads both into ``raw_stdout`` /
``raw_stderr`` so the worker's log upload persists the real failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_swe_agent import CodeSWEAgent


class _CR:
    """Stand-in for ``subprocess.CompletedProcess`` — only the fields
    ``setup_repository`` reads are needed."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def agent(tmp_path, monkeypatch):
    # ``CodeSWEAgent.__init__`` anchors results/predictions paths off
    # ``Path.cwd()``; chdir to ``tmp_path`` so the test creates its
    # artifact dirs inside the pytest sandbox instead of the repo root.
    monkeypatch.chdir(tmp_path)
    return CodeSWEAgent(backend="claude")


def _instance():
    return {
        "instance_id": "acme__widget-abc123",
        "repo": "acme/widget",
        "base_commit": "deadbeef" * 5,
    }


def test_setup_repository_returns_triple_on_clone_failure(agent, monkeypatch):
    """Clone exits non-zero → ``(None, stdout, stderr_with_git_msg)``."""

    def fake_run(cmd, *args, **kwargs):
        # First call is ``git clone``. Return non-zero + a git-style
        # fatal on stderr so the harness treats it as a clone failure.
        return _CR(returncode=128, stdout="", stderr="fatal: repository not found\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    path, stdout, stderr = agent.setup_repository(_instance())
    assert path is None
    assert "fatal: repository not found" in stderr
    # The human-readable "Cloning ..." line lands in stdout.
    assert "Cloning acme/widget" in stdout


def test_setup_repository_returns_triple_on_checkout_failure(agent, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["git", "clone"]:
            # Pretend clone worked; create the directory the next step
            # will chdir into.
            os.makedirs(cmd[-1], exist_ok=True)
            return _CR(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return _CR(
                returncode=128,
                stdout="",
                stderr="fatal: reference is not a tree: deadbeefdeadbeef\n",
            )
        return _CR(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    path, stdout, stderr = agent.setup_repository(_instance())
    assert path is None
    assert "fatal: reference is not a tree" in stderr
    assert any(c[:2] == ["git", "checkout"] for c in calls)


def test_setup_repository_success(agent, monkeypatch, tmp_path):
    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            os.makedirs(cmd[-1], exist_ok=True)
        return _CR(returncode=0, stdout="cloned ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    path, stdout, stderr = agent.setup_repository(_instance())
    assert path is not None
    assert "cloned ok" in stdout
    assert stderr.strip() == ""


def test_process_instance_threads_stderr_into_raw_stderr(agent, monkeypatch):
    """End-to-end: setup fails → early-return dict carries raw_stderr."""

    def fake_run(cmd, *args, **kwargs):
        return _CR(returncode=128, stdout="", stderr="fatal: repository not found\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with patch.object(agent, "setup_repository", wraps=agent.setup_repository):
        prediction = agent.process_instance(_instance())

    assert prediction["error"].startswith("fatal:") or "repository" in prediction["error"]
    assert "fatal: repository not found" in prediction["raw_stderr"]
    assert "Cloning acme/widget" in prediction["raw_stdout"]


def test_process_instance_error_summary_prefers_fatal_line(agent, monkeypatch):
    """The DB ``error`` column (80-char tail) should show the git
    ``fatal:`` line, not the generic wrapper message."""

    def fake_run(cmd, *args, **kwargs):
        return _CR(
            returncode=128,
            stdout="",
            stderr=(
                "Cloning into '/tmp/swe_bench_acme__widget-abc123'...\n"
                "fatal: unable to access 'https://github.com/acme/widget.git/': "
                "Could not resolve host: github.com\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    prediction = agent.process_instance(_instance())
    assert "fatal: unable to access" in prediction["error"]
    assert "Could not resolve host" in prediction["error"]
