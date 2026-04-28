"""Tests for ``_resolve_repo_identifier`` — the per-instance helper that
substitutes the cloned repo's git remote URL into the nudged prompt
template + the injected CLAUDE.md section.

Behavior under test:
  * Real git repo with origin set → returns the literal URL.
  * No origin set (or non-git path) → falls back to
    ``https://github.com/{repo}.git`` derived from the SWE-bench
    instance's ``repo`` field.
  * Empty fallback + no origin → returns empty string (defensive; the
    nudged template still parses with the placeholder empty).

Spec: cl-benchmark/docs/mcp-priming-spec.md.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_swe_agent import _resolve_repo_identifier  # noqa: E402


def _init_git(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init", "-q"], cwd=path, check=True)


def test_returns_origin_url_when_set(tmp_path):
    _init_git(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/acme/widget.git"],
        cwd=tmp_path,
        check=True,
    )
    out = _resolve_repo_identifier(str(tmp_path), fallback_repo="should-not-be-used/x")
    assert out == "https://github.com/acme/widget.git"


def test_falls_back_to_inferred_github_url_when_origin_missing(tmp_path):
    _init_git(tmp_path)
    # No remote configured → ``git remote get-url origin`` exits non-zero.
    out = _resolve_repo_identifier(str(tmp_path), fallback_repo="acme/widget")
    assert out == "https://github.com/acme/widget.git"


def test_falls_back_when_path_is_not_a_git_repo(tmp_path):
    out = _resolve_repo_identifier(str(tmp_path), fallback_repo="acme/widget")
    assert out == "https://github.com/acme/widget.git"


def test_returns_empty_when_no_origin_and_no_fallback(tmp_path):
    """Defensive corner — should never happen for SWE-bench Pro clones,
    but the nudged template still substitutes cleanly into an empty
    string."""
    _init_git(tmp_path)
    out = _resolve_repo_identifier(str(tmp_path), fallback_repo="")
    assert out == ""


def test_returns_empty_when_path_does_not_exist():
    out = _resolve_repo_identifier("/nonexistent/path/should/not/be/here", fallback_repo="")
    assert out == ""
