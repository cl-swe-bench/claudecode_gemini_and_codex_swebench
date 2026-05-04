"""Tests for the post-instance ``~/.claude/projects/`` cleanup helpers
that close cl-benchmark's project-memory contamination channel.

Approach: temporarily relocate HOME to a tmp dir so ``Path.home()`` —
which the helpers use internally — points at a controlled fixture. We
then probe the snapshot/cleanup contract directly: the snapshot freezes
the entries-known-pre-invocation, and the cleanup rms only entries
that appeared after.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Harness lives at the repo root, sibling to ``tests/``. Adding the
# parent dir to ``sys.path`` lets us import the helpers directly without
# packaging the harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_swe_agent import _cleanup_new_project_dirs, _snapshot_projects_dir  # noqa: E402


@pytest.fixture
def tmp_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``Path.home()`` (which uses ``$HOME``) at a tmp dir so the
    helpers operate on a controlled fixture. Yields the projects dir
    path so each test can mkdir / touch entries directly."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Also clear USERPROFILE (Windows) defensively even though tests
    # only run on Unix CI today — keeps this resilient.
    monkeypatch.delenv("USERPROFILE", raising=False)
    projects = tmp_path / ".claude" / "projects"
    return projects


def test_snapshot_returns_empty_when_dir_missing(tmp_home: Path) -> None:
    # Fresh HOME — no .claude tree yet. Snapshot must return an empty
    # set rather than raise; the contamination-analysis doc notes this
    # is the common state on fresh CI hosts.
    assert _snapshot_projects_dir() == set()


def test_snapshot_lists_existing_dirs(tmp_home: Path) -> None:
    tmp_home.mkdir(parents=True)
    (tmp_home / "-tmp-foo").mkdir()
    (tmp_home / "-tmp-bar").mkdir()
    # Files must not be treated as project dirs.
    (tmp_home / "stray.txt").write_text("not a dir")
    snap = _snapshot_projects_dir()
    assert snap == {"-tmp-foo", "-tmp-bar"}


def test_cleanup_removes_only_new_entries(tmp_home: Path) -> None:
    tmp_home.mkdir(parents=True)
    (tmp_home / "-pre-existing").mkdir()
    before = _snapshot_projects_dir()
    # Simulate claude-code creating a residue dir during the instance.
    new_dir = tmp_home / "-var-folders-T-swe_bench_iid-x-y"
    new_dir.mkdir()
    (new_dir / "memory").mkdir()
    (new_dir / "memory" / "auto.md").write_text("# auto-memory residue")
    _cleanup_new_project_dirs(before)
    assert (tmp_home / "-pre-existing").exists(), "must keep entries that pre-existed"
    assert not new_dir.exists(), "must remove entries that appeared after the snapshot"


def test_cleanup_no_op_when_snapshot_predates_dir(tmp_home: Path) -> None:
    """Snapshot ran when the projects dir didn't exist; cleanup runs
    after the dir came into existence. Whatever was created during the
    instance counts as 'new' relative to the empty snapshot, so cleanup
    sweeps it. No exceptions; the whole tree should be empty after."""
    before = _snapshot_projects_dir()
    assert before == set()
    tmp_home.mkdir(parents=True)
    (tmp_home / "-late-arrival").mkdir()
    _cleanup_new_project_dirs(before)
    # Sweeping happened.
    assert list(tmp_home.iterdir()) == []


def test_cleanup_silent_on_filesystem_errors(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup is defensive — a failure to rm must not propagate.
    Simulate by patching ``Path.iterdir`` on the projects dir to raise
    ``OSError``; the helper should swallow and return."""
    before = _snapshot_projects_dir()
    tmp_home.mkdir(parents=True)
    (tmp_home / "-some-residue").mkdir()

    real_iterdir = Path.iterdir

    def boom(self: Path):
        if self.name == "projects":
            raise OSError("simulated EIO")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", boom)
    # No raise.
    _cleanup_new_project_dirs(before)


def test_cleanup_skips_files_at_top_level(tmp_home: Path) -> None:
    tmp_home.mkdir(parents=True)
    before = _snapshot_projects_dir()
    # Some Claude Code versions write files at the projects/ root
    # (timestamps, indexes). Cleanup should ignore those — they aren't
    # per-cwd memory dirs.
    (tmp_home / "index.json").write_text("{}")
    (tmp_home / "-new-residue").mkdir()
    _cleanup_new_project_dirs(before)
    assert (tmp_home / "index.json").exists()
    assert not (tmp_home / "-new-residue").exists()


def test_helpers_are_idempotent(tmp_home: Path) -> None:
    """Running cleanup twice with the same ``before`` snapshot must be
    safe — the second call sees nothing new to remove."""
    tmp_home.mkdir(parents=True)
    (tmp_home / "-keep").mkdir()
    before = _snapshot_projects_dir()
    (tmp_home / "-residue").mkdir()
    _cleanup_new_project_dirs(before)
    _cleanup_new_project_dirs(before)  # second invocation must not raise
    assert (tmp_home / "-keep").exists()
    assert not (tmp_home / "-residue").exists()


def test_does_not_touch_paths_outside_projects(tmp_home: Path) -> None:
    """Sibling state under ``~/.claude/`` (CLAUDE.md, memory/, credentials)
    must remain untouched — they're not cleanup targets and the analysis
    doc treats user-level CLAUDE.md as a separate concern owned by T2."""
    tmp_home.mkdir(parents=True)
    sibling_md = tmp_home.parent / "CLAUDE.md"
    sibling_md.write_text("user-level memory — do not touch")
    sibling_memory = tmp_home.parent / "memory"
    sibling_memory.mkdir()
    (sibling_memory / "note.md").write_text("user-level note")
    before = _snapshot_projects_dir()
    (tmp_home / "-residue").mkdir()
    _cleanup_new_project_dirs(before)
    assert sibling_md.exists()
    assert sibling_memory.exists()
    assert not (tmp_home / "-residue").exists()


def test_deletion_recurses(tmp_home: Path) -> None:
    """The residue includes nested ``memory/auto.md``, ``todos/``, etc.
    Cleanup must remove the whole tree, not just the top-level dir."""
    tmp_home.mkdir(parents=True)
    before = _snapshot_projects_dir()
    res = tmp_home / "-tree"
    (res / "memory").mkdir(parents=True)
    (res / "memory" / "auto.md").write_text("residue auto-memory")
    (res / "todos").mkdir()
    (res / "todos" / "todo.md").write_text("residue todo")
    _cleanup_new_project_dirs(before)
    assert not res.exists()


def test_path_home_picks_up_relocated_home(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cl-benchmark relocates HOME per-shard for api_key runs. The
    helpers must operate on the relocated HOME, not on the host.
    Property check: changing HOME after a snapshot redirects the
    second snapshot to the new location."""
    snap1 = _snapshot_projects_dir()
    other = tmp_path.parent / "other-home"
    other.mkdir()
    (other / ".claude" / "projects").mkdir(parents=True)
    (other / ".claude" / "projects" / "-elsewhere").mkdir()
    monkeypatch.setenv("HOME", str(other))
    snap2 = _snapshot_projects_dir()
    assert snap1 == set()
    assert snap2 == {"-elsewhere"}
