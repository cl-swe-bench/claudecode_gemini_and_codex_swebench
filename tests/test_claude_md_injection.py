"""Tests for ``inject_claude_md_section`` + ``remove_claude_md_section``.

The harness writes a ``Code Lexica MCP`` section into the cloned repo's
CLAUDE.md before launching the agent on MCP-enabled runs. Three lifecycle
shapes have to behave correctly:

  * Fresh write — no upstream CLAUDE.md, no prior section.
  * Append — upstream CLAUDE.md exists, no prior section. Our section
    lands at the end; upstream content survives verbatim.
  * Replace — upstream + a prior section (sentinel present). Our section
    replaces the bracketed region in-place; upstream survives.

Plus the symmetric removal path (idempotent + preserves upstream).
Spec: cl-benchmark/docs/mcp-priming-spec.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mcp_config import (  # noqa: E402
    _CLAUDE_MD_SENTINEL_END,
    _CLAUDE_MD_SENTINEL_START,
    inject_claude_md_section,
    remove_claude_md_section,
)


REPO_ID = "https://github.com/acme/widget.git"


def test_inject_writes_fresh_when_no_claude_md(tmp_path):
    out = inject_claude_md_section(str(tmp_path), REPO_ID)
    assert out == tmp_path / "CLAUDE.md"
    body = out.read_text()
    assert _CLAUDE_MD_SENTINEL_START in body
    assert _CLAUDE_MD_SENTINEL_END in body
    assert REPO_ID in body
    # repo_identifier placeholder fully substituted.
    assert "{repo_identifier}" not in body


def test_inject_appends_to_upstream_claude_md(tmp_path):
    upstream = "# Tutanota CLAUDE.md\n\nProject conventions live here.\n"
    (tmp_path / "CLAUDE.md").write_text(upstream)

    inject_claude_md_section(str(tmp_path), REPO_ID)
    body = (tmp_path / "CLAUDE.md").read_text()

    # Upstream content preserved verbatim.
    assert body.startswith(upstream)
    # Our section appended after.
    assert _CLAUDE_MD_SENTINEL_START in body
    assert _CLAUDE_MD_SENTINEL_END in body
    # No double-blank-line accumulation between upstream + our section.
    assert "\n\n\n" not in body
    # Order: upstream first, then our section.
    upstream_idx = body.index("Project conventions live here.")
    section_idx = body.index(_CLAUDE_MD_SENTINEL_START)
    assert upstream_idx < section_idx


def test_inject_replaces_in_place_when_section_exists(tmp_path):
    """Re-runs swap the section without duplicating. Repo identifier
    changes (e.g. clone w/ different remote) flow through cleanly."""

    (tmp_path / "CLAUDE.md").write_text(
        "# Header\n\n"
        f"{_CLAUDE_MD_SENTINEL_START}\nold body with old-identifier\n{_CLAUDE_MD_SENTINEL_END}\n"
        "## Trailing upstream section\n"
    )

    inject_claude_md_section(str(tmp_path), REPO_ID)
    body = (tmp_path / "CLAUDE.md").read_text()

    # Upstream content above + below preserved.
    assert body.startswith("# Header\n")
    assert "## Trailing upstream section" in body
    # Old body gone, new repo identifier present.
    assert "old body with old-identifier" not in body
    assert REPO_ID in body
    # Sentinels appear exactly once (no duplication on replace).
    assert body.count(_CLAUDE_MD_SENTINEL_START) == 1
    assert body.count(_CLAUDE_MD_SENTINEL_END) == 1


def test_inject_is_idempotent_across_repeat_calls(tmp_path):
    """Calling inject twice with the same identifier produces the same
    file — no growth, no drift."""

    inject_claude_md_section(str(tmp_path), REPO_ID)
    first = (tmp_path / "CLAUDE.md").read_text()

    inject_claude_md_section(str(tmp_path), REPO_ID)
    second = (tmp_path / "CLAUDE.md").read_text()

    assert first == second


def test_remove_strips_only_our_section(tmp_path):
    """remove_claude_md_section preserves upstream content. Useful when a
    run flips MCP off mid-flight — we don't leak our section into a
    non-MCP attempt."""

    upstream_before = "# Header\n\nUpstream paragraph.\n"
    upstream_after = "## Trailing\n\nMore upstream content.\n"
    (tmp_path / "CLAUDE.md").write_text(
        upstream_before
        + "\n"
        + f"{_CLAUDE_MD_SENTINEL_START}\nour body\n{_CLAUDE_MD_SENTINEL_END}\n"
        + upstream_after
    )

    remove_claude_md_section(str(tmp_path))
    body = (tmp_path / "CLAUDE.md").read_text()

    assert "Upstream paragraph." in body
    assert "More upstream content." in body
    assert _CLAUDE_MD_SENTINEL_START not in body
    assert "our body" not in body


def test_remove_deletes_file_when_only_our_section(tmp_path):
    """If our section is the entire file, removal deletes the file
    rather than leaving an empty husk. Avoids confusing follow-on tools
    that expect "no CLAUDE.md" semantics."""

    inject_claude_md_section(str(tmp_path), REPO_ID)
    assert (tmp_path / "CLAUDE.md").exists()

    remove_claude_md_section(str(tmp_path))
    assert not (tmp_path / "CLAUDE.md").exists()


def test_remove_is_noop_when_no_claude_md(tmp_path):
    # No file at all — must not raise.
    remove_claude_md_section(str(tmp_path))
    assert not (tmp_path / "CLAUDE.md").exists()


def test_remove_is_noop_when_no_sentinel_present(tmp_path):
    """Upstream-only CLAUDE.md (no Code Lexica section) is left
    untouched. Important: a run that disables MCP after a fresh clone
    should not nuke the upstream content."""

    upstream = "# Upstream\n\nNo Code Lexica section here.\n"
    (tmp_path / "CLAUDE.md").write_text(upstream)

    remove_claude_md_section(str(tmp_path))
    assert (tmp_path / "CLAUDE.md").read_text() == upstream
