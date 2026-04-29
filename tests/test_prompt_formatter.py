"""Tests for prompt_formatter — mirrors SWE-bench_Pro-os concat.

The core contract under test: when the instance dict carries
``requirements`` and/or ``interface`` (Pro-style rows), the formatted
prompt inlines them under ``Requirements:`` / ``New interfaces
introduced:`` headings the same way
``SWE-bench_Pro-os/helper_code/create_problem_statement.py`` does.
Missing or empty values fall through — Lite runs keep working.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompt_formatter import PromptFormatter, _concat_problem_statement  # noqa: E402


# -------------------- _concat_problem_statement --------------------


def test_concat_includes_requirements_and_interface_when_both_present():
    out = _concat_problem_statement(
        problem_statement="Fix the bug in get_flag.",
        requirements="- Must preserve backwards compatibility\n- Handle nil pointer",
        interface="Type: Method\nName: GetFlag\nPath: internal/flipt/get_flag.go",
    )
    # Matches Pro-os's exact section structure.
    assert "Fix the bug in get_flag." in out
    assert "\n\nRequirements:\n- Must preserve backwards compatibility" in out
    assert "\n\nNew interfaces introduced:\nType: Method" in out
    # Order: problem_statement → Requirements → New interfaces introduced.
    ps_idx = out.index("Fix the bug")
    req_idx = out.index("Requirements:")
    iface_idx = out.index("New interfaces introduced:")
    assert ps_idx < req_idx < iface_idx


def test_concat_falls_through_when_fields_absent():
    """Lite / older-snapshot shape — no ``requirements``, no ``interface``
    keys. Output is the unchanged problem_statement."""
    out = _concat_problem_statement(
        problem_statement="Fix the bug.",
        requirements=None,
        interface=None,
    )
    assert out == "Fix the bug."


def test_concat_treats_empty_and_whitespace_as_absent():
    """Pro-os dataset occasionally ships the keys populated with empty
    strings — don't inject empty ``Requirements:`` blocks."""
    out = _concat_problem_statement(
        problem_statement="Fix the bug.",
        requirements="",
        interface="   \n  ",
    )
    assert out == "Fix the bug."


def test_concat_omits_only_the_missing_section():
    """Partial population — requirements but no interface (or vice-versa)
    — emits just the populated section."""
    out_req = _concat_problem_statement(
        problem_statement="PS",
        requirements="R1",
        interface=None,
    )
    assert "Requirements:\nR1" in out_req
    assert "New interfaces introduced:" not in out_req

    out_iface = _concat_problem_statement(
        problem_statement="PS",
        requirements=None,
        interface="I1",
    )
    assert "New interfaces introduced:\nI1" in out_iface
    assert "Requirements:" not in out_iface


# -------------------- PromptFormatter.format_issue integration --------------------


def test_format_issue_inlines_requirements_and_interface_for_pro_rows():
    """Pro-style instance dict — the final CLI prompt carries the
    requirements + interface sections in the issue-description body."""
    formatter = PromptFormatter()
    instance = {
        "instance_id": "instance_flipt-io__flipt-abc123",
        "repo": "flipt-io/flipt",
        "base_commit": "deadbeef",
        "problem_statement": "Fix GetFlag when flag is disabled.",
        "requirements": "- Return nil\n- Log warning",
        "interface": "Type: Method\nName: GetFlag",
    }
    prompt = formatter.format_issue(instance)
    assert "Fix GetFlag when flag is disabled." in prompt
    assert "Requirements:\n- Return nil\n- Log warning" in prompt
    assert "New interfaces introduced:\nType: Method\nName: GetFlag" in prompt


def test_format_issue_unchanged_for_lite_rows():
    """Lite row (no requirements / interface keys at all). The prompt
    body still contains the problem_statement — and does NOT contain
    a stray ``Requirements:`` or ``New interfaces introduced:`` header
    for empty data."""
    formatter = PromptFormatter()
    instance = {
        "instance_id": "astropy__astropy-12345",
        "repo": "astropy/astropy",
        "base_commit": "deadbeef",
        "problem_statement": "Fix FITS header parsing.",
    }
    prompt = formatter.format_issue(instance)
    assert "Fix FITS header parsing." in prompt
    assert "Requirements:" not in prompt
    assert "New interfaces introduced:" not in prompt


def test_format_issue_keeps_hints_text_appended_after_enriched_body():
    """``hints_text`` append happens *after* the template render, so it
    should still land at the end of the prompt — after the enriched
    problem-statement body — when all three Pro fields + hints are
    present (rare but legal)."""
    formatter = PromptFormatter()
    instance = {
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "PS",
        "requirements": "R",
        "interface": "I",
        "hints_text": "Look at the Foo class",
    }
    prompt = formatter.format_issue(instance)
    # Order: PS → Requirements → New interfaces → hints.
    assert prompt.index("PS") < prompt.index("Requirements:") < prompt.index(
        "New interfaces introduced:"
    ) < prompt.index("Hints:")


def test_format_issue_drops_being_evaluated_framing():
    """Pro-os never tells its agent the task is a benchmark — our
    template dropped ``You are being evaluated on SWE-bench`` to align.
    Regression guard against the string sneaking back in via a future
    template edit."""
    formatter = PromptFormatter()
    prompt = formatter.format_issue({
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "Fix the bug.",
    })
    assert "being evaluated" not in prompt.lower()
    assert "swe-bench" not in prompt.lower()


def test_format_issue_includes_dont_touch_tests_nudge():
    """Matches Pro-os's ``tool_use.yaml`` directive: test files are
    off-limits to the agent. Regression guard against the nudge being
    dropped in a future edit."""
    formatter = PromptFormatter()
    prompt = formatter.format_issue({
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "Fix the bug.",
    })
    # Two-part nudge: claim tests are already handled, then the
    # explicit "don't modify tests" directive.
    assert "I've already taken care of" in prompt
    assert "DON'T have to modify the testing logic" in prompt
    assert "non-test files" in prompt


def test_format_issue_drops_conflicting_tests_should_pass_note():
    """The old Important-notes line ``The tests should pass after
    applying your fix`` conflicted with the don't-touch-tests nudge
    (pass-tests → maybe edit tests). Regression guard that the line
    stays removed."""
    formatter = PromptFormatter()
    prompt = formatter.format_issue({
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "Fix the bug.",
    })
    assert "tests should pass after applying" not in prompt


def test_format_issue_issue_title_is_first_line_of_problem_statement_only():
    """``issue_title`` in the template is still the first line of the
    *original* problem_statement — enriching the body shouldn't
    accidentally promote "Requirements:" into the title slot."""
    formatter = PromptFormatter()
    instance = {
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "One-line title\n\nDetails here.",
        "requirements": "R",
        "interface": "I",
    }
    prompt = formatter.format_issue(instance)
    # The first line of problem_statement appears where the title goes
    # (template: ``Issue: {issue_title}``), not a header from the
    # enriched body.
    assert "Issue: One-line title" in prompt


# -------------------- mcp_prompt_nudge variant --------------------


def _basic_instance() -> dict:
    return {
        "instance_id": "inst_1",
        "repo": "acme/widget",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug\n\nDetails here.",
    }


def test_default_template_unchanged_when_nudge_off():
    """Regression guard: nudge=False preserves the byte-shape of the
    cc-46-full-run-regen baseline. Any drift here breaks comparability
    with non-MCP runs.
    """
    formatter = PromptFormatter()
    prompt = formatter.format_issue(_basic_instance())
    # Sentinel substrings unique to the default 5-step template.
    assert "1. Understand the issue by carefully reading the description" in prompt
    assert "2. Search the codebase to find relevant files using grep, find, or other search tools" in prompt
    # The nudged variant's preamble must NOT appear.
    assert "Codebase context tools" not in prompt
    assert "mcp__code-lexica" not in prompt


def test_nudge_template_inserts_codebase_context_block():
    formatter = PromptFormatter(
        mcp_prompt_nudge=True,
        repo_identifier="https://github.com/acme/widget.git",
    )
    prompt = formatter.format_issue(_basic_instance())
    # Block header (call-once + share-result framing) + both tool names land verbatim.
    assert (
        "Codebase context tools (call these ONCE per task — share the result, don't re-fetch):"
        in prompt
    )
    assert "mcp__code-lexica__get_codebase_context" in prompt
    assert "mcp__code-lexica__get_implementation_guide" in prompt
    # 7-step task list — the new MCP-first step + the get_implementation_guide step.
    # Step 2 says "ONCE at the start" and tells the agent to share via subagent
    # briefs rather than have subagents re-fetch.
    assert "2. Call mcp__code-lexica__get_codebase_context ONCE at the start" in prompt
    assert "INCLUDE the returned context in the subagent brief verbatim" in prompt
    assert "5. Call mcp__code-lexica__get_implementation_guide if your fix" in prompt
    assert "7. Ensure your fix doesn't break existing functionality" in prompt


def test_nudge_template_substitutes_repo_identifier():
    formatter = PromptFormatter(
        mcp_prompt_nudge=True,
        repo_identifier="https://github.com/foo/bar.git",
    )
    prompt = formatter.format_issue(_basic_instance())
    assert 'repoIdentifier="https://github.com/foo/bar.git"' in prompt
    # The {repo_identifier} placeholder is fully substituted (no curly braces).
    assert "{repo_identifier}" not in prompt


def test_nudge_template_handles_missing_repo_identifier():
    """No-identifier path renders an empty string in the placeholder
    rather than raising. The agent can still call ``git remote get-url
    origin`` itself when this happens — but the harness should always
    pass an identifier; this is a defensive fallback."""
    formatter = PromptFormatter(mcp_prompt_nudge=True)
    prompt = formatter.format_issue(_basic_instance())
    assert 'repoIdentifier=""' in prompt
    assert "{repo_identifier}" not in prompt


def test_external_template_path_takes_precedence_over_nudge(tmp_path):
    """When both ``prompt_template_path`` and ``mcp_prompt_nudge=True``
    are set, the external file wins. This keeps the Phase-7 codex/gemini
    custom-template path open without nudge interference."""
    custom = tmp_path / "custom.tmpl"
    custom.write_text("CUSTOM PROMPT for {repo_name}: {issue_title}")
    formatter = PromptFormatter(
        prompt_template_path=str(custom),
        mcp_prompt_nudge=True,
    )
    prompt = formatter.format_issue(_basic_instance())
    assert prompt.startswith("CUSTOM PROMPT for acme/widget")
    # Nudge content does NOT appear when external template is used.
    assert "mcp__code-lexica" not in prompt
