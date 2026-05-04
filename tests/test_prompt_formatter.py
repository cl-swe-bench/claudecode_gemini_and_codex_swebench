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
    dropped or paraphrased in a future edit. Strings are upstream's
    verbatim — including the ``non-tests files`` typo (extra ``s``)
    that we mirror for parity."""
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
    # Upstream's exact spelling — keep verbatim for byte-for-byte
    # parity with the published Pro-os baseline.
    assert "non-tests files" in prompt


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


def test_format_issue_problem_statement_appears_inside_pr_description_block():
    """The byte-for-byte upstream wrapper places ``{problem_statement}``
    inside ``<pr_description>...</pr_description>`` with no separate
    title slot — title is just the first line of the body. Regression
    guard against the wrapper drifting back to the old paraphrase that
    extracted ``Issue: ...`` into a header line.
    """
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
    # The body sits between the XML tags; the title is the first line
    # of the body, not a separate "Issue: ..." header.
    assert "<pr_description>\nOne-line title\n\nDetails here." in prompt
    # The old paraphrased header must not reappear.
    assert "Issue: One-line title" not in prompt
    assert "Repository: acme/widget" not in prompt


# -------------------- mcp_prompt_nudge variant --------------------


def _basic_instance() -> dict:
    return {
        "instance_id": "inst_1",
        "repo": "acme/widget",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug\n\nDetails here.",
    }


def test_default_template_unchanged_when_nudge_off():
    """Regression guard: nudge=False emits the upstream-parity wrapper.
    Sentinel substrings come from
    SWE-bench_Pro-os/SWE-agent/config/tool_use.yaml's instance_template
    verbatim — drift here breaks comparability with the published
    Pro-os baseline.
    """
    formatter = PromptFormatter()
    prompt = formatter.format_issue(_basic_instance())
    # Sentinel substrings unique to upstream's tool_use.yaml steps.
    assert (
        "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>"
        in prompt
    )
    assert (
        "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error"
        in prompt
    )
    assert "Your thinking should be thorough and so it's fine if it's very long." in prompt
    # The nudged variant's preamble must NOT appear.
    assert "Codebase context tools" not in prompt
    assert "mcp__code-lexica" not in prompt


def test_default_template_byte_for_byte_matches_upstream_tool_use_yaml():
    """Strong regression guard: the rendered default-template prompt
    must equal upstream's
    ``SWE-bench_Pro-os/SWE-agent/config/tool_use.yaml`` instance_template
    after substitution, byte-for-byte. The only deviations from
    upstream are functionally-required: ``{{working_dir}}`` →
    ``base_path`` (we run outside swe-agent's docker container) and
    ``{{problem_statement}}`` → ``issue_description`` (which equals
    upstream's ``create_problem_statement`` output for a Pro-shape row).

    If this test breaks, our scores are no longer apples-to-apples
    comparable to Pro-os's published baseline. Update upstream's
    template snapshot below ONLY when we deliberately re-sync to a
    new upstream revision.
    """
    formatter = PromptFormatter()
    instance = {
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "Fix the bug.",
        "requirements": "R1",
        "interface": "I1",
    }
    rendered = formatter.format_issue(instance, base_path="/app")

    # Upstream tool_use.yaml instance_template, with {{working_dir}} →
    # /app and {{problem_statement}} → create_problem_statement(row)
    # for the instance dict above.
    expected = (
        "<uploaded_files>\n"
        "/app\n"
        "</uploaded_files>\n"
        "I've uploaded a python code repository in the directory /app. Consider the following PR description:\n"
        "\n"
        "<pr_description>\n"
        "Fix the bug.\n"
        "\n"
        "Requirements:\n"
        "R1\n"
        "\n"
        "New interfaces introduced:\n"
        "I1\n"
        "</pr_description>\n"
        "\n"
        "Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n"
        "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        "Your task is to make the minimal changes to non-tests files in the /app directory to ensure the <pr_description> is satisfied.\n"
        "Follow these steps to resolve the issue:\n"
        "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\n"
        "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error\n"
        "3. Edit the source code of the repo to resolve the issue\n"
        "4. Rerun your reproduce script and confirm that the error is fixed!\n"
        "5. Think about edgecases and make sure your fix handles them as well\n"
        "Your thinking should be thorough and so it's fine if it's very long."
    )
    assert rendered == expected, (
        "Default-template render diverged from upstream tool_use.yaml. "
        "Diff (rendered → expected):\n"
        + "\n".join(
            f"  {i}: {r!r} != {e!r}"
            for i, (r, e) in enumerate(zip(rendered.splitlines(), expected.splitlines()))
            if r != e
        )
    )


def test_nudge_template_inserts_codebase_context_block():
    """The MCP-nudge variant is upstream's wrapper + 1 inserted block
    (Codebase context tools) + 1 prepended task-list step. Upstream's
    original 5 steps stay verbatim, renumbered to 2-6.
    """
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
    # New step 1 is the MCP context fetch; upstream's "find and read
    # code relevant to the <pr_description>" line is renumbered to step 2.
    assert "1. Call mcp__code-lexica__get_codebase_context ONCE at the start" in prompt
    assert "INCLUDE the returned context in the subagent brief verbatim" in prompt
    assert (
        "2. As a first step, it might be a good idea to find and read code relevant to the <pr_description>"
        in prompt
    )
    assert (
        "3. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error"
        in prompt
    )
    assert "6. Think about edgecases and make sure your fix handles them as well" in prompt
    # Upstream's tail is preserved.
    assert prompt.endswith(
        "Your thinking should be thorough and so it's fine if it's very long."
    )
    # The conditional implementation_guide step from the previous spec
    # is gone; the tool is mentioned in the tools block as available
    # but isn't a hard step.
    assert "5. Call mcp__code-lexica__get_implementation_guide" not in prompt
    # No "Important notes" block in the simplified variant.
    assert "Important notes:" not in prompt


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


def test_nudge_template_byte_for_byte_matches_spec():
    """Strong regression guard for the MCP nudge template's exact
    rendered shape. The opening through the "Your task is to make the
    minimal changes to non-tests files…" line is byte-for-byte upstream
    ``tool_use.yaml``; what follows is byte-for-byte the spec at
    ``cl-benchmark/docs/mcp-priming-spec.md`` "Prompt nudge template".

    If this test breaks, MCP-nudge runs are no longer cleanly comparable
    to either upstream-baseline runs (wrapper drift) or to prior
    nudge-on cohorts (nudge-content drift). Update the snapshot below
    ONLY when re-spec'ing the MCP nudge intentionally.
    """
    formatter = PromptFormatter(
        mcp_prompt_nudge=True,
        repo_identifier="https://github.com/acme/widget.git",
    )
    instance = {
        "instance_id": "x",
        "repo": "acme/widget",
        "base_commit": "",
        "problem_statement": "Fix the bug.",
        "requirements": "R1",
        "interface": "I1",
    }
    rendered = formatter.format_issue(instance, base_path="/app")

    expected = (
        "<uploaded_files>\n"
        "/app\n"
        "</uploaded_files>\n"
        "I've uploaded a python code repository in the directory /app. Consider the following PR description:\n"
        "\n"
        "<pr_description>\n"
        "Fix the bug.\n"
        "\n"
        "Requirements:\n"
        "R1\n"
        "\n"
        "New interfaces introduced:\n"
        "I1\n"
        "</pr_description>\n"
        "\n"
        "Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n"
        "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        "Your task is to make the minimal changes to non-tests files in the /app directory to ensure the <pr_description> is satisfied.\n"
        "\n"
        "Codebase context tools (call these ONCE per task — share the result, don't re-fetch):\n"
        "  - mcp__code-lexica__get_codebase_context — architecture, code map, conventions.\n"
        "    Call BEFORE any grep/find/Read or before delegating to a subagent.\n"
        "  - mcp__code-lexica__get_implementation_guide — workflow recipes + API/data-model reference.\n"
        "    Call BEFORE writing the fix when it touches business logic, endpoints, models, or routes.\n"
        "\n"
        'For all Code Lexica calls, pass repoIdentifier="https://github.com/acme/widget.git".\n'
        "\n"
        "Follow these steps to resolve the issue:\n"
        "1. Call mcp__code-lexica__get_codebase_context ONCE at the start to fetch codebase context. When you delegate to a subagent, INCLUDE the returned context in the subagent brief verbatim — do not have subagents call get_codebase_context themselves; it would re-fetch the same data and bloat the conversation.\n"
        "2. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\n"
        "3. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error\n"
        "4. Edit the source code of the repo to resolve the issue\n"
        "5. Rerun your reproduce script and confirm that the error is fixed!\n"
        "6. Think about edgecases and make sure your fix handles them as well\n"
        "Your thinking should be thorough and so it's fine if it's very long."
    )
    assert rendered == expected, (
        "MCP-nudge render diverged from the locked spec. "
        "Diff (rendered → expected):\n"
        + "\n".join(
            f"  {i}: {r!r} != {e!r}"
            for i, (r, e) in enumerate(zip(rendered.splitlines(), expected.splitlines()))
            if r != e
        )
    )


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
