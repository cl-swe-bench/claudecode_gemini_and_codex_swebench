import tempfile
from pathlib import Path
from typing import Dict, Optional


def _concat_problem_statement(
    *,
    problem_statement: str,
    requirements: Optional[str],
    interface: Optional[str],
) -> str:
    """Mirror SWE-bench_Pro-os's ``create_problem_statement`` concat so
    our agent sees the same contract SWE-agent does on Pro runs.

    Upstream (``SWE-bench_Pro-os/helper_code/create_problem_statement.py``)
    builds the string as::

        {problem_statement}

        Requirements:
        {requirements}

        New interfaces introduced:
        {interface}

    Pro ships both ``requirements`` (~3.6 KB of contract prose) and
    ``interface`` (~1 KB of API-shape prose) populated on every row. Lite
    and older Pro snapshots don't carry them, so missing / empty fields
    fall through to just ``problem_statement`` unchanged — no change in
    behavior for Lite runs. Whitespace-only values are treated as
    absent (no empty section header), but populated values are
    substituted verbatim — including any leading / trailing whitespace
    the dataset row carries. Upstream's ``create_problem_statement``
    f-strings the values straight in; we mirror that for byte-for-byte
    parity with the Pro-os baseline.
    """
    parts = [problem_statement or ""]
    if requirements and requirements.strip():
        parts.append(f"\n\nRequirements:\n{requirements}")
    if interface and interface.strip():
        parts.append(f"\n\nNew interfaces introduced:\n{interface}")
    return "".join(parts)


class PromptFormatter:
    """Format SWE-bench issues into prompts for Claude Code."""

    def __init__(
        self,
        prompt_template_path: Optional[str] = None,
        mcp_prompt_nudge: bool = False,
        repo_identifier: Optional[str] = None,
        commit_hash: Optional[str] = None,
    ):
        """
        Args:
            prompt_template_path: Optional external template file. Overrides
                the inline default + nudge templates when set.
            mcp_prompt_nudge: When True, swap the inline default for the
                MCP-aware variant (Codebase context tool block + 5-step
                task list). cl-benchmark's run-form toggle drives this; the
                external template path takes precedence if both are set.
                Spec: cl-benchmark/docs/mcp-priming-spec.md.
            repo_identifier: Git remote URL (e.g.
                ``https://github.com/owner/repo.git``) substituted into the
                nudged template's ``{repo_identifier}`` placeholder. Inert
                for the default template, which doesn't reference MCP.
            commit_hash: Full 40-char base_commit SHA pinned by the dataset
                row, substituted into the nudged template's
                ``{commit_hash}`` placeholder. Pins the MCP response to the
                exact codebase state the agent is operating against.
                Empty string when unknown — template still parses; the MCP
                server treats missing commits as "current state".
        """
        self.prompt_template_path = prompt_template_path
        self.mcp_prompt_nudge = mcp_prompt_nudge
        self.repo_identifier = repo_identifier
        self.commit_hash = commit_hash
        self.base_template = self._load_base_template()

    def _load_base_template(self) -> str:
        """Load the base prompt template."""
        if self.prompt_template_path:
            try:
                with open(self.prompt_template_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                pass

        if self.mcp_prompt_nudge:
            return self._mcp_nudge_template()

        # Byte-for-byte mirror of SWE-bench_Pro-os's tool_use.yaml
        # `instance_template` (the config Pro-os's README points users
        # at: ``--config config/tool_use.yaml``). The only deviations
        # from upstream are functionally-required substitutions:
        #
        #   * upstream ``{{working_dir}}`` → our ``{base_path}``,
        #     because cl-benchmark runs the agent locally outside
        #     swe-agent's docker container so the repo lives at the
        #     actual cloned path rather than at ``/<repo_name>``.
        #   * upstream ``{{problem_statement}}`` → our
        #     ``{issue_description}``, where ``issue_description`` is
        #     the output of ``_concat_problem_statement()`` — itself
        #     a byte-for-byte mirror of Pro-os's
        #     ``create_problem_statement()``.
        #
        # Anything else (Python-script step, "non-tests files" with the
        # extra ``s``, the "Your thinking should be thorough" tail) is
        # kept verbatim. Drift from this template = drift from the
        # published Pro-os baseline; scores stop being apples-to-apples
        # comparable. Source:
        # SWE-bench_Pro-os/SWE-agent/config/tool_use.yaml.
        return """<uploaded_files>
{base_path}
</uploaded_files>
I've uploaded a python code repository in the directory {base_path}. Consider the following PR description:

<pr_description>
{issue_description}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {base_path} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""

    def _mcp_nudge_template(self) -> str:
        """MCP-aware variant of the default template — emitted when the
        caller passes ``mcp_prompt_nudge=True``.

        Equal to ``_default_template`` (upstream ``tool_use.yaml``)
        with two MCP-specific insertions and one upstream-step deletion:

        * Between the "Your task is to make the minimal changes…" line
          and the "Follow these steps to resolve the issue:" line, a
          ``Codebase context tool`` block names the
          ``get_codebase_context`` Code Lexica MCP tool and describes
          when to call it, followed by the ``repoIdentifier`` +
          ``commitHash`` directive (both required parameters that pin
          the response to the exact repo and commit being worked on).
        * The numbered task list is rebuilt around the MCP call: a
          mandatory ``get_codebase_context`` step is prepended as
          step 1 (with subagent-sharing nudge to avoid redundant
          re-fetches), and upstream's original "find and read code
          relevant to the <pr_description>" step is dropped because
          ``get_codebase_context`` supersedes it. Upstream's remaining
          4 steps (reproduce, edit, rerun, edge-cases) follow as
          steps 2-5, verbatim.

        Upstream's "Your thinking should be thorough…" tail is kept;
        no "Important notes" block is added — keeping the MCP variant
        as close to upstream as possible while still exercising the
        MCP tool. The Code Lexica server has additional tools
        (``get_implementation_guide`` / ``get_testing_guide``) but
        the prompt + CLAUDE.md template intentionally don't promote
        them — early runs showed agents over-fetching when multiple
        tools were surfaced at once, so we narrowed the nudge to the
        single highest-value tool.

        ``{repo_identifier}`` + ``{commit_hash}`` are substituted at
        format time. cl-benchmark threads the resolved git remote URL +
        the dataset row's ``base_commit`` down through ``run_shard``;
        callers without one get an empty string and the prompt still
        parses (the agent can still call ``git remote get-url origin``
        / ``git rev-parse HEAD`` itself). Spec:
        cl-benchmark/docs/mcp-priming-spec.md (Prompt nudge template).
        """
        return """<uploaded_files>
{base_path}
</uploaded_files>
I've uploaded a python code repository in the directory {base_path}. Consider the following PR description:

<pr_description>
{issue_description}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {base_path} directory to ensure the <pr_description> is satisfied.

Codebase context tool (call ONCE per task — share the result, don't re-fetch):
  - mcp__code-lexica__get_codebase_context — architecture, code map, conventions, PRE-FILTERED by the server to the parts relevant to your specific task. Tells you which files/directories are relevant + how they're named so you can skip dead-end reads. Call BEFORE any grep/find/Read or before delegating to a subagent.

For all Code Lexica calls, pass repoIdentifier="{repo_identifier}", commitHash="{commit_hash}", and taskPrompt = the EXACT, COMPLETE text inside the <pr_description> block above, copied verbatim. Do NOT summarize, paraphrase, shorten, or re-word it. The first two pin the response to the exact repo + commit being worked on; taskPrompt drives the server-side relevance filter, which degrades on a lossy summary. All three are required for a task-tailored response.

Follow these steps to resolve the issue:
1. Call mcp__code-lexica__get_codebase_context ONCE at the start to fetch task-relevant codebase context. Pass the <pr_description> body verbatim as taskPrompt — copy it exactly, do not summarize or paraphrase — so the server filters to files relevant to this issue. USE the returned context to direct your subsequent reads and searches — don't ignore it and re-grep the codebase from scratch. When you delegate to a subagent, INCLUDE the returned context in the subagent brief verbatim — do not have subagents call get_codebase_context themselves; it would re-fetch the same data and bloat the conversation.
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""

    def build_issue_description(self, instance: Dict) -> str:
        """Return the exact ``<pr_description>`` body for an instance.

        SWE-bench Pro ships ``requirements`` + ``interface`` alongside
        ``problem_statement``; upstream Pro-os's
        ``helper_code/create_problem_statement.py`` concatenates all
        three. We mirror that (via ``_concat_problem_statement``) so our
        agent sees the same contract SWE-agent does. Lite rows + older
        snapshots lack the extra keys and fall through to just
        ``problem_statement``.

        Exposed as a method (not inlined) so callers that need the
        verbatim task body without rendering the whole prompt — notably
        the Code Lexica ``taskPrompt`` pin in
        ``code_swe_agent.process_instance`` — get a value byte-identical
        to the ``<pr_description>`` block ``format_issue`` emits.
        """
        return _concat_problem_statement(
            problem_statement=instance.get("problem_statement", ""),
            requirements=instance.get("requirements"),
            interface=instance.get("interface"),
        )

    def format_issue(
        self, instance: Dict, *, base_path: Optional[str] = None
    ) -> str:
        """Format a SWE-bench instance into a prompt for Claude Code.

        ``base_path`` overrides the default ``<tempdir>/swe_bench_<iid>``
        path that the formatter would otherwise compute. Pass the actual
        cloned path here so the prompt's ``Base directory:`` line matches
        the agent's real cwd — matters when the caller adds a per-attempt
        suffix to the cwd (cl-benchmark's worker does this for sample
        isolation; see ``setup_repository``'s ``cwd_suffix`` arg).
        Default ``None`` reproduces legacy behavior for direct callers.
        """
        # Extract key information from the instance
        repo_name = instance.get("repo", "")
        problem_statement = instance.get("problem_statement", "")
        issue_title = problem_statement.split('\n')[0]
        # The exact ``<pr_description>`` body (problem_statement +
        # Requirements + interface). Single source so the pinned
        # ``taskPrompt`` (Code Lexica hook) is byte-identical to what
        # the agent sees here. See ``build_issue_description``.
        issue_description = self.build_issue_description(instance)
        base_commit = instance.get("base_commit", "")

        # Get instance_id for tracking
        instance_id = instance.get("instance_id", "")

        # Default base_path matches setup_repository's cwd_suffix=None
        # path so direct CLI users still see the legacy ``Base directory:``
        # line. cl-benchmark callers pass the resolved suffixed path.
        if base_path is None:
            base_path = str(Path(tempfile.gettempdir()) / f"swe_bench_{instance_id}")

        prompt = self.base_template.format(
            repo_name=repo_name,
            issue_title=issue_title,
            issue_description=issue_description,
            base_path=base_path,
            instance_id=instance_id,
            base_commit=base_commit,
            repo_identifier=self.repo_identifier or "",
            commit_hash=self.commit_hash or "",
        )

        # Add any hints if available
        if "hints_text" in instance and instance["hints_text"]:
            prompt += f"\n\nHints:\n{instance['hints_text']}"

        return prompt

    def format_for_cli(
        self, instance: Dict, *, base_path: Optional[str] = None
    ) -> str:
        """Format the prompt for Claude Code CLI execution.

        See ``format_issue`` for the ``base_path`` override semantics.
        """
        base_prompt = self.format_issue(instance, base_path=base_path)

        # Return the raw prompt without escaping for CLI input
        return base_prompt
    
    def extract_instance_info(self, instance: Dict) -> Dict:
        """Extract key information from a SWE-bench instance."""
        return {
            "instance_id": instance.get("instance_id", ""),
            "repo": instance.get("repo", ""),
            "version": instance.get("version", ""),
            "base_commit": instance.get("base_commit", ""),
            "problem_statement": instance.get("problem_statement", ""),
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance.get("created_at", ""),
            "test_patch": instance.get("test_patch", ""),
            "environment_setup_commit": instance.get("environment_setup_commit", "")
        }
