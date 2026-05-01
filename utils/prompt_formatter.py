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
    ):
        """
        Args:
            prompt_template_path: Optional external template file. Overrides
                the inline default + nudge templates when set.
            mcp_prompt_nudge: When True, swap the inline default for the
                MCP-aware variant (Codebase context tools block + 7-step
                task list). cl-benchmark's run-form toggle drives this; the
                external template path takes precedence if both are set.
                Spec: cl-benchmark/docs/mcp-priming-spec.md.
            repo_identifier: Git remote URL (e.g.
                ``https://github.com/owner/repo.git``) substituted into the
                nudged template's ``{repo_identifier}`` placeholder. Inert
                for the default template, which doesn't reference MCP.
        """
        self.prompt_template_path = prompt_template_path
        self.mcp_prompt_nudge = mcp_prompt_nudge
        self.repo_identifier = repo_identifier
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

        Byte-for-byte equal to ``_default_template`` (upstream
        ``tool_use.yaml``) with two MCP-specific insertions:

        * Between the "Your task is to make the minimal changes…" line
          and the "Follow these steps to resolve the issue:" line, a
          ``Codebase context tools`` block names both Code Lexica MCP
          tools and describes when to call each, followed by the
          ``repoIdentifier`` directive.
        * As the new first step in upstream's numbered task list, a
          mandatory ``get_codebase_context`` call (with
          subagent-sharing nudge to avoid redundant re-fetches).
          Upstream's original 5 steps follow as steps 2-6, verbatim.

        Upstream's "Your thinking should be thorough…" tail is kept;
        no "Important notes" block is added — keeping the MCP variant
        as close to upstream as possible while still exercising the
        MCP tools. The implementation_guide tool is described in the
        tools block but isn't a hard step; the agent decides whether
        to call it based on the in-block prose ("Call BEFORE writing
        the fix when it touches business logic…").

        ``{repo_identifier}`` is substituted at format time. cl-benchmark
        threads the resolved git remote URL down through ``run_shard``;
        callers without an identifier get an empty string and the prompt
        still parses (the agent can still call ``git remote get-url
        origin`` itself). Spec:
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

Codebase context tools (call these ONCE per task — share the result, don't re-fetch):
  - mcp__code-lexica__get_codebase_context — architecture, code map, conventions.
    Call BEFORE any grep/find/Read or before delegating to a subagent.
  - mcp__code-lexica__get_implementation_guide — workflow recipes + API/data-model reference.
    Call BEFORE writing the fix when it touches business logic, endpoints, models, or routes.

For all Code Lexica calls, pass repoIdentifier="{repo_identifier}".

Follow these steps to resolve the issue:
1. Call mcp__code-lexica__get_codebase_context ONCE at the start to fetch codebase context. When you delegate to a subagent, INCLUDE the returned context in the subagent brief verbatim — do not have subagents call get_codebase_context themselves; it would re-fetch the same data and bloat the conversation.
2. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
3. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
4. Edit the source code of the repo to resolve the issue
5. Rerun your reproduce script and confirm that the error is fixed!
6. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""

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
        # SWE-bench Pro ships ``requirements`` + ``interface`` fields
        # alongside ``problem_statement``. Upstream Pro-os's
        # ``helper_code/create_problem_statement.py`` concatenates all
        # three into the prompt body — mirror that here so our agent
        # sees the same contract as SWE-agent on Pro runs. Lite rows +
        # older dataset snapshots don't carry these keys; the helper
        # falls through to just ``problem_statement`` unchanged.
        issue_description = _concat_problem_statement(
            problem_statement=problem_statement,
            requirements=instance.get("requirements"),
            interface=instance.get("interface"),
        )
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
