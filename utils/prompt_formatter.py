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
    behavior for Lite runs. Whitespace-only values are treated as empty.
    """
    parts = [problem_statement or ""]
    req = (requirements or "").strip()
    if req:
        parts.append(f"\n\nRequirements:\n{req}")
    iface = (interface or "").strip()
    if iface:
        parts.append(f"\n\nNew interfaces introduced:\n{iface}")
    return "".join(parts)


class PromptFormatter:
    """Format SWE-bench issues into prompts for Claude Code."""
    
    def __init__(self, prompt_template_path: Optional[str] = None):
        self.prompt_template_path = prompt_template_path
        self.base_template = self._load_base_template()
        
    def _load_base_template(self) -> str:
        """Load the base prompt template."""
        if self.prompt_template_path:
            try:
                with open(self.prompt_template_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                pass
        
        # Default template.
        #
        # Aligned with SWE-bench_Pro-os's `tool_use.yaml` instance
        # template in two ways:
        #   * No "being evaluated on SWE-bench" framing. Pro-os never
        #     tells its agent the task is a benchmark; doing so leaks
        #     meta-context and may nudge the model to behave
        #     differently (more cautious, more verbose) than the
        #     reference runs. Opening sentence just describes the
        #     workspace.
        #   * Don't-touch-tests nudge lifted from Pro-os verbatim.
        #     The statement is a directional nudge (Pro-os's flow
        #     doesn't actually apply test_patch before the agent
        #     runs either — ours doesn't either; the evaluator
        #     re-applies test_patch at grading time in both flows).
        #     Dropping the old "tests should pass after applying
        #     your fix" note that conflicted with the nudge.
        return """You have access to a repository with a software issue that needs to be fixed.

Repository: {repo_name}
Issue: {issue_title}

Issue Description:
{issue_description}

I've already taken care of all changes to any of the test files described in the issue description. This means you DON'T have to modify the testing logic or any of the tests in any way. Your task is to make the minimal changes to non-test files in the repository to satisfy the issue description.

Your task:
1. Understand the issue by carefully reading the description
2. Search the codebase to find relevant files using grep, find, or other search tools
3. Analyze the code to understand the root cause
4. Generate a fix that resolves the issue
5. Ensure your fix doesn't break existing functionality

Important notes:
- Focus on making minimal, targeted changes
- Consider edge cases and potential side effects
- Output clear file edits showing exactly what needs to be changed

Base directory: {base_path}
"""
    
    def format_issue(self, instance: Dict) -> str:
        """Format a SWE-bench instance into a prompt for Claude Code."""
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

        # Format the prompt
        base_path = Path(tempfile.gettempdir()) / f"swe_bench_{instance_id}"

        prompt = self.base_template.format(
            repo_name=repo_name,
            issue_title=issue_title,
            issue_description=issue_description,
            base_path=str(base_path),
            instance_id=instance_id,
            base_commit=base_commit,
        )

        # Add any hints if available
        if "hints_text" in instance and instance["hints_text"]:
            prompt += f"\n\nHints:\n{instance['hints_text']}"

        return prompt
    
    def format_for_cli(self, instance: Dict) -> str:
        """Format the prompt for Claude Code CLI execution."""
        base_prompt = self.format_issue(instance)

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
