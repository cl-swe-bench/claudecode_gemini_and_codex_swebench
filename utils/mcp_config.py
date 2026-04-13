"""MCP server configuration for Code Lexica A/B benchmarking."""

import json
import re
from pathlib import Path
from typing import Optional


class McpConfigManager:
    """Manages per-repo MCP token resolution and .mcp.json injection."""

    def __init__(self, registry_path: str = None):
        if registry_path is None:
            # Default to configs/code_lexica_repos.json relative to project root
            registry_path = Path(__file__).parent.parent / "configs" / "code_lexica_repos.json"
        self.registry = self._load_registry(registry_path)

    def _load_registry(self, path) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"MCP registry not found: {p}\n"
                "Copy configs/code_lexica_repos.example.json to "
                "configs/code_lexica_repos.json and fill in your tokens."
            )
        return json.loads(p.read_text())

    def _get_all_repos(self) -> dict:
        """Return a merged dict of all repos across all benchmark sections.

        Supports both the new sectioned format (swe_bench_lite, swe_bench_pro)
        and the legacy flat 'repos' key for backwards compatibility.
        """
        merged = {}
        for key in ("repos", "swe_bench_lite", "swe_bench_pro"):
            section = self.registry.get(key)
            if isinstance(section, dict):
                merged.update(section)
        return merged

    def _get_section_repos(self, section: str) -> dict:
        """Return repos for a specific benchmark section."""
        return self.registry.get(section, {})

    @staticmethod
    def repo_from_instance_id(instance_id: str) -> str:
        """Parse a SWE-bench instance_id to a repo slug.

        Handles both Lite and Pro formats:
          Lite: 'django__django-11133'           -> 'django/django'
          Pro:  'instance_ansible__ansible-abc123-vdef456' -> 'ansible/ansible'
        """
        # Strip the 'instance_' prefix used in Pro IDs
        cleaned = instance_id.removeprefix("instance_")

        # Split on '__' which separates owner from repo-suffix
        parts = cleaned.split("__")
        if len(parts) < 2:
            return cleaned.replace("__", "/")

        owner = parts[0]
        repo_and_suffix = parts[1]

        # For Pro IDs the suffix is a commit hash (hex), possibly followed by
        # -v<hex>.  For Lite IDs it ends with -<issue_number> (digits only).
        #
        # The repo name in the instance_id uses the same casing as the real
        # repo name, so we can match against the known repos in our config
        # to find an exact boundary.  When that fails, we fall back to
        # heuristic stripping of hash/number suffixes.
        segments = repo_and_suffix.split("-")

        # Find the boundary between repo name segments and ID segments.
        # Repo names can contain hyphens (e.g. scikit-learn, element-web).
        # We strip from the right: any segment that looks like a hash/number.
        idx = len(segments) - 1
        _hex_chars = set("0123456789abcdef")
        while idx > 0:
            seg = segments[idx]
            # Pure digits (Lite issue number)
            if seg.isdigit():
                idx -= 1
                continue
            # Hex string (Pro commit hash fragment) -- at least 6 chars
            if len(seg) >= 6 and all(c in _hex_chars for c in seg):
                idx -= 1
                continue
            # v-prefixed hash or 'vnan' (Pro version suffix)
            if seg.startswith("v") and len(seg) >= 2 and all(
                c in _hex_chars for c in seg[1:]
            ):
                idx -= 1
                continue
            if seg == "vnan":
                idx -= 1
                continue
            break

        repo = "-".join(segments[: idx + 1])
        return f"{owner}/{repo}"

    @staticmethod
    def _is_valid_token(token: Optional[str]) -> bool:
        """Check if a token is a real token (not blank or placeholder)."""
        if not token or not token.strip():
            return False
        # Reject common placeholder values
        placeholders = {"cl_your_token_here", "cl_...", "cl_"}
        return token.strip() not in placeholders and token.startswith("cl_") and len(token) > 4

    def get_configured_repos(self) -> set:
        """Return the set of repos that have valid (non-placeholder) tokens."""
        return {
            repo for repo, token in self._get_all_repos().items()
            if self._is_valid_token(token)
        }

    def get_token(self, instance_id: str) -> Optional[str]:
        """Get the MCP token for the repo in this instance.

        Searches all benchmark sections. Returns None if the token is
        missing, blank, or a placeholder.
        """
        repo = self.repo_from_instance_id(instance_id)
        token = self._get_all_repos().get(repo)
        if not self._is_valid_token(token):
            return None
        return token

    def inject_mcp_json(self, instance_id: str, task_dir: str) -> bool:
        """Write .mcp.json into the task directory.

        Returns True if a token was found and the file was written.
        """
        token = self.get_token(instance_id)
        if not token:
            repo = self.repo_from_instance_id(instance_id)
            print(f"  WARNING: No MCP token for repo '{repo}', running without MCP")
            return False

        mcp_config = {
            "mcpServers": {
                "code-lexica": {
                    "type": "http",
                    "url": self.registry["mcp_url"],
                    "headers": {
                        "Authorization": f"Bearer {token}"
                    },
                    "serverUseInstructions": (
                        "Code Lexica provides pre-indexed codebase intelligence "
                        "including architecture, code maps, conventions, and "
                        "implementation patterns. If you are working in a git "
                        "repository, run git remote get-url origin and pass the "
                        "result as the repoIdentifier parameter on all Code Lexica "
                        "tool calls — this ensures you receive context specific to "
                        "your repository within the broader project. BEFORE exploring "
                        "the codebase with grep, glob, semantic search, file reads, "
                        "or explore subagents, call get_codebase_context with your "
                        "repoIdentifier to get the project structure, coding "
                        "conventions, and system architecture. This call does not "
                        "replace the need to grep and read dozens of files to "
                        "understand the project, but it narrows and focuses the "
                        "search space to the most relevant files and directories so "
                        "you can use fewer tokens and waste less time reading "
                        "irrelevant files. Call get_implementation_guide before "
                        "implementing features, adding endpoints, creating models, "
                        "or modifying business logic. Call get_testing_guide before "
                        "writing any tests. Always call Code Lexica tools BEFORE "
                        "delegating to subagents — subagents may not have access to "
                        "these tools, so fetch the context first and include it in "
                        "your subagent instructions."
                    )
                }
            }
        }
        mcp_path = Path(task_dir) / ".mcp.json"
        mcp_path.write_text(json.dumps(mcp_config, indent=2))
        repo = self.repo_from_instance_id(instance_id)
        print(f"  MCP enabled for {repo}")
        return True

    @staticmethod
    def remove_mcp_json(task_dir: str):
        """Remove .mcp.json from task directory if present."""
        mcp_path = Path(task_dir) / ".mcp.json"
        mcp_path.unlink(missing_ok=True)
