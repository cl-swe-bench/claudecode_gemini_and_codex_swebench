"""MCP server configuration for Code Lexica A/B benchmarking."""

import json
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

    def repo_from_instance_id(self, instance_id: str) -> str:
        """Parse SWE-bench instance_id to repo slug.

        Examples:
            'django__django-11133' -> 'django/django'
            'scikit-learn__scikit-learn-12345' -> 'scikit-learn/scikit-learn'
            'pytest-dev__pytest-7432' -> 'pytest-dev/pytest'
        """
        # Split on '__' which separates owner from repo-issue_number
        parts = instance_id.split("__")
        owner = parts[0]
        repo_and_num = parts[1]
        # Remove trailing -<issue_number> by splitting from the right
        # Handle repos with hyphens (e.g., scikit-learn) by only removing
        # the last numeric segment
        segments = repo_and_num.split("-")
        # Walk backwards to find where the issue number starts
        # Issue numbers are purely numeric
        idx = len(segments) - 1
        while idx > 0 and segments[idx].isdigit():
            idx -= 1
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
            repo for repo, token in self.registry.get("repos", {}).items()
            if self._is_valid_token(token)
        }

    def get_token(self, instance_id: str) -> Optional[str]:
        """Get the MCP token for the repo in this instance.

        Returns None if the token is missing, blank, or a placeholder.
        """
        repo = self.repo_from_instance_id(instance_id)
        token = self.registry.get("repos", {}).get(repo)
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
                    }
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
