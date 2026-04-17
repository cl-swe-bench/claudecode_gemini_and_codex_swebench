import os
import json
import subprocess
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class ClaudeCodeInterface:
    """Interface for interacting with Claude Code CLI."""

    def __init__(self, timeout_s: int = 900):
        """Ensure the Claude CLI is available on the system.

        Args:
            timeout_s: Subprocess timeout for each ``claude -p`` invocation.
                Worker callers plumb this through from ``run_shard``; the
                standalone CLI picks up the default.
        """
        self.timeout_s = timeout_s
        try:
            result = subprocess.run([
                "claude", "--version"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "Claude CLI not found. Please ensure 'claude' is installed and in PATH"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Please ensure 'claude' is installed and in PATH"
            )

    def execute_code_cli(self, prompt: str, cwd: str, model: str = None) -> Dict[str, any]:
        """Execute Claude Code via CLI and capture the response.

        Args:
            prompt: The prompt to send to Claude.
            cwd: Working directory to execute in.
            model: Optional model to use (e.g., 'opus-4.1', 'sonnet-3.7').
        """
        try:
            # Save the current directory
            original_cwd = os.getcwd()

            # Change to the working directory
            os.chdir(cwd)

            # Build command with optional model parameter
            # Use -p (print mode) with --output-format json for structured output
            cmd = ["claude", "-p", "--dangerously-skip-permissions", "--output-format", "json"]
            if model:
                cmd.extend(["--model", model])

            # Execute claude command with the prompt via stdin
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            # Restore original directory
            os.chdir(original_cwd)

            # Parse JSON output for token usage and metadata
            # Claude Code -p --output-format json returns:
            # {
            #   "result": "...", "num_turns": N, "duration_ms": N,
            #   "duration_api_ms": N, "is_error": bool, "session_id": "...",
            #   "total_cost_usd": N.NN,
            #   "usage": { "input_tokens": N, "output_tokens": N,
            #              "cache_creation_input_tokens": N, "cache_read_input_tokens": N }
            # }
            token_usage = {}
            if result.stdout:
                try:
                    output_data = json.loads(result.stdout)
                    if isinstance(output_data, dict):
                        usage = output_data.get("usage") or {}
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        token_usage = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        }
                        if output_data.get("total_cost_usd") is not None:
                            token_usage["cost_usd"] = output_data["total_cost_usd"]
                        if output_data.get("num_turns") is not None:
                            token_usage["num_turns"] = output_data["num_turns"]
                        if output_data.get("duration_ms") is not None:
                            token_usage["duration_ms"] = output_data["duration_ms"]
                        if output_data.get("duration_api_ms") is not None:
                            token_usage["duration_api_ms"] = output_data["duration_api_ms"]
                        if output_data.get("session_id") is not None:
                            token_usage["session_id"] = output_data["session_id"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # Fall back to no token tracking

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "token_usage": token_usage,
            }

        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout_s} seconds",
                "returncode": -1,
                "token_usage": {},
                "timed_out": True,
            }
        except Exception as e:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "token_usage": {},
            }

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Claude's response."""
        # This will be implemented by patch_extractor.py
        # For now, return empty list
        return []

    @staticmethod
    def _to_token_usage(usage: Dict) -> Dict[str, int]:
        """Normalize Claude's token_usage dict to the cl-benchmark library schema.

        Returned keys match ``cl_benchmark_core.schemas.library.InstanceTokenUsage``:
        ``input``, ``output``, ``cache_creation``, ``cache_read``.
        """
        usage = usage or {}
        return {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
            "cache_creation": int(usage.get("cache_creation_tokens", 0) or 0),
            "cache_read": int(usage.get("cache_read_tokens", 0) or 0),
        }