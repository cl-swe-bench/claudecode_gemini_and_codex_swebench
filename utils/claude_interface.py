import os
import json
import subprocess
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv

from utils.retry import (
    ClaudeRateLimitDetector,
    RateLimitEvent,
    RateLimitPolicy,
    with_rate_limit_retry,
)

load_dotenv()


class ClaudeCodeInterface:
    """Interface for interacting with Claude Code CLI."""

    def __init__(
        self,
        timeout_s: int = 900,
        env_overrides: Optional[Dict[str, str]] = None,
        retry_policy: Optional[RateLimitPolicy] = None,
        on_rate_limit_retry: Optional[Callable[[RateLimitEvent], None]] = None,
    ):
        """Ensure the Claude CLI is available on the system.

        Args:
            timeout_s: Subprocess timeout for each ``claude -p`` invocation.
                Worker callers plumb this through from ``run_shard``; the
                standalone CLI picks up the default.
            env_overrides: Phase 4. Extra env vars to merge into the
                subprocess env — used by the worker to inject
                ``ANTHROPIC_API_KEY`` and to relocate ``HOME``/``XDG_CONFIG_HOME``
                to a per-shard sandbox. None (the default) leaves the
                current process env untouched.
            retry_policy / on_rate_limit_retry: Phase 4 rate-limit retry
                policy + callback. Defaults: 5 retries, exponential
                backoff with jitter, honoring parsed ``Retry-After``.
        """
        self.timeout_s = timeout_s
        self.env_overrides: Dict[str, str] = dict(env_overrides) if env_overrides else {}
        self.retry_policy = retry_policy or RateLimitPolicy()
        self.on_rate_limit_retry = on_rate_limit_retry
        self._detector = ClaudeRateLimitDetector()
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

    def execute_code_cli(self, prompt: str, cwd: str, model: str = None) -> Dict[str, Any]:
        """Execute Claude Code via CLI, retrying on rate-limit errors.

        Args:
            prompt: The prompt to send to Claude.
            cwd: Working directory to execute in.
            model: Optional model to use (e.g., 'opus-4.1', 'sonnet-3.7').
        """
        return with_rate_limit_retry(
            call=lambda: self._single_invocation(prompt, cwd, model),
            detector=self._detector,
            policy=self.retry_policy,
            on_retry=self.on_rate_limit_retry,
            interface="claude",
        )

    def _single_invocation(
        self, prompt: str, cwd: str, model: Optional[str]
    ) -> Dict[str, Any]:
        """One ``claude -p`` subprocess call, wrapped by the retry loop."""
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

            # Phase 4: merge env_overrides into the current process env so the
            # CLI picks up per-shard ``HOME``/``XDG_CONFIG_HOME`` + the
            # ``ANTHROPIC_API_KEY``. Passing ``env=None`` preserves current
            # behaviour when there are no overrides.
            env = {**os.environ, **self.env_overrides} if self.env_overrides else None

            # Execute claude command with the prompt via stdin
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
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
