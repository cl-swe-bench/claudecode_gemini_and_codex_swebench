import os
import subprocess
from typing import Any, Callable, Dict, List, Optional

from utils.retry import (
    GeminiRateLimitDetector,
    RateLimitEvent,
    RateLimitPolicy,
    with_rate_limit_retry,
)


class GeminiCodeInterface:
    """Interface for interacting with the Google Gemini CLI."""

    def __init__(
        self,
        timeout_s: int = 900,
        env_overrides: Optional[Dict[str, str]] = None,
        retry_policy: Optional[RateLimitPolicy] = None,
        on_rate_limit_retry: Optional[Callable[[RateLimitEvent], None]] = None,
    ):
        """Ensure the Gemini CLI is available on the system."""
        self.timeout_s = timeout_s
        self.env_overrides: Dict[str, str] = dict(env_overrides) if env_overrides else {}
        self.retry_policy = retry_policy or RateLimitPolicy()
        self.on_rate_limit_retry = on_rate_limit_retry
        self._detector = GeminiRateLimitDetector()
        try:
            result = subprocess.run(["gemini", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "Gemini CLI not found. Please ensure 'gemini' is installed and in PATH"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "Gemini CLI not found. Please ensure 'gemini' is installed and in PATH"
            )

    def execute_code_cli(self, prompt: str, cwd: str, model: str = None) -> Dict[str, Any]:
        """Execute Gemini via CLI, retrying on rate-limit errors."""
        return with_rate_limit_retry(
            call=lambda: self._single_invocation(prompt, cwd, model),
            detector=self._detector,
            policy=self.retry_policy,
            on_retry=self.on_rate_limit_retry,
            interface="gemini",
        )

    def _single_invocation(
        self, prompt: str, cwd: str, model: Optional[str]
    ) -> Dict[str, Any]:
        try:
            original_cwd = os.getcwd()
            os.chdir(cwd)

            # Build command
            cmd = ["gemini"]
            if model:
                cmd.extend(["--model", model])

            env = {**os.environ, **self.env_overrides} if self.env_overrides else None

            # Execute gemini command with the prompt via stdin
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
            )

            os.chdir(original_cwd)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout_s} seconds",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Gemini's response (placeholder)."""
        return []

    @staticmethod
    def _to_token_usage(usage: Dict) -> Dict[str, int]:
        """Normalize Gemini token usage to the cl-benchmark library schema.

        TODO(P4): Gemini CLI does not currently emit token usage in a format we
        parse — returns zeros. Verify against the Gemini CLI JSON contract when
        we have real invocation data to reverse-engineer.
        """
        usage = usage or {}
        return {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
            "cache_creation": 0,
            "cache_read": 0,
        }
