import os
import json
import subprocess
from typing import Any, Callable, Dict, List, Literal, Optional
from dotenv import load_dotenv

from utils.cancellable_subprocess import run_with_cancel
from utils.retry import (
    ClaudeRateLimitDetector,
    RateLimitEvent,
    RateLimitPolicy,
    with_rate_limit_retry,
)

load_dotenv()


# Claude Code's ``-p --output-format`` knob. ``json`` emits a single
# summary blob (default; baseline-comparable behavior). ``stream-json``
# emits NDJSON with one event per turn including ``tool_use`` /
# ``tool_result`` blocks — used by cl-benchmark when the run has
# ``capture_full_transcript=True`` for diagnostic visibility into MCP
# tool calls. Both formats emit a final ``result`` event with the same
# token/cost totals; the cl-benchmark cost rollup parses that final
# event in either case so token accounting is identical.
OutputFormat = Literal["json", "stream-json"]


class ClaudeCodeInterface:
    """Interface for interacting with Claude Code CLI."""

    def __init__(
        self,
        timeout_s: int = 900,
        env_overrides: Optional[Dict[str, str]] = None,
        retry_policy: Optional[RateLimitPolicy] = None,
        on_rate_limit_retry: Optional[Callable[[RateLimitEvent], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        output_format: OutputFormat = "json",
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
            is_cancelled: Bug 1 (2026-04). Predicate the worker plumbs
                through so a cancel request reaches the live CLI
                subprocess — when True mid-run, we SIGTERM the CLI's
                process group + wait a grace period before SIGKILL.
                Without this, a long ``claude -p`` session can keep a
                cancelled shard running for many minutes while the
                worker's between-instance cancel check waits for the
                current invocation to finish on its own.
            output_format: ``json`` (default; baseline) emits a single
                summary blob. ``stream-json`` emits NDJSON of every
                turn including ``tool_use`` / ``tool_result`` blocks —
                cl-benchmark workers pass this when the run has
                ``capture_full_transcript=True`` for diagnostic
                visibility. Both formats share an identical final
                ``result`` event so token + cost extraction is
                format-invariant.
        """
        if output_format not in ("json", "stream-json"):
            raise ValueError(
                f"output_format must be 'json' or 'stream-json', got {output_format!r}"
            )
        self.timeout_s = timeout_s
        self.env_overrides: Dict[str, str] = dict(env_overrides) if env_overrides else {}
        self.retry_policy = retry_policy or RateLimitPolicy()
        self.on_rate_limit_retry = on_rate_limit_retry
        self.is_cancelled = is_cancelled
        self.output_format = output_format
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
            is_cancelled=self.is_cancelled,
        )

    def _single_invocation(
        self, prompt: str, cwd: str, model: Optional[str]
    ) -> Dict[str, Any]:
        """One ``claude -p`` subprocess call, wrapped by the retry loop."""
        # ``--output-format`` defaults to ``json`` (single summary blob,
        # baseline behavior). ``stream-json`` emits NDJSON with every
        # turn's tool_use + tool_result content; cl-benchmark callers
        # pass it when the run has ``capture_full_transcript=True``.
        # The CLI rejects ``-p --output-format=stream-json`` without
        # ``--verbose`` ("requires --verbose"), so add it in tandem.
        cmd = [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            self.output_format,
        ]
        if self.output_format == "stream-json":
            cmd.append("--verbose")
        # MCP wiring. ``code_swe_agent.process_instance`` writes a
        # per-instance ``.mcp.json`` into ``cwd`` when MCP is enabled
        # (per-repo Code Lexica token). Claude Code ignores
        # project-scoped ``.mcp.json`` files unless the user has trusted
        # the workspace interactively — and ``--dangerously-skip-permissions``
        # does NOT cover that trust prompt. cl-benchmark runs in a fresh
        # tempdir + isolated ``HOME`` per shard, so no trust ever
        # persists and the CLI silently boots with ``mcp_servers: []``.
        # Passing ``--mcp-config`` explicitly bypasses the trust step
        # (the CLI treats the flag value as user-supplied + auto-trusted),
        # and ``--strict-mcp-config`` ensures we don't accidentally also
        # load any user-scoped MCP config that drifted into ``HOME``.
        # No flag → byte-identical baseline behavior for non-MCP runs.
        mcp_path = os.path.join(cwd, ".mcp.json")
        if os.path.isfile(mcp_path):
            cmd.extend(["--mcp-config", mcp_path, "--strict-mcp-config"])
        if model:
            cmd.extend(["--model", model])

        # Phase 4: merge env_overrides into the current process env so the
        # CLI picks up per-shard ``HOME``/``XDG_CONFIG_HOME`` + the
        # ``ANTHROPIC_API_KEY``. Passing ``env=None`` preserves current
        # behaviour when there are no overrides.
        env = {**os.environ, **self.env_overrides} if self.env_overrides else None

        try:
            # Bug 1 (2026-04): ``run_with_cancel`` replaces the plain
            # ``subprocess.run`` so a cancel from the worker SIGTERMs the
            # CLI's process group instead of waiting out the 15-min
            # session. ``is_cancelled`` may be None — the helper
            # collapses to the old behavior in that case.
            result = run_with_cancel(
                cmd,
                input=prompt,
                cwd=cwd,
                env=env,
                timeout_s=self.timeout_s,
                is_cancelled=self.is_cancelled,
            )
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "token_usage": {},
            }

        if result.timed_out:
            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": -1,
                "token_usage": {},
                "timed_out": True,
            }

        if result.cancelled:
            # Surface the cancel flag all the way up — with_rate_limit_retry
            # + process_instance + run_shard all key off it.
            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "token_usage": {},
                "cancelled": True,
            }

        # Parse output for token usage and metadata. Both ``json`` and
        # ``stream-json`` formats share the same final ``result`` event
        # shape:
        # {
        #   "result": "...", "num_turns": N, "duration_ms": N,
        #   "duration_api_ms": N, "is_error": bool, "session_id": "...",
        #   "total_cost_usd": N.NN,
        #   "usage": { "input_tokens": N, "output_tokens": N,
        #              "cache_creation_input_tokens": N, "cache_read_input_tokens": N }
        # }
        # ``json`` emits this as the single stdout blob. ``stream-json``
        # emits it as the final NDJSON event with ``type: "result"``;
        # ``_extract_result_event`` walks the NDJSON to find it. Either
        # way, downstream parsing is identical so cost rollups don't
        # drift between formats.
        token_usage = {}
        if result.stdout:
            output_data = self._extract_result_event(result.stdout)
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

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "token_usage": token_usage,
        }

    def _extract_result_event(self, stdout: str) -> Dict[str, Any]:
        """Pull the final ``result`` event out of ``stdout``.

        ``json`` mode: stdout is a single JSON blob — parse + return as-is.
        ``stream-json`` mode: stdout is NDJSON of many events — find the
        last event with ``type == "result"`` and return it.

        Empty / unparseable input returns ``{}`` (callers fall back to no
        token tracking, mirroring the prior json-only error path). This
        is deliberate — a malformed transcript shouldn't kill the patch
        pipeline; the patch is extracted via ``git diff`` independently.
        """
        if self.output_format == "json":
            try:
                data = json.loads(stdout)
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}

        # stream-json: walk NDJSON for the final ``type=result`` event.
        last_result: Dict[str, Any] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Malformed line in the middle of the stream — skip it
                # and keep walking. A single bad line shouldn't kill
                # the rest.
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                last_result = event
        return last_result

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
