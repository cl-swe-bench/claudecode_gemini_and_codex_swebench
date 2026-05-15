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
        mcp_debug: bool = False,
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
            mcp_debug: When True, appends ``--debug-file <cwd>/
                mcp-debug.log`` to the CLI invocation. The CLI writes
                its full debug stream (connection lifecycle,
                ``tool_dispatch_start``/``end``, MCP timeout cause
                lines, etc.) to that file. We read it back after the
                subprocess exits and return contents via
                ``result["mcp_debug_log"]``; the file is then deleted.
                cl-benchmark workers pass this when the run has
                ``capture_mcp_debug=True``. Independent of
                ``output_format``. See
                ``cl-benchmark/docs/mcp-debug-capture-spec.md``.
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
        self.mcp_debug = mcp_debug
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

    def execute_code_cli(
        self,
        prompt: str,
        cwd: str,
        model: str = None,
        *,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Claude Code via CLI, retrying on rate-limit errors.

        Args:
            prompt: The prompt to send to Claude.
            cwd: Working directory to execute in.
            model: Optional model to use (e.g., 'opus-4.1', 'sonnet-3.7').
            instance_id: Optional SWE-bench instance id, threaded into
                the ``RateLimitEvent`` payload so the worker can label
                rate-limit telemetry with the instance that hit it.
                Standalone CLI use leaves this ``None``.
        """
        return with_rate_limit_retry(
            call=lambda: self._single_invocation(prompt, cwd, model),
            detector=self._detector,
            policy=self.retry_policy,
            on_retry=self.on_rate_limit_retry,
            interface="claude",
            is_cancelled=self.is_cancelled,
            instance_id=instance_id,
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
        # MCP debug capture: writes the full ``--debug`` stream
        # (connection lifecycle, ``tool_dispatch_start``/``end``,
        # MCP timeout causes, heartbeats) to a sibling file. cl-bench
        # workers set this when ``run.capture_mcp_debug=True``. We
        # also implicitly enable debug mode by passing ``--debug-file``
        # — no separate ``--debug`` flag needed (and avoids the
        # high-volume debug output ending up on stderr, where the
        # harness collects gen-log content). Read back + return
        # contents after the CLI exits; see ``_read_and_clean_mcp_debug``.
        mcp_debug_path = os.path.join(cwd, "mcp-debug.log") if self.mcp_debug else None
        if mcp_debug_path is not None:
            cmd.extend(["--debug-file", mcp_debug_path])
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
            # No stdout available — process management itself failed
            # (FileNotFoundError on ``cmd[0]``, etc.). Empty token usage
            # is honest here. Debug log might still exist if the CLI
            # crashed mid-debug; surface what we have.
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "token_usage": {},
                "mcp_debug_log": self._read_and_clean_mcp_debug(mcp_debug_path),
            }

        if result.timed_out:
            # 30-minute hardcap. ``run_with_cancel`` stashed the partial
            # NDJSON stream the CLI emitted before SIGKILL; parse it so
            # cumulative cost from prior turns lands in ``token_usage``.
            # Without this we systematically under-bill timed-out
            # samples (the previous bug — ran 30 min, recorded $0).
            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": -1,
                "token_usage": self._extract_token_usage(result.stdout),
                "timed_out": True,
                "mcp_debug_log": self._read_and_clean_mcp_debug(mcp_debug_path),
            }

        if result.cancelled:
            # Surface the cancel flag all the way up — with_rate_limit_retry
            # + process_instance + run_shard all key off it. Same partial-
            # stdout parsing as the timeout path: cancels can fire
            # mid-session with real spend already incurred.
            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "token_usage": self._extract_token_usage(result.stdout),
                "cancelled": True,
                "mcp_debug_log": self._read_and_clean_mcp_debug(mcp_debug_path),
            }

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "token_usage": self._extract_token_usage(result.stdout),
            "mcp_debug_log": self._read_and_clean_mcp_debug(mcp_debug_path),
        }

    @staticmethod
    def _read_and_clean_mcp_debug(path: Optional[str]) -> Optional[str]:
        """Read the CLI's ``--debug-file`` output, then delete the file.

        Returns the contents as a string when the file exists + is
        non-empty. Returns ``None`` when the path is None (caller
        didn't ask for debug capture), the file is missing (CLI
        crashed before debug-mode init, or earlier — e.g.,
        FileNotFoundError on the ``claude`` binary), or empty.

        Always attempts the unlink afterward so per-instance task
        dirs don't accumulate logs across re-runs in the same cwd.
        Delete failures are silent — they shouldn't fail the
        instance.
        """
        if path is None or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            return None
        try:
            os.unlink(path)
        except OSError:
            pass
        return body if body else None

    def _extract_token_usage(self, stdout: str) -> Dict[str, Any]:
        """Aggregate ``type=result`` events in ``stdout`` into the
        cl-benchmark library token-usage schema.

        Called from every ``execute_code_cli`` return path — success,
        timeout, cancelled, exception. Timeouts in particular emit
        partial NDJSON with cumulative cost from prior turns; the
        success path used to be the only caller, which silently
        under-billed every 30-min timed-out sample by the full
        accumulated cost (~$5-15/sample empirically).

        Returns ``{}`` when stdout is empty or contains no result
        events (older CLI versions, hard fast-fail before any model
        call, etc.).

        The result event shape (same in ``json`` + ``stream-json``):

            {
              "type": "result", "result": "...",
              "num_turns": N, "duration_ms": N, "duration_api_ms": N,
              "is_error": bool, "session_id": "...",
              "total_cost_usd": N.NN,
              "usage": {
                "input_tokens": N, "output_tokens": N,
                "cache_creation_input_tokens": N,
                "cache_read_input_tokens": N
              },
              "modelUsage": { "<model-id>": {...}, ... }
            }

        ``json`` emits the final event as a single blob; ``stream-json``
        may emit one or more result events as NDJSON (CLI splits at
        compaction / session-resume boundaries). ``_summarize_results``
        walks every event and aggregates per-batch fields (token
        counts, ``duration_ms``, ``num_turns``) by SUM and cumulative
        fields (``total_cost_usd``, ``session_id``, ``modelUsage``)
        from the last event.
        """
        if not stdout:
            return {}
        summary = self._summarize_results(stdout)
        if not summary:
            # No ``type=result`` events parsed. This is the long-agentic-
            # loop-timeout case: the CLI was SIGKILLed at 1800s before
            # ever reaching a compaction boundary or final result, so
            # ``_summarize_results`` has nothing to work with. Fall
            # back to walking per-turn ``type=assistant`` events for
            # token counts — the worker's ``compute_total_cost_usd``
            # then derives cost from tokens × list price. See
            # cl-benchmark/docs/post-regen-eval-spec.md.
            return self._extract_token_usage_from_assistant_events(stdout)
        usage = summary.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        token_usage: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        }
        if summary.get("total_cost_usd") is not None:
            token_usage["cost_usd"] = summary["total_cost_usd"]
        if summary.get("num_turns") is not None:
            token_usage["num_turns"] = summary["num_turns"]
        if summary.get("duration_ms") is not None:
            token_usage["duration_ms"] = summary["duration_ms"]
        if summary.get("duration_api_ms") is not None:
            token_usage["duration_api_ms"] = summary["duration_api_ms"]
        if summary.get("session_id") is not None:
            token_usage["session_id"] = summary["session_id"]
        # Per-model breakdown — cumulative across all batches + all
        # models the CLI invoked (main agent + Task subagents). Empty
        # dict on older CLIs that don't emit ``modelUsage``.
        if summary.get("modelUsage"):
            token_usage["model_usage"] = summary["modelUsage"]
        return token_usage

    def _extract_token_usage_from_assistant_events(
        self, stdout: str
    ) -> Dict[str, Any]:
        """Fallback when no ``type=result`` events are present.

        Walks stream-json for ``type=assistant`` events and sums their
        ``message.usage`` fields. Each turn emits one assistant event
        with cumulative-up-to-now or per-turn token counts (depending
        on the CLI version — both shapes work because we just sum
        across events, and the consumer's ``compute_total_cost_usd``
        uses these as totals).

        Used for the long-agentic-loop timeout case: CLI ran for the
        full 1800s, never reached a compaction / final result, but
        emitted assistant events per turn. Without this fallback we
        record ``cost=$0`` for samples that actually burned $10+ of
        API time. Spec: cl-benchmark/docs/post-regen-eval-spec.md.

        Returns ``{}`` when stdout has no parseable assistant events
        either (CLI died before any model call, etc.).
        """
        if not stdout:
            return {}
        usage_sums = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        turn_count = 0
        # Both ``json`` and ``stream-json`` modes can land here. For
        # ``json`` mode the CLI emits a single blob — if it had any
        # assistant context we'd have already captured it via the
        # result event; landing here for json mode is unusual but
        # handled (parse as single object).
        if self.output_format == "json":
            try:
                obj = json.loads(stdout.strip())
            except (json.JSONDecodeError, ValueError):
                return {}
            candidates = [obj] if isinstance(obj, dict) else []
        else:
            candidates = []
            for line in stdout.splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    candidates.append(json.loads(s))
                except (json.JSONDecodeError, ValueError):
                    continue
        for ev in candidates:
            if not isinstance(ev, dict) or ev.get("type") != "assistant":
                continue
            message = ev.get("message") or {}
            usage = message.get("usage") or {}
            if not usage:
                continue
            turn_count += 1
            for k in usage_sums:
                usage_sums[k] += int(usage.get(k) or 0)
        if turn_count == 0:
            return {}
        # Don't populate cost_usd — leave it for the worker's
        # ``compute_total_cost_usd`` formula path. Same for
        # duration_ms (no source without a result event).
        return {
            "input_tokens": usage_sums["input_tokens"],
            "output_tokens": usage_sums["output_tokens"],
            "total_tokens": usage_sums["input_tokens"] + usage_sums["output_tokens"],
            "cache_creation_tokens": usage_sums["cache_creation_input_tokens"],
            "cache_read_tokens": usage_sums["cache_read_input_tokens"],
            "num_turns": turn_count,
        }

    def _summarize_results(self, stdout: str) -> Dict[str, Any]:
        """Aggregate every ``type=result`` event in stdout into a single
        canonical summary dict.

        Multi-result emission is the general case (CC may split a session
        at compaction or session-resume boundaries on any model). Treat
        one or many uniformly:

        - **Per-batch** fields (``duration_ms``, ``num_turns``, the four
          token counts under ``usage``) sum across events.
        - **Cumulative** fields (``total_cost_usd``, ``duration_api_ms``,
          ``session_id``, ``modelUsage``) are taken from the last event
          since the CLI itself accumulates these across batches.

        Json-mode (single blob) is the single-result degenerate case:
        same code path, same answer.

        Returns ``{}`` on parse failure or when no result events are
        present (CLI crashed, killed by OOM, etc.) so callers fall
        back to wall-clock timing and zero token usage. A malformed
        transcript shouldn't kill the patch pipeline — the patch is
        extracted via ``git diff`` independently.
        """
        events = self._collect_result_events(stdout)
        if not events:
            return {}

        usage_keys = (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
        summed_usage: Dict[str, int] = {k: 0 for k in usage_keys}
        for e in events:
            u = e.get("usage") or {}
            for k in usage_keys:
                summed_usage[k] += int(u.get(k) or 0)

        final = events[-1]
        return {
            # Per-batch — summed across all events.
            "duration_ms": sum(int(e.get("duration_ms") or 0) for e in events),
            "num_turns": sum(int(e.get("num_turns") or 0) for e in events),
            "usage": summed_usage,
            # Cumulative — last event already accumulates these.
            "duration_api_ms": final.get("duration_api_ms"),
            "session_id": final.get("session_id"),
            "total_cost_usd": final.get("total_cost_usd"),
            "modelUsage": final.get("modelUsage") or {},
        }

    def _collect_result_events(self, stdout: str) -> List[Dict[str, Any]]:
        """Return every ``type=result`` event in ``stdout`` in stream order.

        Handles both ``json`` mode (single JSON blob) and ``stream-json``
        mode (NDJSON, one event per line). Malformed lines mid-stream are
        skipped silently — partial-output transcripts shouldn't fail the
        whole pipeline.
        """
        if not stdout:
            return []

        raw = stdout.strip()

        if self.output_format == "json":
            # Single blob. The CLI emits the final result event as the
            # whole stdout in this mode; if there were any intermediate
            # results during the run, json mode discards them, so we
            # only ever see one.
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
            if isinstance(data, dict) and data.get("type") == "result":
                return [data]
            return []

        # stream-json: walk NDJSON.
        out: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                out.append(event)
        return out

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Claude's response."""
        # This will be implemented by patch_extractor.py
        # For now, return empty list
        return []

    @staticmethod
    def _to_token_usage(usage: Dict) -> Dict[str, Any]:
        """Normalize Claude's token_usage dict to the cl-benchmark library schema.

        Returned keys match ``cl_benchmark_core.schemas.library.InstanceTokenUsage``:
        ``input``, ``output``, ``cache_creation``, ``cache_read``, plus an
        optional ``model_usage`` dict carrying per-model breakdown for
        subagent cost accounting.

        The ``model_usage`` source (CC's ``modelUsage`` block) uses
        camelCase keys (``inputTokens``, ``costUSD``); we re-key to
        snake_case to match the rest of the cl-benchmark schema and
        store ``cost_usd`` as a string (Decimal serialization happens
        in Pydantic / SQL JSONB layers without the float-drift risk
        that comes with eager Decimal-from-float conversions here).
        """
        usage = usage or {}
        out: Dict[str, Any] = {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
            "cache_creation": int(usage.get("cache_creation_tokens", 0) or 0),
            "cache_read": int(usage.get("cache_read_tokens", 0) or 0),
        }
        raw_model_usage = usage.get("model_usage") or {}
        if raw_model_usage:
            out["model_usage"] = {
                model: {
                    "input_tokens": int(mu.get("inputTokens") or 0),
                    "output_tokens": int(mu.get("outputTokens") or 0),
                    "cache_creation_input_tokens": int(mu.get("cacheCreationInputTokens") or 0),
                    "cache_read_input_tokens": int(mu.get("cacheReadInputTokens") or 0),
                    "cost_usd": str(mu.get("costUSD") if mu.get("costUSD") is not None else "0"),
                }
                for model, mu in raw_model_usage.items()
                if isinstance(mu, dict)
            }
        return out
