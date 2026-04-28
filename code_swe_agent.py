#!/usr/bin/env python3
"""
SWE-bench agent capable of using Claude Code or Codex backends.
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
import time
from datetime import datetime
from typing import Any, Callable, List, Dict, Optional, Tuple
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import jsonlines

from utils.claude_interface import ClaudeCodeInterface
from utils.codex_interface import CodexCodeInterface
from utils.gemini_interface import GeminiCodeInterface
from utils.prompt_formatter import PromptFormatter
from utils.patch_extractor import PatchExtractor
from utils.model_registry import get_model_name
from utils.mcp_config import McpConfigManager


DEFAULT_BACKEND = os.environ.get("CODE_SWE_BACKEND", "claude")


class CodeSWEAgent:
    """Main agent for running SWE-bench using different code models."""

    def __init__(self, prompt_template: Optional[str] = None,
                 model: Optional[str] = None,
                 backend: str = DEFAULT_BACKEND,
                 mcp_enabled: bool = False,
                 repos_filter: Optional[List[str]] = None,
                 mcp_repos_only: bool = False,
                 mcp_config_path: Optional[str] = None,
                 subprocess_timeout_s: int = 900,
                 env_overrides: Optional[Dict[str, str]] = None,
                 on_rate_limit_retry: Optional[Callable[[Any], None]] = None,
                 is_cancelled: Optional[Callable[[], bool]] = None,
                 output_format: str = "json"):
        # env_overrides + on_rate_limit_retry are Phase 4 additions. The
        # worker plumbs per-shard env + a structlog-emitting callback; the
        # standalone CLI leaves both at None and the behaviour is
        # identical to P3. ``is_cancelled`` is Bug 1 (2026-04): the
        # worker flips an ``threading.Event`` closure True when the DB
        # status moves to ``cancel_requested``, and the interface's
        # ``run_with_cancel`` SIGTERMs the live CLI process group.
        self.is_cancelled = is_cancelled
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend == "codex":
            self.interface = CodexCodeInterface(
                timeout_s=subprocess_timeout_s,
                env_overrides=env_overrides,
                on_rate_limit_retry=on_rate_limit_retry,
                is_cancelled=is_cancelled,
            )
        elif self.backend == "gemini":
            self.interface = GeminiCodeInterface(
                timeout_s=subprocess_timeout_s,
                env_overrides=env_overrides,
                on_rate_limit_retry=on_rate_limit_retry,
                is_cancelled=is_cancelled,
            )
        else:
            self.backend = "claude"
            # ``output_format`` is claude-only for now — codex/gemini
            # interfaces don't accept it. cl-benchmark only sets it
            # to ``stream-json`` for ``capture_full_transcript=True``
            # runs against the claude harness; codex/gemini variants
            # would need their own equivalent in a future slice.
            self.interface = ClaudeCodeInterface(
                timeout_s=subprocess_timeout_s,
                env_overrides=env_overrides,
                on_rate_limit_retry=on_rate_limit_retry,
                is_cancelled=is_cancelled,
                output_format=output_format,
            )

        self.prompt_formatter = PromptFormatter(prompt_template)
        self.patch_extractor = PatchExtractor()
        self.base_dir = Path.cwd()
        self.results_dir = self.base_dir / "results"
        self.predictions_dir = self.base_dir / "predictions"

        # Resolve model name from alias
        self.model = get_model_name(model, self.backend) if model else None
        self.model_alias = model  # Keep original alias for logging

        # MCP configuration
        self.mcp_enabled = mcp_enabled
        self.mcp_manager = None
        if mcp_enabled or mcp_repos_only:
            self.mcp_manager = McpConfigManager(registry_path=mcp_config_path) if mcp_config_path \
                else McpConfigManager()
            if mcp_enabled:
                print("MCP mode enabled — Code Lexica context will be injected per repo")

        # Repo filtering
        self.repos_filter = None
        if mcp_repos_only and self.mcp_manager:
            # Derive filter from repos that have valid (non-placeholder) tokens
            configured_repos = self.mcp_manager.get_configured_repos()
            if repos_filter:
                # Intersect: only repos that are both requested AND have tokens
                self.repos_filter = configured_repos & set(repos_filter)
            else:
                self.repos_filter = configured_repos
            print(f"MCP repos filter: {len(self.repos_filter)} repos with tokens configured")
            for repo in sorted(self.repos_filter):
                print(f"  - {repo}")
        elif repos_filter:
            self.repos_filter = set(repos_filter)
            print(f"Repo filter: {len(self.repos_filter)} repos selected")
            for repo in sorted(self.repos_filter):
                print(f"  - {repo}")

        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        self.pred_timestamp: Optional[str] = None
        self.pred_file: Optional[Path] = None

    def setup_repository(self, instance: Dict) -> Tuple[Optional[str], str, str]:
        """Set up a repository for testing.

        Returns a ``(path, stdout, stderr)`` triple. On failure ``path`` is
        ``None`` and the stderr string carries whatever git (or the
        surrounding ``except``) wrote — callers bubble that up as
        ``raw_stderr`` so downstream log uploads + the run-detail UI show
        the real reason rather than a generic "Failed to set up
        repository" string.
        """
        instance_id = instance["instance_id"]
        repo_name = instance["repo"]
        base_commit = instance["base_commit"]

        # Create temporary directory for this instance (cross-platform)
        temp_dir = Path(tempfile.gettempdir()) / f"swe_bench_{instance_id}"

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        def _log_header(msg: str) -> None:
            # Keep the human-readable trail in the captured stdout too so
            # the blob reads like a debug session.
            stdout_parts.append(msg)
            print(msg)

        try:
            # Remove if exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Save current directory
            original_dir = Path.cwd()

            # Clone repository
            _log_header(f"Cloning {repo_name} to {temp_dir}")
            clone_url = f"https://github.com/{repo_name}.git"

            result = subprocess.run(
                ["git", "clone", clone_url, str(temp_dir)],
                capture_output=True,
                text=True,
                cwd=str(original_dir)  # Ensure we're in a valid directory
            )
            stdout_parts.append(result.stdout or "")
            stderr_parts.append(result.stderr or "")

            if result.returncode != 0:
                msg = f"Failed to clone repository: {result.stderr}"
                stderr_parts.append(msg)
                print(msg)
                return None, "\n".join(stdout_parts), "\n".join(stderr_parts)

            # Checkout base commit.
            #
            # Some repos (notably ProtonMail/WebClients) have GitLab
            # mirrors whose merge bot rewrites ``main`` via rebase +
            # force-push. Intermediate merge commits get orphaned on
            # GitHub — they survive in the object database (the REST
            # API still serves them) but aren't reachable from any ref,
            # so a fresh clone's working tree doesn't have them. When
            # the dataset's ``base_commit`` was captured at an
            # intermediate state, ``git checkout <sha>`` fails with
            # ``reference is not a tree``.
            #
            # GitHub's ``uploadpack.allowAnySHA1InWant`` (enabled by
            # default on public repos) lets us ``git fetch origin
            # <sha>`` to pull the orphan directly. After that, the
            # object lives in the local store and checkout succeeds.
            # We try the cheap path first (plain checkout) and only
            # run the fetch on failure, so most repos pay zero extra
            # latency for this fallback.
            os.chdir(temp_dir)
            result = subprocess.run(
                ["git", "checkout", base_commit],
                capture_output=True,
                text=True
            )
            stdout_parts.append(result.stdout or "")
            stderr_parts.append(result.stderr or "")

            if result.returncode != 0:
                _log_header(
                    f"Checkout failed, attempting direct fetch of {base_commit} "
                    f"(likely an orphan commit — fell off main via force-push)"
                )
                fetch = subprocess.run(
                    ["git", "fetch", "origin", base_commit],
                    capture_output=True,
                    text=True,
                )
                stdout_parts.append(fetch.stdout or "")
                stderr_parts.append(fetch.stderr or "")
                if fetch.returncode == 0:
                    result = subprocess.run(
                        ["git", "checkout", base_commit],
                        capture_output=True,
                        text=True,
                    )
                    stdout_parts.append(result.stdout or "")
                    stderr_parts.append(result.stderr or "")

            if result.returncode != 0:
                msg = f"Failed to checkout commit {base_commit}: {result.stderr}"
                stderr_parts.append(msg)
                print(msg)
                os.chdir(str(original_dir))  # Return to original directory
                return None, "\n".join(stdout_parts), "\n".join(stderr_parts)

            os.chdir(str(original_dir))  # Return to original directory
            return str(temp_dir), "\n".join(stdout_parts), "\n".join(stderr_parts)

        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            msg = f"Error setting up repository: {e}"
            stderr_parts.append(msg)
            stderr_parts.append(tb)
            print(msg)
            # Try to return to original directory if possible
            try:
                os.chdir(str(original_dir))
            except Exception as chdir_error:
                chdir_msg = f"Warning: Failed to return to original directory: {chdir_error}"
                stderr_parts.append(chdir_msg)
                print(chdir_msg)
            return None, "\n".join(stdout_parts), "\n".join(stderr_parts)
            
    def process_instance(self, instance: Dict) -> Dict:
        """Process a single SWE-bench instance."""
        instance_id = instance["instance_id"]
        print(f"\nProcessing {instance_id}")

        original_dir = os.getcwd()

        # Task-context metadata the dataset carries. Included on every
        # return path (setup failure, CLI failure, success, exception)
        # so ``run_shard`` can plumb it into ``InstanceRunResult``
        # regardless of outcome — cl-benchmark's Task tab renders these
        # above the patch for "what was the agent asked to do?" context.
        # Normalize "" → None so pydantic's optional fields don't store
        # empty strings that render as a blank tab.
        task_context = {
            "problem_statement": (instance.get("problem_statement") or None),
            "hints_text": (instance.get("hints_text") or None),
            "base_commit": (instance.get("base_commit") or None),
            # ``formatted_prompt`` is filled in only for paths that
            # reached prompt formatting. Setup failures leave it None.
            "formatted_prompt": None,
        }

        repo_path, setup_stdout, setup_stderr = self.setup_repository(instance)
        # Slice C: setup-phase streams travel on their own keys so the
        # UI's Setup tab can render them distinctly from CLI output.
        # Included on every return path so the Setup tab has something
        # to show even on happy runs (git clone progress is often non-
        # empty and occasionally useful for "why did clone take so long").
        setup_streams = {
            "setup_stdout": setup_stdout,
            "setup_stderr": setup_stderr,
        }
        if not repo_path:
            # Pull a one-line summary out of the captured stderr so the
            # DB's ``error`` column (80-char tail) surfaces the actual git
            # failure rather than the generic "Failed to set up
            # repository" string.
            summary = "Failed to set up repository"
            for line in reversed(setup_stderr.splitlines()):
                line = line.strip()
                if line and "fatal:" in line:
                    summary = line
                    break
            return {
                "instance_id": instance_id,
                "model": f"{self.backend}-code",
                "prediction": "",
                "error": summary,
                "token_usage": {},
                # CLI never ran — leave the CLI streams empty; the Setup
                # streams below carry the actual git-clone failure bytes.
                "raw_stdout": "",
                "raw_stderr": "",
                **setup_streams,
                **task_context,
            }

        try:
            prompt = self.prompt_formatter.format_for_cli(instance)
            # Capture the prompt bytes we actually sent. The formatter
            # already merged the template + issue + hints; this is what
            # the CLI reads from stdin.
            task_context["formatted_prompt"] = prompt

            os.chdir(repo_path)
            subprocess.run(["git", "add", "-A"], capture_output=True)
            subprocess.run(["git", "stash"], capture_output=True)

            # Inject ``.mcp.json`` AFTER the ``git add -A`` + ``git stash``
            # cleanup. The stash hides any retry-leftover working-tree
            # modifications, and ``-A`` would have staged ``.mcp.json``
            # too — so writing it before the stash silently swept it
            # away, leaving Claude Code with no MCP servers to load.
            # Writing it post-stash keeps the file on disk + untracked
            # for the duration of the agent run; ``inject_mcp_json``
            # overwrites any prior copy and ``remove_mcp_json`` clears
            # it for non-MCP runs.
            if self.mcp_enabled and self.mcp_manager:
                self.mcp_manager.inject_mcp_json(instance_id, repo_path)
            else:
                McpConfigManager.remove_mcp_json(repo_path)

            model_info = f" with model {self.model_alias}" if self.model else ""
            print(f"Running {self.backend.title()} Code{model_info}...")
            result = self.interface.execute_code_cli(prompt, repo_path, self.model)

            token_usage = result.get("token_usage", {})

            if not result["success"]:
                print(f"{self.backend.title()} Code execution failed: {result['stderr']}")
                os.chdir(original_dir)
                return {
                    "instance_id": instance_id,
                    "model": self.model_alias or f"{self.backend}-code",
                    "prediction": "",
                    "error": f"Execution failed: {result['stderr']}",
                    "token_usage": token_usage,
                    "raw_stdout": result.get("stdout", "") or "",
                    "raw_stderr": result.get("stderr", "") or "",
                    "timed_out": bool(result.get("timed_out")),
                    **setup_streams,
                    **task_context,
                }

            # Extract patch from git diff (works regardless of output format)
            patch = self.patch_extractor.extract_from_cli_output(result["stdout"], repo_path)

            is_valid, error = self.patch_extractor.validate_patch(patch)
            if not is_valid:
                print(f"Invalid patch: {error}")
                patch = ""

            prediction = self.patch_extractor.format_for_swebench(
                patch, instance_id, self.model_alias or f"{self.backend}-code"
            )
            prediction["token_usage"] = token_usage
            prediction["raw_stdout"] = result.get("stdout", "") or ""
            prediction["raw_stderr"] = result.get("stderr", "") or ""
            prediction.update(setup_streams)
            prediction.update(task_context)

            self._save_result(instance_id, result, patch)

            return prediction

        except Exception as e:
            import traceback
            print(f"Error processing instance: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "instance_id": instance_id,
                "model": self.model_alias or f"{self.backend}-code",
                "prediction": "",
                "error": str(e),
                "token_usage": {},
                "raw_stdout": "",
                "raw_stderr": "",
                **setup_streams,
                **task_context,
            }
        finally:
            try:
                os.chdir(original_dir)
            except Exception as e:
                print(f"Warning: Could not restore directory: {e}")

            if repo_path and os.path.exists(repo_path):
                shutil.rmtree(repo_path)
    def _save_result(self, instance_id: str, result: Dict, patch: str):
        """Save detailed results for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{instance_id}_{timestamp}.json"

        with open(result_file, 'w') as f:
            json.dump({
                "instance_id": instance_id,
                "timestamp": timestamp,
                "token_usage": result.get("token_usage", {}),
                "claude_output": {
                    "success": result.get("success"),
                    "returncode": result.get("returncode"),
                    "stderr": result.get("stderr", ""),
                    # Truncate stdout to avoid huge files (JSON output can be large)
                    "stdout_length": len(result.get("stdout", "")),
                },
                "extracted_patch": patch
            }, f, indent=2)
            
    def run_on_dataset(self, dataset_name: str, split: str = "test",
                      limit: Optional[int] = None) -> List[Dict]:
        """Run on a full dataset."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        # Filter by repos if specified
        if self.repos_filter:
            original_count = len(dataset)
            dataset = dataset.filter(
                lambda instance: instance["repo"] in self.repos_filter
            )
            print(f"Repo filter: {len(dataset)}/{original_count} instances match selected repos")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        if len(dataset) == 0:
            print("No instances to process after filtering.")
            return []

        self.pred_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pred_file = self.predictions_dir / f"predictions_{self.pred_timestamp}.jsonl"
        if self.pred_file.exists():
            self.pred_file.unlink()
        json_file = self.predictions_dir / f"predictions_{self.pred_timestamp}.json"
        if json_file.exists():
            json_file.unlink()

        predictions: List[Dict] = []
        total = len(dataset)
        patches_generated = 0
        total_tokens = 0
        total_cost = 0.0

        for idx, instance in enumerate(dataset, 1):
            instance_id = instance["instance_id"]
            repo = instance["repo"]
            elapsed = (datetime.now() - datetime.strptime(self.pred_timestamp, "%Y%m%d_%H%M%S")).total_seconds()
            avg_time = elapsed / (idx - 1) if idx > 1 else 0
            remaining = avg_time * (total - idx + 1)

            print(f"\n{'='*60}")
            print(f"[{idx}/{total}] {instance_id}")
            print(f"  Repo: {repo}")
            if idx > 1:
                print(f"  Avg time/task: {avg_time:.0f}s | Est. remaining: {remaining/60:.1f}m")
                print(f"  Patches so far: {patches_generated}/{idx-1} ({patches_generated/(idx-1)*100:.0f}%)")
                stats_parts = []
                if total_tokens > 0:
                    stats_parts.append(f"Tokens: {total_tokens:,}")
                if total_cost > 0:
                    stats_parts.append(f"Cost: ${total_cost:.2f}")
                if stats_parts:
                    print(f"  Running totals: {' | '.join(stats_parts)}")
            print(f"{'='*60}")

            prediction = self.process_instance(instance)

            has_patch = bool(prediction.get("prediction", "").strip())
            if has_patch:
                patches_generated += 1

            # Accumulate token usage if available
            task_usage = prediction.get("token_usage", {})
            task_tokens = task_usage.get("total_tokens", 0)
            task_cost = task_usage.get("cost_usd", 0) or 0
            task_turns = task_usage.get("num_turns")
            task_duration = task_usage.get("duration_ms")
            total_tokens += task_tokens
            total_cost += task_cost

            # Per-task result summary
            status = "PATCH" if has_patch else "NO PATCH"
            details = [status]
            if task_tokens:
                details.append(f"{task_tokens:,} tokens")
            if task_cost:
                details.append(f"${task_cost:.3f}")
            if task_turns:
                details.append(f"{task_turns} turns")
            if task_duration:
                details.append(f"{task_duration/1000:.1f}s")
            print(f"  Result: {' | '.join(details)}")

            predictions.append(prediction)
            self._save_predictions(prediction)

        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"  Tasks: {total}")
        print(f"  Patches: {patches_generated}/{total} ({patches_generated/total*100:.0f}%)")
        if total_tokens > 0:
            print(f"  Total tokens: {total_tokens:,} (avg {total_tokens//total:,}/task)")
        if total_cost > 0:
            print(f"  Total cost: ${total_cost:.2f} (avg ${total_cost/total:.3f}/task)")
        print(f"{'='*60}")

        with open(json_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to {self.pred_file}")
        return predictions
    
    def run_on_instance(self, instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Lite") -> Dict:
        """Run on a single instance by ID."""
        dataset = load_dataset(dataset_name, split="test")
        
        # Find the instance
        instance = None
        for item in dataset:
            if item["instance_id"] == instance_id:
                instance = item
                break
                
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in dataset")
            
        return self.process_instance(instance)
    
    def _save_predictions(self, prediction: Dict):
        """Append a single prediction to the jsonl file."""
        if not self.pred_file:
            raise ValueError("Prediction timestamp not initialized. Call run_on_dataset first.")

        with jsonlines.open(self.pred_file, mode='a') as writer:
            writer.write(prediction)


def run_shard(
    instance_ids: List[str],
    dataset_name: str,
    *,
    dataset_revision: Optional[str] = None,
    mcp_enabled: bool = False,
    mcp_config_path: Optional[str] = None,
    model: Optional[str] = None,
    backend: str = "claude",
    prompt_template: Optional[str] = None,
    on_instance_start: Optional[Callable[[str], None]] = None,
    on_instance_complete: Optional[Callable[[Any], None]] = None,
    on_instance_error: Optional[Callable[[str, Exception, str], None]] = None,
    subprocess_timeout_s: int = 900,
    api_key: Optional[str] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    rate_limit_callback: Optional[Callable[[Any], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    output_format: str = "json",
) -> List[Any]:
    """Shardable library entry point used by the cl-benchmark worker.

    Unlike ``CodeSWEAgent.run_on_dataset``, this does NOT write JSONL
    predictions, per-instance result files, or ``benchmark_scores.log`` —
    the caller owns persistence. Callbacks fire at each instance boundary so
    the worker can write to Postgres + MinIO without the harness knowing
    about either.

    Phase 4 additions:
      - ``api_key``: informational — the caller's intent to run API-key
        mode. Not read directly (see ``env_overrides``). Kept as a
        distinct arg so log output can elide the secret while logging
        the intent.
      - ``env_overrides``: merged into each subprocess's ``env=``. The
        worker injects ``ANTHROPIC_API_KEY`` (or the per-harness
        equivalent) here and relocates ``HOME``/``XDG_CONFIG_HOME`` to
        a per-shard sandbox so codex/gemini auth caches don't race and
        so api_key-claude doesn't pick up a host Max OAuth from
        ``~/.claude/auth.json``.
      - ``rate_limit_callback``: invoked per retry with a ``RateLimitEvent``
        so the worker can emit ``instance.rate_limit_retry`` events.

    Returns a list of ``cl_benchmark_core.schemas.library.InstanceRunResult``
    Pydantic models (import is lazy so standalone CLI use of this harness
    does not require ``cl-benchmark-core`` to be installed).
    """
    # ``api_key`` is currently informational — the actual injection is via
    # ``env_overrides``. Referencing it here keeps linters quiet and
    # documents the parameter's role. If we ever want to log the call
    # without the secret, this is where redaction would land.
    _ = api_key
    # Lazy import: keeps the harness usable standalone when cl-benchmark-core
    # is not installed (direct CLI / test_sets workflow).
    from cl_benchmark_core.schemas.library import (
        ErrorKind,
        InstanceRunResult,
        InstanceTokenUsage,
        LibraryInstanceStatus,
    )

    agent = CodeSWEAgent(
        prompt_template=prompt_template,
        model=model,
        backend=backend,
        mcp_enabled=mcp_enabled,
        mcp_config_path=mcp_config_path,
        subprocess_timeout_s=subprocess_timeout_s,
        env_overrides=env_overrides,
        on_rate_limit_retry=rate_limit_callback,
        is_cancelled=is_cancelled,
        output_format=output_format,
    )

    dataset = load_dataset(dataset_name, revision=dataset_revision, split="test")
    id_set = set(instance_ids)
    dataset = dataset.filter(lambda inst: inst["instance_id"] in id_set)
    by_id = {inst["instance_id"]: inst for inst in dataset}

    results: List[InstanceRunResult] = []

    for instance_id in instance_ids:
        # Bug 1: between-instance cancel check. If the worker flipped
        # the cancel predicate during a prior instance's run (or just
        # set it before this shard started), stop before spawning the
        # CLI for the next instance. The worker's own cancel check
        # between instances also catches this — duplicate here is a
        # belt-and-braces exit so the CLI subprocess never spawns.
        if is_cancelled is not None and is_cancelled():
            break
        instance = by_id.get(instance_id)
        if instance is None:
            err = LookupError(f"instance_id '{instance_id}' not in {dataset_name}@{dataset_revision}")
            if on_instance_error is not None:
                on_instance_error(instance_id, err, "")
            result = InstanceRunResult(
                instance_id=instance_id,
                repo=None,
                status=LibraryInstanceStatus.ERROR,
                patch=None,
                error_message=str(err),
                error_kind=ErrorKind.SETUP_ERROR,
            )
            results.append(result)
            if on_instance_complete is not None:
                on_instance_complete(result)
            continue

        if on_instance_start is not None:
            on_instance_start(instance_id)

        start_monotonic = time.monotonic()
        try:
            prediction = agent.process_instance(instance)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            # Map the harness's rate-limit-exhausted exception to a
            # first-class ErrorKind so the worker emits a distinctly
            # filterable event and the UI can surface "hit limits 5x".
            from utils.retry import RateLimitExhausted  # local import — no cycle
            if isinstance(exc, RateLimitExhausted):
                mapped_error_kind = ErrorKind.RATE_LIMIT_EXHAUSTED
            else:
                mapped_error_kind = ErrorKind.UNKNOWN
            if on_instance_error is not None:
                on_instance_error(instance_id, exc, tb)
            result = InstanceRunResult(
                instance_id=instance_id,
                repo=instance.get("repo"),
                status=LibraryInstanceStatus.ERROR,
                patch=None,
                error_message=str(exc),
                error_kind=mapped_error_kind,
                duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                raw_stderr=tb,
                # Even on unhandled exceptions we still have the dataset
                # row — surface the task context so the Task tab isn't
                # blank on crash rows. ``formatted_prompt`` may or may
                # not have been built yet; we don't have visibility into
                # how far ``process_instance`` got, so leave it None.
                problem_statement=(instance.get("problem_statement") or None),
                hints_text=(instance.get("hints_text") or None),
                base_commit=(instance.get("base_commit") or None),
            )
            results.append(result)
            if on_instance_complete is not None:
                on_instance_complete(result)
            continue

        token_usage_raw = prediction.get("token_usage") or {}
        normalized = agent.interface._to_token_usage(token_usage_raw)
        token_usage = InstanceTokenUsage(**normalized)

        # Prefer the harness-reported duration (Claude Code exposes it in its
        # JSON response); fall back to wall-clock for backends that don't.
        duration_ms = token_usage_raw.get("duration_ms")
        if duration_ms is None:
            duration_ms = int((time.monotonic() - start_monotonic) * 1000)
        num_turns = token_usage_raw.get("num_turns")

        raw_stdout = prediction.get("raw_stdout", "") or ""
        raw_stderr = prediction.get("raw_stderr", "") or ""
        patch = prediction.get("prediction") or ""
        error_message = prediction.get("error")

        if error_message:
            status = LibraryInstanceStatus.ERROR
            if prediction.get("timed_out"):
                error_kind = ErrorKind.TIMEOUT
            elif "set up repository" in error_message.lower():
                error_kind = ErrorKind.SETUP_ERROR
            else:
                error_kind = ErrorKind.SUBPROCESS_ERROR
            result_patch = None
        elif patch:
            status = LibraryInstanceStatus.GENERATED
            error_kind = None
            result_patch = patch
        else:
            # CLI exited cleanly (no ``prediction.error``) but the extractor
            # returned an empty string — i.e. ``git diff HEAD`` in the
            # instance clone dir produced zero bytes. This is a reliability
            # signal (agent ran to completion without making net file
            # changes), NOT a parser crash. ``PARSING_ERROR`` is reserved
            # for the extractor-couldn't-read case.
            status = LibraryInstanceStatus.ERROR
            error_message = "no patch produced"
            error_kind = ErrorKind.EMPTY_PATCH
            result_patch = None

        result = InstanceRunResult(
            instance_id=instance_id,
            repo=instance.get("repo"),
            status=status,
            patch=result_patch,
            token_usage=token_usage,
            duration_ms=duration_ms,
            num_turns=num_turns,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            error_message=error_message,
            error_kind=error_kind,
            # Task context — ``process_instance`` populates these on
            # every return path (setup, CLI, success). None for pre-B
            # harness builds since the keys simply don't exist on the
            # returned dict.
            problem_statement=prediction.get("problem_statement"),
            hints_text=prediction.get("hints_text"),
            base_commit=prediction.get("base_commit"),
            formatted_prompt=prediction.get("formatted_prompt"),
            # Setup phase streams — always populated on every return
            # path from slice-C onward. Empty string (not None) for
            # pre-slice-C results so the UI's Setup tab can distinguish
            # "setup ran with no output" from "pre-migration row".
            setup_stdout=prediction.get("setup_stdout", "") or "",
            setup_stderr=prediction.get("setup_stderr", "") or "",
        )
        results.append(result)

        # ``on_instance_error`` is reserved for unhandled harness exceptions
        # — an error InstanceRunResult still goes through the normal
        # completion callback so the worker writes the log + emits a single
        # terminal event per instance.
        if on_instance_complete is not None:
            on_instance_complete(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run code models on SWE-bench")
    parser.add_argument("--dataset_name", type=str,
                       default="princeton-nlp/SWE-bench_Lite",
                       help="Dataset to use")
    parser.add_argument("--instance_id", type=str,
                       help="Run on a specific instance ID")
    parser.add_argument("--limit", type=int,
                       help="Limit number of instances to process")
    parser.add_argument("--prompt_template", type=str,
                       help="Path to custom prompt template")
    parser.add_argument("--model", type=str,
                       help="Model to use (e.g., opus-4.1, codex-4.2, or any name)")
    parser.add_argument("--backend", type=str, choices=["claude", "codex", "gemini"],
                       help="Code model backend to use")
    parser.add_argument("--mcp", action="store_true",
                       help="Enable Code Lexica MCP server for codebase context")
    parser.add_argument("--repos", type=str,
                       help="Comma-separated list of repos to run (e.g., django/django,sympy/sympy)")
    parser.add_argument("--mcp-repos-only", action="store_true",
                       help="Only run on repos that have MCP tokens configured in code_lexica_repos.json")

    args = parser.parse_args()

    backend = args.backend or DEFAULT_BACKEND

    # Check if selected CLI is available
    if backend == "codex":
        cli_cmd = "codex"
    elif backend == "gemini":
        cli_cmd = "gemini"
    else:
        cli_cmd = "claude"

    try:
        result = subprocess.run([cli_cmd, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {cli_cmd} CLI not found. Please ensure '{cli_cmd}' is installed and in PATH")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {cli_cmd} CLI not found. Please ensure '{cli_cmd}' is installed and in PATH")
        sys.exit(1)

    # Parse repos filter
    repos_filter = None
    if args.repos:
        repos_filter = [r.strip() for r in args.repos.split(",") if r.strip()]

    agent = CodeSWEAgent(
        args.prompt_template, args.model, backend,
        mcp_enabled=args.mcp,
        repos_filter=repos_filter,
        mcp_repos_only=args.mcp_repos_only,
    )
    
    # Run on specific instance or dataset
    if args.instance_id:
        print(f"Running on instance: {args.instance_id}")
        prediction = agent.run_on_instance(args.instance_id, args.dataset_name)
        print(f"Prediction saved: {prediction}")
    else:
        print(f"Running on dataset: {args.dataset_name}")
        predictions = agent.run_on_dataset(args.dataset_name, limit=args.limit)
        print(f"Processed {len(predictions)} instances")


if __name__ == "__main__":
    main()
