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
from datetime import datetime
from typing import List, Dict, Optional
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
                 mcp_repos_only: bool = False):
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend == "codex":
            self.interface = CodexCodeInterface()
        elif self.backend == "gemini":
            self.interface = GeminiCodeInterface()
        else:
            self.backend = "claude"
            self.interface = ClaudeCodeInterface()

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
            self.mcp_manager = McpConfigManager()
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

    def setup_repository(self, instance: Dict) -> Optional[str]:
        """Set up a repository for testing."""
        instance_id = instance["instance_id"]
        repo_name = instance["repo"]
        base_commit = instance["base_commit"]

        # Create temporary directory for this instance (cross-platform)
        temp_dir = Path(tempfile.gettempdir()) / f"swe_bench_{instance_id}"

        try:
            # Remove if exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Save current directory
            original_dir = Path.cwd()
            
            # Clone repository
            print(f"Cloning {repo_name} to {temp_dir}")
            clone_url = f"https://github.com/{repo_name}.git"
            
            result = subprocess.run(
                ["git", "clone", clone_url, str(temp_dir)],
                capture_output=True,
                text=True,
                cwd=str(original_dir)  # Ensure we're in a valid directory
            )
            
            if result.returncode != 0:
                print(f"Failed to clone repository: {result.stderr}")
                return None
                
            # Checkout base commit
            os.chdir(temp_dir)
            result = subprocess.run(
                ["git", "checkout", base_commit],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Failed to checkout commit: {result.stderr}")
                os.chdir(str(original_dir))  # Return to original directory
                return None

            os.chdir(str(original_dir))  # Return to original directory
            return str(temp_dir)
            
        except Exception as e:
            print(f"Error setting up repository: {e}")
            # Try to return to original directory if possible
            try:
                os.chdir(str(original_dir))
            except Exception as chdir_error:
                print(f"Warning: Failed to return to original directory: {chdir_error}")
            return None
            
    def process_instance(self, instance: Dict) -> Dict:
        """Process a single SWE-bench instance."""
        instance_id = instance["instance_id"]
        print(f"\nProcessing {instance_id}")

        original_dir = os.getcwd()

        repo_path = self.setup_repository(instance)
        if not repo_path:
            return {
                "instance_id": instance_id,
                "model": f"{self.backend}-code",
                "prediction": "",
                "error": "Failed to set up repository",
                "token_usage": {},
            }

        try:
            # Inject or remove MCP config before running the agent
            if self.mcp_enabled and self.mcp_manager:
                self.mcp_manager.inject_mcp_json(instance_id, repo_path)
            else:
                McpConfigManager.remove_mcp_json(repo_path)

            prompt = self.prompt_formatter.format_for_cli(instance)

            os.chdir(repo_path)
            subprocess.run(["git", "add", "-A"], capture_output=True)
            subprocess.run(["git", "stash"], capture_output=True)

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
