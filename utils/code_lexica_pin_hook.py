#!/usr/bin/env python3
"""Claude Code ``PreToolUse`` hook: pin Code Lexica params to harness values.

Wired up by ``McpConfigManager.inject_pin_hook`` (utils/mcp_config.py) on
runs with ``mcp_pin_task_prompt=true``. Claude Code invokes this script for
every ``mcp__code-lexica__get_codebase_context`` tool call, passing the call
JSON on stdin. We overwrite ``taskPrompt`` / ``repoIdentifier`` /
``commitHash`` in the tool input with the harness's verbatim values
(written by the harness to ``.code_lexica/pinned_params.json`` in the clone)
and emit a ``PreToolUse`` decision with ``updatedInput`` so the *tool*
receives the pinned values regardless of what the agent constructed.

Rationale: at temperature the agent summarizes the task instead of passing
it verbatim, so ``taskPrompt`` drifts wildly across samples of the same
instance. Pinning makes the server's relevance-filter input deterministic.
Spec: cl-benchmark/docs/mcp-priming-spec.md (Amendment 2026-05-27).

**Fail-open by contract:** any error (missing/corrupt params file, unreadable
stdin) results in NO decision emitted, so Claude Code runs the tool call
unchanged. A pinning failure must never block generation.
"""
import json
import os
import sys

# Keep in sync with PINNED_PARAMS_RELPATH in utils/mcp_config.py. Hard-coded
# (not imported) so the hook has zero import dependency on the harness package
# when Claude Code spawns it as a bare subprocess.
PINNED_PARAMS_RELPATH = ".code_lexica/pinned_params.json"

# Only these keys are pinned; any other tool args the agent passes survive.
PINNED_KEYS = ("taskPrompt", "repoIdentifier", "commitHash")


def _params_path() -> str:
    # Claude Code sets CLAUDE_PROJECT_DIR to the project (clone) dir for
    # hooks; fall back to cwd, which is also the clone dir under the harness.
    base = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return os.path.join(base, PINNED_PARAMS_RELPATH)


def main() -> None:
    try:
        event = json.loads(sys.stdin.read())
    except Exception:
        return  # fail open: no decision -> tool proceeds unchanged

    try:
        with open(_params_path()) as f:
            pinned = json.load(f)
    except Exception:
        return  # fail open: params not available -> don't override

    tool_input = dict(event.get("tool_input") or {})
    for key in PINNED_KEYS:
        if key in pinned and pinned[key] is not None:
            tool_input[key] = pinned[key]

    sys.stdout.write(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": tool_input,
        }
    }))


if __name__ == "__main__":
    main()
