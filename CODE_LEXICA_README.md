# Code Lexica MCP Benchmarking Guide

This guide covers how to run SWE-bench Lite benchmarks comparing Claude Code **with and without** the Code Lexica MCP server, which provides AI agents with codebase context (code maps, architecture, conventions, workflows, etc.).

## Prerequisites

1. **Python 3.10+** with `_lzma` support (required by the `datasets` library)
   ```bash
   python --version
   python -c "import lzma"  # Should not error
   ```
   If `import lzma` fails (common with pyenv), reinstall Python with xz support:
   ```bash
   brew install xz
   pyenv install 3.13.1  # Rebuild with xz linked
   ```

2. **Claude Code CLI** installed and authenticated
   ```bash
   claude --version
   ```

3. **Docker** installed and running (only needed for evaluation phase)
   ```bash
   docker --version
   docker ps
   ```
   - 50GB+ free disk, 16GB+ RAM, Docker Desktop memory set to 8GB+

4. **Code Lexica backend** running (local or production)
   - Local: `http://localhost:5050/mcp`
   - Production: `https://api.codelexica.com/mcp`

## Installation

```bash
cd /path/to/claudecode_gemini_and_codex_swebench
pip install -r requirements.txt
```

Verify the install:
```bash
python -c "from datasets import load_dataset; print('OK')"
python swe_bench.py list-models
```

## MCP Token Configuration

### 1. Create the token registry

```bash
cp configs/code_lexica_repos.example.json configs/code_lexica_repos.json
```

### 2. Set the MCP URL

Edit `configs/code_lexica_repos.json` and set the `mcp_url`:

- **Local development:** `http://localhost:5050/mcp`
- **Production:** `https://api.codelexica.com/mcp`

### 3. Add per-repo tokens

Each SWE-bench Lite repo needs a Code Lexica project with an MCP token. The 12 repos are:

| Repo | SWE-bench Lite tasks |
|------|---------------------|
| `django/django` | ~100+ |
| `scikit-learn/scikit-learn` | ~30+ |
| `matplotlib/matplotlib` | ~30+ |
| `sympy/sympy` | ~40+ |
| `pytest-dev/pytest` | ~15+ |
| `pallets/flask` | ~5+ |
| `psf/requests` | ~5+ |
| `astropy/astropy` | ~15+ |
| `pydata/xarray` | ~15+ |
| `sphinx-doc/sphinx` | ~20+ |
| `pylint-dev/pylint` | ~10+ |
| `mwaskom/seaborn` | ~5+ |

For each repo:
1. Create a project in Code Lexica and link the GitHub repo
2. Wait for processing to complete (reports must be generated)
3. Generate an MCP token for the project
4. Add the token to `configs/code_lexica_repos.json`

Example completed config:
```json
{
  "mcp_url": "http://localhost:5050/mcp",
  "repos": {
    "django/django": "cl_a1b2c3d4e5f6...",
    "sympy/sympy": "cl_f6e5d4c3b2a1...",
    "scikit-learn/scikit-learn": "cl_your_token_here"
  }
}
```

You don't need all 12 repos configured to start testing. Use `--mcp-repos-only` to run only against repos that have tokens (see below).

> **Security:** `configs/code_lexica_repos.json` is gitignored. Never commit tokens.

## Running Benchmarks

### Baseline Run (Without MCP)

This runs Claude Code against SWE-bench tasks with no additional context:

```bash
# Smoke test: 1 task, no Docker evaluation
python swe_bench.py run --limit 1 --no-eval

# Quick baseline: 10 tasks with evaluation
python swe_bench.py run --quick

# Specify model explicitly
python swe_bench.py run --limit 10 --model sonnet-4.5

# Full baseline: all 300 tasks
python swe_bench.py run --full
```

### MCP Run (With Code Lexica Context)

Add `--mcp` to enable the Code Lexica MCP server. Claude Code will automatically connect to it and use codebase context (code maps, conventions, architecture, etc.) when working on each task:

```bash
# Smoke test with MCP
python swe_bench.py run --limit 1 --no-eval --mcp

# Quick MCP run
python swe_bench.py run --quick --mcp

# Full MCP run
python swe_bench.py run --full --mcp
```

### Filtering by Repo

#### Explicit repo list (`--repos`)

Run only tasks from specific repos:

```bash
# Only django tasks
python swe_bench.py run --limit 10 --repos django/django

# Multiple repos
python swe_bench.py run --limit 20 --repos django/django,sympy/sympy

# Combine with MCP
python swe_bench.py run --limit 10 --mcp --repos django/django,sympy/sympy
```

#### MCP-configured repos only (`--mcp-repos-only`)

Automatically runs only tasks for repos that have tokens in `code_lexica_repos.json`. This is useful when you're onboarding repos incrementally:

```bash
# Run against all repos that have tokens configured
python swe_bench.py run --mcp --mcp-repos-only

# With a limit
python swe_bench.py run --limit 20 --mcp --mcp-repos-only

# Combine with --repos to further narrow (intersection)
# Only runs django IF it has a token configured
python swe_bench.py run --mcp --mcp-repos-only --repos django/django
```

#### `--mcp-repos-only` without `--mcp`

You can use `--mcp-repos-only` as a standalone filter even without enabling MCP. This is useful for running a baseline on the same subset of repos you'll later test with MCP:

```bash
# Baseline on MCP-configured repos (no MCP context, just filtering)
python swe_bench.py run --limit 20 --mcp-repos-only

# Then the MCP run on the same repos
python swe_bench.py run --limit 20 --mcp --mcp-repos-only
```

## A/B Comparison Workflow

### Step 1: Configure tokens for your ready repos

Edit `configs/code_lexica_repos.json` with tokens for repos that have finished processing.

### Step 2: Run the baseline

```bash
python swe_bench.py run --mcp-repos-only --no-eval \
  --model sonnet-4.5 \
  --notes "baseline: no MCP"
```

### Step 3: Run with MCP

```bash
python swe_bench.py run --mcp --mcp-repos-only --no-eval \
  --model sonnet-4.5 \
  --notes "with Code Lexica MCP"
```

### Step 4: Evaluate both runs

```bash
# Evaluate the most recent predictions
python swe_bench.py eval --last 2 --force
```

### Step 5: Compare results

```bash
python swe_bench.py check
```

Results are logged to `benchmark_scores.log` (JSON lines). Each entry includes:
- `generation_score` — % of tasks that produced a patch
- `evaluation_score` — % of tasks where the patch actually fixed the issue
- `mcp_enabled` — whether MCP was active
- `repos_filter` — which repos were included
- `model` — which model was used

## Verifying MCP is Working

### Check the MCP server is reachable

```bash
# Local
curl -s http://localhost:5050/mcp -H "Authorization: Bearer cl_your_token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

### Check server logs during a run

When running with `--mcp`, watch your Code Lexica backend logs. You should see:
- Incoming requests with the Bearer token for each task's repo
- `lastUsedAt` timestamps updating on the MCP tokens
- Resource requests for agent reports (code-map, system-overview, etc.)

### Verify `.mcp.json` injection

During a run, the harness writes `.mcp.json` into each task's temp directory (e.g., `/tmp/swe_bench_django__django-11133/.mcp.json`). You can inspect it:

```bash
# While a task is running:
cat /tmp/swe_bench_django__django-*/.mcp.json 2>/dev/null
```

## Cost and Time Estimates

| Run Size | Tasks | Est. Cost (per condition) | Est. Time (per condition) |
|----------|-------|---------------------------|---------------------------|
| Smoke | 1 | ~$2 | ~10 min |
| Quick | 10 | ~$20 | ~2 hours |
| Standard | 50 | ~$100 | ~8 hours |
| Full | 300 | ~$600 | ~50 hours |

A full A/B comparison (baseline + MCP) at 300 tasks costs ~$1,200 total.

With `--mcp-repos-only`, cost scales with the number of configured repos. If you have 3 of 12 repos configured, expect roughly 25% of the full task count.

## Troubleshooting

### `ModuleNotFoundError: No module named '_lzma'`

Python was compiled without xz/lzma support. Fix:
```bash
brew install xz                  # macOS
pyenv install 3.13.1             # Rebuild Python
```

### `MCP registry not found`

You haven't created the token config:
```bash
cp configs/code_lexica_repos.example.json configs/code_lexica_repos.json
# Edit and add your tokens
```

### `WARNING: No MCP token for repo 'X', running without MCP`

The task's repo doesn't have a token in `code_lexica_repos.json`. Either:
- Add a token for that repo, or
- Use `--mcp-repos-only` to skip repos without tokens

### MCP server not responding

- Verify the server is running: `curl http://localhost:5050/health`
- Check the `mcp_url` in `configs/code_lexica_repos.json`
- Verify the token is valid and not revoked

### Low evaluation scores

This is normal for SWE-bench. Top models score 30-45% on Lite. Focus on the **relative difference** between baseline and MCP runs, not absolute numbers.

## Architecture

```
Phase 1: Patch Generation (runs locally, NOT in Docker)
─────────────────────────────────────────────────────
For each SWE-bench task:
  1. Clone repo to /tmp/swe_bench_<instance_id>/
  2. Checkout base commit
  3. If --mcp: write .mcp.json with per-repo token
  4. Run: claude --dangerously-skip-permissions < prompt
     └─> Claude Code discovers .mcp.json
     └─> Connects to Code Lexica MCP server
     └─> Gets codebase context (code maps, conventions, etc.)
     └─> Edits files to fix the issue
  5. Extract git diff as patch
  6. Clean up temp directory

Phase 2: Evaluation (runs in Docker)
─────────────────────────────────────
For each prediction:
  1. Build Docker container with repo at base commit
  2. Apply the patch
  3. Run the project's test suite
  4. Report pass/fail
  (No MCP involvement — just testing if patches work)
```
