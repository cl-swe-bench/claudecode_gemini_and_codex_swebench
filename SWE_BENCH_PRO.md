# Running SWE-bench Pro with Claude Code

This guide walks through running SWE-bench Pro tasks with this harness (patch generation + token tracking) and evaluating with the [ScaleAI SWE-bench Pro evaluation harness](https://github.com/scaleapi/SWE-bench_Pro-os).

## Architecture

```
This Harness (generation)              ScaleAI Harness (evaluation)
┌─────────────────────────┐            ┌──────────────────────────┐
│ 1. Load SWE-bench Pro   │            │ 4. Load patches JSON     │
│ 2. Run Claude Code      │  convert   │ 5. Run in Docker         │
│ 3. Save predictions     │ ────────>  │ 6. Check test results    │
│    (.jsonl + tokens)    │            │ 7. Output eval_results   │
└─────────────────────────┘            └──────────────────────────┘
```

- **Generation** happens here: clone repo, prompt Claude Code, extract git diff, track tokens/cost
- **Evaluation** happens in the ScaleAI repo: apply patch in Docker, run tests, report pass/fail

## Prerequisites

### This harness

```bash
cd /Users/trees/Source/claudecode_gemini_and_codex_swebench
pip install -r requirements.txt
```

### ScaleAI evaluation harness

```bash
cd /Users/trees/Source/SWE-bench_Pro-os
pip install -r requirements.txt
```

### Docker

Docker must be installed and running. The ScaleAI harness uses prebuilt Docker images from Docker Hub (`jefzda/sweap-images`).

On Apple Silicon Macs, the evaluator auto-detects and uses `--platform linux/amd64`.

## Quick Start: Single Instance

### Step 1: Find an instance ID

List available instances from the dataset:

```bash
cd /Users/trees/Source/claudecode_gemini_and_codex_swebench

python -c "
from datasets import load_dataset
ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
for row in ds:
    print(f\"{row['instance_id']}  repo={row['repo']}\")
" | head -20
```

Pick an instance ID, e.g. `instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2`.

### Step 2: Generate a patch with Claude Code

```bash
python swe_bench.py run \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 1 \
    --no-eval \
    --notes "swe-bench-pro single test"
```

To filter by repo:

```bash
python swe_bench.py run \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 1 \
    --repos ansible/ansible \
    --no-eval
```

To run with MCP (Code Lexica):

```bash
python swe_bench.py run \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 1 \
    --repos ansible/ansible \
    --mcp \
    --no-eval \
    --notes "swe-bench-pro with MCP"
```

This outputs:
- `predictions/predictions_YYYYMMDD_HHMMSS.jsonl` -- predictions with token usage
- `predictions/predictions_YYYYMMDD_HHMMSS.json` -- same data as JSON array

### Step 3: Convert predictions to ScaleAI format

```bash
python swe_bench.py convert-pro predictions/predictions_YYYYMMDD_HHMMSS.jsonl
```

This creates `predictions/patches_pro_YYYYMMDD_HHMMSS.json` in the format the ScaleAI evaluator expects:

```json
[
  {
    "instance_id": "instance_...",
    "patch": "diff --git ...",
    "prefix": "claude-code"
  }
]
```

Options:
- `--output custom_path.json` -- custom output path
- `--prefix my-model` -- custom prefix for output filenames
- `--include-empty` -- include predictions with empty patches

### Step 4: Evaluate with ScaleAI harness

```bash
cd /Users/trees/Source/SWE-bench_Pro-os

python swe_bench_pro_eval.py \
    --raw_sample_path helper_code/sweap_eval_full_v2.jsonl \
    --patch_path /Users/trees/Source/claudecode_gemini_and_codex_swebench/predictions/patches_pro_YYYYMMDD_HHMMSS.json \
    --output_dir eval_output \
    --scripts_dir run_scripts \
    --dockerhub_username jefzda \
    --use_local_docker
```

Results are written to `eval_output/eval_results.json`:

```json
{
  "instance_ansible__ansible-0ea40...": true,
  ...
}
```

`true` = all required tests pass, `false` = at least one test fails.

## Comparing Baseline vs MCP

### Run baseline

```bash
cd /Users/trees/Source/claudecode_gemini_and_codex_swebench

python swe_bench.py run \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 10 \
    --no-eval \
    --notes "baseline"
# Note the prediction file name, e.g. predictions_20260413_120000.jsonl
```

### Run with MCP

```bash
python swe_bench.py run \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 10 \
    --mcp \
    --no-eval \
    --notes "with-mcp"
# Note the prediction file name, e.g. predictions_20260413_130000.jsonl
```

### Convert both

```bash
python swe_bench.py convert-pro predictions/predictions_20260413_120000.jsonl \
    --prefix baseline

python swe_bench.py convert-pro predictions/predictions_20260413_130000.jsonl \
    --prefix with-mcp
```

### Evaluate both

```bash
cd /Users/trees/Source/SWE-bench_Pro-os

python swe_bench_pro_eval.py \
    --raw_sample_path helper_code/sweap_eval_full_v2.jsonl \
    --patch_path /Users/trees/Source/claudecode_gemini_and_codex_swebench/predictions/patches_pro_20260413_120000.json \
    --output_dir eval_baseline \
    --scripts_dir run_scripts \
    --dockerhub_username jefzda \
    --use_local_docker

python swe_bench_pro_eval.py \
    --raw_sample_path helper_code/sweap_eval_full_v2.jsonl \
    --patch_path /Users/trees/Source/claudecode_gemini_and_codex_swebench/predictions/patches_pro_20260413_130000.json \
    --output_dir eval_mcp \
    --scripts_dir run_scripts \
    --dockerhub_username jefzda \
    --use_local_docker
```

### Compare token usage

The prediction JSONL files contain per-instance `token_usage` fields. Compare them:

```bash
cd /Users/trees/Source/claudecode_gemini_and_codex_swebench

python -c "
import json, jsonlines

def summarize(path):
    total_tokens, total_cost, count = 0, 0, 0
    with jsonlines.open(path) as r:
        for obj in r:
            usage = obj.get('token_usage', {})
            total_tokens += usage.get('total_tokens', 0)
            total_cost += usage.get('cost_usd', 0) or 0
            count += 1
    print(f'  Instances: {count}')
    print(f'  Total tokens: {total_tokens:,}')
    print(f'  Avg tokens/task: {total_tokens // max(count,1):,}')
    print(f'  Total cost: \${total_cost:.2f}')
    print(f'  Avg cost/task: \${total_cost / max(count,1):.3f}')

print('BASELINE:')
summarize('predictions/predictions_20260413_120000.jsonl')
print()
print('WITH MCP:')
summarize('predictions/predictions_20260413_130000.jsonl')
"
```

## Notes

- **`--no-eval` is required** when running SWE-bench Pro with this harness. The built-in evaluation uses the `swebench` library which only supports SWE-bench Lite/Verified, not Pro.
- **Docker images are ~2-5 GB each**. The first evaluation run for an instance will pull the image; subsequent runs use the cached image.
- **The `jefzda` Docker Hub account** hosts the official ScaleAI prebuilt images. No additional setup is needed.
- **The ScaleAI repo includes `sweap_eval_full_v2.jsonl`** in `helper_code/` which serves as the `--raw_sample_path`. You do not need to export the dataset separately unless you want a filtered subset.
- **Instance IDs in SWE-bench Pro** look like `instance_org__repo-commithash-vhash`, which is different from Lite's `org__repo-number` format. This is handled automatically when you load the `ScaleAI/SWE-bench_Pro` dataset.

## Exporting a Dataset Subset (Optional)

If you want to evaluate only a subset, you can export filtered dataset instances:

```bash
python swe_bench.py export-dataset \
    --dataset ScaleAI/SWE-bench_Pro \
    --limit 50 \
    --output swe_bench_pro_subset.jsonl
```

Then use `swe_bench_pro_subset.jsonl` as the `--raw_sample_path` in the ScaleAI evaluator.
