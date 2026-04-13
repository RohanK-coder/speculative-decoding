# SpecuPipe: Streaming Speculative Decoding Playground

SpecuPipe is a hackathon project that demonstrates and evaluates **speculative decoding** for LLM inference.
It compares:

- **Baseline greedy decoding** (target model only),
- **Speculative decoding** (draft + target verification),
- and an optional **naive multi-token method** (fast but can drift from baseline output).

The project includes a live Streamlit dashboard, reproducible experiment scripts, correctness checks, and plotting/export utilities.

---

## Why this project

Autoregressive decoding is often latency-bound because each token must be generated sequentially.
Speculative decoding addresses this by:

1. letting a draft model propose multiple tokens,
2. verifying them with the target model,
3. committing accepted tokens and rolling back rejected ones.

This project helps you study the trade-off between **speed**, **acceptance rate**, **rollback overhead**, and **output correctness**.

---

## Core Features

- Baseline target-model greedy generation
- Speculative decoding with:
  - configurable depth `k`
  - adaptive depth mode
  - hybrid gating mode
  - category-aware policy mode in grid experiments
- Acceptance/rejection and rollback accounting
- Logical KV-cache state tracking and history
- Rich per-run metrics:
  - speedup, tokens/sec, acceptance rate
  - rollbacks, wasted draft tokens, bottleneck/utilization proxies
  - energy and KV-overhead proxy metrics
- Streamlit live decode dashboard
- Batch experiment pipelines + CSV/JSON outputs + plots
- Correctness validation against baseline output

---

## Repository Structure

```text
specupipe/
  app.py                         # Streamlit dashboard
  core/
    baseline.py                  # Baseline greedy decode
    speculative.py               # Speculative algorithm
    streaming_decode.py          # Token-by-token streaming generators
    naive_multitoken.py          # Naive draft-only comparator
    models.py                    # HF model/tokenizer wrapper
    cache.py                     # KV-cache manager + history
    metrics.py                   # Step and aggregate metric containers
    prompting.py                 # Family-specific prompt formatting
    analysis.py                  # Speedup and metric helpers
    utils.py                     # Device helpers and utilities
  experiments/
    run_single.py
    run_grid.py
    compare_algorithms.py
    compare_model_families.py
    validate_correctness.py
    plot_results.py
    export_best_configs.py
  tests/
    test_equivalence.py
    test_cache.py
  outputs/
    results/
    plots/
  requirements.txt
```

---

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (Optional) Log exact environment for reproducibility

```bash
python -V
pip freeze > outputs/results/repro_requirements_lock.txt
```

---

## Quick Start

### Launch the dashboard

```bash
streamlit run app.py
```

In the sidebar:
- pick model family (`gpt2`, `tinyllama`, `qwen`),
- choose mode (`fixed`, `adaptive`, `hybrid`),
- set speculation depth `k`,
- run and compare baseline vs speculative results live.

---

## CLI Usage

Run all commands from the `specupipe/` directory.

### 1) Single baseline vs speculative run

```bash
python experiments/run_single.py \
  --prompt "Speculative decoding is useful because" \
  --draft_model distilgpt2 \
  --target_model gpt2 \
  --max_new_tokens 50 \
  --k 4 \
  --device cpu
```

### 2) Compare baseline vs naive vs speculative

```bash
python experiments/compare_algorithms.py \
  --prompt "Explain why efficient inference matters in production systems." \
  --draft_model distilgpt2 \
  --target_model gpt2 \
  --max_new_tokens 32 \
  --k 3 \
  --device cpu \
  --mode hybrid
```

### 3) Compare model families

```bash
python experiments/compare_model_families.py \
  --family gpt2 \
  --question "What is speculative decoding?" \
  --max_new_tokens 64 \
  --k 3 \
  --device cpu \
  --mode hybrid
```

### 4) Run full grid experiment (main benchmark pipeline)

```bash
python experiments/run_grid.py
```

Generates:
- `outputs/results/grid_results.csv`
- `outputs/results/grid_results.json`
- `outputs/results/grid_summary.csv`
- `outputs/results/category_summary.csv`

### 5) Plot benchmark results

```bash
python experiments/plot_results.py
```

Generates PNG plots in `outputs/plots/`.

### 6) Export best/worst configurations

```bash
python experiments/export_best_configs.py
```

Generates:
- `outputs/results/best_overall_configs.csv`
- `outputs/results/worst_overall_configs.csv`
- `outputs/results/best_by_output_length.csv`
- `outputs/results/best_by_category.csv`

### 7) Validate output correctness

```bash
python experiments/validate_correctness.py
```

Generates:
- `outputs/results/validation_results.csv`

---

## Reproducibility Guide

Use this section directly in your hackathon submission to show reproducibility discipline.

### Reproducibility checklist

1. **Use the same Python + dependency versions**
   - Create a clean virtual environment.
   - Install from `requirements.txt`.
   - Save `pip freeze` output.
2. **Use fixed command sequence**
   - First run `run_grid.py`,
   - then `plot_results.py`,
   - then `export_best_configs.py`,
   - then `validate_correctness.py`.
3. **Use a consistent device**
   - Prefer `--device cpu` for portability.
4. **Avoid changing prompt/model sets during reproduction**
   - Keep built-in prompt lists and depth sets unchanged.
5. **Store artifacts**
   - Keep all generated files in `outputs/results/` and `outputs/plots/`.

### Recommended reproducible run script

```bash
source .venv/bin/activate
python -V
pip freeze > outputs/results/repro_requirements_lock.txt

python experiments/run_grid.py
python experiments/plot_results.py
python experiments/export_best_configs.py
python experiments/validate_correctness.py
```

### Notes on variance

- GPU/MPS execution can change timing behavior; compare speedups on the same hardware class.
- First run may include model download overhead from Hugging Face.
- Throughput metrics are hardware-dependent; correctness (`output_match`) is the key algorithmic sanity signal.

---

## Git Hygiene

This repository ignores generated run artifacts so commits stay clean and reviewable.

- Ignored on purpose:
  - `outputs/results/*`
  - `outputs/plots/*`
  - `outputs/logs/*`
  - local virtualenv/caches (`.venv/`, `.cache/`, etc.)
- Tracked on purpose:
  - source code under `core/`, `experiments/`, `tests/`, and `app.py`
  - `README.md` and project configuration files
  - output directory placeholders (`outputs/**/.gitkeep`)

For reproducibility reporting, include these as attachments in your submission:

- `outputs/results/repro_requirements_lock.txt` (your `pip freeze`)
- summary artifacts such as:
  - `outputs/results/grid_summary.csv`
  - `outputs/results/category_summary.csv`
  - `outputs/results/validation_results.csv`

---

## Running Tests

```bash
pytest -q
```

Current tests focus on:
- speculative-vs-baseline equivalence behavior,
- KV-cache manager transitions.

---

## Default Model Pairs

- GPT-2 family:
  - Draft: `distilgpt2`
  - Target: `gpt2`
- TinyLlama family:
  - Draft/Target: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Qwen family (dashboard):
  - Draft/Target: `Qwen/Qwen2.5-1.5B-Instruct`

---

## Hackathon Pitch (one-liner)

SpecuPipe shows how speculative decoding can reduce LLM inference latency while preserving baseline output quality through explicit verification, rollback, and measurement.

