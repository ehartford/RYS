# RYS

Reproducibility framework for relayering experiments:

- `Scanner`: `(i, j)` sweeps and explicit-config benchmarking
- `Probes`: canonical public Math and EQ probe sets
- `Beam search`: multi-block composition search
- `Surrogate`: XGBoost training, candidate generation, and top-k benchmarking
- `Model builder`: export relayered Hugging Face checkpoints
- `Heatmaps`: brain-scan plotting and balanced Math+EQ analysis

This repo is the clean public subset of the original private experiment code. It is intended to let you rerun the published workflows without the historical runs, one-off notebooks, or blog-specific analysis.

## Layout

- `datasets/`
  - `math_16.json`
  - `math_120.json`
  - `eq_16.json`
  - `eq_140.json`
  - `manifest.json`
- `src/core/`
  - config parsing and layer-list expansion
  - Hugging Face relayer wrappers for dense and MoE-style decoder stacks
- `src/workers/`
  - Math and EQ benchmark workers
  - queue handling
  - model loading helpers
- `src/utils/`
  - balanced Math+EQ analysis
  - heatmap helpers
  - surrogate feature utilities
- `scripts/`
  - sweep setup
  - ExLlama workers
  - beam search
  - surrogate pipeline
  - heatmap analysis
- `hf_export/`
  - relayered checkpoint export
  - Hugging Face upload helper

## Probe Sets

This repo ships the canonical public probe files used in this work:

- `datasets/math_16.json`
- `datasets/math_120.json`
- `datasets/eq_16.json`
- `datasets/eq_140.json`

`datasets/manifest.json` records provenance, checksums, and basic metadata. Dataset generation and calibration code is intentionally not included here; this repo publishes the fixed benchmark subsets used for measurement.

## Environment

Python workflow uses `uv`.

Install project dependencies:

```bash
uv sync
```

The ExLlama scanner requires:

- a local `exllamav3` checkout
- an ExLlama-compatible model directory
- the Docker image in `docker/Dockerfile.exllama` or `docker/Dockerfile.exllama_precompiled`

Set:

```bash
export EXLLAMAV3_PATH=/path/to/exllamav3
```

## Scanner

### 1. Prepare a full `(i, j)` sweep queue

Example for a 64-layer model:

```bash
uv run python scripts/init_queue.py \
  --num-layers 64 \
  --queue-file results/demo/queue.json \
  --results-file results/demo/combined_results.pkl
```

This produces a queue of canonical layer-list configs. The baseline `(0,0)` is included by default.

### 2. Run the ExLlama combined Math+EQ worker

This worker loads the EXL3 model once and evaluates all Math and EQ prompts in one mixed generation pass per config:

```bash
uv run python scripts/run_exllama_math_eq_combined_worker.py \
  --queue-file results/demo/queue.json \
  --combined-results-file results/demo/combined_results.pkl \
  --math-results-file results/demo/math_results.pkl \
  --eq-results-file results/demo/eq_results.pkl \
  --model-dir /path/to/model.exl3 \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --auto-cache
```

For containerized runs, use:

```bash
./scripts/run_exllama_docker.sh
```

Override via environment variables such as `MODEL_DIR`, `QUEUE_FILE`, `MATH_RESULTS_FILE`, `EQ_RESULTS_FILE`, `MATH_DATASET`, and `EQ_DATASET`.

### 3. Render balanced Math+EQ analysis and heatmaps

```bash
uv run python scripts/analyze_results.py \
  --math-scores results/demo/math_results.pkl \
  --eq-scores results/demo/eq_results.pkl \
  --out-dir results/demo/analysis \
  --num-layers 64
```

Outputs include:

- `top10_balanced_z_delta.csv`
- `top10_balanced_z_delta.json`
- scatter plots
- balanced heatmap data

## Beam Search

Beam search composes multiple repeated blocks and benchmarks only unseen configs.

Example:

```bash
uv run python scripts/beam_search.py \
  --model-path /path/to/hf-model \
  --num-layers 64 \
  --seed-math-results results/demo/math_results.pkl \
  --seed-eq-results results/demo/eq_results.pkl \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --work-dir results/demo/beam-search
```

Notes:

- Beam search benchmarks candidates with the Hugging Face worker path in `src/workers/`.
- The input seed result pickles should come from an already measured single-block scan.
- Search state is resume-friendly inside `--work-dir`.

## Surrogate Pipeline

The surrogate uses per-layer repeat counts as features.

### 1. Train

```bash
uv run python scripts/train_surrogate.py \
  --single-block-math-results results/demo/math_results.pkl \
  --single-block-eq-results results/demo/eq_results.pkl \
  --beam-math-results results/demo/beam-search/beam_math_results.pkl \
  --beam-eq-results results/demo/beam-search/beam_eq_results.pkl \
  --out-dir results/demo/surrogate \
  --num-layers 64
```

### 2. Generate candidate count vectors

```bash
uv run python scripts/generate_candidates.py \
  --num-layers 64 \
  --max-extra-layers 12 \
  --count 2000000 \
  --output-csv results/demo/surrogate/candidates.csv
```

### 3. Score candidate vectors with the trained model

```bash
uv run python scripts/score_candidates.py \
  --model-dir results/demo/surrogate \
  --candidates-file results/demo/surrogate/candidates.csv \
  --output-csv results/demo/surrogate/top_scored.csv \
  --top-k 100
```

### 4. Convert top candidates back to config specs for real benchmarking

```bash
uv run python scripts/build_topk_config.py \
  --top-candidates-csv results/demo/surrogate/top_scored.csv \
  --num-layers 64 \
  --output-config results/demo/surrogate/top100.config
```

Benchmark those configs with the same Math/EQ worker harness used elsewhere. Predicted scores are for ranking only; measured benchmark scores are the final experiment outputs.

## Model Builder

The exporter creates a normal Hugging Face `safetensors` checkpoint with duplicated layers physically written into the shard files.

Single block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-block-30-34 \
  --blocks "30,34"
```

Multi-block:

```bash
uv run python -m hf_export.export_model \
  --source /path/to/base-model \
  --source-repo-id some/model \
  --output exports/model-31_34__43_45 \
  --blocks "31,34;43,45"
```

Upload:

```bash
export HF_TOKEN=...

uv run python -m hf_export.upload_to_hf \
  --folder exports/model-block-30-34 \
  --repo-id your-name/model-block-30-34
```

The export manifest is written to `rys_export_manifest.json`.

## Heatmaps

`src/utils/heatmaps.py` contains the reusable plotting helpers.

For normal `(i, j)` brain scans, use `scripts/analyze_results.py`. For per-layer repeat sweeps, use:

```bash
uv run python scripts/plot_repeat_heatmaps.py \
  --results-file results/demo/repeatx8_math_results.pkl \
  --manifest-file results/demo/repeatx8_manifest.json \
  --out-dir results/demo/repeatx8_heatmaps
```

## Notes

- `eq_16` and `eq_140` are first-pass-only EQ subsets.
- The public repo keeps the benchmark datasets fixed; it does not regenerate or recalibrate them.
- The Hugging Face builder currently supports decoder-layer prefixes detected under:
  - `model.language_model.layers.`
  - `model.layers.`
  - `language_model.layers.`
- Unsupported architectures should fail early rather than silently exporting a bad checkpoint.
