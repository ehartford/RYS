# Runbook: Kimi K2.6 RYS experiments with vLLM

This runbook assumes the current repo is `/home/hotaisle/RYS`, the model is
`/home/hotaisle/models/Kimi-K2.6`, and vLLM is already installed into the RYS
virtualenv from `/home/hotaisle/vllm-rys`.

The vLLM scan path should be run as one vLLM engine group across the 8 GPUs.
That engine uses tensor-parallel ranks internally, so you may see multiple
vLLM worker processes, but they are shards of one model replica, not multiple
full copies of Kimi. Within that one engine run, RYS changes the layer
execution order in memory between configs. It does not rewrite weights or
reload the model between configs.

Do not launch multiple independent vLLM scan workers against the same 8 GPUs.
Each independent `scripts/run_vllm_math_eq_combined_worker.py` command would
load another model replica, which will not fit alongside the first one.

## 0. Start a clean shell

Use `tmux` or another long-running shell for the actual scan.

```bash
cd /home/hotaisle/RYS
source ~/RYS/.venv/bin/activate

export MODEL=/home/hotaisle/models/Kimi-K2.6
export TP=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ROCm/vLLM settings from the known-good Kimi K2.6 serve environment.
export LLVM_PATH=/opt/rocm/llvm
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx942
export VLLM_ROCM_VARIANT=rocm721
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=0
export TRITON_CACHE_DIR=/home/hotaisle/.triton/cache
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# vLLM engine settings that mirror the successful serve command where they
# apply to offline LLM inference.
export RYS_BLOCK_SIZE=1
export RYS_REASONING_PARSER=kimi_k2
export RYS_MM_ENCODER_TP_MODE=data

export RUN_ROOT=/home/hotaisle/RYS/results/kimi-k26-vllm/$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT"/{configs,queues,logs,analysis,exports,beam,surrogate}
```

Adjust `TP` and `CUDA_VISIBLE_DEVICES` together. For example, if you expose
four GPUs, use `export TP=4`.

Known-good API server launch on this machine:

```bash
vllm serve "$MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --block-size 1 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data
```

The RYS scanner uses vLLM's offline `LLM` API, not `vllm serve`. The relevant
serve settings for offline scanning are `--block-size`, `--reasoning-parser`,
and `--mm-encoder-tp-mode`. `--tool-call-parser` and
`--enable-auto-tool-choice` are OpenAI-compatible server options and are not
passed to the offline probe worker.

## 1. Record and verify the local environment

These checks do not load the full Kimi model.

```bash
python - <<'PY' | tee "$RUN_ROOT/environment.txt"
import json
import os
from pathlib import Path

import vllm

model = Path(os.environ["MODEL"])
cfg = json.loads((model / "config.json").read_text())
text_cfg = cfg.get("text_config", cfg)

print("vllm_version:", vllm.__version__)
print("vllm_file:", vllm.__file__)
print("model:", model)
for name in [
    "LLVM_PATH",
    "ROCM_PATH",
    "HIP_PLATFORM",
    "PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_VARIANT",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_RMSNORM",
    "TRITON_CACHE_DIR",
    "XLA_FLAGS",
]:
    print(f"{name}:", os.environ.get(name, ""))
print("architectures:", cfg.get("architectures"))
print("text_num_hidden_layers:", text_cfg.get("num_hidden_layers"))
print("first_k_dense_replace:", text_cfg.get("first_k_dense_replace"))
print("n_routed_experts:", text_cfg.get("n_routed_experts"))
print("num_experts_per_tok:", text_cfg.get("num_experts_per_tok"))
PY
```

Expected local facts:

- vLLM version: `0.19.2rc1.dev66+gb47840019.d20260421`
- text layers: `61`
- dense prefix: `first_k_dense_replace = 1`
- MoE band: layers `1..60`

If vLLM has intentionally changed, either update
`src/workers/vllm_relayer_patch.py` after checking the forward signature, or
set one of these only after confirming compatibility:

```bash
export RYS_VLLM_EXPECTED_VERSION="$(python - <<'PY'
import vllm
print(vllm.__version__)
PY
)"

# Last resort for local hacking only:
# export RYS_VLLM_ALLOW_VERSION_MISMATCH=1
```

Run the fast repo checks:

```bash
python -m pytest \
  tests/test_vllm_relayer_patch.py \
  tests/test_probe_harness.py \
  tests/test_hf_export.py

python scripts/run_vllm_math_eq_combined_worker.py --help >/dev/null
python scripts/init_queue.py --help >/dev/null
python -m hf_export.export_model --help >/dev/null
```

The scanner itself does not need plotting or surrogate-training packages. The
analysis step needs `matplotlib`; the surrogate step needs `xgboost`.

## 2. Run a two-config smoke scan

This is the first command that loads the Kimi checkpoint into one tensor-parallel
vLLM engine group. It runs baseline and one canary repeat. The worker also runs
its own short preflight before the listed configs.

```bash
cat > "$RUN_ROOT/configs/smoke.txt" <<'EOF'
0,0
blocks:30,32
EOF

python scripts/run_vllm_math_eq_combined_worker.py \
  --config-file "$RUN_ROOT/configs/smoke.txt" \
  --combined-results-file "$RUN_ROOT/smoke_combined_results.pkl" \
  --math-results-file "$RUN_ROOT/smoke_math_results.pkl" \
  --eq-results-file "$RUN_ROOT/smoke_eq_results.pkl" \
  --model "$MODEL" \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --block-size "$RYS_BLOCK_SIZE" \
  --reasoning-parser "$RYS_REASONING_PARSER" \
  --mm-encoder-tp-mode "$RYS_MM_ENCODER_TP_MODE" \
  2>&1 | tee "$RUN_ROOT/logs/smoke_vllm.log"
```

Smoke success criteria:

- the log says `Loading vLLM model once (reusing across configs)...`
- preflight baseline and canary both produce non-empty text
- patch status shows a patched DeepseekV2 model
- the canary does not fail with a "patch did not observe non-baseline" error
- the result pickle has two entries

Inspect the smoke results:

```bash
python - <<'PY'
import os
import pickle
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
for name in ["smoke_combined_results.pkl", "smoke_math_results.pkl", "smoke_eq_results.pkl"]:
    path = root / name
    data = pickle.loads(path.read_bytes())
    print(name, "entries=", len(data))
    for key, value in data.items():
        print("  key_len=", len(key), "score=", value.get("score", value.get("combined_score")))
PY
```

If vLLM has NCCL/custom all-reduce issues on your hardware, rerun the worker
with:

```bash
--disable-custom-all-reduce
```

Do not use `--no-enforce-eager` for scans. CUDA graphs can capture one layer
order and invalidate dynamic reordering.

## 3. Build the first coarse MoE-band queue

Start with layer starts in the MoE band (`i >= 1`) and a stride-2 grid. This is
deliberately smaller than the full 1,891-config triangular sweep.

Dry run first:

```bash
python scripts/init_queue.py \
  --num-layers 61 \
  --min-i 1 \
  --i-stride 2 \
  --j-stride 2 \
  --max-span 12 \
  --queue-file "$RUN_ROOT/queues/coarse_queue.json" \
  --results-file "$RUN_ROOT/coarse_combined_results.pkl" \
  --dry-run
```

Create the queue:

```bash
python scripts/init_queue.py \
  --num-layers 61 \
  --min-i 1 \
  --i-stride 2 \
  --j-stride 2 \
  --max-span 12 \
  --queue-file "$RUN_ROOT/queues/coarse_queue.json" \
  --results-file "$RUN_ROOT/coarse_combined_results.pkl"
```

Notes:

- baseline is included unless you pass `--exclude-baseline`
- `--results-file` is used to skip completed configs if you rebuild the queue
- queue entries are explicit layer orders, so later arbitrary multi-block
  configs can use the same worker path

## 4. Run the coarse scan

This keeps one tensor-parallel vLLM engine resident and dynamically changes the
execution order between queue entries. Run one copy of this command for the
whole 8-GPU model replica.

```bash
python scripts/run_vllm_math_eq_combined_worker.py \
  --queue-file "$RUN_ROOT/queues/coarse_queue.json" \
  --combined-results-file "$RUN_ROOT/coarse_combined_results.pkl" \
  --math-results-file "$RUN_ROOT/coarse_math_results.pkl" \
  --eq-results-file "$RUN_ROOT/coarse_eq_results.pkl" \
  --model "$MODEL" \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --block-size "$RYS_BLOCK_SIZE" \
  --reasoning-parser "$RYS_REASONING_PARSER" \
  --mm-encoder-tp-mode "$RYS_MM_ENCODER_TP_MODE" \
  2>&1 | tee "$RUN_ROOT/logs/coarse_vllm.log"
```

If the process stops, rerun the same command after rebuilding the queue with
the same `--results-file`. Completed configs will be skipped:

```bash
python scripts/init_queue.py \
  --num-layers 61 \
  --min-i 1 \
  --i-stride 2 \
  --j-stride 2 \
  --max-span 12 \
  --queue-file "$RUN_ROOT/queues/coarse_queue.json" \
  --results-file "$RUN_ROOT/coarse_combined_results.pkl"
```

## 5. Analyze the coarse scan

Install or activate plotting dependencies if this fails with
`ModuleNotFoundError: matplotlib`.

```bash
python scripts/analyze_results.py \
  --math-scores "$RUN_ROOT/coarse_math_results.pkl" \
  --eq-scores "$RUN_ROOT/coarse_eq_results.pkl" \
  --out-dir "$RUN_ROOT/analysis/coarse" \
  --num-layers 61 \
  --top-n 25 \
  --title kimi-k26-vllm-coarse-shared-cache
```

Primary outputs:

- `$RUN_ROOT/analysis/coarse/top25_balanced_z_delta.json`
- `$RUN_ROOT/analysis/coarse/top25_balanced_z_delta.csv`
- `$RUN_ROOT/analysis/coarse/balanced_summary.json`
- `$RUN_ROOT/analysis/coarse/analysis_scores.pkl`

The vLLM scan stores canonical execution-order keys. For those keys, the
analysis summary and top tables are the source of truth; heatmap rendering may
be skipped for non-legacy keyspaces.

## 6. Densify around promising regions

Use the coarse top table and logs to choose windows. Example: densify starts
`21..43`, all spans up to 16, stride 1.

```bash
python scripts/init_queue.py \
  --num-layers 61 \
  --min-i 21 \
  --max-i 43 \
  --i-stride 1 \
  --j-stride 1 \
  --max-span 16 \
  --queue-file "$RUN_ROOT/queues/dense_21_43_queue.json" \
  --results-file "$RUN_ROOT/dense_21_43_combined_results.pkl" \
  --skip-existing "$RUN_ROOT/coarse_combined_results.pkl" \
  --dry-run

python scripts/init_queue.py \
  --num-layers 61 \
  --min-i 21 \
  --max-i 43 \
  --i-stride 1 \
  --j-stride 1 \
  --max-span 16 \
  --queue-file "$RUN_ROOT/queues/dense_21_43_queue.json" \
  --results-file "$RUN_ROOT/dense_21_43_combined_results.pkl" \
  --skip-existing "$RUN_ROOT/coarse_combined_results.pkl"

python scripts/run_vllm_math_eq_combined_worker.py \
  --queue-file "$RUN_ROOT/queues/dense_21_43_queue.json" \
  --combined-results-file "$RUN_ROOT/dense_21_43_combined_results.pkl" \
  --math-results-file "$RUN_ROOT/dense_21_43_math_results.pkl" \
  --eq-results-file "$RUN_ROOT/dense_21_43_eq_results.pkl" \
  --model "$MODEL" \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --block-size "$RYS_BLOCK_SIZE" \
  --reasoning-parser "$RYS_REASONING_PARSER" \
  --mm-encoder-tp-mode "$RYS_MM_ENCODER_TP_MODE" \
  2>&1 | tee "$RUN_ROOT/logs/dense_21_43_vllm.log"
```

Analyze each dense run the same way:

```bash
python scripts/analyze_results.py \
  --math-scores "$RUN_ROOT/dense_21_43_math_results.pkl" \
  --eq-scores "$RUN_ROOT/dense_21_43_eq_results.pkl" \
  --out-dir "$RUN_ROOT/analysis/dense_21_43" \
  --num-layers 61 \
  --top-n 25 \
  --title kimi-k26-vllm-dense-21-43-shared-cache
```

## 7. Optional: beam refinement with vLLM

Beam search uses the vLLM worker for candidate evaluation. Inside each vLLM
worker run, the model is loaded once and configs are dynamic. With
`--worker-backend vllm`, the current beam driver launches one combined vLLM
worker at a time for each seed/depth evaluation phase; it does not run parallel
Math and EQ model replicas. Expect a reload at phase or depth boundaries.

```bash
python scripts/beam_search.py \
  --model-path "$MODEL" \
  --num-layers 61 \
  --seed-math-results "$RUN_ROOT/coarse_math_results.pkl" \
  --seed-eq-results "$RUN_ROOT/coarse_eq_results.pkl" \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --work-dir "$RUN_ROOT/beam" \
  --worker-backend vllm \
  --vllm-tensor-parallel-size "$TP" \
  --vllm-gpu-memory-utilization 0.90 \
  --vllm-max-model-len 4096 \
  --vllm-max-num-seqs 32 \
  --vllm-max-num-batched-tokens 8192 \
  --vllm-block-size "$RYS_BLOCK_SIZE" \
  --vllm-reasoning-parser "$RYS_REASONING_PARSER" \
  --vllm-mm-encoder-tp-mode "$RYS_MM_ENCODER_TP_MODE" \
  --beam-width 8 \
  --seed-top-k 24 \
  --pool-size 48 \
  --expand-per-node 16 \
  --max-candidates-per-depth 128 \
  --max-depth 3 \
  --max-extra-layers 16 \
  2>&1 | tee "$RUN_ROOT/logs/beam_vllm.log"
```

If avoiding all reloads is more important than using beam expansion, build an
explicit config file and send it through
`scripts/run_vllm_math_eq_combined_worker.py --config-file ...` instead.

## 8. Optional: surrogate candidate pass

This step needs `xgboost`. It is for ranking candidates, not proving a final
result.

Train from the single-block and beam/dense results you trust:

```bash
python scripts/train_surrogate.py \
  --single-block-math-results "$RUN_ROOT/coarse_math_results.pkl" \
  --single-block-eq-results "$RUN_ROOT/coarse_eq_results.pkl" \
  --beam-math-results "$RUN_ROOT/beam/beam_math_results.pkl" \
  --beam-eq-results "$RUN_ROOT/beam/beam_eq_results.pkl" \
  --out-dir "$RUN_ROOT/surrogate/model" \
  --num-layers 61
```

Generate and score candidates:

```bash
python scripts/generate_candidates.py \
  --out-file "$RUN_ROOT/surrogate/candidates.csv" \
  --num-candidates 2000000 \
  --num-layers 61 \
  --max-extra-layers 16 \
  --max-repeat-per-layer 3 \
  --seed 42

python scripts/score_candidates.py \
  --candidates-file "$RUN_ROOT/surrogate/candidates.csv" \
  --model-method "$RUN_ROOT/surrogate/model/model_method_b.json" \
  --model-math "$RUN_ROOT/surrogate/model/model_math_delta.json" \
  --model-eq "$RUN_ROOT/surrogate/model/model_eq_delta.json" \
  --out-dir "$RUN_ROOT/surrogate/scored" \
  --num-layers 61 \
  --top-k 100

python scripts/build_topk_config.py \
  --top-candidates-csv "$RUN_ROOT/surrogate/scored/top_candidates.csv" \
  --out-config "$RUN_ROOT/configs/surrogate_top100.txt" \
  --out-manifest "$RUN_ROOT/surrogate/top100_manifest.json" \
  --num-layers 61 \
  --top-k 100
```

Measure surrogate-selected candidates with one resident vLLM worker:

```bash
python scripts/run_vllm_math_eq_combined_worker.py \
  --config-file "$RUN_ROOT/configs/surrogate_top100.txt" \
  --combined-results-file "$RUN_ROOT/surrogate_top100_combined_results.pkl" \
  --math-results-file "$RUN_ROOT/surrogate_top100_math_results.pkl" \
  --eq-results-file "$RUN_ROOT/surrogate_top100_eq_results.pkl" \
  --model "$MODEL" \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --block-size "$RYS_BLOCK_SIZE" \
  --reasoning-parser "$RYS_REASONING_PARSER" \
  --mm-encoder-tp-mode "$RYS_MM_ENCODER_TP_MODE" \
  2>&1 | tee "$RUN_ROOT/logs/surrogate_top100_vllm.log"
```

## 9. Optional: add a small code probe

The current code-probe scorer is a lightweight deterministic placeholder: it
checks whether the configured reference string appears in the raw output. Use
it only as a small extra signal unless you replace the harness with real test
execution.

Dataset shape can be a JSON object or list. Example object:

```json
{
  "two_sum": {
    "prompt": "Write a Python function two_sum(nums, target) that returns indices.",
    "reference": "def two_sum"
  }
}
```

Worker flags:

```bash
--code-dataset-path datasets/code_probe.json \
--code-results-file "$RUN_ROOT/code_results.pkl" \
--code-max-new 256
```

## 10. Export top candidates as physical HF checkpoints

The vLLM scan uses shared-cache runtime relayering. Final candidates should be
exported as physical duplicated-layer checkpoints and rescored in an unpatched
runtime before you treat them as real winners.

Extract the top spec from an analysis JSON:

```bash
export TOP_SPEC="$(python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / "analysis/coarse/top25_balanced_z_delta.json"
rows = json.loads(path.read_text())
print(rows[0]["relayer"])
PY
)"

echo "$TOP_SPEC"
```

Dry-run export first:

```bash
python -m hf_export.export_model \
  --source "$MODEL" \
  --output "$RUN_ROOT/exports/top1_dry_run" \
  --spec "$TOP_SPEC" \
  --dry-run \
  --overwrite
```

Then export the actual checkpoint:

```bash
python -m hf_export.export_model \
  --source "$MODEL" \
  --output "$RUN_ROOT/exports/top1" \
  --spec "$TOP_SPEC" \
  --overwrite
```

The dry run should write config/manifest files only. The real export rewrites
safetensor shards and will need substantial disk space.

## 11. Rescore exported checkpoints in unmodified vLLM

The scan harness monkey-patches vLLM. The export check should not. Use a normal
vLLM or HF evaluation path against `$RUN_ROOT/exports/top1`.

Minimum confirmation:

- baseline score from the original Kimi checkpoint
- candidate score from the exported checkpoint
- same Math/EQ datasets and generation settings
- no `src.workers.vllm_relayer_patch` import in the validation runner

If exported scores do not match the scanned candidate within expected sampling
noise, investigate cache semantics and export-layer ordering before trusting
the candidate.

## 12. Important semantics and failure modes

- Dynamic scan: one loaded tensor-parallel vLLM engine group, execution order
  changed through `collective_rpc` between configs.
- vLLM may spawn multiple worker processes/ranks for tensor parallelism. Those
  ranks are one sharded model replica. They are not separate full Kimi loads.
- Do not run multiple independent vLLM scan commands on the same 8 GPUs unless
  you intentionally have enough free VRAM for another full model replica.
- Repeated layer cache: `shared_cache`. A repeated source layer reuses the same
  physical attention cache slot during scan-time scoring.
- Final artifact: exported checkpoints physically duplicate layers and get
  normal per-physical-layer cache slots.
- Always scan with eager execution. Do not enable CUDA graph capture for dynamic
  relayering.
- Pipeline parallelism is not supported by the patch for non-baseline orders.
  Use tensor parallelism.
- If every config scores exactly like baseline, assume the patch did not land
  until proven otherwise. Re-run the smoke scan and inspect patch statuses.
- If you change vLLM, re-check the target forward method before allowing a
  version mismatch.
