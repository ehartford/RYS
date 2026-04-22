# Proposal: vLLM-backed RYS scan for Kimi K2.6

## Goal

Add a vLLM scan backend to this RYS repo so Kimi K2.6 can be evaluated without EXL3 conversion. The scan should keep the model resident in vLLM, set a relayer execution order for each queue entry, score the existing Math/EQ probes plus an optional small code probe, and write the same result pickle shapes used by the existing ExLlama and HF paths.

This proposal is scoped to changes inside `/home/hotaisle/RYS`. The installed vLLM tree is `/home/hotaisle/vllm-rys` and should not need a source fork for the first implementation.

## Verified Local Facts

- Environment: `source ~/RYS/.venv/bin/activate`
- Installed vLLM package path: `/home/hotaisle/vllm-rys/vllm`
- Installed vLLM version: `0.19.2rc1.dev66+gb47840019.d20260421`
- Model path: `~/models/Kimi-K2.6`
- Model architecture: outer `KimiK25ForConditionalGeneration`, inner vLLM text model resolved through `DeepseekV2ForCausalLM`
- Text layer count: `text_config.num_hidden_layers = 61`
- Checkpoint text-layer weight prefix: `language_model.model.layers.`
- vLLM module path: `model.language_model.model.layers` inside the loaded Kimi wrapper
- Layer 0 is dense MLP (`first_k_dense_replace = 1`); layers 1-60 are MoE with 384 routed experts and 8 active experts per token
- Current HF exporter prefix detection does not include `language_model.model.layers.`

## Existing RYS Pieces To Reuse

- `src/core/layer_config.py`
  - Keep canonical layer orders as explicit `layers:` lists.
  - Reuse `parse_queue_entry_layers`, `layer_key`, `layer_spec_string`, and block expansion helpers.

- `src/workers/shared_queue.py`
  - Reuse queue claiming and result pickle locking.

- `scripts/run_exllama_math_eq_combined_worker.py`
  - Use as the closest behavioral template for a new vLLM combined worker: one model load, one mixed Math/EQ prompt set per config, three result pickles.

- `src/workers/math_worker.py` and `src/workers/eq_worker.py`
  - Reuse scoring and extraction helpers: integer extraction, Math score, EQ score, and EQ parser.

- `scripts/analyze_results.py`, `src/utils/math_eq_analysis.py`, surrogate scripts, and heatmap scripts
  - No format changes should be required if vLLM writes canonical tuple keys and `{"score": ..., "responses": ...}` values.

## Important Semantic Choice

The fastest monkey patch is to replace the Python loop in vLLM's `DeepseekV2Model.forward` and call existing physical layers in a custom order. This shares the physical attention object and KV cache slot for a repeated source layer.

That shared-cache scan mode is useful for coarse search because it avoids reloading or materializing duplicated weights, but it is not perfectly identical to an exported checkpoint during multi-token generation. In an exported model, each duplicated execution position owns a distinct KV cache slot. In shared-cache mode, a repeated source layer overwrites its own cache slot on the later visit. The first generated token after prefill is close to the intended semantics; later generated tokens can diverge because earlier tokens' first-visit cache has been replaced by later-visit cache.

The plan therefore has two tiers:

1. `shared_cache` vLLM scan mode for coarse search and heatmaps.
2. Export-and-rescore confirmation for the top candidates in unmodified vLLM, so final pattern selection is based on physical duplicated-layer semantics.

An exact in-memory cache mode can be added later, but it should not block the first experiment.

## Proposed Changes

### 1. Add a reusable probe harness

Create `src/workers/probe_harness.py`.

Responsibilities:

- Build Math prompts with the same system prompt and `/no_think` behavior used today.
- Build EQ prompts with the same `/no_think` behavior used today.
- Apply the tokenizer chat template with `enable_thinking=False` when supported.
- Preserve the existing closed direct think seed defaults from the ExLlama combined worker.
- Score generated strings into the same Math/EQ result dictionaries already produced by `scripts/run_exllama_math_eq_combined_worker.py`.
- Optionally support a small code probe using a JSON dataset with fields like:
  - `prompt`
  - `entry_point` or `test`
  - `reference` or `expected`

Initial implementation can keep code scoring simple: greedy completion, store responses, and either exact hidden-test execution behind an explicit flag or text-only placeholder scoring. The key is to reserve the schema so code probes can be added without changing the vLLM worker.

Why: the ExLlama combined worker currently contains useful prompt/scoring logic inline. Moving the common parts into `src/workers/probe_harness.py` avoids copying that logic into the vLLM worker and makes future HF/ExLlama cleanup easier.

### 2. Add the vLLM relayer monkey patch

Create `src/workers/vllm_relayer_patch.py`.

Responsibilities:

- Assert the pinned vLLM version by default:
  - expected: `0.19.2rc1.dev66+gb47840019.d20260421`
  - allow override with `RYS_VLLM_ALLOW_VERSION_MISMATCH=1`
- Import `vllm.model_executor.models.deepseek_v2`.
- Patch `DeepseekV2Model.forward` with a function based on the pinned source signature:
  - `forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None)`
- Preserve existing behavior for:
  - first pipeline rank input embedding
  - non-first pipeline rank `IntermediateTensors`
  - residual handling
  - `llama_4_scaling`
  - `aux_hidden_state_layers`
  - final norm and pipeline-parallel return shape
- Read the current execution order from module-local state.
- If no execution order is set, use the normal order.
- If an execution order is set, iterate over that explicit list.
- Reject unsupported pipeline-parallel execution for now:
  - require `pipeline_parallel_size == 1`, or fail loudly if `self.start_layer != 0` or `self.end_layer != len(self.layers)`
  - tensor parallelism and expert parallelism remain acceptable

Control API:

```python
patch_vllm()
set_exec_order(order: list[int] | None)
get_exec_order() -> list[int] | None
get_patch_status() -> dict
reset_exec_order()
```

The patched forward should record lightweight counters for smoke tests:

- number of forward calls
- last execution order length
- last execution order hash
- whether a non-baseline order was observed

### 3. Add a vLLM worker extension for cross-process control

Create `src/workers/vllm_worker_extension.py`.

Responsibilities:

- Import `vllm_relayer_patch` at module import time so worker processes patch before model warmup.
- Define a uniquely named extension class, for example `RYSVllmWorkerExtension`.
- Add collective RPC methods:

```python
def rys_set_exec_order(self, order: list[int] | None) -> dict: ...
def rys_reset_exec_order(self) -> dict: ...
def rys_get_patch_status(self) -> dict: ...
```

vLLM supports `worker_extension_cls`; this repo can pass:

```python
worker_extension_cls="src.workers.vllm_worker_extension.RYSVllmWorkerExtension"
```

The scan script should set the order through:

```python
llm.llm_engine.collective_rpc("rys_set_exec_order", args=(layer_indices,))
```

This is necessary because module-level state in the driver process does not automatically synchronize to tensor-parallel or multiprocess workers.

### 4. Add a vLLM combined worker script

Create `scripts/run_vllm_math_eq_combined_worker.py`.

Behavior should mirror `scripts/run_exllama_math_eq_combined_worker.py`.

Core CLI:

```bash
python scripts/run_vllm_math_eq_combined_worker.py \
  --queue-file results/kimi-k26/queue.json \
  --combined-results-file results/kimi-k26/combined_results.pkl \
  --math-results-file results/kimi-k26/math_results.pkl \
  --eq-results-file results/kimi-k26/eq_results.pkl \
  --model ~/models/Kimi-K2.6 \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 64 \
  --eq-max-new 64 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90
```

Additional CLI:

- `--enforce-eager`, default true
- `--trust-remote-code`, default true
- `--dtype`, default `auto`
- `--quantization`, default `None`
- `--max-model-len`, optional
- `--max-num-seqs`, optional
- `--max-num-batched-tokens`, optional
- `--worker-extension-cls`, default `src.workers.vllm_worker_extension.RYSVllmWorkerExtension`
- `--kv-semantics`, default `shared_cache`, with only `shared_cache` implemented initially
- `--canary-block`, default `1,2` or `30,32`
- `--run-canary`, default true
- `--code-dataset-path`, optional
- `--code-max-new`, optional

Implementation outline:

1. Add repo root to `sys.path`.
2. Import `src.workers.vllm_relayer_patch` before constructing `vllm.LLM`.
3. Load tokenizer with `transformers.AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.
4. Build all Math/EQ prompts once through `probe_harness`.
5. Construct `vllm.LLM` once:

```python
llm = LLM(
    model=args.model,
    tokenizer=args.model,
    trust_remote_code=True,
    tensor_parallel_size=args.tensor_parallel_size,
    enforce_eager=True,
    worker_extension_cls=args.worker_extension_cls,
    gpu_memory_utilization=args.gpu_memory_utilization,
    dtype=args.dtype,
    max_model_len=args.max_model_len,
    max_num_seqs=args.max_num_seqs,
    max_num_batched_tokens=args.max_num_batched_tokens,
)
```

6. Verify patch status through `collective_rpc("rys_get_patch_status")`.
7. Run a baseline preflight and canary preflight:
   - baseline should produce parseable Math/EQ outputs
   - canary should report a non-baseline execution order observed
   - optionally compare canary output against baseline to confirm it changes
8. Loop over `SharedWorkQueue`.
9. For each queue entry:
   - parse with `parse_queue_entry_layers(61, entry)` using actual model layer count when available
   - call `rys_set_exec_order(layer_indices)` for every config, including baseline
   - generate all mixed prompts with greedy `SamplingParams`
   - split outputs back to Math/EQ/code items
   - score and persist after each config
10. Save combined result metadata:
   - `config_key`
   - `config_layers`
   - `config_spec`
   - `execution_order`
   - `execution_order_hash`
   - `kv_semantics`
   - `vllm_version`
   - `model_path`
   - `elapsed`
   - `math_score`
   - `eq_score`
   - optional `code_score`
   - `combined_score`

### 5. Add Kimi export support

Patch `hf_export/common.py`:

```python
TEXT_LAYER_PREFIX_CANDIDATES = (
    "language_model.model.layers.",
    "model.language_model.layers.",
    "model.layers.",
    "language_model.layers.",
)
```

Add tests in `tests/test_hf_export.py`:

- prefix detection finds `language_model.model.layers.`
- `build_tensor_name_mapping` duplicates Kimi-style expert tensors:
  - `weight_packed`
  - `weight_scale`
  - `weight_shape`
- `build_exported_config` updates `text_config.num_hidden_layers`

The existing exporter maps all tensors under the detected layer prefix, so compressed-tensors expert tensors should copy correctly once the prefix is detected. The test should prove that explicitly.

### 6. Add exported-checkpoint confirmation workflow

Create `scripts/rescore_exported_candidates.py` or document a manual workflow first.

Purpose:

- Take top candidates from `scripts/analyze_results.py` or surrogate output.
- Export each candidate with `hf_export.export_model`.
- Load the exported checkpoint in unmodified vLLM.
- Run the same Math/EQ/code probes.
- Write a confirmation table comparing:
  - shared-cache scan score
  - exported-checkpoint score
  - deltas by metric

This is the step that should decide the final "best repeating patterns."

Recommended first version can be manual and top-k only, for example top 5 or top 10, because each export is very large.

### 7. Add beam-search backend support

Update `scripts/beam_search.py` after the combined vLLM worker is stable.

Add:

```bash
--worker-backend hf|vllm
```

For `--worker-backend vllm`, route both Math and EQ depth queues through `scripts/run_vllm_math_eq_combined_worker.py` instead of launching separate HF `math_worker` and `eq_worker` processes.

This avoids two independent Kimi loads and keeps the beam path consistent with the single-block scan.

### 8. Add documentation

Update `README.md` with a vLLM/Kimi section:

- environment activation
- vLLM version pin
- Kimi model path assumptions
- queue creation examples for:
  - baseline smoke queue
  - stride/coarse queue
  - full triangular queue
- vLLM worker command
- analysis command
- export confirmation command
- explicit warning about `shared_cache` semantics

## Suggested Initial Run Plan

### Smoke queue

Create a tiny config file:

```text
0,0
1,2
30,32
```

Run:

```bash
source ~/RYS/.venv/bin/activate
python scripts/init_queue.py \
  --num-layers 61 \
  --config-file results/kimi-k26/smoke.config \
  --queue-file results/kimi-k26/smoke_queue.json \
  --results-file results/kimi-k26/smoke_combined_results.pkl
```

Then:

```bash
python scripts/run_vllm_math_eq_combined_worker.py \
  --queue-file results/kimi-k26/smoke_queue.json \
  --combined-results-file results/kimi-k26/smoke_combined_results.pkl \
  --math-results-file results/kimi-k26/smoke_math_results.pkl \
  --eq-results-file results/kimi-k26/smoke_eq_results.pkl \
  --model ~/models/Kimi-K2.6 \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --math-max-new 32 \
  --eq-max-new 32 \
  --tensor-parallel-size 8
```

### Coarse MoE-band scan

Prefer a config-file generator rather than full grid cold. Add a small script or extend `scripts/init_queue.py` with stride support:

```bash
python scripts/init_queue.py \
  --num-layers 61 \
  --min-span 1 \
  --max-span 12 \
  --exclude-baseline \
  --queue-file results/kimi-k26/coarse_queue.json \
  --results-file results/kimi-k26/coarse_combined_results.pkl
```

For a true stride-2 or stride-3 grid, add `--i-stride` and `--j-stride` to `scripts/init_queue.py` rather than manually maintaining config files.

Recommended starting constraints:

- include baseline
- skip `i = 0` initially except baseline
- scan `i,j` in layers 1-60
- stride 2 or 3
- span cap around 12-20 until throughput is known

### Analyze

```bash
python scripts/analyze_results.py \
  --math-scores results/kimi-k26/coarse_math_results.pkl \
  --eq-scores results/kimi-k26/coarse_eq_results.pkl \
  --out-dir results/kimi-k26/analysis \
  --num-layers 61 \
  --title kimi-k26-vllm-shared-cache
```

### Confirm top candidates

For each top candidate:

```bash
python -m hf_export.export_model \
  --source ~/models/Kimi-K2.6 \
  --output exports/kimi-k26-candidate-N \
  --layer-list "..." \
  --overwrite
```

Then rescore that exported checkpoint in unmodified vLLM with the same probe harness. The exported checkpoint should be treated as authoritative for final selection.

## Validation Checklist

### Unit tests

- `vllm_relayer_patch` refuses unpinned vLLM unless override is set.
- `vllm_relayer_patch` expands baseline and non-baseline orders exactly as `layer_config` expects.
- `vllm_worker_extension` methods return patch status dictionaries.
- `probe_harness` produces the same Math/EQ scores from sample strings as the current workers.
- `hf_export.common.detect_text_layer_prefix` detects `language_model.model.layers.`
- Kimi-style `weight_packed`, `weight_scale`, and `weight_shape` tensor names are duplicated by `build_tensor_name_mapping`.

### Small runtime tests

- Load a tiny vLLM-supported model with the patch and verify baseline greedy output matches unpatched vLLM.
- Run a tiny queue with baseline and one repeated-layer config.
- Assert `rys_get_patch_status` reports non-baseline execution for the repeated config.
- Assert output pickle keys are canonical explicit layer tuples.
- Run `scripts/analyze_results.py` on the vLLM-generated pickles.

### Kimi smoke tests

- Baseline only: 1 Math prompt and 1 EQ prompt, short `max_new`.
- Canary: `blocks:1,2`, short `max_new`, confirm patch status changes and outputs are parseable.
- Practical canary: `blocks:30,32`, short `max_new`, confirm no vLLM scheduler/cache crash.

### Export confirmation tests

- Dry-run export for `blocks:30,32`.
- Full export for one small candidate if disk space allows.
- Rescore exported checkpoint in unmodified vLLM.
- Compare exported result to shared-cache scan and record the delta in the candidate report.

## Main Risks

- Shared-cache scan semantics are not identical to physical duplicated layers during multi-token generation. This is acceptable for coarse search only if top candidates are exported and rescored before final claims.
- vLLM's Python forward loop can be captured by CUDA graphs or torch compile. Use `enforce_eager=True` for all scan runs until the patch is proven.
- Pipeline parallelism is not compatible with arbitrary relayer order in the initial patch. Require pipeline parallel size 1.
- Multiprocess workers need RPC state updates. Setting a module global only in the driver is insufficient.
- Kimi's checkpoint is very large. Export only a small number of confirmed candidates.
- Code-probe design should be added before final conclusions, because Math/EQ-only optimization may select configs that hurt coding or agentic behavior.

## Implementation Order

1. Add `probe_harness.py` and tests for scoring parity.
2. Add `vllm_relayer_patch.py` and tests around order state/version assertions.
3. Add `vllm_worker_extension.py` and patch-status RPC plumbing.
4. Add `scripts/run_vllm_math_eq_combined_worker.py`.
5. Run tiny-model runtime tests.
6. Run Kimi baseline/canary smoke tests.
7. Add Kimi prefix support to `hf_export/common.py` and tests.
8. Run a coarse Kimi scan.
9. Analyze with existing `scripts/analyze_results.py`.
10. Export and rescore top candidates in unmodified vLLM.
11. Add vLLM backend support to `scripts/beam_search.py`.
12. Document the workflow in `README.md`.

## Definition Of Done

- A Kimi K2.6 queue can be scanned through vLLM without EXL3.
- The scan loads Kimi once per worker process and changes layer order through RPC between configs.
- Result pickles are compatible with existing RYS analysis and surrogate tools.
- The proposal's smoke commands run end-to-end.
- At least one top candidate can be exported with `language_model.model.layers.` support and rescored in unmodified vLLM.
- Final reported "best repeating patterns" are based on exported-checkpoint confirmation, not only shared-cache scan results.
