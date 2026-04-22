#!/usr/bin/env python
"""vLLM combined Math/EQ worker for RYS relayer scans."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def enable_line_buffered_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def add_repo_to_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root, *parts])


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_num_layers(model_path: str) -> int:
    model_dir = Path(model_path).expanduser()
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = load_json(config_path)
        text_cfg = cfg.get("text_config")
        if isinstance(text_cfg, dict) and isinstance(text_cfg.get("num_hidden_layers"), int):
            return int(text_cfg["num_hidden_layers"])
        if isinstance(cfg.get("num_hidden_layers"), int):
            return int(cfg["num_hidden_layers"])

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "num_hidden_layers", None) is not None:
        return int(text_cfg.num_hidden_layers)
    if getattr(cfg, "num_hidden_layers", None) is not None:
        return int(cfg.num_hidden_layers)
    raise ValueError(f"Could not resolve num_hidden_layers for {model_path}")


def save_pickle_result(path: Path, config_key: tuple[int, ...], value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                pickle.dump({}, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    with path.open("r+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = pickle.load(f)
            data[config_key] = value
            f.seek(0)
            f.truncate()
            pickle.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def collective_rpc(llm: Any, method: str, args: tuple = ()) -> list[Any]:
    if hasattr(llm, "collective_rpc"):
        return llm.collective_rpc(method, args=args)
    return llm.llm_engine.collective_rpc(method, args=args)


def set_vllm_exec_order(llm: Any, layer_indices: list[int] | None) -> list[dict[str, Any]]:
    return collective_rpc(llm, "rys_set_exec_order", args=(layer_indices,))


def get_vllm_patch_status(llm: Any) -> list[dict[str, Any]]:
    return collective_rpc(llm, "rys_get_patch_status")


def extract_vllm_texts(request_outputs: list[Any]) -> list[str]:
    texts: list[str] = []
    for request_output in request_outputs:
        outputs = getattr(request_output, "outputs", None) or []
        if not outputs:
            texts.append("")
            continue
        texts.append(str(getattr(outputs[0], "text", "")))
    return texts


def run_generate(llm: Any, prompts: list[str], *, max_new_tokens: int, seed: int) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return extract_vllm_texts(outputs)


def make_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "worker_extension_cls": args.worker_extension_cls,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.reasoning_parser:
        kwargs["reasoning_parser"] = args.reasoning_parser
    if args.mm_encoder_tp_mode:
        kwargs["mm_encoder_tp_mode"] = args.mm_encoder_tp_mode
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    return LLM(**kwargs)


def run_preflight(
    *,
    llm: Any,
    prompts: list[str],
    num_layers: int,
    canary_spec: str,
    max_new_tokens: int,
    seed: int,
) -> None:
    from src.core.layer_config import normalize_to_layers

    if not prompts:
        raise RuntimeError("Preflight requested but no prompts were built.")

    print("[preflight] baseline short generation")
    set_vllm_exec_order(llm, None)
    baseline_text = run_generate(
        llm,
        [prompts[0]],
        max_new_tokens=max_new_tokens,
        seed=seed,
    )[0]
    if not baseline_text.strip():
        raise RuntimeError("vLLM baseline preflight produced empty text.")

    print(f"[preflight] canary {canary_spec}")
    canary_layers = normalize_to_layers(num_layers, canary_spec)
    set_vllm_exec_order(llm, canary_layers)
    canary_text = run_generate(
        llm,
        [prompts[0]],
        max_new_tokens=max_new_tokens,
        seed=seed,
    )[0]
    if not canary_text.strip():
        raise RuntimeError("vLLM canary preflight produced empty text.")

    statuses = get_vllm_patch_status(llm)
    observed = any(
        int(status.get("stats", {}).get("nonbaseline_forward_calls", 0)) > 0
        for status in statuses
    )
    if not observed:
        raise RuntimeError(
            "vLLM canary preflight did not observe a non-baseline patched forward. "
            "The worker extension or monkey patch did not land."
        )
    set_vllm_exec_order(llm, None)
    print("[preflight] passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM combined Math/EQ worker using a runtime RYS relayer patch."
    )
    parser.add_argument("--queue-file", default=None)
    parser.add_argument("--config-file", default=None, help="Optional direct config file. Bypasses queue mode.")
    parser.add_argument("--combined-results-file", required=True)
    parser.add_argument("--math-results-file", required=True)
    parser.add_argument("--eq-results-file", required=True)
    parser.add_argument("--code-results-file", default=None)
    parser.add_argument("--model", required=True, help="HF model path or id.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path or id.")
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--code-dataset-path", default=None)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=64)
    parser.add_argument("--code-max-new", type=int, default=256)
    parser.add_argument("--preflight-max-new", type=int, default=8)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--canary-block", default="30,32")
    parser.add_argument("--disable-no-think-prefix", action="store_true")
    parser.add_argument("--disable-eq-no-think-prefix", action="store_true")
    parser.add_argument("--think-seed-mode", choices=["off", "closed_direct"], default="closed_direct")
    parser.add_argument("--eq-think-seed-mode", choices=["off", "closed_direct"], default="closed_direct")
    parser.add_argument("--code-think-seed-mode", choices=["off", "closed_direct"], default="off")
    parser.add_argument("--think-seed-text", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--mm-encoder-tp-mode", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument(
        "--worker-extension-cls",
        default="src.workers.vllm_worker_extension.RYSVllmWorkerExtension",
    )
    parser.add_argument("--kv-semantics", choices=["shared_cache"], default="shared_cache")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=0.0,
        help="When in queue mode, wait this long for more queued work before exiting. 0 exits immediately.",
    )
    parser.add_argument(
        "--queue-poll-interval-sec",
        type=float,
        default=1.0,
        help="Polling interval while waiting for more queued work.",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Optional file whose existence tells an idle queue-mode worker to exit.",
    )
    args = parser.parse_args()
    if bool(args.queue_file) == bool(args.config_file):
        parser.error("Specify exactly one of --queue-file or --config-file.")
    return args


def main() -> None:
    enable_line_buffered_output()
    add_repo_to_path()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    args.model = str(Path(args.model).expanduser()) if args.model.startswith("~") else args.model
    if args.tokenizer and args.tokenizer.startswith("~"):
        args.tokenizer = str(Path(args.tokenizer).expanduser())

    if args.think_seed_text is None:
        from src.workers.probe_harness import DEFAULT_THINK_SEED_TEXT

        args.think_seed_text = DEFAULT_THINK_SEED_TEXT

    if args.worker_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        args.worker_id = f"VLLM-COMB-{cuda_visible}"

    from transformers import AutoTokenizer

    from src.core.layer_config import (
        is_baseline_layers,
        layer_spec_string,
        normalize_to_layers,
        parse_queue_entry_layers,
    )
    from src.workers.probe_harness import (
        build_code_prompts,
        build_math_eq_prompts,
        score_probe_outputs,
    )
    from src.workers.shared_queue import SharedWorkQueue
    from src.workers import vllm_relayer_patch

    vllm_relayer_patch.patch_vllm()

    print("=" * 80)
    print(f"vLLM Combined Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file:            {args.queue_file}")
    print(f"Config file:           {args.config_file}")
    print(f"Combined results file: {args.combined_results_file}")
    print(f"Math results file:     {args.math_results_file}")
    print(f"EQ results file:       {args.eq_results_file}")
    print(f"Model:                 {args.model}")
    print(f"Tensor parallel size:  {args.tensor_parallel_size}")
    print(f"Enforce eager:         {args.enforce_eager}")
    print(f"KV semantics:          {args.kv_semantics}")

    num_layers = resolve_num_layers(args.model)
    print(f"Resolved text layers:  {num_layers}")

    math_dataset = load_json(args.math_dataset_path)
    eq_dataset = load_json(args.eq_dataset_path)
    code_dataset = load_json(args.code_dataset_path) if args.code_dataset_path else None

    hf_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prompts, items = build_math_eq_prompts(
        hf_tokenizer=hf_tokenizer,
        math_dataset=math_dataset,
        eq_dataset=eq_dataset,
        use_math_no_think_prefix=not args.disable_no_think_prefix,
        use_eq_no_think_prefix=not args.disable_eq_no_think_prefix,
        math_think_seed_mode=args.think_seed_mode,
        eq_think_seed_mode=args.eq_think_seed_mode,
        think_seed_text=args.think_seed_text,
    )
    if code_dataset is not None:
        code_prompts, code_items = build_code_prompts(
            hf_tokenizer=hf_tokenizer,
            code_dataset=code_dataset,
            think_seed_mode=args.code_think_seed_mode,
            think_seed_text=args.think_seed_text,
        )
        prompts.extend(code_prompts)
        items.extend(code_items)

    max_new_tokens = max(
        args.math_max_new,
        args.eq_max_new,
        args.code_max_new if code_dataset is not None else 0,
    )
    print(f"Prompt count:          {len(prompts)}")
    print(f"Generation max tokens: {max_new_tokens}")

    print("Loading vLLM model once (reusing across configs)...")
    llm = make_llm(args)
    statuses = get_vllm_patch_status(llm)
    print(f"Patch statuses:        {json.dumps(statuses, sort_keys=True)[:1000]}")

    if not args.skip_preflight:
        run_preflight(
            llm=llm,
            prompts=prompts,
            num_layers=num_layers,
            canary_spec=args.canary_block,
            max_new_tokens=args.preflight_max_new,
            seed=args.seed,
        )

    queue = SharedWorkQueue(args.queue_file, args.combined_results_file) if args.queue_file else None
    initial_queue_total: int | None = None
    progress_start = time.time()
    processed_this_run = 0
    direct_total = 0
    direct_entries: list[dict[str, Any]] = []
    if args.config_file:
        for line_no, raw in enumerate(Path(args.config_file).read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            layers = normalize_to_layers(num_layers, line)
            direct_entries.append(
                {
                    "idx": len(direct_entries),
                    "layers": layers,
                    "spec": line if ":" in line else layer_spec_string(layers),
                    "source_line": line_no,
                }
            )
        print(f"Loaded {len(direct_entries)} direct configs from {args.config_file}")
        direct_total = len(direct_entries)
    elif queue is not None:
        initial_remaining, initial_completed = queue.get_queue_status()
        if args.idle_timeout_sec <= 0:
            initial_queue_total = initial_remaining + initial_completed
        print(
            f"Queue progress start:  completed={initial_completed} "
            f"remaining={initial_remaining} "
            f"total={initial_queue_total if initial_queue_total is not None else 'dynamic'}"
        )

    idle_started_at: float | None = None
    while True:
        if queue is not None:
            entry = queue.get_next_config()
        elif direct_entries:
            entry = direct_entries.pop(0)
        else:
            entry = None
        if entry is None:
            stop_requested = bool(args.stop_file and Path(args.stop_file).exists())
            if queue is not None and args.idle_timeout_sec > 0 and not stop_requested:
                if idle_started_at is None:
                    idle_started_at = time.time()
                    print(
                        f"Queue empty; waiting up to {format_duration(args.idle_timeout_sec)} "
                        "for more configs."
                    )
                idle_elapsed = time.time() - idle_started_at
                if idle_elapsed < args.idle_timeout_sec:
                    time.sleep(max(args.queue_poll_interval_sec, 0.1))
                    continue
                print(f"Queue idle timeout reached after {format_duration(idle_elapsed)}.")
            elif stop_requested:
                print(f"Stop file detected: {args.stop_file}")
            print("No configs left. Exiting.")
            break
        idle_started_at = None

        try:
            parsed_entry = parse_queue_entry_layers(num_layers, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] Invalid queue entry {entry!r}: {exc}")
            continue

        config_key = parsed_entry["layer_key"]
        layer_indices = parsed_entry["layers"]
        config_spec = parsed_entry["spec"]
        if queue is not None:
            remaining, completed = queue.get_queue_status()
            progress_total = initial_queue_total if initial_queue_total is not None else remaining + completed + 1
            progress_done_before = progress_total - remaining - 1
        else:
            remaining, completed = len(direct_entries), processed_this_run
            progress_total = direct_total
            progress_done_before = processed_this_run
        print(
            f"\n[{args.worker_id}] Running config {config_spec} "
            f"({layer_spec_string(layer_indices)}) "
            f"(progress={progress_done_before + 1}/{progress_total}, "
            f"remaining={remaining}, completed={completed})"
        )

        order_for_worker = None if is_baseline_layers(layer_indices, num_layers) else list(layer_indices)
        set_statuses = set_vllm_exec_order(llm, order_for_worker)
        t0 = time.time()
        raw_outputs = run_generate(
            llm,
            prompts,
            max_new_tokens=max_new_tokens,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        math_result, eq_result, code_result = score_probe_outputs(raw_outputs, items)
        patch_statuses = get_vllm_patch_status(llm)

        metric_scores = [math_result["score"], eq_result["score"]]
        if code_result is not None:
            metric_scores.append(code_result["score"])
        combined_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0

        combined_result: dict[str, Any] = {
            "config_key": config_key,
            "config_layers": list(layer_indices),
            "config_spec": config_spec,
            "execution_order": list(layer_indices),
            "execution_order_hash": set_statuses[0].get("current_exec_order_hash") if set_statuses else None,
            "elapsed": elapsed,
            "mode": "vllm_single_pass_all",
            "kv_semantics": args.kv_semantics,
            "num_prompts": len(prompts),
            "math_score": math_result["score"],
            "eq_score": eq_result["score"],
            "combined_score": combined_score,
            "math_valid_final_answer_count": math_result["valid_final_answer_count"],
            "math_valid_final_answer_rate": math_result["valid_final_answer_rate"],
            "math_fallback_used_count": math_result["fallback_used_count"],
            "math_fallback_used_rate": math_result["fallback_used_rate"],
            "patch_statuses": patch_statuses,
            "model_path": args.model,
        }
        if code_result is not None:
            combined_result["code_score"] = code_result["score"]

        if queue is not None:
            queue.save_result(config_key, combined_result)
        else:
            save_pickle_result(Path(args.combined_results_file), config_key, combined_result)
        save_pickle_result(Path(args.math_results_file), config_key, math_result)
        save_pickle_result(Path(args.eq_results_file), config_key, eq_result)
        if code_result is not None and args.code_results_file:
            save_pickle_result(Path(args.code_results_file), config_key, code_result)

        log_bits = [
            f"math={math_result['score']:.4f}",
            f"eq={eq_result['score']:.4f}",
        ]
        if code_result is not None:
            log_bits.append(f"code={code_result['score']:.4f}")
        log_bits.append(f"combined={combined_score:.4f}")
        log_bits.append(
            f"valid={math_result['valid_final_answer_count']}/{len(math_result['responses'])}"
        )
        log_bits.append(f"elapsed={elapsed:.1f}s")
        print(f"[{args.worker_id}] " + " ".join(log_bits))

        processed_this_run += 1
        done = progress_done_before + 1
        if queue is not None:
            remaining_after, completed_after = queue.get_queue_status()
            done = (initial_queue_total or (remaining_after + completed_after)) - remaining_after
            total = initial_queue_total or (remaining_after + completed_after)
            remaining_for_eta = max(total - done, 0)
        else:
            total = direct_total
            remaining_after = len(direct_entries)
            completed_after = processed_this_run
            remaining_for_eta = remaining_after
        elapsed_total = time.time() - progress_start
        avg = elapsed_total / processed_this_run if processed_this_run else 0.0
        eta = avg * remaining_for_eta
        pct = (done / total * 100.0) if total else 100.0
        print(
            f"[{args.worker_id}] progress {done}/{total} ({pct:.1f}%) "
            f"remaining={remaining_for_eta} completed={completed_after} "
            f"avg={avg:.1f}s/config elapsed={format_duration(elapsed_total)} "
            f"eta={format_duration(eta)}",
            flush=True,
        )

    set_vllm_exec_order(llm, None)


if __name__ == "__main__":
    main()
