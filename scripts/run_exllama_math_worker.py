#!/usr/bin/env python
import argparse
import json
import os
import sys
import time


def add_exllamav3_to_path():
    env_path = os.environ.get("EXLLAMAV3_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.append(env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exllamav3"))
    if os.path.isdir(repo_root):
        sys.path.append(repo_root)


def add_repo_to_path():
    env_path = os.environ.get("RYS_PATH") or os.environ.get("RYS_REPRO_PATH") or os.environ.get("LEVELGEN_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.insert(0, env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(repo_root, "src")):
        sys.path.insert(0, repo_root)


def _strip_forced_think(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt[:-len("<think>\n")]
    if prompt.endswith("<think>"):
        return prompt[:-len("<think>")]
    return prompt


def _append_think_seed(prompt: str, think_seed_mode: str, think_seed_text: str) -> str:
    if think_seed_mode == "off":
        return prompt
    if think_seed_mode == "closed_direct":
        return f"{prompt}<think>{think_seed_text}</think>\n"
    raise ValueError(f"Unknown think seed mode: {think_seed_mode}")


def apply_chat_template(
    hf_tokenizer,
    messages,
    think_seed_mode: str = "off",
    think_seed_text: str = "I can answer this directly.",
):

    try:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _append_think_seed(_strip_forced_think(prompt), think_seed_mode, think_seed_text)
    except TypeError:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return _append_think_seed(_strip_forced_think(prompt), think_seed_mode, think_seed_text)


MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)


def generate_math_messages(question: str, use_no_think_prefix: bool = True) -> list[dict]:
    user_text = f"/no_think {question}" if use_no_think_prefix else question
    return [
        {
            "role": "system",
            "content": MATH_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None
    return [float(v) for v in items]


def estimate_max_prompt_tokens(
    hf_tokenizer,
    dataset,
    use_no_think_prefix: bool = True,
    think_seed_mode: str = "off",
    think_seed_text: str = "I can answer this directly.",
) -> int:
    max_tokens = 0
    for _, sample in dataset.items():
        messages = generate_math_messages(sample["question"], use_no_think_prefix)
        prompt = apply_chat_template(
            hf_tokenizer,
            messages,
            think_seed_mode=think_seed_mode,
            think_seed_text=think_seed_text,
        )
        tokenized = hf_tokenizer(prompt, return_tensors="pt")
        max_tokens = max(max_tokens, tokenized["input_ids"].shape[1])
    return max_tokens


def build_layer_map(num_blocks: int, block: tuple[int, int]) -> list[int] | None:
    start, end = block
    if start == 0 and end == 0:
        return None
    if start < 0 or end < 0 or start >= end or end > num_blocks:
        raise ValueError(f"Invalid block {block} for {num_blocks} blocks")
    return list(range(0, end)) + list(range(start, num_blocks))


def load_exllama_model(
    model_dir,
    max_chunk_size,
    max_output_size,
    device: str | None = None,
    reserve_per_device: list[float] | None = None,
    use_per_device: list[float] | None = None,
):
    from exllamav3 import Config, Model, Tokenizer

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    load_kwargs = dict(
        progressbar=True,
        max_chunk_size=max_chunk_size,
        max_output_size=max_output_size,
    )
    if device:
        load_kwargs["device"] = device
    else:
        if reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = reserve_per_device
        if use_per_device is not None:
            load_kwargs["use_per_device"] = use_per_device
    model.load(
        **load_kwargs,
    )
    tokenizer = Tokenizer.from_config(config)
    return config, model, tokenizer


def build_cache_and_generator(model, tokenizer, layer_map, cache_size, max_chunk_size, max_batch_size):
    from exllamav3 import Cache, Generator

    if layer_map is not None:
        model.layer_map = layer_map
    else:
        model.layer_map = None
    cache = Cache(model, max_num_tokens=cache_size, layer_map=layer_map)
    # Cache tensors are allocated during model load; we attach after load so allocate here.
    for module in model.get_cache_layers():
        if getattr(module, "num_kv_heads", 1) == 0:
            continue
        for cl in module.cache_layers:
            cl.alloc(module.device)
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_chunk_size=max_chunk_size,
        max_batch_size=max_batch_size,
    )
    return cache, generator


def run_math(generator, hf_tokenizer, prompts, answers, qids, batch_size, max_new_tokens):
    from src.workers.math_worker import calculate_score, extract_integers, strip_thinking
    from exllamav3.generator.sampler import GreedySampler

    scores = []
    responses = []
    sampler = GreedySampler()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_answers = answers[start:start + batch_size]
        batch_qids = qids[start:start + batch_size]
        outputs = generator.generate(
            prompt=batch_prompts,
            max_new_tokens=max_new_tokens,
            completion_only=True,
            encode_special_tokens=True,
            sampler=sampler,
        )
        for output, answer, qid in zip(outputs, batch_answers, batch_qids):
            raw_text = output
            stripped_text = strip_thinking(raw_text)
            stripped_integers = extract_integers(stripped_text)
            integers = stripped_integers
            has_valid_final_answer = len(stripped_integers) > 0
            fallback_used = False
            if len(integers) == 0:
                integers = extract_integers(raw_text)
                fallback_used = len(integers) > 0
            if len(integers) == 0:
                question_score = 0
            else:
                question_score = max(calculate_score(answer, i) for i in integers)
            scores.append(question_score)
            responses.append({
                "qid": qid,
                "raw_output": raw_text,
                "stripped_output": stripped_text,
                "extracted": integers,
                "has_valid_final_answer": has_valid_final_answer,
                "fallback_used": fallback_used,
                "reference": answer,
                "score": question_score,
            })

    avg_score = sum(scores) / len(scores) if scores else 0
    valid_final_answer_count = sum(1 for r in responses if r["has_valid_final_answer"])
    fallback_used_count = sum(1 for r in responses if r["fallback_used"])
    return {
        "score": avg_score,
        "valid_final_answer_count": valid_final_answer_count,
        "valid_final_answer_rate": valid_final_answer_count / len(responses) if responses else 0,
        "fallback_used_count": fallback_used_count,
        "fallback_used_rate": fallback_used_count / len(responses) if responses else 0,
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description="ExLlamaV3 math worker for layer duplication sweeps.")
    parser.add_argument(
        "--queue-file",
        type=str,
        required=True,
        help="Path to shared queue file (canonical layers entries or legacy key entries).",
    )
    parser.add_argument("--results-file", type=str, required=True, help="Path to shared results file")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to EXL3 model directory")
    parser.add_argument("--dataset-path", type=str, default="datasets/math_16.json",
                        help="Path to math dataset")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for inference")
    parser.add_argument("--max-new", type=int, default=16, help="Max new tokens")
    parser.add_argument("--cache-size", type=int, default=16384, help="Cache size in tokens")
    parser.add_argument("--auto-cache", action="store_true", help="Auto-size cache based on prompt length")
    parser.add_argument("--cache-page", type=int, default=256, help="Round seq len to this multiple")
    parser.add_argument("--max-chunk-size", type=int, default=2048, help="Max chunk size for load")
    parser.add_argument("--max-output-size", type=int, default=32, help="Max output size hint for load")
    parser.add_argument("--device", type=str, default=None, help="Single device to load model on (e.g. cuda:0).")
    parser.add_argument(
        "--reserve-per-device",
        type=str,
        default=None,
        help="Comma-separated GB to reserve per device for autosplit (e.g. 8,8).",
    )
    parser.add_argument(
        "--use-per-device",
        type=str,
        default=None,
        help="Comma-separated GB target use per device for autosplit (e.g. 70,70).",
    )
    parser.add_argument("--worker-id", type=str, default=None, help="Worker ID for logging")
    parser.add_argument(
        "--disable-no-think-prefix",
        action="store_true",
        help="Do not prefix math prompts with '/no_think'. Useful for models that degrade with it.",
    )
    parser.add_argument(
        "--think-seed-mode",
        choices=["off", "closed_direct"],
        default="off",
        help="Prompt seed strategy after stripping forced '<think>' from chat templates.",
    )
    parser.add_argument(
        "--think-seed-text",
        default="I can answer this directly.",
        help="Text inserted inside the closed-think seed when --think-seed-mode=closed_direct.",
    )
    args = parser.parse_args()

    add_exllamav3_to_path()
    add_repo_to_path()

    import torch
    from transformers import AutoTokenizer
    from src.core.layer_config import layer_spec_string, parse_queue_entry_layers
    from src.workers.shared_queue import SharedWorkQueue
    if args.worker_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        args.worker_id = f"EXL3-{cuda_visible}"

    print("=" * 80)
    print(f"ExLlamaV3 Math Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file:   {args.queue_file}")
    print(f"Results file: {args.results_file}")
    print(f"Model:        {args.model_dir}")
    print(f"Dataset:      {args.dataset_path}")
    print(f"Batch size:   {args.batch_size}")

    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    use_no_think_prefix = not args.disable_no_think_prefix
    reserve_per_device = parse_float_list(args.reserve_per_device)
    use_per_device = parse_float_list(args.use_per_device)

    if args.auto_cache:
        max_prompt = estimate_max_prompt_tokens(
            hf_tokenizer,
            dataset,
            use_no_think_prefix,
            think_seed_mode=args.think_seed_mode,
            think_seed_text=args.think_seed_text,
        )
        seq_len = round_up(max_prompt + args.max_new, args.cache_page)
        args.cache_size = seq_len * args.batch_size
        print(f"[auto-cache] seq_len={seq_len} cache_size={args.cache_size}")

    prompts = []
    answers = []
    qids = []
    for qid, sample in dataset.items():
        messages = generate_math_messages(sample["question"], use_no_think_prefix)
        prompts.append(
            apply_chat_template(
                hf_tokenizer,
                messages,
                think_seed_mode=args.think_seed_mode,
                think_seed_text=args.think_seed_text,
            )
        )
        answers.append(sample["answer"])
        qids.append(qid)

    print("Loading model weights once (reusing across configs)...")
    config, model, exllama_tokenizer = load_exllama_model(
        args.model_dir,
        max_chunk_size=args.max_chunk_size,
        max_output_size=args.max_output_size,
        device=args.device,
        reserve_per_device=reserve_per_device,
        use_per_device=use_per_device,
    )
    num_blocks = config.num_hidden_layers

    queue = SharedWorkQueue(args.queue_file, args.results_file)

    while True:
        entry = queue.get_next_config()
        if entry is None:
            print("Queue empty. Exiting.")
            break

        try:
            parsed_entry = parse_queue_entry_layers(num_blocks, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] Invalid queue entry {entry!r}: {exc}")
            continue
        config_key = parsed_entry["layer_key"]
        layer_indices = parsed_entry["layers"]
        config_spec = parsed_entry["spec"]
        remaining, completed = queue.get_queue_status()
        print(f"\n[{args.worker_id}] Running config {config_spec} ({layer_spec_string(layer_indices)}) "
              f"(remaining={remaining}, completed={completed})")

        layer_map = None if layer_indices == list(range(num_blocks)) else list(layer_indices)
        max_batch_size = max(1, args.cache_size // args.cache_page)

        t0 = time.time()
        cache, generator = build_cache_and_generator(
            model,
            exllama_tokenizer,
            layer_map=layer_map,
            cache_size=args.cache_size,
            max_chunk_size=args.max_chunk_size,
            max_batch_size=max_batch_size,
        )

        result = run_math(generator, hf_tokenizer, prompts, answers, qids, args.batch_size, args.max_new)
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        result["config_key"] = config_key
        result["config_layers"] = list(layer_indices)
        result["config_spec"] = config_spec
        result["think_seed_mode"] = args.think_seed_mode
        result["think_seed_text"] = args.think_seed_text if args.think_seed_mode != "off" else ""

        queue.save_result(config_key, result)
        print(
            f"[{args.worker_id}] Score={result['score']:.4f} "
            f"valid={result['valid_final_answer_count']}/{len(result['responses'])} "
            f"fallback={result['fallback_used_count']}/{len(result['responses'])} "
            f"elapsed={elapsed:.1f}s"
        )

        for layer in cache.layers.values():
            layer.free()
        cache.detach_from_model(model)
        del generator, cache
        torch.cuda.empty_cache()

    model.unload()
    del exllama_tokenizer, model


if __name__ == "__main__":
    main()
