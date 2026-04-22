"""Shared probe prompt construction and scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.workers.eq_worker import calculate_eq_score, extract_emotion_scores, generate_eq_messages
from src.workers.math_worker import calculate_score, extract_integers
from src.workers.model_utils import strip_thinking


MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)
DEFAULT_THINK_SEED_TEXT = "I can answer this now, and will do so succinctly."


def strip_forced_think(prompt: str) -> str:
    """Remove a trailing forced think tag added by some chat templates."""
    if prompt.endswith("<think>\n"):
        return prompt[: -len("<think>\n")]
    if prompt.endswith("<think>"):
        return prompt[: -len("<think>")]
    return prompt


def append_think_seed(prompt: str, think_seed_mode: str, think_seed_text: str) -> str:
    """Append a short closed think block when requested."""
    if think_seed_mode == "off":
        return prompt
    if think_seed_mode == "closed_direct":
        return f"{prompt}<think>{think_seed_text}</think>\n"
    raise ValueError(f"Unknown think seed mode: {think_seed_mode}")


def apply_chat_template(
    hf_tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    think_seed_mode: str = "off",
    think_seed_text: str = DEFAULT_THINK_SEED_TEXT,
) -> str:
    """Apply a tokenizer chat template with thinking disabled when supported."""
    try:
        prompt = hf_tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = hf_tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    return append_think_seed(strip_forced_think(str(prompt)), think_seed_mode, think_seed_text)


def generate_math_messages(question: str, *, use_no_think_prefix: bool = True) -> list[dict[str, str]]:
    user_text = f"/no_think {question}" if use_no_think_prefix else question
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def add_no_think_prefix(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []
    updated = [dict(m) for m in messages]
    last = updated[-1]
    if last.get("role") == "user":
        content = str(last.get("content", ""))
        last["content"] = f"/no_think {content}"
    return updated


def build_math_eq_prompts(
    *,
    hf_tokenizer: Any,
    math_dataset: Mapping[str, Mapping[str, Any]],
    eq_dataset: Mapping[str, Mapping[str, Any]],
    use_math_no_think_prefix: bool = True,
    use_eq_no_think_prefix: bool = True,
    math_think_seed_mode: str = "closed_direct",
    eq_think_seed_mode: str = "closed_direct",
    think_seed_text: str = DEFAULT_THINK_SEED_TEXT,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build one mixed prompt list and task metadata list."""
    prompts: list[str] = []
    items: list[dict[str, Any]] = []

    for qid, sample in math_dataset.items():
        messages = generate_math_messages(
            str(sample["question"]),
            use_no_think_prefix=use_math_no_think_prefix,
        )
        prompts.append(
            apply_chat_template(
                hf_tokenizer,
                messages,
                think_seed_mode=math_think_seed_mode,
                think_seed_text=think_seed_text,
            )
        )
        items.append({"task": "math", "qid": qid, "answer": sample["answer"]})

    for qid, sample in eq_dataset.items():
        messages = generate_eq_messages(str(sample["prompt"]), use_no_think_prefix=False)
        if use_eq_no_think_prefix:
            messages = add_no_think_prefix(messages)
        prompts.append(
            apply_chat_template(
                hf_tokenizer,
                messages,
                think_seed_mode=eq_think_seed_mode,
                think_seed_text=think_seed_text,
            )
        )
        items.append(
            {
                "task": "eq",
                "qid": qid,
                "reference": sample.get(
                    "reference_answer",
                    sample.get("reference_answer_fullscale", {}),
                ),
            }
        )

    return prompts, items


def build_code_prompts(
    *,
    hf_tokenizer: Any,
    code_dataset: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    think_seed_mode: str = "off",
    think_seed_text: str = DEFAULT_THINK_SEED_TEXT,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build optional code-probe prompts with a conservative scoring schema."""
    if isinstance(code_dataset, Mapping):
        rows = list(code_dataset.items())
    else:
        rows = [(str(i), row) for i, row in enumerate(code_dataset)]

    prompts: list[str] = []
    items: list[dict[str, Any]] = []
    for qid, sample in rows:
        prompt = str(sample.get("prompt") or sample.get("question") or "")
        messages = [{"role": "user", "content": prompt}]
        prompts.append(
            apply_chat_template(
                hf_tokenizer,
                messages,
                think_seed_mode=think_seed_mode,
                think_seed_text=think_seed_text,
            )
        )
        items.append(
            {
                "task": "code",
                "qid": qid,
                "reference": sample.get("reference", sample.get("expected")),
                "entry_point": sample.get("entry_point"),
                "test": sample.get("test"),
            }
        )
    return prompts, items


def score_probe_outputs(
    outputs: Sequence[str],
    items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Score mixed Math/EQ/code outputs using the public RYS result schema."""
    math_scores: list[float] = []
    math_responses: list[dict[str, Any]] = []
    eq_scores: list[float] = []
    eq_responses: list[dict[str, Any]] = []
    code_scores: list[float] = []
    code_responses: list[dict[str, Any]] = []

    for output, item in zip(outputs, items):
        task = str(item["task"])
        raw_text = str(output)
        stripped_text = strip_thinking(raw_text)

        if task == "math":
            answer = item["answer"]
            qid = item["qid"]
            stripped_integers = extract_integers(stripped_text)
            integers = stripped_integers
            has_valid_final_answer = len(stripped_integers) > 0
            fallback_used = False
            if not integers:
                integers = extract_integers(raw_text)
                fallback_used = len(integers) > 0
            question_score = (
                max(calculate_score(answer, integer) for integer in integers)
                if integers
                else 0.0
            )
            math_scores.append(float(question_score))
            math_responses.append(
                {
                    "qid": qid,
                    "raw_output": raw_text,
                    "stripped_output": stripped_text,
                    "extracted": integers,
                    "has_valid_final_answer": has_valid_final_answer,
                    "fallback_used": fallback_used,
                    "reference": answer,
                    "score": float(question_score),
                }
            )
            continue

        if task == "eq":
            reference = item["reference"]
            qid = item["qid"]
            predicted, confidence = extract_emotion_scores(stripped_text)
            question_score = calculate_eq_score(predicted, reference, confidence)
            eq_scores.append(float(question_score))
            eq_responses.append(
                {
                    "qid": qid,
                    "raw_output": raw_text,
                    "stripped_output": stripped_text,
                    "extracted": predicted,
                    "confidence": confidence,
                    "reference": reference,
                    "score": float(question_score),
                }
            )
            continue

        if task == "code":
            reference = item.get("reference")
            qid = item["qid"]
            # Placeholder deterministic score for lightweight code probes. A
            # proper execution harness can replace this without changing result
            # consumers.
            question_score = 1.0 if reference and str(reference) in raw_text else 0.0
            code_scores.append(float(question_score))
            code_responses.append(
                {
                    "qid": qid,
                    "raw_output": raw_text,
                    "stripped_output": stripped_text,
                    "reference": reference,
                    "entry_point": item.get("entry_point"),
                    "test": item.get("test"),
                    "score": float(question_score),
                }
            )

    math_valid_count = sum(1 for row in math_responses if row["has_valid_final_answer"])
    math_fallback_count = sum(1 for row in math_responses if row["fallback_used"])
    math_result = {
        "score": (sum(math_scores) / len(math_scores)) if math_scores else 0.0,
        "valid_final_answer_count": math_valid_count,
        "valid_final_answer_rate": (math_valid_count / len(math_responses)) if math_responses else 0.0,
        "fallback_used_count": math_fallback_count,
        "fallback_used_rate": (math_fallback_count / len(math_responses)) if math_responses else 0.0,
        "responses": math_responses,
    }
    eq_result = {
        "score": (sum(eq_scores) / len(eq_scores)) if eq_scores else 0.0,
        "responses": eq_responses,
    }
    code_result = None
    if code_responses:
        code_result = {
            "score": (sum(code_scores) / len(code_scores)) if code_scores else 0.0,
            "responses": code_responses,
        }
    return math_result, eq_result, code_result
