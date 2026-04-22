from __future__ import annotations

from src.workers.probe_harness import (
    add_no_think_prefix,
    apply_chat_template,
    build_math_eq_prompts,
    score_probe_outputs,
)


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        body = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        return body + "\n<think>"


def test_apply_chat_template_strips_forced_think_and_adds_seed() -> None:
    prompt = apply_chat_template(
        DummyTokenizer(),
        [{"role": "user", "content": "hi"}],
        think_seed_mode="closed_direct",
        think_seed_text="done",
    )
    assert prompt.endswith("<think>done</think>\n")
    assert "user:hi" in prompt


def test_add_no_think_prefix_only_updates_last_user_message() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "question"},
    ]
    updated = add_no_think_prefix(messages)
    assert updated[-1]["content"] == "/no_think question"
    assert messages[-1]["content"] == "question"


def test_build_and_score_math_eq_prompts() -> None:
    math_dataset = {"m1": {"question": "2+2?", "answer": 4}}
    eq_dataset = {
        "e1": {
            "prompt": "Rate emotions.\nRevised scores:",
            "reference_answer": {
                "emotion1_score": 7.0,
                "emotion2_score": 6.0,
                "emotion3_score": 5.0,
                "emotion4_score": 4.0,
            },
        }
    }
    prompts, items = build_math_eq_prompts(
        hf_tokenizer=DummyTokenizer(),
        math_dataset=math_dataset,
        eq_dataset=eq_dataset,
        math_think_seed_mode="off",
        eq_think_seed_mode="off",
    )
    assert len(prompts) == 2
    assert [item["task"] for item in items] == ["math", "eq"]

    math_result, eq_result, code_result = score_probe_outputs(
        [
            "The final answer is 4.",
            "Revised scores:\nEmotionA: 7\nEmotionB: 6\nEmotionC: 5\nEmotionD: 4",
        ],
        items,
    )
    assert math_result["score"] == 1.0
    assert math_result["valid_final_answer_count"] == 1
    assert eq_result["score"] == 0.95
    assert code_result is None
