"""Runtime vLLM monkey patch for RYS relayer scans.

This patch targets the native vLLM DeepseekV2/DeepseekV3 text model used by
Kimi K2.6. It intentionally runs in shared-cache scan mode: repeated source
layers reuse their physical attention module and KV slot.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from typing import Any


EXPECTED_VLLM_VERSION = "0.19.2rc1.dev66+gb47840019.d20260421"

_PATCHED = False
_ORIGINAL_FORWARD: Any = None
_CURRENT_EXEC_ORDER: tuple[int, ...] | None = None
_STATS: dict[str, Any] = {
    "forward_calls": 0,
    "nonbaseline_forward_calls": 0,
    "last_exec_order_length": None,
    "last_exec_order_hash": None,
    "last_exec_order_preview": None,
    "last_error": None,
}


def _hash_order(order: tuple[int, ...] | None) -> str | None:
    if order is None:
        return None
    raw = ",".join(str(x) for x in order).encode("ascii")
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalize_order(order: Iterable[int] | None) -> tuple[int, ...] | None:
    if order is None:
        return None
    normalized = tuple(int(x) for x in order)
    if not normalized:
        raise ValueError("Execution order must be non-empty or None.")
    bad = [idx for idx in normalized if idx < 0]
    if bad:
        raise ValueError(f"Execution order contains negative layer indices: {bad[:8]}")
    return normalized


def _baseline_order_for_model(model: Any) -> tuple[int, ...]:
    return tuple(range(int(model.start_layer), int(model.end_layer)))


def _is_baseline_order(order: tuple[int, ...] | None, model: Any) -> bool:
    return order is None or order == _baseline_order_for_model(model)


def _record_forward(order: tuple[int, ...] | None, model: Any) -> None:
    _STATS["forward_calls"] = int(_STATS["forward_calls"]) + 1
    _STATS["last_exec_order_length"] = None if order is None else len(order)
    _STATS["last_exec_order_hash"] = _hash_order(order)
    _STATS["last_exec_order_preview"] = None if order is None else list(order[:16])
    if not _is_baseline_order(order, model):
        _STATS["nonbaseline_forward_calls"] = int(_STATS["nonbaseline_forward_calls"]) + 1


def _validate_order_for_model(order: tuple[int, ...], model: Any) -> None:
    num_layers = len(model.layers)
    out_of_range = [idx for idx in order if idx >= num_layers]
    if out_of_range:
        raise ValueError(
            f"Execution order contains layer indices outside [0, {num_layers}): "
            f"{out_of_range[:8]}"
        )

    # The initial monkey patch is designed for TP/EP, not pipeline parallelism.
    if int(model.start_layer) != 0 or int(model.end_layer) != num_layers:
        raise RuntimeError(
            "RYS vLLM shared-cache relayer patch requires pipeline_parallel_size=1 "
            f"(got local layer range [{model.start_layer}, {model.end_layer}) of {num_layers})."
        )


def set_exec_order(order: Iterable[int] | None) -> dict[str, Any]:
    """Set the process-local execution order used by patched forwards."""
    global _CURRENT_EXEC_ORDER
    _CURRENT_EXEC_ORDER = _normalize_order(order)
    return get_patch_status()


def reset_exec_order() -> dict[str, Any]:
    return set_exec_order(None)


def get_exec_order() -> list[int] | None:
    if _CURRENT_EXEC_ORDER is None:
        return None
    return list(_CURRENT_EXEC_ORDER)


def get_patch_status() -> dict[str, Any]:
    return {
        "patched": _PATCHED,
        "expected_vllm_version": os.environ.get(
            "RYS_VLLM_EXPECTED_VERSION",
            EXPECTED_VLLM_VERSION,
        ),
        "current_exec_order": get_exec_order(),
        "current_exec_order_hash": _hash_order(_CURRENT_EXEC_ORDER),
        "stats": dict(_STATS),
        "kv_semantics": "shared_cache",
    }


def _assert_vllm_version() -> str:
    import vllm

    actual = str(vllm.__version__)
    expected = os.environ.get("RYS_VLLM_EXPECTED_VERSION", EXPECTED_VLLM_VERSION)
    if actual != expected and os.environ.get("RYS_VLLM_ALLOW_VERSION_MISMATCH") != "1":
        raise RuntimeError(
            "RYS vLLM patch targets vLLM "
            f"{expected}, but imported vLLM is {actual}. "
            "Set RYS_VLLM_ALLOW_VERSION_MISMATCH=1 to bypass this guard."
        )
    return actual


def patch_vllm() -> dict[str, Any]:
    """Patch vLLM's DeepseekV2Model.forward once in the current process."""
    global _PATCHED, _ORIGINAL_FORWARD

    actual_version = _assert_vllm_version()
    from vllm.model_executor.models import deepseek_v2 as dsv2

    if _PATCHED:
        status = get_patch_status()
        status["vllm_version"] = actual_version
        return status

    model_cls = dsv2.DeepseekV2Model
    _ORIGINAL_FORWARD = model_cls.forward

    def patched_forward(
        self: Any,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Any,
        inputs_embeds: Any = None,
    ) -> Any:
        order = _CURRENT_EXEC_ORDER
        _record_forward(order, self)

        if dsv2.get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to DeepseekV2Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError("intermediate_tensors is required on non-first PP ranks")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is not None:
            llama_4_scaling = dsv2._get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        if order is None:
            execution_order = tuple(range(int(self.start_layer), int(self.end_layer)))
        else:
            _validate_order_for_model(order, self)
            execution_order = order

        aux_hidden_states = []
        for idx in execution_order:
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )

        if not dsv2.get_pp_group().is_last_rank:
            return dsv2.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    model_cls.forward = patched_forward
    _PATCHED = True
    status = get_patch_status()
    status["vllm_version"] = actual_version
    return status


def restore_vllm_forward_for_tests() -> None:
    """Restore the original forward method in tests."""
    global _PATCHED, _ORIGINAL_FORWARD
    if not _PATCHED or _ORIGINAL_FORWARD is None:
        return
    from vllm.model_executor.models import deepseek_v2 as dsv2

    dsv2.DeepseekV2Model.forward = _ORIGINAL_FORWARD
    _PATCHED = False
    _ORIGINAL_FORWARD = None
