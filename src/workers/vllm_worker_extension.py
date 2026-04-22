"""vLLM worker extension for RYS relayer scan control."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.workers import vllm_relayer_patch


vllm_relayer_patch.patch_vllm()


class RYSVllmWorkerExtension:
    """Methods injected into vLLM workers and exposed through collective_rpc."""

    def rys_set_exec_order(self, order: Iterable[int] | None) -> dict[str, Any]:
        return vllm_relayer_patch.set_exec_order(order)

    def rys_reset_exec_order(self) -> dict[str, Any]:
        return vllm_relayer_patch.reset_exec_order()

    def rys_get_patch_status(self) -> dict[str, Any]:
        return vllm_relayer_patch.get_patch_status()
