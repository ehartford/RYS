from __future__ import annotations

import pytest

from src.workers import vllm_relayer_patch


def test_exec_order_state_roundtrip() -> None:
    vllm_relayer_patch.reset_exec_order()
    status = vllm_relayer_patch.set_exec_order([0, 1, 1, 2])
    assert status["current_exec_order"] == [0, 1, 1, 2]
    assert status["current_exec_order_hash"]
    assert vllm_relayer_patch.get_exec_order() == [0, 1, 1, 2]

    reset_status = vllm_relayer_patch.reset_exec_order()
    assert reset_status["current_exec_order"] is None
    assert vllm_relayer_patch.get_exec_order() is None


def test_exec_order_rejects_empty_and_negative() -> None:
    with pytest.raises(ValueError):
        vllm_relayer_patch.set_exec_order([])
    with pytest.raises(ValueError):
        vllm_relayer_patch.set_exec_order([0, -1])
    vllm_relayer_patch.reset_exec_order()


def test_patch_vllm_import_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("vllm")
    monkeypatch.setenv("RYS_VLLM_ALLOW_VERSION_MISMATCH", "1")
    status = vllm_relayer_patch.patch_vllm()
    try:
        assert status["patched"] is True
        assert status["kv_semantics"] == "shared_cache"
    finally:
        vllm_relayer_patch.restore_vllm_forward_for_tests()
        vllm_relayer_patch.reset_exec_order()
