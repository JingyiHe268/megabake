"""Tuned TMA GEMM bodies inside the table-driven launch.

Large graphs use the table launch, whose GEMM phases build their TMA
descriptors in the kernel prologue from the device address table. These tests
force that launch on a two-layer model so it compiles in seconds; the task 31
tests cover the full 30-layer model.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.fixtures import SmolLMArchitecture, StatefulSmolLMArchitecture


def _sm90_available() -> bool:
    if os.environ.get("MEGABAKE_RUN_GPU_TESTS") != "1":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


pytestmark = pytest.mark.skipif(
    not _sm90_available(),
    reason="table GEMM tests require an SM90 CUDA device",
)

LAYERS = 2
# Q, K, V, O, gate, up, and down per layer, plus the vocabulary projection.
PROJECTIONS = 7 * LAYERS + 1


@pytest.fixture
def table_launch(monkeypatch):
    import megabake.compiler as compiler

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)


def _assert_tuned_table_projections(backend) -> None:
    compiled = backend.megabake_last_compiled
    launch = compiled.generated.launch
    assert launch.launch_policy == "megabake_cooperative_table"
    gemm_phases = [
        phase for phase in compiled.plan.phases if phase.body.provider == "g0_range"
    ]
    assert len(gemm_phases) == PROJECTIONS
    assert not any(
        phase.body.name == "generic_contraction" for phase in compiled.plan.phases
    )
    assert len(launch.gemm_table) == PROJECTIONS
    # Equal shapes share one variant: Q/O, K/V, gate/up, down, and vocabulary.
    assert len(launch.gemm_template_phases) == 5
    assert compiled.plan.launch.grid_size == compiled.resources.max_cooperative_workers
    assert compiled.resources.local_memory_bytes_per_thread <= 64
    trace = backend.megabake_last_trace[-1]
    assert trace.recurring_gpu_operations == (("workload_kernel", 1),)


def test_table_launch_runs_tuned_gemms_in_prefill(tmp_path, table_launch):
    from megabake import make_backend

    torch.manual_seed(7)
    model = SmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    calls = (
        (torch.tensor([[1, 2, 3, 4]], device="cuda"), torch.ones(1, 4, dtype=torch.int64, device="cuda")),
        (torch.tensor([[9, 8, 7, 6]], device="cuda"), torch.tensor([[1, 1, 1, 0]], device="cuda")),
    )
    with torch.no_grad():
        for input_ids, mask in calls:
            actual = optimized(input_ids, mask).logits
            expected = model(input_ids, mask).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    _assert_tuned_table_projections(backend)


def test_table_launch_runs_tuned_gemms_in_stateful_decode(tmp_path, table_launch):
    from megabake import make_backend

    torch.manual_seed(7)
    model = StatefulSmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)

    def state():
        return (
            *(torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2 * LAYERS)),
            *(torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(LAYERS)),
        )

    actual_state, reference_state = state(), state()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for position, token in enumerate((5, 6, 7, 8)):
            token_ids = torch.tensor([[token]], device="cuda")
            position_ids = torch.tensor([[position]], device="cuda")
            actual = optimized(token_ids, position_ids, *actual_state)
            expected = model(token_ids, position_ids, *reference_state)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_tensor, reference_tensor in zip(actual_state, reference_state, strict=True):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)

    _assert_tuned_table_projections(backend)
