from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    make_phase_schedule,
    select_bodies,
    verify_plan,
)
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.memory import InvalidStorage
from megabake.regions import form_regions


def _target() -> TargetFacts:
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    )


def _graph(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    return form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )


def _kernel_plan(model, *inputs):
    selection = select_bodies(_graph(model, *inputs), _target())
    dependencies = derive_dependencies(selection)
    return assign_storage(make_phase_schedule(selection, dependencies))


def test_assign_storage_separates_live_intermediates_and_boundary_outputs():
    class Model(torch.nn.Module):
        def forward(self, value):
            left = torch.relu(value)
            right = torch.neg(value)
            return left + right, left * right

    plan = _kernel_plan(Model(), torch.randn(2, 3))
    verify_plan(plan)

    assert len(plan.storage.workspace) == 2
    first, second = plan.storage.workspace
    assert first.offset_bytes == 0
    assert second.offset_bytes == 128
    assert first.size_bytes == second.size_bytes == 128
    assert plan.storage.workspace_bytes == 256
    workspace_values = {
        assignment.value_id for assignment in plan.storage.workspace
    }
    assert all(
        binding.kind == "workspace"
        for binding in plan.storage.bindings
        if binding.value_id in workspace_values
    )
    output = next(
        binding
        for binding in plan.storage.bindings
        if binding.value_id == plan.graph.outputs[0]
    )
    assert output.kind == "output"
    assert output.storage_id not in {
        assignment.storage_id for assignment in plan.storage.workspace
    }

    scratch = plan.storage.worker_scratch[0]
    assert len(scratch.phases) == len(plan.phases)
    assert all(phase.barrier_offsets == {} for phase in scratch.phases)
    assert all(phase.descriptor_offsets == {} for phase in scratch.phases)
    assert plan.launch.cooperative
    assert plan.launch.grid_size == len(plan.storage.worker_scratch)


def test_assign_storage_adds_worker_barriers_and_descriptors_for_gemm():
    class Model(torch.nn.Module):
        def forward(self, source, first_weight, second_weight):
            hidden = source @ first_weight.T
            return hidden @ second_weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    first_weight = torch.randn(256, 128, dtype=torch.bfloat16)
    second_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    plan = _kernel_plan(Model(), source, first_weight, second_weight)
    verify_plan(plan)

    assert len(plan.storage.workspace) == 1
    assert plan.storage.workspace[0].size_bytes == 65_536
    assert plan.storage.workspace[0].alignment_bytes == 128
    gemm_scratch = next(
        phase for phase in plan.storage.worker_scratch[0].phases if phase.barrier_offsets
    )
    assert gemm_scratch.barrier_offsets == {
        "producer_tail": 0,
        "wgmma_wait_group_0": 8,
        "tma_store_producer_tail": 16,
    }
    assert gemm_scratch.descriptor_offsets == {"A": 128, "B": 256, "C": 384}
    assert gemm_scratch.body_data_offset_bytes == 512
    assert plan.launch.threads_per_block == 256
    assert plan.launch.dynamic_shared_memory_bytes == plan.storage.worker_scratch_bytes


def test_verify_plan_rejects_missing_output_undersized_and_overlapping_storage():
    class Model(torch.nn.Module):
        def forward(self, value):
            left = value.relu()
            right = value.neg()
            return left + right, left * right

    plan = _kernel_plan(Model(), torch.randn(2, 3))
    missing_output = replace(
        plan,
        storage=replace(
            plan.storage,
            bindings=tuple(
                binding
                for binding in plan.storage.bindings
                if not (binding.value_id == plan.graph.outputs[0] and binding.kind == "output")
            ),
        ),
    )
    with pytest.raises(InvalidStorage, match="missing output"):
        verify_plan(missing_output)

    undersized = replace(
        plan,
        storage=replace(
            plan.storage,
            workspace=(replace(plan.storage.workspace[0], size_bytes=8),),
        ),
    )
    with pytest.raises(InvalidStorage, match="undersized"):
        verify_plan(undersized)

    overlapping = replace(
        plan,
        storage=replace(
            plan.storage,
            workspace=(
                plan.storage.workspace[0],
                replace(plan.storage.workspace[1], offset_bytes=0),
            ),
        ),
    )
    with pytest.raises(InvalidStorage, match="overlaps"):
        verify_plan(overlapping)


def test_verify_plan_rejects_incompatible_worker_scratch():
    class Model(torch.nn.Module):
        def forward(self, value):
            left = torch.relu(value)
            right = torch.neg(value)
            return left + right

    plan = _kernel_plan(Model(), torch.randn(2, 3))
    worker = plan.storage.worker_scratch[0]
    incompatible_phase = replace(worker.phases[0], size_bytes=1)
    broken = replace(
        plan,
        storage=replace(
            plan.storage,
            worker_scratch=(
                replace(
                    worker,
                    size_bytes=0,
                    phases=(incompatible_phase, *worker.phases[1:]),
                ),
                *plan.storage.worker_scratch[1:],
            ),
        ),
    )
    with pytest.raises(InvalidStorage, match="scratch is undersized"):
        verify_plan(broken)
