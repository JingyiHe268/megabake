from __future__ import annotations

import torch

from megabake import (
    derive_dependencies,
    make_phase_schedule,
    select_bodies,
    simulate_phase_schedule,
)
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from megabake.schedule import DependencyKind


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


def _scheduled(model, *inputs):
    selection = select_bodies(_graph(model, *inputs), _target())
    dependencies = derive_dependencies(selection)
    return selection, dependencies, make_phase_schedule(selection, dependencies)


def test_phase_schedule_handles_fork_join_and_keeps_idle_workers_in_barriers():
    class Model(torch.nn.Module):
        def forward(self, value):
            left = torch.relu(value)
            right = torch.neg(value)
            return left + right

    value = torch.randn(2, 3)
    selection, dependencies, schedule = _scheduled(Model(), value)
    join = next(
        selected
        for selected in selection.selections
        if selected.outputs[0] == selection.graph.outputs[0]
    )

    assert len(dependencies.prerequisites_for(join.region_id)) == 2
    assert len(schedule.phases) == 3
    assert len(schedule.barriers) == 2
    assert all(
        barrier.participating_workers == tuple(range(schedule.worker_count))
        for barrier in schedule.barriers
    )

    simulation = simulate_phase_schedule(schedule)
    assert simulation.phase_order.index(join.region_id) == 2
    assert len(simulation.barrier_order) == 2


def test_phase_schedule_allows_aliasing_view_without_an_unscheduled_phase():
    class Model(torch.nn.Module):
        def forward(self, source, weight, bias):
            return torch.nn.functional.linear(source, weight, bias)

    source = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    _, dependencies, schedule = _scheduled(Model(), source, weight, bias)

    assert dependencies.dependencies == ()
    assert len(schedule.phases) == 1
    assert schedule.barriers == ()
    simulation = simulate_phase_schedule(schedule)
    assert simulation.barrier_order == ()


def test_dependencies_include_all_reduction_input_tiles():
    class Model(torch.nn.Module):
        def forward(self, source, first_weight, second_weight):
            hidden = source @ first_weight.T
            return hidden @ second_weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    first_weight = torch.randn(256, 128, dtype=torch.bfloat16)
    second_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    _, dependencies, schedule = _scheduled(
        Model(), source, first_weight, second_weight
    )
    dependency = next(
        dependency
        for dependency in dependencies.dependencies
        if dependency.kind == DependencyKind.PRODUCER_TO_CONSUMER
    )

    assert dependency.producer_tiles == (
        (0, 0),
        (0, 128),
        (64, 0),
        (64, 128),
    )
    assert dependency.consumer_tiles == ((0, 0), (64, 0))
    assert not dependency.conservative

    simulation = simulate_phase_schedule(schedule)
    assert simulation.work_item_counts[schedule.phases[0].region_id] == 4
    assert simulation.work_item_counts[schedule.phases[-1].region_id] == 2
    assert simulation.idle_worker_counts[schedule.phases[-1].region_id] == 2
