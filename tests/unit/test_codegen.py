from __future__ import annotations

import ast
from dataclasses import replace
import json

import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    emit_cute,
    make_phase_schedule,
    select_bodies,
)
from megabake.plan import ConversionBody, CuTeLayout
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
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


def _plan(model, *inputs):
    prepared = prepare_fx_graph(
        torch.export.export(model, inputs).graph_module, inputs
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(
        graph, _target(), SelectionPolicy(allow_tuned=False)
    )
    return assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )


def test_emit_cute_deduplicates_equivalent_bodies_and_saves_diagnostics(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value) + torch.relu(value)

    plan = _plan(Model(), torch.randn(5, 70))
    generated = emit_cute(plan, tmp_path)

    ast.parse(generated.source)
    assert generated.source.count("def _body_0(") == 1
    assert generated.source.count("def _body_1(") == 1
    assert generated.source.count("_body_0(") == 3
    assert generated.source.count("grid_barrier(ready, workers, epoch +") == 2
    assert "cooperative_worker_loop(" in generated.source
    assert generated.source_path.exists()
    assert generated.source_map_path.exists()

    diagnostics = json.loads(generated.source_map_path.read_text())
    kinds = {entry["kind"] for entry in diagnostics["entries"]}
    assert {"device_body", "phase_barrier", "enclosing_kernel", "host_launch_wrapper"} <= kinds
    assert all(generated.launch.work_iterators)
    assert all(iterator.dtype == "torch.int64" for iterator in generated.launch.work_iterators)


def test_emit_cute_emits_explicit_layout_conversion(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu().sum(dim=-1)

    value = torch.randn(2, 3)
    prepared = prepare_fx_graph(
        torch.export.export(Model(), (value,)).graph_module, (value,)
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(
        graph, _target(), SelectionPolicy(allow_tuned=False)
    )
    producer = next(
        selected
        for selected in selection.selections
        if selected.program.family == "elementwise"
    )
    consumer = next(
        selected
        for selected in selection.selections
        if selected.program.family == "reduction"
    )
    value_id = producer.outputs[0]
    source_layout = CuTeLayout.from_contract(
        producer.body.layouts.outputs[value_id]
    )
    target_layout = replace(source_layout, strides=(1, 3))
    load = consumer.program.loads[0]
    consumer_program = replace(
        consumer.program,
        loads=(replace(load, strides=target_layout.strides),),
    )
    consumer_body = replace(
        consumer.body,
        layouts=replace(
            consumer.body.layouts,
            inputs={
                value_id: replace(
                    consumer.body.layouts.inputs[value_id],
                    strides=target_layout.strides,
                )
            },
        ),
    )
    conversion = ConversionBody(
        value_id,
        producer.region_id,
        consumer.region_id,
        source_layout,
        target_layout,
        24,
        (producer.region_id,),
    )
    converted_selection = replace(
        selection,
        selections=(
            producer,
            replace(
                consumer, program=consumer_program, body=consumer_body
            ),
        ),
        conversions=(conversion,),
    )
    plan = assign_storage(
        make_phase_schedule(
            converted_selection, derive_dependencies(converted_selection)
        )
    )

    generated = emit_cute(plan, tmp_path)
    ast.parse(generated.source)
    assert "_conversion_0_0(" in generated.source
    assert generated.launch.phases[0].conversion_symbols == ("_conversion_0_0",)
    assert any(
        entry.kind == "layout_conversion"
        for entry in generated.source_map
    )
