from __future__ import annotations

from dataclasses import replace

import torch
import pytest

import megabake.planner as planner_module
from megabake import make_tile_maps, select_bodies
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.plan import CuTeLayout, CuTeSwizzle, InvalidTileMap
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


def test_select_bodies_prefers_measured_gemm_and_maps_complete_reduction_domain():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    selection = select_bodies(_graph(Model(), source, weight), _target())
    selected = next(
        selected
        for selected in selection.selections
        if selected.program.family == "contraction"
    )

    assert selected.body.provider == "g0_range"
    assert selected.body.evidence.measured
    assert selection.report.complete

    tile_map = make_tile_maps(selection)[0]
    assert tile_map.output_grid.tile_shape == (64, 128)
    assert tile_map.output_grid.counts == (2, 1)
    assert len(tile_map.accesses) == 2
    assert tile_map.work_items[0].reduction_range == (0, 128)

    access = next(
        access for access in tile_map.accesses if access.output.origin == (64, 0)
    )
    left, right = access.inputs
    assert left.shape == (64, 128)
    assert right.shape == (128, 128)
    assert left.element_offset((64, 0)) == 64 * 128
    assert right.element_offset((0, 0)) == 0


def test_select_bodies_uses_legal_vector_chain_and_preserves_fallback():
    class Model(torch.nn.Module):
        def forward(self, value):
            return (torch.relu(value) + 1) * 2

    class ExternalIntermediate(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return activated, activated + 1

    torch.manual_seed(71)
    value = torch.randn(2, 3)
    selection = select_bodies(_graph(Model(), value), _target())

    assert [selected.region_id for selected in selection.selections] == [
        "region:mul:fused"
    ]
    selected = selection.selections[0]
    assert selected.source_op_ids == ("op:relu", "op:add", "op:mul")
    assert selected.inputs == ("value:arg0_1",)
    assert selected.outputs == ("value:mul",)
    assert selected.body.provider == "generic"
    assert selected.body.name == "generic_elementwise_fused"
    assert selected.body.semantic.expression_hooks == ("vector_chain",)
    assert selected.body.evidence.measurements["fusion_saved_bytes"] == 2 * 3 * 4 * 4
    assert selection.report.complete

    fallback = select_bodies(_graph(ExternalIntermediate(), value), _target())
    assert [selected.region_id for selected in fallback.selections] == [
        "region:relu",
        "region:add",
    ]
    assert any(
        rejection.region_id == "region:add:fused"
        and rejection.reason == "an intermediate value has an external use"
        for rejection in fallback.report.rejections
    )


def test_make_tile_maps_clips_partial_output_tiles():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(65, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    selection = select_bodies(_graph(Model(), source, weight), _target())
    tile_map = make_tile_maps(selection)[0]
    partial = next(
        access for access in tile_map.accesses if access.output.origin[0] == 64
    )
    left = partial.inputs[0]

    assert partial.output.shape == (1, 128)
    assert left.shape == (1, 128)
    assert left.element_offset((64, 0)) == 64 * 128


def test_make_tile_maps_handles_unit_dimensions_and_boundary_strides():
    class TransposeReLU(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value.t())

    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    transpose_graph = _graph(TransposeReLU(), torch.randn(3, 5))
    transpose_selection = select_bodies(transpose_graph, _target())
    transpose_map = make_tile_maps(transpose_selection)[0]
    input_tile = transpose_map.accesses[0].inputs[0]

    assert transpose_map.output_grid.tile_shape == (5, 3)
    assert input_tile.element_offset((2, 1)) == 2 * 1 + 1 * 5
    assert transpose_selection.report.rejections

    sum_graph = _graph(Sum(), torch.randn(1, 7))
    sum_map = make_tile_maps(select_bodies(sum_graph, _target()))[0]
    assert sum_map.output_grid.tile_shape == (1,)
    assert sum_map.accesses[0].inputs[0].shape == (1, 7)
    assert sum_map.accesses[0].reduction_range == (0, 7)

    row = CuTeLayout(
        dtype="torch.float32",
        shape=(1, 3),
        strides=(99, 1),
        alignment_bytes=4,
    )
    equivalent = replace(row, strides=(7, 1))
    assert not row.needs_conversion_to(equivalent)

    with pytest.raises(InvalidTileMap, match="swizzle"):
        CuTeLayout(
            dtype="torch.float32",
            shape=(2, 2),
            strides=(2, 1),
            alignment_bytes=4,
            swizzle=CuTeSwizzle(mask=3, shift=1),
        )


def test_make_tile_maps_inserts_layout_conversion_with_cost_and_dependency(
    monkeypatch,
):
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu().sum(dim=-1)

    real_get_candidates = planner_module.get_candidates

    def changed_input_layout(region, target, policy):
        result = real_get_candidates(region, target, policy)
        program = region.tensor_programs[0]
        if program.family != "reduction":
            return result
        config = result.configurations[0]
        value_id = program.loads[0].value_id
        layouts = replace(
            config.layouts,
            inputs={
                value_id: replace(
                    config.layouts.inputs[value_id], strides=(1, 3)
                )
            },
        )
        return replace(result, configurations=(replace(config, layouts=layouts),))

    monkeypatch.setattr(planner_module, "get_candidates", changed_input_layout)
    selection = select_bodies(_graph(Model(), torch.randn(2, 3)), _target())
    relu = next(
        selected
        for selected in selection.selections
        if selected.program.family in {"elementwise", "copy"}
    )
    reduction = next(
        selected
        for selected in selection.selections
        if selected.program.family == "reduction"
    )

    assert len(selection.conversions) == 1
    conversion = selection.conversions[0]
    assert conversion.value_id == relu.outputs[0]
    assert conversion.source_region_id == relu.region_id
    assert conversion.target_region_id == reduction.region_id
    assert conversion.cost_bytes == 2 * 3 * 4 * 2
    assert conversion.dependencies == (relu.region_id,)
    reduction_map = make_tile_maps(selection)[-1]
    assert reduction_map.conversion_bodies == (conversion,)
