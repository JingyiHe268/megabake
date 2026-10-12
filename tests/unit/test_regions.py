from __future__ import annotations

from types import SimpleNamespace

import torch

from megabake import (
    canonicalize_semantics,
    dump_region_graph,
    form_regions,
    prepare_fx_graph,
)
from megabake.graph import import_fx


def _context(graph: torch.fx.GraphModule, *values: torch.Tensor) -> SimpleNamespace:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    return SimpleNamespace(
        input_order=[node.name for node in placeholders],
        input_specs=[
            SimpleNamespace(
                name=node.name,
                kind="runtime",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
            )
            for node, value in zip(placeholders, values)
        ],
    )


def _import(model: torch.nn.Module, *values: torch.Tensor):
    exported = torch.export.export(model, values)
    prepared = prepare_fx_graph(
        exported.graph_module,
        values,
        output_tree_spec=exported.call_spec.out_spec,
    )
    return canonicalize_semantics(import_fx(prepared.graph, prepared.context))


def test_form_regions_preserves_intermediate_graph_output_and_operation_coverage():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            projected = torch.nn.functional.linear(x, weight, bias)
            activated = torch.relu(projected)
            return activated.sum(dim=-1), activated

    torch.manual_seed(41)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))

    assert len(graph.regions) == len(graph.ops)
    assert [region.source_op_ids for region in graph.regions] == [
        (operation_id,) for operation_id in graph.ops
    ]
    activated = graph.values["value:relu"]
    activated_region = next(
        region for region in graph.regions if activated.producer in region.source_op_ids
    )
    assert activated_region.live_outputs == (activated.id,)
    assert graph.outputs == (
        graph.values["value:sum_1"].id,
        activated.id,
    )


def test_form_regions_matches_view_plus_gemm_and_preserves_alias_index_map():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(43)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))

    gemm_region = next(
        region for region in graph.regions if region.id == "region:addmm"
    )
    facts = gemm_region.semantic_facts
    assert facts["family"] == "view_plus_gemm"
    assert facts["view"] == "op:permute"
    assert facts["view_source"] == "value:arg1_1"
    assert facts["materialization"] == "none"
    assert facts["alias"] is True
    assert facts["composed_index_map"] == {
        "terms": (
            {"axis": "r0", "view_axis": "o0", "stride": 1},
            {"axis": "o1", "view_axis": "o1", "stride": 3},
        ),
        "offset": 0,
    }


def test_dump_region_graph_reports_regions_dependencies_and_outputs(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            projected = torch.nn.functional.linear(x, weight, bias)
            activated = torch.relu(projected)
            return activated.sum(dim=-1), activated

    torch.manual_seed(53)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))
    path = dump_region_graph(graph, tmp_path / "nested" / "region_graph.txt")

    text = path.read_text(encoding="utf-8")
    assert "# RegionGraph" in text
    assert "values=7 operations=4 regions=4 inputs=3 outputs=2" in text
    assert "region:addmm source_ops=['op:addmm']" in text
    assert "external_inputs=['value:arg2_1', 'value:arg0_1', 'value:permute']" in text
    assert "region:permute -> region:addmm value=value:permute" in text
    assert "region:addmm -> region:relu value=value:addmm" in text
    assert "region:relu -> region:sum_1 value=value:relu" in text
    assert "region:sum_1 -> output value=value:sum_1" in text
    assert "region:relu -> output value=value:relu" in text


def test_form_regions_keeps_unfamiliar_operations_representable():
    def unknown_operator(value):
        return value * 2

    class Model(torch.nn.Module):
        def forward(self, x):
            return unknown_operator(x) + 1

    torch.manual_seed(47)
    x = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    unknown_node = next(
        node for node in graph_module.graph.nodes if node.op == "call_function"
    )
    unknown_node.meta["val"] = x
    unknown_node.args = ()
    unknown_operation_id = f"op:{unknown_node.name}"
    graph = form_regions(
        canonicalize_semantics(import_fx(graph_module, _context(graph_module, x)))
    )

    unknown_region = graph.regions[0]
    assert unknown_region.source_op_ids == (unknown_operation_id,)
    assert unknown_region.semantic_facts == {}
    assert graph.ops[unknown_operation_id].tensor_program is None
    assert len(graph.regions) == len(graph.ops)
