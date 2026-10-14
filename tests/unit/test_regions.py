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


def test_form_regions_builds_executable_vector_chain_candidate_and_keeps_baselines():
    class Model(torch.nn.Module):
        def forward(self, value):
            return (torch.relu(value) + 1) * 2

    torch.manual_seed(59)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    assert len(graph.regions) == len(graph.ops)
    candidate = graph.regions[-1].fusion_candidates[0]
    assert candidate.id == "region:mul:fused"
    assert candidate.source_op_ids == ("op:relu", "op:add", "op:mul")
    assert candidate.external_inputs == ("value:arg0_1",)
    assert candidate.live_outputs == ("value:mul",)
    assert candidate.executable
    assert candidate.cost_bytes == 2 * 3 * 4 * 4
    assert candidate.tensor_program.loads[0].value_id == "value:arg0_1"
    assert candidate.semantic_facts["source_casts"] == "preserved"


def test_form_regions_builds_executable_scalar_dag_candidate():
    class Model(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return (activated + 1) * (activated * 2)

    torch.manual_seed(59)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    candidate = graph.regions[-1].fusion_candidates[0]
    assert candidate.id == "region:mul_1:fused"
    assert set(candidate.source_op_ids) == {
        "op:relu",
        "op:add",
        "op:mul_1",
        "op:mul",
    }
    assert candidate.executable
    assert candidate.tensor_program is not None
    assert candidate.tensor_program.loads[0].value_id == "value:arg0_1"


def test_form_regions_rejects_vector_chain_when_intermediate_is_external():
    class Model(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return activated, activated + 1

    torch.manual_seed(61)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    candidate = graph.regions[-1].fusion_candidates[0]
    assert not candidate.executable
    assert candidate.rejections == ("an intermediate value has an external use",)
    assert len(graph.regions) == len(graph.ops)


def test_form_regions_matches_gated_mlp_and_counts_candidate_costs():
    class Model(torch.nn.Module):
        def forward(self, source, gate_weight, up_weight):
            gate = torch.nn.functional.linear(source, gate_weight)
            up = torch.nn.functional.linear(source, up_weight)
            activated = torch.nn.functional.silu(gate)
            return activated * up, gate, activated

    torch.manual_seed(67)
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    gate_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    up_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, gate_weight, up_weight))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if candidate.semantic_facts["family"] == "gated_mlp"
    )
    facts = candidate.semantic_facts
    assert candidate.source_op_ids == (
        "op:mm",
        "op:mm_1",
        "op:convert_element_type_4",
        "op:neg",
        "op:exp",
        "op:add",
        "op:div",
        "op:convert_element_type_5",
        "op:mul",
    )
    assert facts["gate_contraction"] == "op:mm"
    assert facts["up_contraction"] == "op:mm_1"
    assert facts["activation"] == "op:convert_element_type_5"
    assert set(facts["secondary_outputs"]) == {
        "value:mul",
        "value:mm",
        "value:convert_element_type_5",
    }
    assert facts["separate_work_bytes"] == 1056
    assert facts["paired_work_bytes"] == 144
    assert facts["packing_cost_bytes"] == 4 * 6 * 2
    assert facts["conversion_cost_bytes"] == 0
    assert facts["scalar_fusion_saved_bytes"] == 960
    assert not candidate.executable
    assert "paired GEMM provider is not implemented" in candidate.rejections


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


def test_form_regions_matches_exact_rms_norm_with_residual_and_casts():
    class Model(torch.nn.Module):
        def forward(self, source, weight, residual):
            widened = source.float()
            variance = torch.pow(widened, 2).mean(dim=-1, keepdim=True)
            scale = torch.rsqrt(variance + 1e-5)
            normalized = scale * widened
            affine = normalized * weight
            return affine.to(torch.bfloat16) + residual

    torch.manual_seed(24)
    source = torch.randn(3, 7, dtype=torch.bfloat16)
    weight = torch.randn(7, dtype=torch.bfloat16)
    residual = torch.randn(3, 7, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, weight, residual))

    tail = next(region for region in graph.regions if region.id == "region:add_1")
    candidate = next(
        candidate
        for candidate in tail.fusion_candidates
        if candidate.semantic_facts["family"] == "rms_norm"
    )
    assert candidate.semantic_facts == {
        "family": "rms_norm",
        "reduction_axis": 1,
        "width": 7,
        "epsilon": 1e-5,
        "affine": "value:arg1_1",
        "residual": "value:arg2_1",
        "source_casts": ("torch.bfloat16->torch.float32", "torch.float32->torch.bfloat16"),
    }
    program = candidate.tensor_program
    assert program is not None
    assert program.source_target == "rms_norm"
    assert program.reduction_axes == ("r0",)
    assert len(program.loads) == 5
    assert {load.value_id for load in program.loads} == {
        "value:arg0_1",
        "value:arg1_1",
        "value:arg2_1",
    }


def test_form_regions_matches_exact_rope_and_keeps_unfamiliar_variants_generic():
    class Model(torch.nn.Module):
        def forward(self, source, cosine, sine):
            half = source.shape[-1] // 2
            first = (
                source[..., :half] * cosine[..., :half]
                - source[..., half:] * sine[..., :half]
            )
            second = (
                source[..., half:] * sine[..., half:]
                + source[..., :half] * cosine[..., half:]
            )
            return torch.cat((first, second), dim=-1)

    torch.manual_seed(24)
    source = torch.randn(3, 6, dtype=torch.bfloat16)
    cosine = torch.randn(3, 6, dtype=torch.bfloat16)
    sine = torch.randn(3, 6, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, cosine, sine))

    tail = next(region for region in graph.regions if region.id == "region:cat")
    candidate = next(
        candidate
        for candidate in tail.fusion_candidates
        if candidate.semantic_facts["family"] == "rope"
    )
    assert candidate.semantic_facts == {
        "family": "rope",
        "axis": 1,
        "width": 6,
        "rotary_width": 6,
        "rotation": "i <-> i + rotary_width / 2",
        "pass_through": 0,
        "source_casts": (),
    }
    program = candidate.tensor_program
    assert program is not None
    assert program.source_target == "rope"
    assert len(program.loads) == 5
    assert {load.value_id for load in program.loads} == {
        "value:arg0_1",
        "value:arg1_1",
        "value:arg2_1",
    }

    class CastModel(Model):
        def forward(self, source, cosine, sine):
            return super().forward(source.float(), cosine.float(), sine.float())

    cast_graph = form_regions(_import(CastModel(), source, cosine, sine))
    assert not any(
        candidate.semantic_facts.get("family") == "rope"
        for region in cast_graph.regions
        for candidate in region.fusion_candidates
    )

    half_graph = form_regions(
        _import(
            Model(),
            source.float().to(torch.float16),
            cosine.float().to(torch.float16),
            sine.float().to(torch.float16),
        )
    )
    assert not any(
        candidate.semantic_facts.get("family") == "rope"
        for region in half_graph.regions
        for candidate in region.fusion_candidates
    )
