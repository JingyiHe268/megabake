from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from megabake.bodies import (
    CandidateSet,
    InvalidBodyConfig,
    SelectionPolicy,
    TargetFacts,
    emit_map_body,
    emit_reduction_body,
    validate_body_config,
)
from megabake.bodies.catalog import get_candidates
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.graph import Region
from megabake.lowering import (
    Accumulator,
    Binary,
    Cast,
    Constant,
    Unary,
    canonicalize_semantics,
)
from megabake.regions import form_regions


def _target(**changes):
    values = {
        "architecture": "sm90a",
        "threads_per_warp": 32,
        "max_threads_per_cta": 1024,
        "max_shared_memory_per_cta": 232_448,
        "max_registers_per_thread": 255,
        "multiprocessor_count": 60,
        "supports_tma": True,
        "supports_wgmma": True,
    }
    values.update(changes)
    return TargetFacts(**values)


def _policy(**changes):
    values = {
        "allow_generic": True,
        "allow_tuned": True,
        "numerical_policy": "source_semantics",
        "require_measured": False,
    }
    values.update(changes)
    return SelectionPolicy(**values)


def _regions(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    return graph.regions


def test_catalog_returns_generic_map_contract():
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value)

    region = _regions(Model(), torch.randn(3, 5))[0]
    result = get_candidates(region, _target(), _policy())

    assert isinstance(result, CandidateSet)
    assert len(result.configurations) == 1
    config = result.configurations[0]
    assert config.provider == "generic"
    assert config.semantic.operation_family == "elementwise"
    assert config.entrypoint.name == "run"
    assert config.entrypoint.arguments == (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )
    assert not config.entrypoint.whole_kernel_launch
    assert config.participants.participating_threads == 128
    assert config.completion.outputs_ready.startswith("assigned output elements")
    assert config.completion.scratch_reusable == "scratch is reusable when run returns"


def test_emit_map_body_generates_copy_source():
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.clone()

    region = _regions(Model(), torch.randn(5, 70))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    body = emit_map_body(region.tensor_programs[0], config)

    assert body.function_name == "run"
    assert body.tile_count == 3
    assert body.work_iterator_dtype == "torch.int64"
    assert "tensor_bindings[0][o0, o1]" in body.source
    assert "output[o0, o1] = value" in body.source


def test_emit_reduction_body_generates_sum_mean_and_norm_sources():
    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    class Mean(torch.nn.Module):
        def forward(self, value):
            return value.mean(dim=-1)

    sum_region = _regions(Sum(), torch.randn(5, 1003))[0]
    mean_region = _regions(Mean(), torch.randn(5, 1003))[0]
    sum_config = get_candidates(sum_region, _target(), _policy()).configurations[0]
    mean_config = get_candidates(mean_region, _target(), _policy()).configurations[0]
    sum_body = emit_reduction_body(
        sum_region.tensor_programs[0],
        sum_config,
    )
    mean_body = emit_reduction_body(
        mean_region.tensor_programs[0],
        mean_config,
    )

    assert sum_config.scratch.size_bytes == 16
    assert sum_config.tiles.work_partition == "one output row per CTA"
    assert sum_config.evidence.revision.startswith(
        "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce"
    )
    assert sum_body.tile_count == mean_body.tile_count == 5
    assert sum_body.reduction_tile == 128
    assert sum_body.scratch_elements == 4
    assert sum_body.scratch_encoding == (
        "scratch[block, warp] holds one accumulator partial per warp"
    )
    assert sum_body.phases == (
        "thread_partial",
        "warp_partial",
        "cta_merge",
        "epilogue",
    )
    for body in (sum_body, mean_body):
        assert "cutlass.range(thread, 1003, 128)" in body.source
        assert "cute.arch.warp_reduction_sum(accumulator)" in body.source
        assert "cute.arch.sync_threads()" in body.source
        assert "output[o0] = value" in body.source
    assert "cutlass.Int64(1003)).to(cutlass.Float32)" in mean_body.source


def test_emit_reduction_body_preserves_norm_epsilon_and_cast():
    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    region = _regions(Sum(), torch.randn(4, 129))[0]
    program = region.tensor_programs[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    post = Cast(
        Unary(
            "rsqrt",
            Binary(
                "add",
                Binary(
                    "div",
                    Cast(
                        Accumulator(program.accumulator_dtype),
                        program.accumulator_dtype,
                        "none",
                    ),
                    Constant(129, program.accumulator_dtype),
                    program.accumulator_dtype,
                ),
                Constant(1e-6, program.accumulator_dtype),
                program.accumulator_dtype,
            ),
            program.accumulator_dtype,
        ),
        "torch.bfloat16",
        "none",
    )
    program = replace(program, post=post, store=replace(program.store, dtype="torch.bfloat16"))
    output_layout = replace(
        config.layouts.outputs[program.store.value_id], dtype="torch.bfloat16"
    )
    config = replace(
        config,
        layouts=replace(
            config.layouts, outputs={program.store.value_id: output_layout}
        ),
    )

    body = emit_reduction_body(program, config)

    assert "cutlass.Float32(1e-06)" in body.source
    assert ".to(cutlass.BFloat16)" in body.source


def test_catalog_describes_g0_range_and_rejects_incompatible_target():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight)
        if region.semantic_facts.get("family") == "view_plus_gemm"
    )
    result = get_candidates(region, _target(), _policy())

    assert [config.provider for config in result.configurations] == [
        "generic",
        "g0_range",
    ]
    tuned = result.configurations[1]
    assert tuned.name == "g0_range_64x64"
    assert tuned.pipeline.stage_count == 13
    assert tuned.participants.registers_per_thread == 232
    assert tuned.scratch.size_bytes == 230_400
    assert tuned.evidence.revision == "0b55a2f691d69981583568fd9eb69687b1f0de8a"
    assert "WGMMA accumulators" in tuned.completion.register_fragments

    rejected = get_candidates(region, _target(max_shared_memory_per_cta=230_399), _policy())
    assert [config.provider for config in rejected.configurations] == ["generic"]
    assert rejected.rejections[-1].provider == "g0_range"
    assert "shared memory" in rejected.rejections[-1].reason


def test_region_without_tensor_program_is_rejected():
    region = Region(
        id="region:unknown",
        source_op_ids=("op:unknown",),
        external_inputs=(),
        live_outputs=(),
        semantic_facts={},
        tensor_programs=(),
    )
    result = get_candidates(region, _target(), _policy())

    assert result.configurations == ()
    assert result.rejections[0].reason == "region has no tensor program"


def test_invalid_participation_and_layout_are_rejected():
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value)

    region = _regions(Model(), torch.randn(3, 5))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]

    too_many_threads = replace(
        config.participants, participating_threads=2048, waiting_threads=2048
    )
    with pytest.raises(InvalidBodyConfig, match="threads"):
        validate_body_config(replace(config, participants=too_many_threads), _target())

    first_input = next(iter(config.layouts.inputs))
    bad_layout = replace(config.layouts.inputs[first_input], strides=(1,))
    bad_inputs = {**config.layouts.inputs, first_input: bad_layout}
    with pytest.raises(InvalidBodyConfig, match="mismatched axes and strides"):
        validate_body_config(
            replace(config, layouts=replace(config.layouts, inputs=bad_inputs))
        )
