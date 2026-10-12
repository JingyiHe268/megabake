from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from megabake.bodies import (
    CandidateSet,
    InvalidBodyConfig,
    SelectionPolicy,
    TargetFacts,
    validate_body_config,
)
from megabake.bodies.catalog import get_candidates
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.graph import Region
from megabake.lowering import canonicalize_semantics
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
