from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megabake.bodies.blackwell import BlackwellGemmProvider
from megabake.bodies import SelectionPolicy, validate_body_config
from megabake.bodies.catalog import get_candidates
from megabake.lowering import Constant
from tests.unit.test_bodies import _regions, _target
from benchmarks.isolated import _graph_ordered_inputs


class Gemm(torch.nn.Module):
    def forward(self, source, weight):
        return source @ weight.T


def _region():
    return next(r for r in _regions(Gemm(), torch.randn(5, 70, dtype=torch.bfloat16),
                                    torch.randn(33, 70, dtype=torch.bfloat16))
                if r.tensor_programs[0].family == "contraction")


@pytest.mark.parametrize("architecture", ["sm100", "sm103", "sm120", "sm121"])
def test_blackwell_provider_is_tensor_core_without_hopper_resources(architecture):
    target = _target(architecture=architecture, supports_wgmma=False,
                     max_shared_memory_per_cta=99_328)
    result = get_candidates(_region(), target, SelectionPolicy())
    config = next(c for c in result.configurations if c.provider == "blackwell_mma")
    assert not any(c.provider == "g0_range" for c in result.configurations)
    assert config.tiles.mma_family == "mma.sync"
    assert config.scratch.size_bytes == 0
    assert not config.pipeline.async_operations
    validate_body_config(config, target, SelectionPolicy())


def test_blackwell_provider_rejects_other_targets_policy_and_changed_semantics():
    provider = BlackwellGemmProvider()
    region = _region()
    target = _target(architecture="sm121", supports_wgmma=False)
    assert not provider.get_candidates(region, _target(), SelectionPolicy()).configurations
    assert not provider.get_candidates(region, target, SelectionPolicy(allow_tuned=False)).configurations
    program = region.tensor_programs[0]
    for changed in (replace(program, init=Constant(1, "torch.float32")),
                    replace(program, post=Constant(0, "torch.bfloat16"))):
        result = provider.get_candidates(replace(region, tensor_programs=(changed,)), target, SelectionPolicy())
        assert not result.configurations


def test_isolated_benchmark_maps_non_self_inverse_input_permutation():
    inputs = tuple(object() for _ in range(4))
    compiled = SimpleNamespace(input_order=("l_gate_weight_", "l_up_weight_", "l_source_", "l_down_weight_"))
    assert _graph_ordered_inputs(compiled, inputs) == (inputs[1], inputs[2], inputs[0], inputs[3])
