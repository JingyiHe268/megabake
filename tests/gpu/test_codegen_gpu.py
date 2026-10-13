from __future__ import annotations

import importlib.util
import sys

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import pytest
import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    emit_cute,
    make_phase_schedule,
    select_bodies,
)
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.bodies.gemm import (
    compile_g0_range_gemm,
    cute_gemm_tensors,
    g0_range_body,
)
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="generated CuTe modules require an SM90 GPU",
)


def _target() -> TargetFacts:
    properties = torch.cuda.get_device_properties(0)
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=properties.multi_processor_count,
        supports_tma=True,
        supports_wgmma=True,
    )


def _plan(model, *inputs, allow_tuned=True):
    prepared = prepare_fx_graph(
        torch.export.export(model, inputs).graph_module, inputs
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(
        graph, _target(), SelectionPolicy(allow_tuned=allow_tuned)
    )
    return assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )


def _load(generated):
    spec = importlib.util.spec_from_file_location(
        generated.module_name, generated.source_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[generated.module_name] = module
    spec.loader.exec_module(module)
    return module


def _view(tensor):
    result = from_dlpack(tensor, assumed_align=16)
    leading_dimension = next(
        (index for index, stride in enumerate(tensor.stride()) if stride == 1),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _run_generic(generated, source):
    module = _load(generated)
    tensors = []
    for argument in generated.launch.tensors:
        if argument.kind == "input":
            tensors.append(source)
        else:
            tensors.append(
                torch.empty(
                    argument.layout.shape,
                    dtype=getattr(torch, argument.layout.dtype.removeprefix("torch.")),
                    device="cuda",
                )
            )
    ready = torch.full(
        (generated.launch.grid_size + 1,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    work_iterators = [
        torch.arange(iterator.length, device="cuda", dtype=torch.int64) * iterator.step
        for iterator in generated.launch.work_iterators
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    views = [
        _view(ready),
        cutlass.Int32(generated.launch.grid_size),
        cutlass.Int32(0),
        *(_view(iterator) for iterator in work_iterators),
        *(_view(tensor) for tensor in tensors),
    ]
    compiled = cute.compile(module.launch, *views, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    return tensors[-1]


def test_generated_cooperative_module_compiles_and_matches_reference(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value) + torch.relu(value)

    source = torch.randn(5, 70)
    plan = _plan(Model(), source, allow_tuned=False)
    generated = emit_cute(plan, tmp_path)
    output = _run_generic(generated, source.cuda())

    torch.testing.assert_close(output.cpu(), torch.relu(source) + torch.relu(source))


def test_generated_g0_adapter_matches_existing_g0_executable(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    plan = _plan(Model(), source, weight)
    generated = emit_cute(plan, tmp_path)
    assert generated.launch.phases[0].provider == "g0_range"

    module = _load(generated)
    gpu_source = source.cuda()
    gpu_weight = weight.cuda()
    output = torch.empty(4, 1536, device="cuda", dtype=torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    tensor_a, tensor_b, tensor_c = cute_gemm_tensors(
        gpu_source, gpu_weight, output
    )
    compiled = cute.compile(
        module.launch, tensor_a, tensor_b, tensor_c, max_clusters, stream
    )
    compiled(tensor_a, tensor_b, tensor_c, stream)
    torch.cuda.synchronize()
    generated_output = output.clone()

    existing = compile_g0_range_gemm(
        g0_range_body(plan.phases[0].body),
        gpu_source,
        gpu_weight,
        output,
        max_clusters,
        stream,
    )
    existing(gpu_source, gpu_weight, output, stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(generated_output, output, rtol=0, atol=0)
