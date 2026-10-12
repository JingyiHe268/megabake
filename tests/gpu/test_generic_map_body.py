from __future__ import annotations

import importlib.util
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from megabake.bodies.catalog import get_candidates
from megabake.bodies.generic import emit_map_body
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="map-body launcher requires CUDA"
)


def _target():
    from megabake.bodies import SelectionPolicy, TargetFacts

    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    ), SelectionPolicy()


def _program(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    region = graph.regions[0]
    target, policy = _target()
    config = get_candidates(region, target, policy).configurations[0]
    return region.tensor_programs[0], config


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_body(program, config, name, tmp_path):
    body = emit_map_body(program, config)
    path = tmp_path / f"{name}_body.py"
    path.write_text(body.source, encoding="utf-8")
    return _load_module(path, f"megabake_{name}_body"), body


def _load_launcher(body_name, tensor_count, threads, tmp_path):
    source = f'''from {body_name} import run
import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.jit
def launch(
    work_iterator: cute.Tensor,
{"".join(f"    tensor_{index}: cute.Tensor,\n" for index in range(tensor_count))}
    stream: cuda.CUstream,
):
    run(
        work_iterator,
        ({", ".join(f"tensor_{index}" for index in range(tensor_count))},),
        None,
        None,
    ).launch(
        grid=[cute.size(work_iterator), 1, 1],
        block=[{threads}, 1, 1],
        stream=stream,
    )
'''
    module_name = f"megabake_{body_name}_launcher"
    path = tmp_path / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")
    return _load_module(path, module_name).launch


def _view(tensor):
    result = from_dlpack(tensor, assumed_align=16)
    leading_dimension = next(
        (
            index
            for index, stride in enumerate(tensor.stride())
            if stride == 1
        ),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _run_body(body, tensors, threads):
    count = tensors[-1].numel()
    tile_count = (count + threads - 1) // threads
    work = (
        torch.arange(tile_count, device="cuda", dtype=torch.int64) * threads
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    views = [_view(work), *(_view(tensor) for tensor in tensors)]
    compiled = cute.compile(body, *views, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()


def test_map_body_launcher_handles_two_expressions_partial_tiles_and_strides(
    tmp_path,
):
    class Add(torch.nn.Module):
        def forward(self, left, right):
            return left + right

    class Cast(torch.nn.Module):
        def forward(self, value):
            return value.to(torch.bfloat16)

    rows, columns = 5, 70
    left_source = torch.randn(columns, rows)
    left = left_source.t()
    right = torch.randn(1, columns)
    value = torch.randn(rows, columns)

    add_program, add_config = _program(Add(), left, right)
    cast_program, cast_config = _program(Cast(), value)
    add_module, add_body = _load_body(add_program, add_config, "add", tmp_path)
    cast_module, cast_body = _load_body(cast_program, cast_config, "cast", tmp_path)

    assert add_body.threads_per_cta == cast_body.threads_per_cta == 128
    assert "tensor_bindings[1][0, o1]" in add_body.source
    assert "linear_index < output_count" in add_body.source
    assert ".to(cutlass.BFloat16)" in cast_body.source

    add_launcher = _load_launcher(
        "megabake_add_body", len(add_program.loads) + 1, 128, tmp_path
    )
    cast_launcher = _load_launcher(
        "megabake_cast_body", len(cast_program.loads) + 1, 128, tmp_path
    )

    gpu_left = left.cuda()
    gpu_right = right.cuda()
    gpu_value = value.cuda()
    add_output = torch.full(
        (rows, columns), float("nan"), device="cuda", dtype=torch.float32
    )
    cast_output = torch.full(
        (rows, columns), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    _run_body(add_launcher, (gpu_left, gpu_right, add_output), 128)
    _run_body(cast_launcher, (gpu_value, cast_output), 128)

    torch.testing.assert_close(add_output, gpu_left + gpu_right)
    torch.testing.assert_close(cast_output, gpu_value.to(torch.bfloat16))
