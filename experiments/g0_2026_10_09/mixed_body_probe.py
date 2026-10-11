"""One cooperative kernel with two CuTe GEMMs and generated map/reduction work."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "design_2026_10_08"))
import resident_range_probe as range_probe
from probe import REV, HASHES, source


@cute.jit
def _grid_barrier(ready: cute.Tensor, workers: cutlass.Int32, epoch: cutlass.Int32):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        cute.arch.atomic_exch(
            ready.iterator + block,
            epoch,
            sem="release",
            scope="gpu",
        )
        if block == 0:
            all_ready = False
            while not all_ready:
                all_ready = True
                for i in cutlass.range(workers):
                    seen = cute.arch.atomic_cas(
                        ready.iterator + i,
                        cmp=epoch,
                        val=epoch,
                        sem="acquire",
                        scope="gpu",
                    )
                    if seen != epoch:
                        all_ready = False
            cute.arch.atomic_exch(
                ready.iterator + workers,
                epoch,
                sem="release",
                scope="gpu",
            )
        else:
            released = False
            while not released:
                seen = cute.arch.atomic_cas(
                    ready.iterator + workers,
                    cmp=epoch,
                    val=epoch,
                    sem="acquire",
                    scope="gpu",
                )
                released = seen == epoch
    cute.arch.sync_threads()


@dataclass(frozen=True)
class _TypedScalar:
    code: str
    dtype: str


def _cast(value, dtype):
    if dtype != "f32":
        raise TypeError(f"unsupported expression cast to {dtype}")
    return _TypedScalar(f"({value.code}).to(cutlass.Float32)", dtype)


def _binary(operator, left, right):
    if left.dtype != right.dtype:
        raise TypeError(f"{operator} requires matching types: {left.dtype}, {right.dtype}")
    return _TypedScalar(f"({left.code} {operator} {right.code})", left.dtype)


def _emit_vector(name, expression, row_broadcast=False):
    if expression.dtype != "f32":
        raise TypeError("vector expression must produce FP32 before the output cast")
    if row_broadcast:
        signature = """    source: cute.Tensor,
    rows: cute.Tensor,
    target: cute.Tensor,
    columns: cutlass.Int32,
    workers: cutlass.Int32,
    threads: cutlass.Constexpr,
    scale: cutlass.Float32,
    offset: cutlass.Float32,"""
    else:
        signature = """    source: cute.Tensor,
    target: cute.Tensor,
    workers: cutlass.Int32,
    threads: cutlass.Constexpr,
    scale: cutlass.Float32,
    offset: cutlass.Float32,"""
    body = f"""@cute.jit
def {name}(
{signature}
):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    count = cute.size(source)
    first = block * threads + thread
    step = workers * threads
    for i in cutlass.range(first, count, step):
        value = {expression.code}
        target[i] = value.to(cutlass.BFloat16)
"""
    return body


def _emit_row_sum():
    accumulator = _TypedScalar("total", "f32")
    element = _cast(_TypedScalar("source[row * columns + column]", "bf16"), "f32")
    expression = _binary("+", accumulator, element)
    body = f"""@cute.jit
def _row_reduce(
    source: cute.Tensor,
    target: cute.Tensor,
    rows: cutlass.Int32,
    columns: cutlass.Int32,
    workers: cutlass.Int32,
):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        for row in cutlass.range(block, rows, workers):
            total = cutlass.Float32(0.0)
            for column in cutlass.range(columns):
                total = {expression.code}
            target[row] = total.to(cutlass.BFloat16)
"""
    return body


_source_value = _TypedScalar("source[i]", "bf16")
_row_value = _TypedScalar("rows[i // columns]", "bf16")
_scale = _TypedScalar("scale", "f32")
_offset = _TypedScalar("offset", "f32")
_vector_body_sources = [
    _emit_vector(
        "_vector_affine",
        _binary("+", _binary("*", _cast(_source_value, "f32"), _scale), _offset),
    ),
    _emit_vector(
        "_vector_row_bias",
        _binary(
            "+",
            _binary(
                "+",
                _cast(_source_value, "f32"),
                _binary("*", _cast(_row_value, "f32"), _scale),
            ),
            _offset,
        ),
        row_broadcast=True,
    ),
    _emit_row_sum(),
]
_generated_path = Path(__file__).with_name("generated_typed_bodies.py")
_generated_path.write_text(
    "import cutlass\nimport cutlass.cute as cute\n\n"
    + "\n".join(_vector_body_sources)
)
_generated_module = range_probe._load(_generated_path, "megabake_generated_typed_bodies")
_vector_affine = _generated_module._vector_affine
_vector_row_bias = _generated_module._vector_row_bias
_row_reduce = _generated_module._row_reduce
GENERATED_BODY_SPECS = [
    {
        "name": name,
        "expression": expression,
        "input_dtype": input_dtype,
        "accumulator_dtype": accumulator_dtype,
        "output_dtype": "bf16",
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
    }
    for name, expression, input_dtype, accumulator_dtype, source_text in (
        ("_vector_affine", "cast_f32(x) * scale + offset", "bf16", None, _vector_body_sources[0]),
        ("_vector_row_bias", "cast_f32(x) + cast_f32(row) * scale + offset", "bf16", None, _vector_body_sources[1]),
        ("_row_reduce", "sum(cast_f32(x))", "bf16", "f32", _vector_body_sources[2]),
    )
]


def _mixed_class(base):
    class MixedBody(base):
        @cute.kernel
        def mixed_kernel(
            self,
            tma_a0,
            tensor_a0,
            tma_b0,
            tensor_b0,
            tma_t,
            tensor_t,
            work0,
            tma_a1,
            tensor_a1,
            tma_b1,
            tensor_b1,
            tma_y,
            tensor_y,
            work1,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_smem_layout_staged: cute.ComposedLayout,
            tensor_t_flat: cute.Tensor,
            tensor_u_flat: cute.Tensor,
            tensor_rows: cute.Tensor,
            tensor_v_flat: cute.Tensor,
            ready: cute.Tensor,
            delay_source: cute.Tensor,
            delay_sink: cute.Tensor,
            workers: cutlass.Int32,
            epoch_base: cutlass.Int32,
            delay_cta: cutlass.Int32,
        ):
            _, _, block = cute.arch.block_idx()
            thread, _, _ = cute.arch.thread_idx()
            scratch = cutlass.memory.SmemAllocator().allocate(self.shared_storage)

            # Retain at least two tiles per GEMM worker. Extra outer workers
            # bypass the GEMM pipeline and still join the phase barriers.
            tiles0 = cute.size(work0.problem_shape_ntile_mnl)
            active0 = workers // 2
            if active0 > tiles0 // 2:
                active0 = tiles0 // 2
            if active0 < 1:
                active0 = 1
            if block < active0:
                self.run_range(
                    work0,
                    (
                        tma_a0,
                        tensor_a0,
                        tma_b0,
                        tensor_b0,
                        tiled_mma,
                        cta_layout_mnk,
                        a_smem_layout_staged,
                        b_smem_layout_staged,
                    ),
                    scratch,
                    (tma_t, tensor_t, epi_smem_layout_staged),
                    (cutlass.Int32(1), cutlass.Int32(1), active0),
                )

            # A selected producer does extra live global reads before it
            # publishes. This makes early arrival visible in the barrier test.
            if block == delay_cta and thread == 0:
                total = cutlass.Float32(0.0)
                for i in cutlass.range(100_000):
                    total += delay_source[i % 257]
                delay_sink[block] = total
            cute.arch.sync_threads()
            _grid_barrier(ready, workers, epoch_base)

            _vector_affine(
                tensor_t_flat,
                tensor_u_flat,
                workers,
                self.threads_per_cta,
                cutlass.Float32(0.5),
                cutlass.Float32(0.125),
            )
            _grid_barrier(ready, workers, epoch_base + 1)

            rows = cute.size(tensor_t, mode=[0])
            columns = cute.size(tensor_t, mode=[1])
            _row_reduce(tensor_u_flat, tensor_rows, rows, columns, workers)
            _grid_barrier(ready, workers, epoch_base + 2)

            _vector_row_bias(
                tensor_u_flat,
                tensor_rows,
                tensor_v_flat,
                cutlass.Int32(columns),
                workers,
                self.threads_per_cta,
                cutlass.Float32(0.25),
                cutlass.Float32(-0.5),
            )
            # The next GEMM reads every row of V, so publish all its writes.
            _grid_barrier(ready, workers, epoch_base + 3)

            tiles1 = cute.size(work1.problem_shape_ntile_mnl)
            active1 = workers // 2
            if active1 > tiles1 // 2:
                active1 = tiles1 // 2
            if active1 < 1:
                active1 = 1
            if block < active1:
                self.run_range(
                    work1,
                    (
                        tma_a1,
                        tensor_a1,
                        tma_b1,
                        tensor_b1,
                        tiled_mma,
                        cta_layout_mnk,
                        a_smem_layout_staged,
                        b_smem_layout_staged,
                    ),
                    scratch,
                    (tma_y, tensor_y, epi_smem_layout_staged),
                    (cutlass.Int32(1), cutlass.Int32(1), active1),
                )

        @cute.jit
        def __call__(
            self,
            a0: cute.Tensor,
            b0: cute.Tensor,
            t: cute.Tensor,
            a1: cute.Tensor,
            b1: cute.Tensor,
            y: cute.Tensor,
            t_flat: cute.Tensor,
            u_flat: cute.Tensor,
            rows: cute.Tensor,
            v_flat: cute.Tensor,
            ready: cute.Tensor,
            delay_source: cute.Tensor,
            delay_sink: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            workers: cutlass.Int32,
            epoch_base: cutlass.Int32,
            delay_cta: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.a_dtype = a0.element_type
            self.b_dtype = b0.element_type
            self.c_dtype = t.element_type
            self.a_layout = cutlass.utils.LayoutEnum.from_tensor(a0)
            self.b_layout = cutlass.utils.LayoutEnum.from_tensor(b0)
            self.c_layout = cutlass.utils.LayoutEnum.from_tensor(t)
            self._setup_attributes()

            tma_a0, tensor_a0 = self._make_tma_atoms_and_tensors(
                a0,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[1],
            )
            tma_b0, tensor_b0 = self._make_tma_atoms_and_tensors(
                b0,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[0],
            )
            tma_t, tensor_t = self._make_tma_store_atoms_and_tensors(
                t,
                self.epi_smem_layout_staged,
                self.epi_tile,
            )
            work0, _ = self._compute_grid(
                tensor_t,
                self.tile_shape_mnk,
                self.cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
                max_active_clusters,
            )

            tma_a1, tensor_a1 = self._make_tma_atoms_and_tensors(
                a1,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[1],
            )
            tma_b1, tensor_b1 = self._make_tma_atoms_and_tensors(
                b1,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[0],
            )
            tma_y, tensor_y = self._make_tma_store_atoms_and_tensors(
                y,
                self.epi_smem_layout_staged,
                self.epi_tile,
            )
            work1, _ = self._compute_grid(
                tensor_y,
                self.tile_shape_mnk,
                self.cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
                max_active_clusters,
            )

            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage
            self.mixed_kernel(
                tma_a0,
                tensor_a0,
                tma_b0,
                tensor_b0,
                tma_t,
                tensor_t,
                work0,
                tma_a1,
                tensor_a1,
                tma_b1,
                tensor_b1,
                tma_y,
                tensor_y,
                work1,
                self.tiled_mma,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
                t_flat,
                u_flat,
                rows,
                v_flat,
                ready,
                delay_source,
                delay_sink,
                workers,
                epoch_base,
                delay_cta,
            ).launch(
                grid=[1, 1, workers],
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )

    return MixedBody


def _matrix(tensor):
    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )


def _flat(tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _bindings(m, max_workers, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, 576), device="cuda", dtype=torch.bfloat16) / (576**0.5)
    gate = torch.randn((1536, 576), device="cuda", dtype=torch.bfloat16) / (576**0.5)
    down = torch.randn((576, 1536), device="cuda", dtype=torch.bfloat16) / (1536**0.5)
    t = torch.empty((m, 1536), device="cuda", dtype=torch.bfloat16)
    u = torch.empty_like(t)
    row_sums = torch.empty((m,), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(t)
    y = torch.empty((m, 576), device="cuda", dtype=torch.bfloat16)
    ready = torch.empty((max_workers + 1,), device="cuda", dtype=torch.int32)
    delay_source = torch.ones((257,), device="cuda", dtype=torch.float32)
    delay_sink = torch.empty((max_workers,), device="cuda", dtype=torch.float32)
    matrices = tuple(map(_matrix, (x, gate, t, v, down, y)))
    flats = tuple(map(_flat, (t, u, row_sums, v, ready, delay_source, delay_sink)))
    return (matrices, flats), (x, gate, down, t, u, row_sums, v, y, ready, delay_sink)


def _reference(inputs):
    x, gate, down = inputs[:3]
    t = (x.float() @ gate.float().T).bfloat16()
    u = (t.float() * 0.5 + 0.125).bfloat16()
    rows = u.float().sum(dim=1).bfloat16()
    v = (u.float() + rows.float().unsqueeze(1) * 0.25 - 0.5).bfloat16()
    y = (v.float() @ down.float().T).bfloat16()
    return {"t": t, "u": u, "rows": rows, "v": v, "y": y}


def _check(compiled, args, inputs, expected, workers, epoch, delay_cta, stream, profile=False):
    matrix, flat = args
    x, gate, down, t, u, rows, v, y, ready, delay_sink = inputs
    launch_args = (*matrix, *flat, cutlass.Int32(workers), cutlass.Int32(epoch),
                   cutlass.Int32(delay_cta), stream)
    if profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            compiled(*launch_args)
            torch.cuda.synchronize()
    else:
        compiled(*launch_args)
        torch.cuda.synchronize()
        profiler = None
    for name, actual in (("t", t), ("u", u), ("rows", rows), ("v", v), ("y", y)):
        err = (actual.float() - expected[name].float()).norm() / expected[name].float().norm().clamp_min(1e-6)
        assert torch.isfinite(actual).all().item(), name
        assert err.item() < 0.01, {
            "name": name,
            "relative_l2": err.item(),
            "max_abs": (actual.float() - expected[name].float()).abs().max().item(),
            "max_abs_index": (actual.float() - expected[name].float()).abs().argmax().item(),
            "mismatch_count": (actual != expected[name]).sum().item(),
            "max_actual": actual.flatten()[
                (actual.float() - expected[name].float()).abs().argmax().item()
            ].item(),
            "max_expected": expected[name].flatten()[
                (actual.float() - expected[name].float()).abs().argmax().item()
            ].item(),
            "actual": actual.flatten()[:8].float().tolist(),
            "expected": expected[name].flatten()[:8].float().tolist(),
            "rows": rows.float().tolist(),
        }
    assert delay_sink[delay_cta].item() == 100_000.0
    assert torch.equal(ready[:workers], torch.full_like(ready[:workers], epoch + 3))
    trace = None
    if profiler is not None:
        trace_path = Path(__file__).with_name("mixed_kernel_trace.json")
        profiler.export_chrome_trace(str(trace_path))
        raw = json.loads(trace_path.read_text())
        names = [e["name"] for e in raw["traceEvents"] if e.get("cat") == "kernel"]
        trace = {"path": str(trace_path), "kernel_events": names, "launch_count": len(names)}
    return trace


def _query_resources(compiled, threads, shared_bytes):
    rows = []
    for symbol in compiled.kernel_info:
        status, kernel = cuda.cuLibraryGetKernel(
            cuda.CUlibrary(int(compiled.library)), symbol.encode()
        )
        assert int(status) == 0, status
        values = {"name": symbol}
        for label, attribute in (
            ("registers_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
            ("static_shared_bytes", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            ("local_bytes_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        ):
            status, value = cuda.cuKernelGetAttribute(attribute, kernel, 0)
            assert int(status) == 0, status
            values[label] = value
        status, function = cuda.cuKernelGetFunction(kernel)
        assert int(status) == 0, status
        status, = cuda.cuFuncSetAttribute(
            function,
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_bytes,
        )
        assert int(status) == 0, status
        status, active = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, threads, shared_bytes
        )
        assert int(status) == 0, status
        values["shared_storage_bytes"] = shared_bytes
        values["active_blocks_per_sm"] = active
        rows.append(values)
    assert len(rows) == 1, rows
    return rows[0]


def _admit_workers(requested, maximum):
    if requested <= 0 or requested > maximum:
        raise ValueError(f"worker grid {requested} exceeds compiled limit {maximum}")
    return requested


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20261009)
    torch.cuda.set_stream(torch.cuda.Stream())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    root = Path("/tmp/megabake-design-research")
    root.mkdir(exist_ok=True)
    upstream = source(root, "dense_gemm_persistent.py")
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == HASHES["dense_gemm_persistent.py"]
    generated = range_probe._adapt(upstream)
    range_module = range_probe._load(generated, "megabake_mixed_range_source")
    mixed_type = _mixed_class(range_module.HopperWgmmaGemmPersistentKernel)
    op = mixed_type(cutlass.Float32, (64, 64), (1, 1), 1, True)

    first_m = int(os.environ.get("MIXED_COMPILE_M", "4"))
    first_args, first_tensors = _bindings(first_m, 60, 501)
    matrices, flats = first_args
    properties = torch.cuda.get_device_properties(0)
    device, = cuda.cuDeviceGet(0)[1:]
    status, cooperative = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device
    )
    assert int(status) == 0 and cooperative == 1
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    compiled = cute.compile(
        op,
        *matrices,
        *flats,
        max_clusters,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(23),
        stream,
    )
    shared_bytes = op.shared_storage.__sizeof__()
    resources = _query_resources(compiled, op.threads_per_cta, shared_bytes)
    max_workers = properties.multi_processor_count * resources["active_blocks_per_sm"]
    assert max_workers > 0
    assert len(first_tensors[8]) >= max_workers
    admission_limit = min(
        max_workers,
        int(os.environ.get("MIXED_WORKER_CAP", str(max_workers))),
    )
    try:
        _admit_workers(admission_limit + 1, admission_limit)
    except ValueError:
        oversized_rejected = True
    else:
        raise AssertionError("oversized cooperative grid was admitted")
    # Keep the complete kernel at the largest worker grid verified end to
    # end. Extra blocks bypass GEMM work and still join every phase barrier.
    requested_workers = int(os.environ.get("MIXED_WORKERS", str(admission_limit)))
    workers = _admit_workers(requested_workers, admission_limit)

    profile_first = first_m != 1 and os.environ.get("MIXED_PROFILE", "1") != "0"
    calls = [
        (first_args, first_tensors, 1, min(23, workers - 1), profile_first),
        (*_bindings(first_m, max_workers, 502), 5, min(23, workers - 1), False),
        (*_bindings(first_m, max_workers, 503), 9, min(23, workers - 1), False),
    ]
    if os.environ.get("MIXED_SHAPE_REBIND_ONLY"):
        rebind_m = int(os.environ.get("MIXED_REBIND_M", "1"))
        prior_m4 = int(os.environ.get("MIXED_REBIND_PRIOR_M4", "0"))
        repeats = int(os.environ.get("MIXED_REBIND_REPEATS", "1"))
        calls = [
            (*_bindings(4, max_workers, 510 + i), 1 + 4 * i, min(23, workers - 1), False)
            for i in range(prior_m4)
        ]
        next_epoch = 1 + 4 * prior_m4
        calls.extend(
            (
                *_bindings(rebind_m, max_workers, 520 + i),
                next_epoch + 4 * i,
                min(23, workers - 1),
                False,
            )
            for i in range(repeats)
        )
    elif os.environ.get("MIXED_SHAPE_REBIND"):
        calls.append((*_bindings(1, max_workers, 504), 13, min(23, workers - 1), False))
        calls.append((*_bindings(256, max_workers, 505), 17, min(23, workers - 1), False))
    if os.environ.get("MIXED_ONE_CALL"):
        calls = calls[:1]
    traces = []
    correctness = []
    for call in calls:
        args, tensors, epoch, delay_cta, profile = call
        expected = _reference(tensors)
        print(f"running correctness call M={tensors[0].shape[0]} workers={workers}", flush=True)
        trace = _check(
            compiled,
            args,
            tensors,
            expected,
            workers,
            epoch,
            delay_cta,
            stream,
            profile=profile,
        )
        if trace:
            traces.append(trace)
        correctness.append({
            "m": tensors[0].shape[0],
            "epoch": epoch,
            "delay_cta": delay_cta,
            "delay_sum": tensors[9][delay_cta].item(),
            "max_abs_error": {
                key: (tensor.float() - expected[key].float()).abs().max().item()
                for key, tensor in (("t", tensors[3]), ("u", tensors[4]),
                                    ("rows", tensors[5]), ("v", tensors[6]),
                                    ("y", tensors[7]))
            },
        })

    gpu_samples = []
    host_samples = []
    if not os.environ.get("MIXED_SKIP_TIMING"):
        for index in range(11):
            epoch = 20 + 4 * index
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            host_start = time.perf_counter()
            compiled(
                *matrices,
                *flats,
                cutlass.Int32(workers),
                cutlass.Int32(epoch),
                cutlass.Int32(min(23, workers - 1)),
                stream,
            )
            end.record()
            end.synchronize()
            host_samples.append((time.perf_counter() - host_start) * 1e6)
            gpu_samples.append(start.elapsed_time(end) * 1000)

    report = {
        "status": "passed",
        "gpu": str(properties),
        "cooperative_launch": bool(cooperative),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cute": cutlass.__version__,
        "checkpoint_dimensions": {"hidden": 576, "intermediate": 1536},
        "provider_source": {
            "repository": "NVIDIA/cutlass",
            "revision": REV,
            "path": "examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py",
            "sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
            "license": "BSD-3-Clause; original notice retained in generated source",
        },
        "body_configuration": {"tile_m": 64, "tile_n": 64, "cluster": [1, 1]},
        "body_contract": {
            "work_iterator": "CuTe static persistent tile-scheduler parameters; z dimension is the worker range",
            "bindings": "TMA descriptors and tensors rebuilt from current input/output addresses on each host call",
            "thread_participation": {
                "threads_per_cta": op.threads_per_cta,
                "dma_warp_group": "warps 0-3; warp 0 issues TMA loads",
                "mma_warp_group": "warps 4-7 issue WGMMA and perform the epilogue reduction/store work",
                "tma_store_warp": 4
            },
            "scratch": "one CTA-local SharedStorage allocated by the enclosing kernel and reused only after TMA store tail completion",
            "epilogue": "source GEMM TMA output followed by generated BF16 vector and row-reduction work",
            "visibility": "each CTA publishes a device-scope release marker; CTA 0 acquires all markers then broadcasts a release marker that every other CTA acquires before dependent reads",
            "state": "caller-owned marker slots plus one release slot; host supplies a fresh four-value epoch range; no clearing launch",
            "pipeline_exit": "input TMA producer tail, WGMMA wait_group(0), and output TMA store tail complete before each inter-phase grid barrier",
            "shortcut": "fixed-argument CUDA Graph replay repeats the epoch; add graph-local epoch ownership before enabling replay",
        },
        "resource_admission": {
            "threads_per_cta": op.threads_per_cta,
            "compiled_kernel": resources,
            "sm_count": properties.multi_processor_count,
            "max_workers": max_workers,
            "admission_limit": admission_limit,
            "launched_workers": workers,
            "oversized_grid_rejected": oversized_rejected,
        },
        "phases": [
            "GEMM [M,576] x [1536,576] -> BF16 T",
            "grid barrier",
            "generated BF16 vector: cast(T,FP32)*0.5+0.125 -> BF16 U",
            "grid barrier",
            "generated FP32 row sum(U) -> BF16 R",
            "grid barrier",
            "generated BF16 vector: U+cast(R,FP32)*0.25-0.5 -> BF16 V",
            "grid barrier",
            "GEMM [M,1536] x [576,1536] -> BF16 Y",
        ],
        "generated_bodies": GENERATED_BODY_SPECS,
        "scalar_variants": {
            "first_vector": {"scale_fp32": 0.5, "offset_fp32": 0.125},
            "row_broadcast_vector": {"scale_fp32": 0.25, "offset_fp32": -0.5},
        },
        "correctness_tolerance": "relative L2 < 0.01 for every intermediate and final output",
        "repeated_calls": correctness,
        "launch_trace": traces,
        "launch_count": 1,
        "gpu_latency_us": (
            {"samples": gpu_samples, "median": statistics.median(gpu_samples)}
            if gpu_samples
            else None
        ),
        "host_call_latency_us": (
            {"samples": host_samples, "median": statistics.median(host_samples)}
            if host_samples
            else None
        ),
    }
    output = Path(os.environ.get("MIXED_OUTPUT", Path(__file__).with_name("mixed_body.json")))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
