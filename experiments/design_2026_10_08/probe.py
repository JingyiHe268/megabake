"""Bounded architecture probe. Official kernels are fetched at a fixed revision.

This measures GEMM providers; it does not implement MegaBake. See RESEARCH.md.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request

REV = '0b55a2f691d69981583568fd9eb69687b1f0de8a'
HASHES = {'dense_gemm.py': 'bb7b76d893219757e2f3701abf1a7c8c819b9966e38aaed30a768596a55aca9f', 'dense_gemm_persistent.py': '65088a1fd7655ed46ad54fecd08e1461a334fb38f1b2d1da7cf252e465afc130'}
BASE = f'https://raw.githubusercontent.com/NVIDIA/cutlass/{REV}/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/'


def source(root, name):
    path = root / name
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(BASE + name, timeout=30).read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[name], f'Source hash mismatch: {path}'
    return path


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = result
    spec.loader.exec_module(result)
    return result


def samples(fn, launches, count=11, output=None):
    """Replay a fixed graph to remove Python gaps; retain each batch average."""
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(launches):
            fn()
    if output is not None:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    if output is not None:
        assert torch.isfinite(output).all().item(), "Graph replay did not write output"
    result = []
    for _ in range(count):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end) * 1000 / launches)
    return {'median_us': statistics.median(result), 'batch_mean_us': result}


def host_samples(fn, count=11):
    """Measure a complete synchronized Python call, retaining every sample."""
    import torch
    result = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        result.append((time.perf_counter() - start) * 1e6)
    return {'median_us': statistics.median(result), 'samples_us': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--single-m', type=int)
    parser.add_argument('--large-shape', action='store_true')
    parser.add_argument('--resources', action='store_true')
    parser.add_argument('--large-tiles', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--shape', help='one M,N,K projection shape, for example 4,576,1536')
    args = parser.parse_args()
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    root = Path('/tmp/megabake-design-research')
    root.mkdir(exist_ok=True)
    paths = [source(root, name) for name in ('dense_gemm.py', 'dense_gemm_persistent.py')]
    dense, persistent = map(module, paths)
    from restart_body import adapt
    restart = module(adapt(paths[0]))
    properties = torch.cuda.get_device_properties(0)
    report = {'torch': torch.__version__, 'torch_git': torch.version.git_version,
              'cuda_runtime': torch.version.cuda, 'cute': cutlass.__version__,
              'gpu': str(properties), 'cutlass_revision': REV,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'seed': 123, 'validation_only': args.validate_only, 'precision': 'BF16 inputs/output, FP32 accumulation; reduced precision reduction disabled',
              'timing': 'CUDA event batch averages, CUDA Graph replay; no Python gaps; no host latency claim',
              'cache': 'rotating weights, at least 64 MiB' if args.streaming else 'repeated same buffers',
              'records': []}
    shapes = [(1, 1536, 576), (4, 1536, 576), (256, 1536, 576), (1024, 1536, 576),
              (1, 11008, 4096), (256, 11008, 4096), (1024, 11008, 4096)]
    if args.shape:
        shapes = [tuple(int(value) for value in args.shape.split(','))]
        if len(shapes[0]) != 3 or min(shapes[0]) <= 0:
            parser.error('--shape must contain positive M,N,K values')
    elif args.quick:
        shapes = [(args.single_m or 4, 11008, 4096)] if args.large_shape else [(args.single_m or 4, 1536, 576)]
    torch.cuda.set_stream(torch.cuda.Stream())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    for m, n, k in shapes:
        pool = max(1, math.ceil((64 * 1024**2) / (2 * n * k))) if args.streaming else 1
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        weights = [torch.randn((n, k), device='cuda', dtype=torch.bfloat16) for _ in range(pool)]
        c = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        # FP32 reference uses exactly the quantized BF16 input values.
        ref = (a.float() @ weights[0].float().T).bfloat16()
        ca = from_dlpack(a.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cb = [from_dlpack(w.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1) for w in weights]
        cc = from_dlpack(c.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        variants = [('torch_mm', None, None)]
        tiles = [(64, 256), (128, 256)] if args.large_tiles else [(64, 64), (64, 128), (128, 128)]
        for tile in tiles:
            variants.extend([('dense', tile, dense.HopperWgmmaGemmKernel),
                             ('persistent', tile, persistent.HopperWgmmaGemmPersistentKernel),
                             ('restart', tile, restart.HopperWgmmaGemmKernel)])
        for name, tile, cls in variants:
            row = {'m': m, 'n': n, 'k': k, 'provider': name, 'tile': tile,
                   'weight_buffers': pool, 'launch_count': 1}
            start = time.perf_counter()
            try:
                if cls is None:
                    functions = [lambda w=w: torch.mm(a, w.T, out=c) for w in weights]
                else:
                    op = cls(cutlass.Float32, tile, (1, 1), 1, True) if name == 'persistent' else cls(cutlass.Float32, tile, (1, 1))
                    if name == "restart":
                        op.workers = properties.multi_processor_count
                    jit_args = (ca, cb[0], cc, max_clusters, stream) if name == 'persistent' else (ca, cb[0], cc, stream)
                    compiled = cute.compile(op, *jit_args)
                    row['launch_count'] = len(compiled.kernel_info)
                    functions = [lambda b=b: compiled(ca, b, cc, stream) for b in cb]
                    row.update(threads=op.threads_per_cta, stages=op.ab_stage)
                    if args.resources:
                        row['shared_storage_bytes'] = op.shared_storage.__sizeof__()
                        row['kernels'] = []
                        for symbol in compiled.kernel_info:
                            status, kernel = cuda.cuLibraryGetKernel(cuda.CUlibrary(int(compiled.library)), symbol.encode())
                            assert int(status) == 0, status
                            values = {'name': symbol}
                            for label, attr in [('registers', cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS), ('local_bytes', cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]:
                                status, value = cuda.cuKernelGetAttribute(attr, kernel, 0)
                                assert int(status) == 0, status
                                values[label] = value
                            status, fun = cuda.cuKernelGetFunction(kernel)
                            assert int(status) == 0, status
                            status, = cuda.cuFuncSetAttribute(fun, cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, row['shared_storage_bytes'])
                            assert int(status) == 0, status
                            status, count = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(fun, op.threads_per_cta, row['shared_storage_bytes'])
                            assert int(status) == 0, status
                            values['resident_blocks_per_sm'] = count
                            row['kernels'].append(values)
                row['setup_seconds'] = time.perf_counter() - start
                c.fill_(float("nan"))
                first_gpu_start = torch.cuda.Event(enable_timing=True)
                first_gpu_end = torch.cuda.Event(enable_timing=True)
                first_host_start = time.perf_counter()
                first_gpu_start.record()
                functions[0]()
                first_gpu_end.record()
                first_gpu_end.synchronize()
                row['first_call_event_span_us'] = first_gpu_start.elapsed_time(first_gpu_end) * 1000
                row['first_call_host_us'] = (time.perf_counter() - first_host_start) * 1e6
                error = (c.float() - ref.float()).abs()
                row['max_abs_error'] = error.max().item()
                row['relative_l2_error'] = (error.norm() / ref.float().norm()).item()
                # Fixed diagnostic threshold, chosen before measurements.
                assert row['relative_l2_error'] < 0.005, row
                assert torch.isfinite(c).all().item()
                def batch():
                    for fn in functions:
                        fn()
                if args.validate_only:
                    a.neg_()
                    c.fill_(float("nan"))
                    functions[0]()
                    torch.cuda.synchronize()
                    changed_error = (c.float() + ref.float()).norm() / ref.float().norm()
                    assert torch.isfinite(c).all().item()
                    assert changed_error.item() < 0.005
                    a.neg_()
                else:
                    timing = samples(batch, max(1, 64 // pool), output=c)
                    assert torch.isfinite(c).all().item()
                    timing['median_us'] /= pool
                    timing['batch_mean_us'] = [v / pool for v in timing['batch_mean_us']]
                    row['warm_gpu_per_weight'] = timing
                    row['complete_host_call'] = host_samples(batch)
                row['status'] = 'passed_diagnostic'
            except Exception as exc:
                row.update(status='error', error=str(exc))
                if isinstance(exc, torch.AcceleratorError):
                    report['records'].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    raise
            report['records'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({key: val for key, val in row.items() if key != 'batch_mean_us'}), flush=True)
    torch.cuda.synchronize()
    assert all(row['status'] == 'passed_diagnostic' for row in report['records']), 'One or more candidates failed; see the report'


if __name__ == '__main__':
    main()
