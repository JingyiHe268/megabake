"""Measure the fusion budget against an actual Inductor MLP on this GPU."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/megabake-design-research/inductor')
import torch
from probe import samples


def host_samples(fn, count=11):
    values = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1e6)
    return {'median_us': statistics.median(values), 'samples_us': values}


def mlp(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(456)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_stream(torch.cuda.Stream())
    from torch._inductor.utils import is_big_gpu
    report = {'inductor_gemm_autotune_eligible': is_big_gpu(0), 'torch': torch.__version__, 'gpu': str(torch.cuda.get_device_properties(0)),
              'mode': 'max-autotune-no-cudagraphs, then externally captured for GPU timing',
              'seed': 456, 'cache': 'repeated buffers', 'records': []}
    compile_wrapper_start = time.perf_counter()
    compiled = torch.compile(mlp, fullgraph=True, mode='max-autotune-no-cudagraphs')
    report['compile_wrapper_setup_seconds'] = time.perf_counter() - compile_wrapper_start
    for k, n in [(576, 1536), (4096, 11008)]:
        for m in [1, 256, 1024]:
            setup_start = time.perf_counter()
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) / (k ** 0.5)
            gate, up = [torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / (k ** 0.5) for _ in range(2)]
            down = torch.randn((k, n), device='cuda', dtype=torch.bfloat16) / (n ** 0.5)
            ref = mlp(x, gate, up, down)
            row = {'m': m, 'n': n, 'k': k,
                   'input_and_reference_setup_seconds': time.perf_counter() - setup_start}
            for name, fn in [('eager_graph', mlp), ('inductor_graph', compiled)]:
                first_gpu_start = torch.cuda.Event(enable_timing=True)
                first_gpu_end = torch.cuda.Event(enable_timing=True)
                first_host_start = time.perf_counter()
                first_gpu_start.record()
                result = fn(x, gate, up, down)
                first_gpu_end.record()
                first_gpu_end.synchronize()
                torch.cuda.synchronize()
                first_host_us = (time.perf_counter() - first_host_start) * 1e6
                error = (result.float() - ref.float()).norm() / ref.float().norm()
                assert error.item() < 0.01, error.item()
                assert torch.isfinite(result).all().item()
                post_compile_gpu_start = torch.cuda.Event(enable_timing=True)
                post_compile_gpu_end = torch.cuda.Event(enable_timing=True)
                post_compile_host_start = time.perf_counter()
                post_compile_gpu_start.record()
                fn(x, gate, up, down)
                post_compile_gpu_end.record()
                post_compile_gpu_end.synchronize()
                torch.cuda.synchronize()
                row[name] = samples(lambda: fn(x, gate, up, down), 32)
                row[name].update(
                    first_call_event_span_us=first_gpu_start.elapsed_time(first_gpu_end) * 1000,
                    first_call_host_us=first_host_us,
                    first_post_compile_event_span_us=post_compile_gpu_start.elapsed_time(post_compile_gpu_end) * 1000,
                    first_post_compile_host_us=(time.perf_counter() - post_compile_host_start) * 1e6,
                    complete_host_call=host_samples(lambda: fn(x, gate, up, down)),
                    relative_l2_error=error.item(),
                )
                if name == 'inductor_graph':
                    row[name]['compile_and_first_call_host_us'] = first_host_us
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    fn(x, gate, up, down)
                    torch.cuda.synchronize()
                trace_path = Path('/tmp/megabake-design-research') / f'{name}-trace.json'
                prof.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
                row[name]['kernel_names'] = [
                    e['name'] for e in trace['traceEvents'] if e.get('cat') == 'kernel'
                ]
                row[name]['launch_count'] = len(row[name]['kernel_names'])
            # Library GEMMs with actual boundary shapes. This sum is a diagnostic,
            # not a lower bound or an equivalent standalone end-to-end workload.
            intermediate = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            output = torch.empty_like(x)
            components = [(x, gate, intermediate), (x, up, intermediate),
                          (intermediate, down, output)]
            # Initialize every input before diagnostic timing.
            intermediate.normal_()
            row['library_gemm_sum_us'] = sum(
                samples(lambda a=a, w=w, c=c: torch.mm(a, w.T, out=c), 32)['median_us']
                for a, w, c in components)
            row['inductor_kernel_names'] = row['inductor_graph']['kernel_names']
            report['records'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'m':m, 'n':n, 'k':k, 'eager_us':row['eager_graph']['median_us'],
                              'inductor_us':row['inductor_graph']['median_us'],
                              'gemm_sum_us':row['library_gemm_sum_us'],
                              'kernel_count':len(row['inductor_kernel_names'])}), flush=True)


if __name__ == '__main__':
    main()
