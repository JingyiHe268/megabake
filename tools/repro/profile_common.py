"""Small, separate CUPTI windows; traces stay private, aggregates are publishable."""
import collections
import json
import time

WINDOW_STEPS = 16
LEADS = [0, 2032, 4079]


def category(name):
    text = name.lower()
    if 'megabake' in text or 'cooperative' in text:
        return 'megakernel'
    if any(s in text for s in ('argmax', 'sample', 'sampling', 'topk')):
        return 'sampling'
    if any(s in text for s in ('norm', 'rotary', 'rope', 'silu', 'gelu')):
        return 'norm / position / activation'
    if any(s in text for s in ('attention', 'flashattn', 'flash_attn', 'flash_fwd', 'fmha', 'batch_decode', 'batchdecode', 'softmax')):
        return 'attention / softmax'
    if any(s in text for s in ('gemm', 'gemv', 'matmul', 'cublas', 'cutlass', 'nvjet_tst')):
        return 'linear algebra'
    if any(s in text for s in ('memcpy', 'memset', 'copy', 'index')):
        return 'copy / indexing'
    return 'other / fused'


def summarize_trace(trace, steps):
    events = trace['traceEvents']
    gpu = [e for e in events if e.get('ph') == 'X' and
           e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('dur', 0) > 0]
    if not gpu:
        raise RuntimeError('No GPU activities in profiling window')
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in gpu)
    busy = 0.; start, end = intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            busy += end-start; start, end = a, b
        else:
            end = max(end, b)
    busy += end-start
    span = max(b for a, b in intervals)-intervals[0][0]
    names = collections.defaultdict(lambda: [0, 0.])
    categories = collections.defaultdict(float)
    for e in gpu:
        names[e['name']][0] += 1; names[e['name']][1] += e['dur']
        categories[category(e['name'])] += e['dur']
    apis = collections.defaultdict(lambda: [0, 0.])
    for e in events:
        if e.get('ph') == 'X' and e.get('cat') == 'cuda_runtime':
            apis[e['name']][0] += 1; apis[e['name']][1] += e.get('dur', 0.)
    return dict(steps=steps, gpu_busy_us_per_step=busy/steps,
                gpu_span_us_per_step=span/steps, gpu_gap_us_per_step=(span-busy)/steps,
                gpu_activities_per_step=len(gpu)/steps,
                kernel_count_per_step=sum(e.get('cat') == 'kernel' for e in gpu)/steps,
                gpu_categories_us_per_step={k:v/steps for k,v in categories.items()},
                kernels=[dict(name=k,calls_per_step=v[0]/steps,us_per_step=v[1]/steps)
                         for k,v in sorted(names.items(),key=lambda kv:-kv[1][1])],
                cuda_apis=[dict(name=k,calls_per_step=v[0]/steps,us_per_step=v[1]/steps)
                           for k,v in sorted(apis.items(),key=lambda kv:-kv[1][1])])


def profile_window(folder, engine, lead, step):
    import torch
    torch.cuda.synchronize()
    path = folder/f'{engine}-lead{lead}-trace.json'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as p:
        begin=time.perf_counter()
        for i in range(WINDOW_STEPS):
            with torch.profiler.record_function('decode_step'):
                step(lead+i)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
    p.export_chrome_trace(str(path))
    result=summarize_trace(json.loads(path.read_text()),WINDOW_STEPS)
    result.update(lead_decode_steps=lead,first_cached_tokens=128+lead,
                  last_cached_tokens=128+lead+WINDOW_STEPS-1,
                  profiled_wall_us_per_step=elapsed*1e6/WINDOW_STEPS)
    return result
