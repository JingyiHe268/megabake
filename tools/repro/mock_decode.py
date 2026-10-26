#!/usr/bin/env python3
"""Fixed-context batch-one decode with random weights and synthetic nonzero KV state."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time
import types


def tensor_kv(source, key, value, key_cache, value_cache, valid_length):
    """GPU-only reference for the declared op; no CPU scalar extraction."""
    import torch
    index = valid_length.to(torch.int64)
    key_cache.index_copy_(1, index, key)
    value_cache.index_copy_(1, index, value)
    valid_length.add_(1)
    query = source.permute(0, 2, 1, 3)
    keys = key_cache.permute(0, 2, 1, 3)
    values = value_cache.permute(0, 2, 1, 3)
    group = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(group, dim=1)
    values = values.repeat_interleave(group, dim=1)
    scores = (query @ keys.transpose(-2, -1)).float() * source.shape[-1] ** -.5
    mask = torch.arange(key_cache.shape[1], device=source.device) < valid_length[0]
    scores = scores.masked_fill(~mask, float('-inf'))
    output = torch.softmax(scores, dim=-1).to(source.dtype) @ values
    return output.permute(0, 2, 1, 3).reshape(source.shape[0] * source.shape[2], 1, source.shape[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', default='smollm-decode')
    parser.add_argument('--rows', type=int, required=True, help='already cached tokens')
    parser.add_argument('--layers', type=int, default=30)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.layers, args.samples, args.rounds) < 1:
        parser.error('positive context, layers, samples and rounds required')
    import torch
    from megabake import make_backend
    from tests.fixtures import SmolLMArchitecture, StatefulSmolLM
    torch.set_grad_enabled(False)
    torch.manual_seed(20261026)
    model = SmolLMArchitecture(layers=args.layers)
    model.config.max_position_embeddings = args.rows + 1
    declared = StatefulSmolLM(model).eval()
    # Reuse the upstream wrapper verbatim, changing only its attention reference.
    namespace = dict(StatefulSmolLM.forward.__globals__, kv_attention=tensor_kv)
    reference_forward = types.FunctionType(StatefulSmolLM.forward.__code__, namespace)
    class TensorReference(StatefulSmolLM):
        forward = reference_forward
    reference = TensorReference(model).eval()
    # Dynamo and backend calibration execute the mutable graph while compiling.
    # Reserve headroom, then restore the entire initial state before validation.
    capacity = args.rows + 128
    shape = (1, capacity, model.config.num_key_value_heads,
             model.config.hidden_size // model.config.num_attention_heads)
    initial = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .02 for _ in range(2 * args.layers)]
    initial += [torch.full((1,), args.rows, device='cuda', dtype=torch.int32) for _ in range(args.layers)]
    token = torch.tensor([[504]], device='cuda')
    position = torch.tensor([[args.rows]], device='cuda')
    def clone_state():
        return [v.clone() for v in initial]
    def reset_lengths(state):
        for v in state[2 * args.layers:]:
            v.fill_(args.rows)
    def l2(actual, expected):
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError('nonfinite decode output')
        return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
    expected_state = clone_state()
    expected = reference(token, position, *expected_state).clone()
    declared_state = clone_state()
    declared_output = declared(token, position, *declared_state)
    torch.testing.assert_close(declared_output, expected)
    for a, b in zip(declared_state, expected_state):
        torch.testing.assert_close(a, b)
    print('Reference logits and full KV state match original declared-op eager', flush=True)
    del declared_state, declared_output
    backend = make_backend(mode='decode', compile_kernel=True, cache=False, scheduler='phase',
                           dump_dir=args.output.parent / 'generated')
    functions, states, errors, cache_errors, compile_seconds = {}, {}, {}, {}, {}
    builders = {'eager': lambda: reference,
                'eager_cuda_graph': lambda: reference,
                'inductor': lambda: torch.compile(reference, fullgraph=True, dynamic=False,
                                                  options={'emulate_precision_casts': True}),
                'megabake': lambda: torch.compile(declared, backend=backend, fullgraph=True, dynamic=False)}
    graph = None
    for name, build in builders.items():
        print(f'BUILD {name}, cached_tokens={args.rows}', flush=True)
        state = clone_state()
        states[name] = state
        start = time.perf_counter()
        function = build()
        actual = function(token, position, *state)
        torch.cuda.synchronize()
        compile_seconds[name] = time.perf_counter() - start
        for current, original in zip(state, initial):
            current.copy_(original)
        actual = function(token, position, *state)
        torch.cuda.synchronize()
        errors[name] = l2(actual, expected)
        if errors[name] >= .02:
            raise RuntimeError(f'{name} logits relative L2 {errors[name]} exceeds .02')
        maxima = []
        for i, (current, expected_cache) in enumerate(zip(state[:2*args.layers], expected_state[:2*args.layers])):
            if not torch.equal(current[:, :args.rows], initial[i][:, :args.rows]):
                raise RuntimeError(f'{name} modified existing KV entries')
            if not torch.equal(current[:, args.rows+1:], initial[i][:, args.rows+1:]):
                raise RuntimeError(f'{name} modified unused KV entries')
            maxima.append(l2(current[:, args.rows:args.rows+1], expected_cache[:, args.rows:args.rows+1]))
        cache_errors[name] = max(maxima)
        if cache_errors[name] >= .02:
            raise RuntimeError(f'{name} new KV entry relative L2 {cache_errors[name]} exceeds .02')
        if any(int(v.item()) != args.rows + 1 for v in state[2*args.layers:]):
            raise RuntimeError(f'{name} did not advance all KV lengths by one')
        if name == 'eager_cuda_graph':
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reset_lengths(state)
                    function(token, position, *state)
            torch.cuda.current_stream().wait_stream(stream)
            reset_lengths(state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = function(token, position, *state)
            reset_lengths(state)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, expected)
            functions[name] = graph.replay
        else:
            functions[name] = lambda function=function, state=state: function(token, position, *state)
        for _ in range(5):
            reset_lengths(state)
            functions[name]()
        torch.cuda.synchronize()
    if backend.megabake_contexts[-1].requirements.get('execution') != 'megakernel':
        raise RuntimeError('decode did not execute a megakernel')
    measurements = {name: {'samples_us': [], 'round_medians_us': []} for name in functions}
    names = list(functions)
    for trial in range(args.rounds):
        for name in names[trial % 4:] + names[:trial % 4]:
            samples = []
            for _ in range(args.samples):
                # Restore fixed logical context OUTSIDE the timed interval.
                reset_lengths(states[name])
                torch.cuda.synchronize()
                start = time.perf_counter()
                functions[name]()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1e6)
            measurements[name]['samples_us'].extend(samples)
            measurements[name]['round_medians_us'].append(statistics.median(samples))
    for record in measurements.values():
        record['median_call_us'] = statistics.median(record['samples_us'])
    reset_lengths(states['megabake'])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        functions['megabake']()
        torch.cuda.synchronize()
    kernels = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if len(kernels) != 1:
        raise RuntimeError(f'Expected one workload kernel, observed {len(kernels)}')
    result = dict(state='complete', mock_weights=True, seed=20261026, workload='smollm-decode', mode='decode',
                  rows=args.rows, context_length=args.rows, batch_size=1, output_tokens=1,
                  cache_capacity=capacity, cache_state_validation=True, cache_relative_l2=cache_errors,
                  baseline_reference='GPU-only tensor KV op; logits and full state checked against upstream eager',
                  inductor_options={'emulate_precision_casts': True},
                  kv_contents='seeded nonzero random KV; not produced by prompt prefill',
                  layers=args.layers, dtype='bfloat16', rounds=args.rounds, samples_per_round=args.samples,
                  samples=args.samples*args.rounds, measurements=measurements,
                  profile='experimental CUDA 12; not the upstream cu130 environment',
                  timing_scope='fixed-context one-token decode including KV updates; reset excluded; synchronized host+GPU calls; no SGLang engine comparison',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  cute=importlib.metadata.version('nvidia-cutlass-dsl'), relative_l2=errors['megabake'],
                  inductor_relative_l2=errors['inductor'], all_relative_l2=errors,
                  workload_kernel_count=1, compile_seconds=compile_seconds,
                  execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
