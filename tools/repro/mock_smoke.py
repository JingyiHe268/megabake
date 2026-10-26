#!/usr/bin/env python3
"""Synthetic-weight GPU compatibility and forward benchmark; no checkpoint downloads.

Run under srun with the isolated CUDA 12 experimental environment. Successful
measurements are comparable within this harness, not to SGLang engine timings.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("mlp", "smollm"), default="mlp")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.layers, args.rows, args.samples, args.rounds) < 1:
        parser.error("layers, rows and samples must be positive")

    import torch
    from megabake import make_backend
    from benchmarks.workloads import mlp

    torch.set_grad_enabled(False)
    torch.manual_seed(20261026)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tensor initialization must pass before benchmarking")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    if args.workload == "mlp":
        model = mlp
        inputs = (torch.randn(args.rows, 576, device="cuda", dtype=torch.bfloat16) / 24,
                  *(torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
                    for m, k in ((1536, 576), (1536, 576), (576, 1536))))
    else:
        from tests.fixtures import SmolLMArchitecture
        module = SmolLMArchitecture(layers=args.layers)
        module.config.max_position_embeddings = max(4, args.rows)
        def model(ids, mask):
            return module(ids, attention_mask=mask).logits
        ids = torch.arange(1, args.rows + 1, device="cuda").view(1, -1)
        padding_mask = torch.ones((1, args.rows), device="cuda", dtype=torch.int64)
        # HF's eager mask builder creates a pageable CPU scalar, which torch
        # 2.14 cannot copy during graph capture. All variants receive the same
        # prebuilt causal mask; verify it preserves the original eager output.
        causal_mask = torch.full((1, 1, args.rows, args.rows), torch.finfo(torch.bfloat16).min,
                                 device="cuda", dtype=torch.bfloat16).triu(1)
        torch.testing.assert_close(model(ids, causal_mask), model(ids, padding_mask))
        inputs = (ids, causal_mask)

    expected = model(*inputs).clone()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           scheduler="phase", dump_dir=args.output.parent / "generated")
    started = time.perf_counter()
    megabake = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    actual = megabake(*inputs)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    relative_l2 = float((actual.float() - expected.float()).norm() / expected.float().norm())
    limit = .01 if args.workload == "mlp" else .02
    if not torch.isfinite(actual).all() or relative_l2 >= limit:
        raise RuntimeError(f"Mock output correctness failed: relative L2 {relative_l2}")
    if backend.megabake_contexts[-1].requirements.get("execution") != "megakernel":
        raise RuntimeError("The mock run did not execute a megakernel")

    inductor = torch.compile(model, fullgraph=True, dynamic=False)
    candidate = inductor(*inputs)
    torch.cuda.synchronize()
    inductor_l2 = float((candidate.float() - expected.float()).norm() / expected.float().norm())
    if not torch.isfinite(candidate).all() or inductor_l2 >= limit:
        raise RuntimeError(f"Inductor correctness failed: relative L2 {inductor_l2}")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = model(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected)

    functions = {"eager": lambda: model(*inputs), "eager_cuda_graph": graph.replay,
                 "inductor": lambda: inductor(*inputs), "megabake": lambda: megabake(*inputs)}
    measurements = {name: {"samples_us": [], "round_medians_us": []} for name in functions}
    for name, function in functions.items():
        for _ in range(5):
            function()
        torch.cuda.synchronize()
    # Rotate the variant order across rounds to reduce fixed order bias.
    names = list(functions)
    for trial in range(args.rounds):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            samples = []
            for _ in range(args.samples):
                torch.cuda.synchronize()
                started = time.perf_counter()
                functions[name]()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1e6)
            measurements[name]["samples_us"].extend(samples)
            measurements[name]["round_medians_us"].append(statistics.median(samples))
    for record in measurements.values():
        record["median_call_us"] = statistics.median(record["samples_us"])

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        megabake(*inputs)
        torch.cuda.synchronize()
    kernels = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    if len(kernels) != 1:
        raise RuntimeError(f"Expected one workload kernel, observed {len(kernels)} GPU events")
    result = {"state": "complete", "mock_weights": True, "seed": 20261026,
              "execution_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
              "workload": args.workload, "layers": args.layers if args.workload == "smollm" else None,
              "rows": args.rows, "samples": args.samples * args.rounds,
              "rounds": args.rounds, "samples_per_round": args.samples, "dtype": "bfloat16",
              "profile": "experimental CUDA 12; not the upstream cu130 environment",
              "timing_scope": "synchronized model-forward calls; SmolLM uses a prebuilt causal mask; no SGLang engine comparison",
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda,
              "cute": importlib.metadata.version("nvidia-cutlass-dsl"),
              "relative_l2": relative_l2, "inductor_relative_l2": inductor_l2,
              "workload_kernel_count": len(kernels), "compile_seconds": compile_seconds,
              "measurements": measurements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
