"""Time the final kernels in isolation: launch plus device execution only.

Every mode is reduced to a prebuilt launch with fixed tensors, so no Python
argument binding, dispatcher work, or output allocation is inside a sample.

    eager_graph      CUDA Graph replay of the eager kernels
    inductor_graph   CUDA Graph replay of the torch.compile (Inductor) kernels
    megabake_fast    prebuilt CuTe execution arguments, launch entry only
    megabake_call    prebuilt CuTe arguments through the normal CuTe call
    megabake_bound   the public compiled callable (reference, not isolated)
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
from cuda.bindings import driver as cuda

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import make_backend
from megabake.compiler import _input_tensors, _RuntimeState
from benchmarks.measure import kernel_trace
from benchmarks.workloads import mlp


def _single(launch, prepare, count):
    """One launch per sample, then wait for the stream."""
    stream = torch.cuda.current_stream()
    host, device = [], []
    for _ in range(count):
        token = prepare()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        launch(token)
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - started) * 1_000_000)
        device.append(start.elapsed_time(end) * 1_000)
    return host, device


def _burst(launch, prepare, count, burst):
    """Back-to-back launches with one wait; removes the per-call wake-up cost."""
    stream = torch.cuda.current_stream()
    host, device = [], []
    for _ in range(count):
        tokens = [prepare() for _ in range(burst)]
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        for token in tokens:
            launch(token)
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - started) * 1_000_000 / burst)
        device.append(start.elapsed_time(end) * 1_000 / burst)
    return host, device


def _summary(samples):
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[(len(ordered) * 9) // 10],
        "min_us": ordered[0],
        "samples_us": samples,
    }


def _measure(name, launch, prepare, arguments, traces, batch_size):
    for _ in range(arguments.warmup):
        launch(prepare())
    torch.cuda.synchronize()
    single_host, single_device = _single(launch, prepare, arguments.samples)
    burst_host, burst_device = _burst(
        launch, prepare, arguments.samples, arguments.burst
    )
    trace = kernel_trace(
        lambda: launch(prepare()), traces / f"m{batch_size}-{name}-trace.json"
    )
    return {
        "single_launch_and_wait": _summary(single_host),
        "single_device_span": _summary(single_device),
        "burst_per_launch": _summary(burst_host),
        "burst_device_span_per_launch": _summary(burst_device),
        "kernel_count": trace["launch_count"],
        "kernel_duration_us": trace["kernel_duration_us"],
        "kernel_names": trace["kernel_names"],
    }


def _capture(function, inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    return graph, result


def _relative_l2(actual, expected):
    return (
        (actual.float() - expected.float()).norm() / expected.float().norm()
    ).item()


class _MegaBakeKernel:
    """The compiled megakernel with every binding fixed ahead of the launch."""

    def __init__(self, kernel, inputs):
        self.kernel = kernel
        self.executor = kernel.compiled
        plan = kernel.plan
        self.state = _RuntimeState.create(
            plan,
            kernel.generated,
            _input_tensors(plan.graph, inputs),
            ready=kernel.ready,
            work_iterators=kernel.work_iterators,
        )
        self.output = self.state.tensor_for_value(plan.graph.outputs[0])
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if kernel.event_schedule is not None:
            kernel.ready.zero_()
            self.stride = 1
        else:
            kernel.ready.fill_(-1)
            self.stride = max(
                max(
                    (phase.index for phase in plan.phases if phase.barrier is not None),
                    default=-1,
                )
                + 1,
                1,
            )
        torch.cuda.synchronize()
        self.epoch = 0
        self.fast_path = hasattr(self.executor, "generate_execution_args") and hasattr(
            self.executor, "run_compiled_program"
        )

    def arguments(self):
        # Each invocation owns a fresh epoch, exactly as the bound callable does.
        epoch = self.epoch
        self.epoch += self.stride
        return self.state.arguments(epoch, self.stream)

    def packed(self):
        arguments = self.arguments()
        packed = self.executor.generate_execution_args(*arguments)
        execution = (
            packed[0] if isinstance(packed, tuple) and len(packed) == 2 else packed
        )
        # Keep the adapted arguments alive until the launch has consumed them.
        return execution, packed, arguments

    def launch_fast(self, token):
        self.executor.run_compiled_program(token[0])

    def launch_call(self, arguments):
        self.executor(*arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/isolated/mlp.json"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 256])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--burst", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument(
        "--scheduler", choices=("auto", "phase", "region-events"), default="auto"
    )
    arguments = parser.parse_args()
    root = arguments.output.parent
    traces = root / "traces"

    torch.manual_seed(16)
    torch.set_grad_enabled(False)
    report = {
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(0)),
        "samples": arguments.samples,
        "burst": arguments.burst,
        "records": [],
    }
    for batch_size in arguments.batches:
        source = torch.randn(
            batch_size, 576, dtype=torch.bfloat16, device="cuda"
        ) / (576**0.5)
        weights = tuple(
            torch.randn(rows, columns, dtype=torch.bfloat16, device="cuda")
            / (columns**0.5)
            for rows, columns in ((1536, 576), (1536, 576), (576, 1536))
        )
        inputs = (source, *weights)
        reference = mlp(*inputs)

        modes = {}
        eager_graph, eager_result = _capture(mlp, inputs)
        modes["eager_graph"] = _measure(
            "eager_graph",
            lambda _: eager_graph.replay(),
            lambda: None,
            arguments,
            traces,
            batch_size,
        )
        modes["eager_graph"]["relative_l2_error"] = _relative_l2(
            eager_result, reference
        )

        inductor = torch.compile(
            mlp, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
        inductor(*inputs)
        inductor_graph, inductor_result = _capture(inductor, inputs)
        modes["inductor_graph"] = _measure(
            "inductor_graph",
            lambda _: inductor_graph.replay(),
            lambda: None,
            arguments,
            traces,
            batch_size,
        )
        modes["inductor_graph"]["relative_l2_error"] = _relative_l2(
            inductor_result, reference
        )

        backend = make_backend(scheduler=arguments.scheduler)
        bound = torch.compile(mlp, backend=backend, fullgraph=True, dynamic=False)
        bound(*inputs)
        compiled = backend.megabake_last_compiled
        # The bound callable is measured first; the isolated launches below
        # take over the kernel's epoch state and it must not be called again.
        modes["megabake_bound"] = _measure(
            "megabake_bound",
            lambda _: bound(*inputs),
            lambda: None,
            arguments,
            traces,
            batch_size,
        )

        megabake = _MegaBakeKernel(compiled, inputs)
        if megabake.fast_path:
            modes["megabake_fast"] = _measure(
                "megabake_fast",
                megabake.launch_fast,
                megabake.packed,
                arguments,
                traces,
                batch_size,
            )
            modes["megabake_fast"]["relative_l2_error"] = _relative_l2(
                megabake.output, reference
            )
        modes["megabake_call"] = _measure(
            "megabake_call",
            megabake.launch_call,
            megabake.arguments,
            arguments,
            traces,
            batch_size,
        )
        modes["megabake_call"]["relative_l2_error"] = _relative_l2(
            megabake.output, reference
        )

        artifact = root / f"m{batch_size}-megabake-kernel"
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy(compiled.generated.source_path, artifact / "module.py")
        shutil.copy(compiled.generated.source_map_path, artifact / "source_map.json")
        resources = compiled.resources
        (artifact / "kernel.json").write_text(
            json.dumps(
                {
                    "kernel_name": resources.kernel_name,
                    "selected_plan": getattr(
                        compiled.search, "selected_description", None
                    ),
                    "workers": compiled.plan.launch.grid_size,
                    "threads_per_block": compiled.plan.launch.threads_per_block,
                    "registers_per_thread": resources.registers_per_thread,
                    "dynamic_shared_memory_bytes": resources.dynamic_shared_memory_bytes,
                    "active_blocks_per_sm": resources.active_blocks_per_sm,
                    "phases": [
                        f"{phase.region_id}:{phase.body.provider}/{phase.body.name}"
                        for phase in compiled.plan.phases
                    ],
                    "launch_arguments": list(compiled.generated.launch.arguments),
                    "cute_fast_path": megabake.fast_path,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        report["records"].append(
            {
                "batch_size": batch_size,
                "megabake_artifact": str(artifact),
                "modes": modes,
            }
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

        print(f"\nM={batch_size}  (microseconds, medians)", flush=True)
        print(
            f"  {'mode':16} {'kernels':>7} {'kernel':>9} {'burst/launch':>13} "
            f"{'single+wait':>12} {'rel_l2':>9}"
        )
        for name, mode in modes.items():
            error = mode.get("relative_l2_error")
            print(
                f"  {name:16} {mode['kernel_count']:7d} "
                f"{mode['kernel_duration_us']:9.2f} "
                f"{mode['burst_per_launch']['median_us']:13.2f} "
                f"{mode['single_launch_and_wait']['median_us']:12.2f} "
                f"{'' if error is None else format(error, '9.2e')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
