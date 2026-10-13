from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import make_backend
from benchmarks.measure import event_samples, kernel_trace
from benchmarks.workloads import mlp


def _first_call(function, *arguments):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_host = time.perf_counter()
    start_event.record()
    result = function(*arguments)
    end_event.record()
    end_event.synchronize()
    return result, {
        "event_span_us": start_event.elapsed_time(end_event) * 1_000,
        "host_us": (time.perf_counter() - start_host) * 1_000_000,
    }


def _capture_cuda_graph(function, *arguments):
    started = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(*arguments)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function(*arguments)
    return graph, result, {"capture_us": (time.perf_counter() - started) * 1_000_000}


def _relative_l2(actual, expected):
    actual_values = tuple(actual)
    expected_values = tuple(expected)
    if len(actual_values) != len(expected_values):
        return float("inf")
    numerator = 0.0
    denominator = 0.0
    for actual_value, expected_value in zip(actual_values, expected_values):
        difference = actual_value.float() - expected_value.float()
        numerator += difference.pow(2).sum().item()
        denominator += expected_value.float().pow(2).sum().item()
    return (numerator / denominator) ** 0.5 if denominator else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/task16/mlp.json"))
    parser.add_argument("--traces", type=Path, default=Path("artifacts/task16/traces"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256])
    parser.add_argument(
        "--scheduler", choices=("auto", "phase", "region-events"), default="auto"
    )
    arguments = parser.parse_args()

    torch.manual_seed(16)
    torch.set_grad_enabled(False)
    report = {
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(0)),
        "seed": 16,
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
        reference, eager_first = _first_call(mlp, *inputs)
        graph, graph_result, graph_setup = _capture_cuda_graph(mlp, *inputs)
        graph.replay()
        graph_error = _relative_l2(graph_result, reference)
        _, graph_first_replay = _first_call(lambda: graph.replay())
        inductor = torch.compile(
            mlp,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        inductor_result, inductor_first = _first_call(inductor, *inputs)
        backend_setup_started = time.perf_counter()
        backend = make_backend(scheduler=arguments.scheduler)
        backend_setup_us = (time.perf_counter() - backend_setup_started) * 1_000_000
        wrapper_setup_started = time.perf_counter()
        compiled = torch.compile(
            mlp, backend=backend, fullgraph=True, dynamic=False
        )
        wrapper_setup_us = (time.perf_counter() - wrapper_setup_started) * 1_000_000
        result, megabake_first = _first_call(compiled, *inputs)
        error = (
            (result.float() - reference.float()).norm() / reference.float().norm()
        ).item()
        inductor_error = (
            (inductor_result.float() - reference.float()).norm()
            / reference.float().norm()
        ).item()
        if error >= 0.01:
            raise RuntimeError(f"MLP relative error {error}")
        if inductor_error >= 0.01:
            raise RuntimeError(f"Inductor MLP relative error {inductor_error}")
        if graph_error >= 0.01:
            raise RuntimeError(f"CUDA Graph MLP relative error {graph_error}")

        record = {
            "batch_size": batch_size,
            "relative_l2_error": error,
            "inductor_relative_l2_error": inductor_error,
            "cuda_graph_relative_l2_error": graph_error,
            "cuda_graph_setup": graph_setup,
            "cuda_graph_first_replay": graph_first_replay,
            "eager_first_call": eager_first,
            "inductor_first_call": inductor_first,
            "megabake_first_call": megabake_first,
            "megabake_cost_breakdown": {
                "backend_setup_us": backend_setup_us,
                "dynamo_wrapper_setup_us": wrapper_setup_us,
                "megabake_compilation_us": getattr(
                    backend, "megabake_last_compile_us", None
                ),
                "first_call_host_us": megabake_first["host_us"],
            },
            "eager": event_samples(lambda: mlp(*inputs)),
            "cuda_graph": event_samples(lambda: graph.replay()),
            "inductor": event_samples(lambda: inductor(*inputs)),
            "megabake": event_samples(lambda: compiled(*inputs)),
        }
        search = getattr(backend.megabake_last_compiled, "search", None)
        if search is not None:
            record["plan_search"] = search.report()
        functions = {
            "eager": lambda: mlp(*inputs),
            "cuda_graph": lambda: graph.replay(),
            "inductor": lambda: inductor(*inputs),
            "megabake": lambda: compiled(*inputs),
        }
        for name, function in functions.items():
            record[name].update(
                kernel_trace(
                    function,
                    arguments.traces / f"m{batch_size}-{name}-trace.json",
                )
            )
        report["records"].append(record)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        baseline_output = arguments.output.with_name("mlp_baseline.json")
        baseline_output.write_text(
            json.dumps(
                {
                    "torch": report["torch"],
                    "gpu": report["gpu"],
                    "seed": report["seed"],
                    "records": [
                        {
                            "batch_size": record["batch_size"],
                            "eager": record["eager"],
                            "cuda_graph": record["cuda_graph"],
                            "inductor": record["inductor"],
                            "eager_first_call": record["eager_first_call"],
                            "inductor_first_call": record["inductor_first_call"],
                        }
                        for record in report["records"]
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "batch_size": batch_size,
                    "eager_complete_us": record["eager"]["complete_call_us"],
                    "cuda_graph_complete_us": record["cuda_graph"][
                        "complete_call_us"
                    ],
                    "inductor_complete_us": record["inductor"]["complete_call_us"],
                    "megabake_complete_us": record["megabake"]["complete_call_us"],
                    "megabake_kernel_duration_us": record["megabake"][
                        "kernel_duration_us"
                    ],
                    "megabake_first_launch_to_last_kernel_us": record["megabake"][
                        "first_launch_to_last_kernel_us"
                    ],
                    "megabake_kernel_count": record["megabake"]["launch_count"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
