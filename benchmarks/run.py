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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/task16/mlp.json"))
    parser.add_argument("--traces", type=Path, default=Path("artifacts/task16/traces"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256])
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
        inductor = torch.compile(
            mlp,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        inductor_result, inductor_first = _first_call(inductor, *inputs)
        backend = make_backend()
        compiled = torch.compile(
            mlp, backend=backend, fullgraph=True, dynamic=False
        )
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

        record = {
            "batch_size": batch_size,
            "relative_l2_error": error,
            "inductor_relative_l2_error": inductor_error,
            "eager_first_call": eager_first,
            "inductor_first_call": inductor_first,
            "megabake_first_call": megabake_first,
            "eager": event_samples(lambda: mlp(*inputs)),
            "inductor": event_samples(lambda: inductor(*inputs)),
            "megabake": event_samples(lambda: compiled(*inputs)),
        }
        search = getattr(backend.megabake_last_compiled, "search", None)
        if search is not None:
            record["plan_search"] = search.report()
        functions = {
            "eager": lambda: mlp(*inputs),
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
