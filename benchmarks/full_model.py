"""Time the pinned SmolLM-135M checkpoint: eager vs Inductor vs MegaBake.

Prefill runs a four-token prompt; decode advances a stateful KV cache one
token per call. Each variant reports the median complete call (host and GPU,
measured with CUDA events around the call), the profiled kernel time and launch
count of one call, and its first-call (compile) time.

    .venv/bin/python benchmarks/full_model.py --out artifacts/full_model
    .venv/bin/python benchmarks/full_model.py --modes decode --capacity 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import make_backend  # noqa: E402
from benchmarks.measure import kernel_trace  # noqa: E402
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION, StatefulSmolLM  # noqa: E402

PROMPT = [1, 504, 3458, 314]


def _load() -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM_MODEL_NAME,
        revision=SMOLLM_REVISION,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    return model.to("cuda").eval()


def _time_calls(function, count: int) -> dict[str, object]:
    host, device = [], []
    for _ in range(count):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - started) * 1e6)
        device.append(start.elapsed_time(end) * 1e3)
    return {
        "complete_call_us": statistics.median(host),
        "device_span_us": statistics.median(device),
        "complete_call_samples_us": host,
        "device_span_samples_us": device,
    }


def _measure(name: str, build, step, args: argparse.Namespace, traces: Path) -> dict:
    torch._dynamo.reset()
    function = build()
    started = time.perf_counter()
    step(function)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - started
    for _ in range(args.warmup):
        step(function)
    entry = _time_calls(lambda: step(function), args.samples)
    trace = kernel_trace(lambda: step(function), traces / f"{name}.json")
    entry.update(
        first_call_s=first_call,
        launch_count=trace["launch_count"],
        kernel_duration_us=trace["kernel_duration_us"],
    )
    summary = {key: value for key, value in entry.items() if not key.endswith("samples_us")}
    print(f"{name}: {json.dumps(summary)}", flush=True)
    return entry


def _megabake(mode: str, out: Path):
    return make_backend(
        mode=mode, compile_kernel=True, cache=False, dump_dir=out / "dumps" / mode, scheduler="phase"
    )


def bench_prefill(model, args) -> dict:
    ids = torch.tensor([PROMPT], device="cuda")
    mask = torch.ones_like(ids)
    variants = {
        "eager": lambda: model,
        "inductor": lambda: torch.compile(model, fullgraph=True, dynamic=False),
        "megabake": lambda: torch.compile(
            model, backend=_megabake("prefill", args.out), fullgraph=True, dynamic=False
        ),
    }
    with torch.no_grad():
        return {
            "prompt_tokens": len(PROMPT),
            **{
                name: _measure(
                    f"prefill_{name}",
                    build,
                    lambda function: function(ids, attention_mask=mask),
                    args,
                    args.out / "traces",
                )
                for name, build in variants.items()
            },
        }


def bench_decode(model, args) -> dict:
    stateful = StatefulSmolLM(model).eval()
    config = model.config
    layers = config.num_hidden_layers
    shape = (
        1,
        args.capacity,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    token = torch.tensor([[504]], device="cuda")
    variants = {
        "eager": lambda: stateful,
        "inductor": lambda: torch.compile(stateful, fullgraph=True, dynamic=False),
        "megabake": lambda: torch.compile(
            stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
        ),
    }
    calls = 1 + args.warmup + args.samples + 1
    if calls > args.capacity:
        raise SystemExit(f"--capacity must be at least {calls} for this many calls")
    results: dict[str, object] = {"capacity": args.capacity}
    with torch.no_grad():
        for name, build in variants.items():
            state = [
                torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
                for _ in range(2 * layers)
            ] + [torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(layers)]
            position = [0]

            def step(function, state=state, position=position):
                position_ids = torch.tensor([[position[0]]], device="cuda")
                position[0] += 1
                return function(token, position_ids, *state)

            results[name] = _measure(f"decode_{name}", build, step, args, args.out / "traces")
            results[name]["final_position"] = position[0]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=128, help="decode KV cache capacity")
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_model"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = _load()
    report: dict[str, object] = {
        "model": f"{SMOLLM_MODEL_NAME}@{SMOLLM_REVISION}",
        "gpu": torch.cuda.get_device_name(),
        "samples": args.samples,
    }
    for mode in args.modes:
        report[mode] = (bench_prefill if mode == "prefill" else bench_decode)(model, args)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
