"""Time Hugging Face causal LMs: eager, Inductor, Inductor with CUDA graphs, and MegaBake.

Prefill runs a four-token prompt; decode advances a stateful KV cache one
token per call. Each variant reports the median complete call (host and GPU,
measured with CUDA events around the call), the profiled kernel time and launch
count of one call, and its first-call (compile) time. The default model is the
pinned SmolLM-135M checkpoint; pass several `--model`s to write one report per
model under `--out/<org>--<name>`. A variant that fails is recorded with its
error and the run continues.

    .venv/bin/python benchmarks/full_model.py --out artifacts/full_model
    .venv/bin/python benchmarks/full_model.py --modes decode --capacity 256
    .venv/bin/python benchmarks/full_model.py --model Qwen/Qwen3.5-0.8B google/gemma-3-270m
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import make_backend  # noqa: E402
from benchmarks.measure import kernel_trace  # noqa: E402
from benchmarks.stateful_models import StatefulCausalLM, layer_types, text_config  # noqa: E402
from megabake.fx_handler.hf_attention import (  # noqa: E402
    ATTENTION_IMPLEMENTATION,
    DecodeStep,
    register_transformers_attention,
)
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION, StatefulSmolLM  # noqa: E402

PROMPT = [1, 504, 3458, 314]
VARIANTS = ("eager", "inductor", "inductor_cudagraphs", "megabake")


def _load(name: str, revision: str | None, attention: str = "eager") -> torch.nn.Module:
    """``attention="megabake"`` declares attention to MegaBake through the
    transformers attention interface instead of leaving it to pattern matching."""
    from transformers import AutoModelForCausalLM

    if attention == ATTENTION_IMPLEMENTATION:
        register_transformers_attention()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        attn_implementation=attention,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    text_config(model).use_cache = False
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


def _pipelined_calls(calls) -> float:
    """Mean time per call when ``calls`` are issued back to back with one sync."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    for call in calls:
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6 / len(calls)


def _measure(
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None
) -> dict:
    torch._dynamo.reset()
    function = build()
    started = time.perf_counter()
    step(function)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - started
    for _ in range(args.warmup):
        step(function)
    entry = _time_calls(lambda: step(function), args.samples)
    if pipelined is not None:
        # Back-to-back calls, as a decode loop with on-device sampling issues
        # them: host work of one call overlaps the previous call's kernels.
        entry["pipelined_call_us"] = _pipelined_calls(pipelined(function))
    trace = kernel_trace(lambda: step(function), traces / f"{name}.json")
    entry.update(
        first_call_s=first_call,
        launch_count=trace["launch_count"],
        kernel_duration_us=trace["kernel_duration_us"],
    )
    summary = {key: value for key, value in entry.items() if not key.endswith("samples_us")}
    print(f"{name}: {json.dumps(summary)}", flush=True)
    return entry


def _measure_or_error(
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None
) -> dict:
    try:
        return _measure(name, build, step, args, traces, pipelined)
    except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
        traceback.print_exc()
        print(f"{name}: FAILED {type(error).__name__}: {error}", flush=True)
        return {"error": f"{type(error).__name__}: {error}"}


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
        # CUDA Graph replay through Inductor: the strongest launch-overhead baseline.
        "inductor_cudagraphs": lambda: torch.compile(
            model, mode="reduce-overhead", fullgraph=True, dynamic=False
        ),
        "megabake": lambda: torch.compile(
            model, backend=_megabake("prefill", args.out), fullgraph=True, dynamic=False
        ),
    }
    with torch.no_grad():
        return {
            "prompt_tokens": len(PROMPT),
            **{
                name: _measure_or_error(
                    f"prefill_{name}",
                    build,
                    lambda function: function(ids, attention_mask=mask),
                    args,
                    args.out / "traces",
                )
                for name, build in variants.items()
                if name in args.variants
            },
        }


def decode_module(model, capacity: int) -> tuple[torch.nn.Module, list]:
    """The one-token decode wrapper and the shapes and dtypes of its state tensors."""
    if model.config.model_type == "llama":
        # Keep the hand-written Llama wrapper the tests and published numbers use.
        config = model.config
        layers = config.num_hidden_layers
        shape = (
            1,
            capacity,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
        )
        state_shapes = [[(shape, torch.bfloat16)]] * 2 * layers + [[((1,), torch.int32)]] * layers
        return StatefulSmolLM(model).eval(), state_shapes
    if set(layer_types(model)) <= {"full_attention", "sliding_attention"}:
        # Any Hugging Face attention model: declared attention over explicit KV state.
        step = DecodeStep(model, capacity).eval()
        return step, step.state_shapes()
    # Hybrid token mixers (linear attention) are not declared ops yet.
    stateful = StatefulCausalLM(model).eval()
    return stateful, stateful.state_shapes(capacity)


def bench_decode(model, args) -> dict:
    stateful, state_shapes = decode_module(model, args.capacity)
    token = torch.tensor([[504]], device="cuda")
    variants = {
        "eager": lambda: stateful,
        "inductor": lambda: torch.compile(stateful, fullgraph=True, dynamic=False),
        "inductor_cudagraphs": lambda: torch.compile(
            stateful, mode="reduce-overhead", fullgraph=True, dynamic=False
        ),
        "megabake": lambda: torch.compile(
            stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
        ),
    }
    calls = 1 + args.warmup + 2 * args.samples + 1
    if calls > args.capacity:
        raise SystemExit(f"--capacity must be at least {calls} for this many calls")
    results: dict[str, object] = {"capacity": args.capacity}
    with torch.no_grad():
        for name, build in variants.items():
            if name not in args.variants:
                continue
            state = [
                torch.zeros(shape, device="cuda", dtype=dtype)
                for layer in state_shapes
                for shape, dtype in layer
            ]
            position = [0]

            def step(function, state=state, position=position):
                position_ids = torch.tensor([[position[0]]], device="cuda")
                position[0] += 1
                return function(token, position_ids, *state)

            def pipelined_steps(function, state=state, position=position):
                # Positions made on the device ahead of time: a pageable host
                # copy would synchronize the stream on every call.
                # Separate tensors: views at different storage offsets can
                # trip alignment guards and recompile.
                ahead = [
                    row.clone()
                    for row in torch.arange(
                        position[0], position[0] + args.samples, device="cuda"
                    ).view(-1, 1, 1)
                ]
                position[0] += args.samples
                return [lambda row=row: function(token, row, *state) for row in ahead]

            results[name] = _measure_or_error(
                f"decode_{name}", build, step, args, args.out / "traces", pipelined_steps
            )
            results[name]["final_position"] = position[0]
            torch.cuda.empty_cache()
    return results


def _bench_model(name: str, revision: str | None, args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"== {name}@{revision or 'main'} -> {args.out}", flush=True)
    model = _load(name, revision, args.attention)
    report: dict[str, object] = {
        "model": f"{name}@{revision or 'main'}",
        "attention": args.attention,
        "layer_types": sorted(set(layer_types(model))),
        "gpu": torch.cuda.get_device_name(),
        "samples": args.samples,
    }
    for mode in args.modes:
        report[mode] = (bench_prefill if mode == "prefill" else bench_decode)(model, args)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'report.json'}")
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", nargs="+", default=[SMOLLM_MODEL_NAME], help="Hugging Face causal LM id(s)"
    )
    parser.add_argument(
        "--revision", help=f"checkpoint revision (default: pinned for {SMOLLM_MODEL_NAME}, else main)"
    )
    parser.add_argument(
        "--attention",
        choices=("eager", ATTENTION_IMPLEMENTATION),
        default="eager",
        help="prefill attention: eager (pattern-matched by MegaBake) or the declared megabake op",
    )
    parser.add_argument("--modes", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="variants to time (default: all)",
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=128, help="decode KV cache capacity")
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_model"))
    args = parser.parse_args()
    root = args.out
    for name in args.model:
        revision = args.revision or (SMOLLM_REVISION if name == SMOLLM_MODEL_NAME else None)
        args.out = root / name.replace("/", "--") if len(args.model) > 1 else root
        _bench_model(name, revision, args)


if __name__ == "__main__":
    main()
