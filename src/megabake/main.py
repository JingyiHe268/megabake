from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import torch

from .frontend import make_backend
from .fx_handler.export import capture_exported_program
from .fx_handler.transformers import capture_transformer
from .graph import dump_compute_graph, import_fx
from .lowering import canonicalize_semantics, dump_lowering_graph
from .regions import dump_region_graph, form_regions


def _backend(args: argparse.Namespace) -> Any:
    return make_backend(
        mode=getattr(args, "mode", "prefill"),
        fallback=getattr(args, "fallback", "error"),
        cache=not getattr(args, "no_cache", False),
        dump_dir=getattr(args, "dump_dir", "fx_traces"),
        model_name=getattr(args, "model_name", "model"),
    )


def _summary(result: Any) -> str:
    if isinstance(result, torch.Tensor):
        return f"tensor(shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device})"
    if isinstance(result, (tuple, list)):
        return f"{type(result).__name__}({', '.join(_summary(value) for value in result)})"
    return type(result).__name__


def _run_demo(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, device=device),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3, device=device),
    ).eval()
    inputs = (torch.randn(5, 4, device=device),)
    backend = _backend(args)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    with torch.no_grad():
        expected = model(*inputs)
        actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected)
    print(f"demo: PASS {_summary(actual)}")
    print(
        f"capture: phase={backend.megabake_contexts[-1].phase} "
        f"cache_state={backend.megabake_contexts[-1].cache_state}"
    )
    return 0


def _run_transformers(args: argparse.Namespace) -> int:
    device = args.device
    input_ids = torch.tensor([args.input_ids], device=device)
    backend = _backend(args)
    capture = capture_transformer(
        args.model_name,
        input_ids,
        revision=args.revision,
        device=device,
        backend=backend,
    )
    output = capture.run(input_ids)
    compute_graph_path = None
    lowering_graph_path = None
    region_graph_path = None
    if getattr(args, "dump_compute_graph", False) or getattr(
        args, "dump_lowering_graph", False
    ) or getattr(args, "dump_region_graph", False):
        compute_graph = import_fx(capture.graph, capture.context)
        model_name = args.model_name
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("._")
        output_directory = Path(args.dump_dir or "fx_traces") / (
            safe_model_name or "model"
        )
        if getattr(args, "dump_compute_graph", False):
            compute_graph_path = dump_compute_graph(
                compute_graph, output_directory / "compute_graph.txt"
        )
        if getattr(args, "dump_lowering_graph", False):
            canonical_graph = canonicalize_semantics(compute_graph)
            lowering_graph_path = dump_lowering_graph(
                canonical_graph, output_directory / "lowering_graph.txt"
            )
        if getattr(args, "dump_region_graph", False):
            region_graph = form_regions(canonicalize_semantics(compute_graph))
            region_graph_path = dump_region_graph(
                region_graph, output_directory / "region_graph.txt"
            )
    print(f"transformers: PASS {_summary(output)}")
    print(
        f"capture: phase={capture.context.phase} "
        f"cache_state={capture.context.cache_state} "
        f"inputs={len(capture.context.input_order)}"
    )
    if compute_graph_path is not None:
        print(f"compute graph: {compute_graph_path}")
    if lowering_graph_path is not None:
        print(f"lowering graph: {lowering_graph_path}")
    if region_graph_path is not None:
        print(f"region graph: {region_graph_path}")
    return 0


def _json_value(value: str) -> Any:
    return json.loads(value)


def _run_export(args: argparse.Namespace) -> int:
    module_name, _, attribute = args.callable.partition(":")
    if not module_name or not attribute:
        raise ValueError("--callable must use module:attribute syntax")
    callable_obj = getattr(importlib.import_module(module_name), attribute)
    args_values = [_json_value(value) for value in args.arg]
    kwargs = dict(item.split("=", 1) for item in args.kwarg)
    kwargs = {name: _json_value(value) for name, value in kwargs.items()}
    if args.model_name is None:
        args.model_name = args.callable
    backend = _backend(args)
    with torch.no_grad():
        exported = torch.export.export(callable_obj, tuple(args_values), kwargs)
        capture = capture_exported_program(
            exported,
            args=args_values,
            kwargs=kwargs,
            backend=backend,
        )
        output = capture.run(*args_values, **kwargs)
    print(f"export: PASS {_summary(output)}")
    print(
        f"capture: phase={capture.context.phase} "
        f"cache_state={capture.context.cache_state} "
        f"inputs={len(capture.context.input_order)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--mode", choices=("prefill", "decode"), default=argparse.SUPPRESS
    )
    common.add_argument("--fallback", choices=("error",), default=argparse.SUPPRESS)
    common.add_argument("--no-cache", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--dump-dir", default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(prog="megabake", parents=[common])
    parser.add_argument("--version", action="version", version="megabake 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", parents=[common])
    demo.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    demo.add_argument("--model-name", default="demo")
    demo.set_defaults(handler=_run_demo)

    transformers_parser = subparsers.add_parser("transformers", parents=[common])
    transformers_parser.add_argument(
        "--model-name", default="HuggingFaceTB/SmolLM-135M"
    )
    transformers_parser.add_argument("--revision")
    transformers_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    transformers_parser.add_argument(
        "--input-ids", type=lambda value: [int(item) for item in value.split(",")],
        default="1,2,3,4",
    )
    transformers_parser.add_argument(
        "--dump-compute-graph", action="store_true", default=False
    )
    transformers_parser.add_argument(
        "--dump-lowering-graph", action="store_true", default=False
    )
    transformers_parser.add_argument(
        "--dump-region-graph", action="store_true", default=False
    )
    transformers_parser.set_defaults(handler=_run_transformers)

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--callable", required=True)
    export_parser.add_argument("--model-name")
    export_parser.add_argument("--arg", action="append", default=[], metavar="JSON")
    export_parser.add_argument(
        "--kwarg", action="append", default=[], metavar="NAME=JSON"
    )
    export_parser.set_defaults(handler=_run_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
