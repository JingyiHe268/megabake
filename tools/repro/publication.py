"""Export successful measurements as tables and data-generated comparison charts only."""
import csv
import html
import io
import json
import math


def measurements(directory):
    config = json.loads((directory / "config.json").read_text())
    rows = []
    for path in sorted(directory.glob("sglang-cudagraph-*.jsonl")):
        stage = path.with_suffix(".status.json")
        if not stage.exists() or json.loads(stage.read_text()).get("state") != "complete":
            continue
        for line in path.read_text().splitlines():
            item = json.loads(line)
            values = [item.get("prefill_latency"), item.get("median_decode_latency")]
            if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                       and math.isfinite(v) and v > 0 for v in values):
                continue
            rows.append({
                "framework": "SGLang", "variant": path.stem, "scope": "engine; advancing decode",
                "model": config["model_id"], "revision": config["model_revision"],
                "dtype": "bfloat16", "batch_size": item["batch_size"],
                "input_tokens": item["input_len"], "output_tokens": item["output_len"],
                "prefill_ms": values[0] * 1000, "decode_median_ms_per_token": values[1] * 1000,
            })
    upstream = directory / "megabake-upstream/report.json"
    stage = directory / "megabake-upstream.status.json"
    if upstream.exists() and stage.exists() and json.loads(stage.read_text()).get("state") == "complete":
        data = json.loads(upstream.read_text())
        for variant in ("eager", "inductor", "inductor_cudagraphs", "megabake"):
            prefill = data.get("prefill", {}).get(variant, {})
            decode = data.get("decode", {}).get(variant, {})
            values = [prefill.get("complete_call_us"), decode.get("complete_call_us")]
            if "error" in prefill or "error" in decode or not all(
                isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in values
            ):
                continue
            rows.append({
                "framework": "PyTorch/MegaBake harness", "variant": variant,
                "scope": "model forward; one token per decode call",
                "model": config["model_id"], "revision": config["model_revision"],
                "dtype": "bfloat16", "batch_size": config["batch_size"],
                "input_tokens": config["prompt_tokens"], "output_tokens": 1,
                "prefill_ms": values[0] / 1000, "decode_median_ms_per_token": values[1] / 1000,
            })
    return rows


def chart(rows, metric, title, unit, description=None, footnote=None):
    """A linear, zero-based bar chart; only rows with identical workload are compared."""
    maximum = max(row[metric] for row in rows)
    bars = []
    for index, row in enumerate(rows):
        y = 110 + index * 72
        value = row[metric]
        width = value / maximum * 500
        label = html.escape(row["variant"])
        bars.append(f'<text x="24" y="{y+22}" font-size="14">{label}</text>'
                    f'<rect x="270" y="{y}" width="{width:.2f}" height="32" fill="#4278b7"/>'
                    f'<text x="{280+width:.2f}" y="{y+22}" font-size="14">{value:.4f}</text>')
    height = 190 + 72 * len(rows)
    description = description or (f'{rows[0]["model"]} · BF16 · batch={rows[0]["batch_size"]} · '
                                  f'input={rows[0]["input_tokens"]} · output={rows[0]["output_tokens"]}')
    footnote = footnote or "Each bar is one recorded trial; decode median covers advancing tokens. No MegaBake comparison."
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">'
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#152030">'
            f'<text x="24" y="36" font-size="22">{html.escape(title)}</text>'
            f'<text x="24" y="64" font-size="14">{html.escape(description)}</text>'
            + ''.join(bars) + f'<path d="M270 94V{100+72*len(rows)}" stroke="#555"/>'
            f'<text x="270" y="{height-55}" font-size="13">0 — {maximum:.4f} {unit}; lower is better</text>'
            f'<text x="24" y="{height-24}" font-size="12">{html.escape(footnote)}</text>'
            '</g></svg>')


def export_mock(directory):
    data = json.loads((directory / "mock-forward.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    decode = data.get("mode") == "decode"
    limit = .01 if data.get("workload") == "mlp" else .02
    errors = [data.get("relative_l2"), data.get("inductor_relative_l2")]
    if decode:
        cache_errors = data.get("cache_relative_l2", {})
        if (data.get("cache_state_validation") is not True or data.get("batch_size") != 1
                or data.get("output_tokens") != 1 or data.get("context_length") != data.get("rows")
                or data.get("cache_capacity") != data.get("rows", 0) + 128
                or set(cache_errors) != {"eager", "eager_cuda_graph", "inductor", "megabake"}
                or not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v < limit
                           for v in cache_errors.values())):
            raise ValueError("Decode context or KV-state validation has not passed")
    if (status.get("state") != "complete" or data.get("state") != "complete"
            or data.get("mock_weights") is not True or data.get("workload_kernel_count") != 1
            or not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v < limit for v in errors)):
        raise ValueError("Mock correctness or single-kernel validation has not passed")
    rows = []
    for name in ("eager", "eager_cuda_graph", "inductor", "megabake"):
        record = data.get("measurements", {}).get(name, {})
        samples = record.get("samples_us", [])
        value = record.get("median_call_us")
        if (not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
                or len(samples) != data.get("samples") or not samples
                or not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in samples)):
            raise ValueError(f"Incomplete mock measurements for {name}")
        rows.append({"variant": name, "workload": data["workload"], "mock_weights": True,
                     "rows": data["rows"], "layers": data.get("layers"), "dtype": data["dtype"],
                     "seed": data["seed"], "samples": len(samples), "median_call_ms": value / 1000,
                     "torch": data["torch"], "cute": data["cute"], "profile": data["profile"],
                     "gpu": data["gpu"], "execution_commit": data["execution_commit"]})
    destination = directory / "published"
    destination.mkdir(exist_ok=True)
    csv_path = destination / "performance.csv"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(stream.getvalue())
    md_path = destination / "performance.md"
    md_path.write_text(
        f'# MOCK {data["workload"]} {"single-token decode" if decode else "forward"} performance\n\n'
        f'Random weights, seed {data["seed"]}, rows {data["rows"]}, layers {data.get("layers")}, '
        f'{data["dtype"]}, {data["samples"]} samples per variant.\n\n'
        f'Profile: {data["profile"]}. Torch {data["torch"]}; CuTe {data["cute"]}; GPU {data["gpu"]}.\n\n'
        f'Timing scope: {data.get("timing_scope", "synchronized model-forward calls")}. '
        'No SGLang engine comparison or pretrained-model accuracy claim.\n\n'
        f'Correctness relative L2: MegaBake {data["relative_l2"]:.6g}, Inductor {data["inductor_relative_l2"]:.6g}. '
        f'MegaBake workload kernels: 1. Execution commit: `{data["execution_commit"]}`.\n\n'
        f'| Variant | Median {"decode" if decode else "forward"} latency (ms) |\n| --- | ---: |\n'
        + ''.join(f'| {row["variant"]} | {row["median_call_ms"]:.6f} |\n' for row in rows))
    svg_path = destination / "mock_forward_ms.svg"
    svg_path.write_text(chart(
        rows, "median_call_ms", f'MOCK {data["workload"]}: synchronized {"decode" if decode else "forward"} latency', "ms",
        description=f'Random weights · BF16 · rows={data["rows"]} · layers={data.get("layers")} · seed={data["seed"]}',
        footnote="One trial; CUDA 12 experimental profile. Same-model forward comparison, not SGLang engine latency."))
    return [csv_path, md_path, svg_path]


def export(directory):
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text()).get('profiling'):
        from profile_publication import export_profile
        return export_profile(directory)
    if (directory / "sglang-generation.json").exists():
        from generation_publication import export_generation
        return export_generation(directory)
    if (directory / "sweep.json").exists():
        from sweep_publication import export_sweep
        return export_sweep(directory)
    if (directory / "mock-forward.json").exists():
        return export_mock(directory)
    rows = measurements(directory)
    if not rows:
        raise ValueError("No successful, finite performance measurements to publish")
    destination = directory / "published"
    destination.mkdir(exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = destination / "performance.csv"
    csv_path.write_text(stream.getvalue())
    text = ("# Recorded performance\n\n"
            "Successful measurements only. Each row is one trial; no confidence interval. "
            "Engine and model-forward scopes differ; no cross-harness speedup is implied.\n\n"
            "| Variant | Scope | Model | Batch | Input | Output | Prefill ms | Decode median ms/token |\n"
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        text += (f'| {row["variant"]} | {row["scope"]} | {row["model"]} | {row["batch_size"]} | {row["input_tokens"]} | '
                 f'{row["output_tokens"]} | {row["prefill_ms"]:.4f} | {row["decode_median_ms_per_token"]:.4f} |\n')
    md_path = destination / "performance.md"
    md_path.write_text(text)
    files = [csv_path, md_path]
    # Prevent comparisons across different models, dtypes, or shapes.
    chart_rows = [row for row in rows if row["framework"] == "SGLang"]
    contracts = {tuple(row[k] for k in ("model", "revision", "dtype", "batch_size", "input_tokens", "output_tokens", "scope")) for row in chart_rows}
    if len(chart_rows) >= 2 and len(contracts) == 1:
        for metric, title, unit in [("prefill_ms", "SGLang prefill latency", "ms"),
                                    ("decode_median_ms_per_token", "SGLang decode median latency", "ms/token")]:
            path = destination / f"{metric}.svg"
            path.write_text(chart(chart_rows, metric, title, unit))
            files.append(path)
    # Return only outputs produced from validated data; arbitrary PNGs/logs are never staged.
    return files
