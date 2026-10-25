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


def chart(rows, metric, title, unit):
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
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">'
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#152030">'
            f'<text x="24" y="36" font-size="22">{html.escape(title)}</text>'
            f'<text x="24" y="64" font-size="14">{html.escape(rows[0]["model"])} · BF16 · '
            f'batch={rows[0]["batch_size"]} · input={rows[0]["input_tokens"]} · output={rows[0]["output_tokens"]}</text>'
            + ''.join(bars) + f'<path d="M270 94V{100+72*len(rows)}" stroke="#555"/>'
            f'<text x="270" y="{height-55}" font-size="13">0 — {maximum:.4f} {unit}; lower is better</text>'
            f'<text x="24" y="{height-24}" font-size="12">Each bar is one recorded trial; decode median covers advancing tokens. No MegaBake comparison.</text>'
            '</g></svg>')


def export(directory):
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
