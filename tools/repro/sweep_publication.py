"""Validate token-sweep evidence and export performance tables and Matplotlib plots."""
import csv
import io
import json
import math
import statistics

VARIANTS = ("eager", "eager_cuda_graph", "inductor", "megabake")


def load_points(directory):
    from publication import export_mock
    manifest = json.loads((directory / "sweep.json").read_text())
    points = []
    for tokens in sorted(manifest["tokens"]):
        point = directory / f"tokens-{tokens}"
        status = json.loads((point / "status.json").read_text()) if (point / "status.json").exists() else {}
        if status.get("state") != "complete":
            continue
        export_mock(point)  # Shared finite-value, correctness and single-kernel gates.
        data = json.loads((point / "mock-forward.json").read_text())
        if data["rows"] != tokens or data["execution_commit"] != manifest["execution_commit"]:
            raise ValueError("Sweep token count or source commit mismatch")
        if data.get("rounds") != manifest["rounds"] or data["samples"] != data["rounds"] * data["samples_per_round"]:
            raise ValueError("Sweep round count mismatch")
        for record in data["measurements"].values():
            rounds = record.get("round_medians_us", [])
            if len(rounds) != data["rounds"] or not all(math.isfinite(v) and v > 0 for v in rounds):
                raise ValueError("Invalid per-round measurements")
            actual_rounds = [statistics.median(record["samples_us"][i:i + data["samples_per_round"]])
                             for i in range(0, data["samples"], data["samples_per_round"])]
            if rounds != actual_rounds or record["median_call_us"] != statistics.median(record["samples_us"]):
                raise ValueError("Summary does not match raw samples")
        points.append(data)
    if len(points) < 2:
        raise ValueError("At least two validated token lengths are required for a sweep")
    contracts = {tuple(p[k] for k in ("workload", "layers", "dtype", "seed", "torch", "cute", "gpu",
                                     "profile", "execution_commit", "rounds", "samples_per_round", "timing_scope"))
                 for p in points}
    if len(contracts) != 1:
        raise ValueError("Incompatible workload or timing contracts in sweep")
    return points


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def export_sweep(directory):
    points = load_points(directory)
    target = directory / "published"
    target.mkdir(exist_ok=True)
    summary, rounds, samples = [], [], []
    decode = points[0].get("mode") == "decode"
    length_field = "cached_tokens" if decode else "tokens"
    for data in points:
        for variant in VARIANTS:
            record = data["measurements"][variant]
            round_ms = [v / 1000 for v in record["round_medians_us"]]
            summary.append(dict({length_field: data["rows"]}, variant=variant,
                                median_call_ms=record["median_call_us"] / 1000,
                                round_min_ms=min(round_ms), round_max_ms=max(round_ms),
                                rounds=data["rounds"], samples_per_round=data["samples_per_round"],
                                layers=data["layers"], seed=data["seed"],
                                megabake_relative_l2=data["relative_l2"],
                                inductor_relative_l2=data["inductor_relative_l2"],
                                execution_commit=data["execution_commit"],
                                mode=data.get("mode", "prefill"), batch_size=1,
                                inductor_emulate_precision_casts=data.get("inductor_options", {}).get("emulate_precision_casts", False),
                                cache_capacity=data.get("cache_capacity", ""),
                                max_new_kv_relative_l2=max(data.get("cache_relative_l2", {"unused": 0}).values())))
            for trial, value in enumerate(round_ms, 1):
                rounds.append(dict({length_field: data["rows"]}, variant=variant, round=trial, median_call_ms=value))
            for i, value in enumerate(record["samples_us"]):
                samples.append(dict({length_field: data["rows"]}, variant=variant,
                                    round=i // data["samples_per_round"] + 1,
                                    sample=i % data["samples_per_round"] + 1, call_ms=value / 1000))
    files = []
    for name, rows in (("performance.csv", summary), ("rounds.csv", rounds), ("samples.csv", samples)):
        path = target / name
        write_csv(path, rows)
        files.append(path)
    data = points[0]
    text = (f'# MOCK SmolLM {data["layers"]}-layer {"decode context" if decode else "prefill token"} sweep\n\n'
            f'BF16, batch 1, seed {data["seed"]}, {data["rounds"]} timing rounds × '
            f'{data["samples_per_round"]} samples per variant and length. '
            'Round order rotates; weights are fixed across lengths; compilation and warmup excluded. '
            'The three rounds share one process and compilation, not independent process repeats.\n\n'
            f'{data["timing_scope"]}. Random weights; no pretrained-model or '
            f'{"end-to-end generation" if decode else "decode"} claim.\n\n'
            f'{data["profile"]}; torch {data["torch"]}, CuTe {data["cute"]}, {data["gpu"]}. '
            f'Execution commit `{data["execution_commit"]}`.\n\n'
            f'| {"Cached tokens" if decode else "Tokens"} | Eager ms | CUDA Graph ms | Inductor ms | MegaBake ms | CUDA Graph / MegaBake | Inductor / MegaBake |\n'
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for point in points:
        values = [point["measurements"][v]["median_call_us"] / 1000 for v in VARIANTS]
        text += '| ' + str(point["rows"]) + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + f' | {values[1]/values[3]:.3f}× | {values[2]/values[3]:.3f}× |\n'
    text += ('\nOnly validated successful lengths are plotted. CSV includes round ranges, correctness errors and raw performance samples. '
             'Ranges describe the observed round medians, not confidence intervals. '
             'Speedup >1 means MegaBake is faster; no SGLang engine comparison.\n')
    if decode:
        text += ('\nEach call consumes one token at the stated cached context and updates KV state. '
                 'Capacity is context+128, specialized per length. Logical lengths are restored before each call outside timing. '
                 'Cached K/V are seeded nonzero random tensors, not computed from a prompt. '
                 'Eager, CUDA Graph and Inductor use a GPU-only tensor KV reference checked against upstream eager logits and full KV state. '
                 'MegaBake uses the original declared KV op; existing cache contents and one-token updates are validated.\n')
        text += ('\nInductor enables `emulate_precision_casts=True` to preserve eager BF16 intermediate rounding. '
                 'This is an explicitly configured baseline, not the default Inductor precision-fusion configuration.\n')
    path = target / "performance.md"
    path.write_text(text)
    files.append(path)
    files.extend(plots(target, points))
    return files


def plots(target, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "megabake-token-sweep", "font.size": 10})
    lengths = [p["rows"] for p in points]
    decode = points[0].get("mode") == "decode"
    colors = ("#687587", "#d99026", "#387bc1", "#14866a")
    files = []
    for kind in ("latency", "speedup"):
        fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
        for variant, color in zip(VARIANTS, colors):
            if kind == "speedup" and variant in ("megabake", "eager"):
                continue
            values = [p["measurements"][variant]["median_call_us"] / 1000 for p in points]
            if kind == "speedup":
                values = [v / (p["measurements"]["megabake"]["median_call_us"] / 1000) for v, p in zip(values, points)]
            label = "inductor (BF16-aligned)" if decode and variant == "inductor" else variant
            ax.plot(lengths, values, "o-", label=label, color=color, linewidth=2)
            if kind == "latency":
                low = [min(p["measurements"][variant]["round_medians_us"]) / 1000 for p in points]
                high = [max(p["measurements"][variant]["round_medians_us"]) / 1000 for p in points]
                ax.fill_between(lengths, low, high, color=color, alpha=.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths, labels=lengths)
        ax.set_xlabel("Already cached tokens (batch = 1; one new token)" if decode else "Input tokens (batch = 1)")
        ax.set_ylabel(("Synchronized decode latency (ms/token)" if decode else "Synchronized forward latency (ms)")
                      if kind == "latency" else "Baseline latency / MegaBake latency")
        ax.set_ylim(bottom=0)
        if kind == "speedup":
            ax.axhline(1, color="#666", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
        ax.legend()
        ax.set_title(f'MOCK SmolLM {points[0]["layers"]} layers · BF16 · H100\n'
                     + (("Decode latency vs cached context (lower is better)" if decode else "Latency vs input length (lower is better)")
                        if kind == "latency" else "MegaBake speedup (above 1 is faster)"))
        fig.supxlabel(f'Random weights · {points[0]["rounds"]} rounds × {points[0]["samples_per_round"]} samples · CUDA 12 experimental profile\n' +
                      ('Random KV; capacity=context+128; reset excluded; host+GPU decode; no SGLang comparison.' if decode else
                       'Prebuilt causal mask; host-call + synchronization included; no SGLang comparison.'), fontsize=8)
        path = target / f"token_{kind}.svg"
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)
        files.append(path)
    return files
