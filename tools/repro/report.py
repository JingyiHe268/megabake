#!/usr/bin/env python3
"""Render an auditable local HTML report from recorded evidence, with no dependencies."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

from evidence import evidence_files


def render(folder):
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text())
    environment = json.loads((folder / "environment.json").read_text()) if (folder / "environment.json").exists() else {}
    config = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
    esc = lambda value: html.escape(str(value))
    blockers = status.get("blockers", [])
    if status.get("error"):
        blockers = [*blockers, status["error"]]
    rows = []
    for path in sorted(folder.glob("sglang-cudagraph-*.jsonl")):
        stage_file = path.with_suffix(".status.json")
        if not stage_file.exists() or json.loads(stage_file.read_text()).get("state") != "complete":
            continue
        for line in path.read_text().splitlines():
            item = json.loads(line)
            rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [
                path.stem, item.get("batch_size"), item.get("input_len"), item.get("output_len"),
                f"{item['prefill_latency'] * 1e3:.4f}" if "prefill_latency" in item else "—",
                f"{item['median_decode_latency'] * 1e3:.4f}" if "median_decode_latency" in item else "—",
            ]) + "</tr>")
    commit = environment.get("megabake_commit", {}).get("output", "unknown").strip()
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MegaBake / SGLang reproduction — {esc(folder.name)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#101823;color:#e6edf6;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1180px;margin:40px auto;padding:0 28px 60px}}h1{{font-size:34px;margin-bottom:6px}}h2{{font-size:22px;margin-top:32px}}
.muted{{color:#a9bacd}}.badge{{display:inline-block;background:#45341a;color:#ffdb84;padding:5px 16px;border-radius:20px}}
.card{{background:#1b2736;padding:20px 24px;margin:16px 0;border-radius:12px;border:1px solid #334258}}
code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,monospace}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #3e4b5c}}
a{{color:#8dc8ff}}li{{margin:7px 0}}
</style><main>
<div class="muted">JingyiHe268 · reproducible experiment evidence</div>
<h1>MegaBake × SGLang · H100 reproduction</h1>
<p class="muted">Run {esc(folder.name)} · {esc(status.get('finished_at', status.get('started_at')))}</p>
<span class="badge">{esc(status['state'].upper())}{' · SGLang baseline only' if status.get('baseline_only') else ''}</span>
<div class="card"><b>Comparison status: {esc(status.get('comparison', 'not_available'))}</b>
<p>{esc(status.get('comparison_note', 'No completed comparable measurements are available.'))}</p>
<p>{'Baseline smoke test; MegaBake was not executed. Trials per CUDA Graph setting: ' + esc(config.get('repeats')) + '.' if status.get('baseline_only') else 'Full reproduction preflight / execution.'}</p>
<b>No cross-framework speedup claimed.</b></div>
<h2>Version & workload</h2><div class="card">
<div>Execution commit: <code>{esc(commit)}</code></div>
<div>Original MegaBake source: <code>{esc(config.get('source_revision'))}</code></div>
<div>SGLang commit: <code>{esc(environment.get('sglang_commit', {}).get('output', 'unknown').strip())}</code></div>
<div>Node: {esc(environment.get('hostname'))} · Slurm job: {esc(environment.get('job_id'))}</div>
<div>Model: {esc(config.get('model_id'))} · revision: <code>{esc(config.get('model_revision'))}</code></div>
<div>BF16 · TP=1 · batch={esc(config.get('batch_size'))} · input={esc(config.get('prompt_tokens'))} · output={esc(config.get('output_tokens'))}</div>
</div><h2>Blockers / missing prerequisites</h2><div class="card"><ul>
{''.join('<li>' + esc(item) + '</li>' for item in blockers) or '<li>No prerequisite blocker recorded. See stage statuses.</li>'}
</ul></div><h2>SGLang raw measurements (milliseconds)</h2><div class="card">
{'<table><tr><th>Variant</th><th>Batch</th><th>Input</th><th>Output</th><th>Prefill ms</th><th>Decode median ms</th></tr>' + ''.join(rows) + '</table>' if rows else '<p>No successful SGLang measurements. Missing values are not zero latency.</p>'}
</div><h2>GPU evidence</h2><div class="card"><pre>{esc(environment.get('gpu', {}).get('output', 'No GPU evidence'))}</pre></div>
<h2>Stages</h2><div class="card"><pre>{esc(json.dumps(status.get('stages', []), indent=2, ensure_ascii=False))}</pre></div>
<p class="muted">Raw evidence: status.json · environment.json · config.json · *.log · *.jsonl. Screenshot is a capture of this evidence report, not a profiler screenshot.</p>
</main></html>'''
    (folder / "report.html").write_text(report)
    (folder / "SUMMARY.md").write_text(
        f"# {folder.name}\n\nStatus: **{status['state']}**\n\n"
        f"Execution commit: `{commit}`\n\n"
        f"Comparison: `{status.get('comparison', 'not_available')}`. No cross-framework speedup claimed.\n\n"
        + "\n".join(f"- {item}" for item in blockers) + "\n\n"
        + "See [report](report.html), [status](status.json), [environment](environment.json).\n"
    )
    checksums = {}
    for path in evidence_files(folder):
        if path.name != "checksums.json":
            checksums[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return folder / "report.html"


if __name__ == "__main__":
    print(render(sys.argv[1]))
