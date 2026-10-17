"""Attribute MegaBake kernel time to phases with global-timer stamps.

Builds the timed kernel (MEGABAKE_PHASE_TIMER=1), runs warm calls, and reads
the stamp block 0 writes after every grid barrier. Phases between two
barriers run concurrently, so each barrier interval's time is split across
its phases in proportion to their work items; per-category totals are
therefore approximate, while interval totals are exact.

    .venv/bin/python tools/phase_profile.py --model HuggingFaceTB/SmolLM-1.7B --mode prefill
    .venv/bin/python tools/phase_profile.py --mode decode --skip-accuracy-gate
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

os.environ["MEGABAKE_PHASE_TIMER"] = "1"

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import megabake.compiler as compiler  # noqa: E402
from megabake import make_backend  # noqa: E402
from megabake.runtime import PHASE_TIMER_SLOTS  # noqa: E402
from benchmarks.full_model import PROMPT, _load, decode_module  # noqa: E402
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION  # noqa: E402


def _category(phase) -> str:
    name = re.sub(r"_\d+x\d+$", "", phase.body.name)
    if phase.body.tiles.mma_family == "wgmma":
        return f"{name} (tensor-core GEMM)"
    kind = phase.region_id.rsplit(":", 1)[-1]
    if kind.startswith(("mm", "bmm", "addmm")) or kind in name:
        return name
    return f"{name} [{kind}]"


def _stamps(callable_, compiled) -> list[int]:
    state = callable_.megabake_runtime_state
    workers = compiled.plan.launch.grid_size
    count = len(compiled.plan.phases)
    raw = state.ready[workers + 2 : workers + 3 + count].cpu().to(torch.int64)
    return [value & 0xFFFFFFFF for value in raw.tolist()]


def profile(callable_, compiled, run, calls: int) -> dict:
    phases = compiled.plan.phases
    barriers = [row[1] != 0 for row in compiled.generated.launch.phase_table]
    samples = []
    for _ in range(calls):
        run()
        torch.cuda.synchronize()
        samples.append(_stamps(callable_, compiled))
    median = lambda values: sorted(values)[len(values) // 2]  # noqa: E731
    intervals, members, previous = [], [], len(phases)
    for index, phase in enumerate(phases):
        members.append(phase)
        if barriers[index]:
            durations = [(stamp[index] - stamp[previous]) & 0xFFFFFFFF for stamp in samples]
            intervals.append((median(durations), tuple(members)))
            members, previous = [], index
    by_category: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.Counter()
    for duration, group in intervals:
        weights = [max(1, len(phase.work_items)) for phase in group]
        for phase, weight in zip(group, weights, strict=True):
            by_category[_category(phase)] += duration * weight / sum(weights)
            counts[_category(phase)] += 1
    total = sum(duration for duration, _ in intervals)
    gemm_tiles = collections.Counter(
        len(phase.work_items) for phase in phases if phase.body.tiles.mma_family == "wgmma"
    )
    return {
        "workers": compiled.plan.launch.grid_size,
        "phases": len(phases),
        "barrier_intervals": len(intervals),
        "timed_total_us": total / 1e3,
        "trailing_phases_untimed": len(members),
        "gemm_phases_by_tile_count": dict(sorted(gemm_tiles.items())),
        "categories": sorted(
            (
                {"category": name, "phases": counts[name], "us": value / 1e3}
                for name, value in by_category.items()
            ),
            key=lambda row: -row["us"],
        ),
        "slowest_intervals": [
            {
                "us": duration / 1e3,
                "phases": [
                    f"{_category(phase)} tiles={len(phase.work_items)}" for phase in group
                ],
            }
            for duration, group in sorted(intervals, key=lambda item: -item[0])[:12]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=SMOLLM_MODEL_NAME)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--attention", choices=("eager", "megabake"), default="eager")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--calls", type=int, default=9)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--skip-accuracy-gate",
        action="store_true",
        help="profile a plan even if it fails the accuracy check (diagnostic only)",
    )
    args = parser.parse_args()
    if args.skip_accuracy_gate:
        compiler._TABLE_PLAN_RELATIVE_L2_LIMIT = float("inf")
        compiler._PLAN_RELATIVE_L2_LIMIT = float("inf")
    revision = args.revision or (SMOLLM_REVISION if args.model == SMOLLM_MODEL_NAME else None)
    model = _load(args.model, revision, args.attention)
    backend = make_backend(mode=args.mode, compile_kernel=True, cache=False, scheduler="phase")
    with torch.no_grad():
        if args.mode == "prefill":
            ids = torch.tensor([PROMPT], device="cuda")
            mask = torch.ones_like(ids)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            run = lambda: compiled_model(ids, attention_mask=mask)  # noqa: E731
        else:
            stateful, shapes = decode_module(model, args.capacity)
            state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
            token = torch.tensor([[504]], device="cuda")
            compiled_model = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)
            position = [0]

            if args.calls + 4 > args.capacity:
                raise SystemExit("--capacity must exceed --calls + 4")

            def run():
                position_ids = torch.tensor([[position[0]]], device="cuda")
                position[0] += 1
                return compiled_model(token, position_ids, *state)

        run()
        callable_ = backend.megabake_last_callable
        compiled = backend.megabake_last_compiled
        if callable_ is None or compiled is None:
            raise SystemExit("MegaBake did not compile a kernel for this model")
        if compiled.generated.launch.launch_policy != "megabake_cooperative_table":
            raise SystemExit("phase timing needs a table launch")
        if len(compiled.plan.phases) >= PHASE_TIMER_SLOTS:
            raise SystemExit("plan has more phases than timer slots")
        for _ in range(3):
            run()
        report = {"model": args.model, "mode": args.mode, **profile(callable_, compiled, run, args.calls)}
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
