#!/usr/bin/env python3
"""Run the CUDA 12 synthetic profile under the existing Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_node import now, run_stage, tool_environment, write
from workflow import validate_id

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=validate_id, required=True)
    parser.add_argument("--workload", choices=("mlp", "smollm", "smollm-decode"), required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    lengths = args.tokens or [config["prompt_tokens"]]
    if (config.get("profile") != "cuda12-mock" or args.layers < 1 or args.rounds < 1
            or min(lengths) < 1 or len(set(lengths)) != len(lengths)):
        raise SystemExit("Use the explicit cuda12-mock profile and positive layer count")
    if os.environ.get("SLURM_JOB_ID") != str(config["job_id"]) or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("Run inside the configured Slurm job step")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Source worktree must be clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    folder = ROOT / "results" / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists():
        raise SystemExit("Run ID already used")
    status = {"state": "running", "started_at": now(), "run_id": args.run_id,
              "mock_weights": True, "execution_commit": commit}
    write(folder / "status.json", status)
    write(folder / "config.json", config)
    env = tool_environment(os.environ, config["megabake_python"], config["cuda_home"], ROOT / "src")
    env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
               CUDA_ROOT=config["cuda_home"], LD_LIBRARY_PATH=config["cuda_home"] + "/lib64",
               TRITON_PTXAS_PATH=config["cuda_home"] + "/bin/ptxas",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    stages = []
    sweep = len(lengths) > 1
    if sweep:
        write(folder / "sweep.json", {"tokens": lengths, "rounds": args.rounds, "execution_commit": commit})
    for length in lengths:
        point = folder / f"tokens-{length}" if sweep else folder
        point.mkdir(exist_ok=True)
        print(f"START tokens={length}, rounds={args.rounds}", flush=True)
        script = "mock_decode.py" if args.workload == "smollm-decode" else "mock_smoke.py"
        command = [config["megabake_python"], "-u", str(ROOT / "tools/repro" / script),
                   "--workload", args.workload, "--layers", str(args.layers),
                   "--rows", str(length), "--samples", str(config["samples"]),
                   "--rounds", str(args.rounds), "--output", str(point / "mock-forward.json")]
        stage = run_stage("mock-forward", command, point, env, config["timeout_seconds"], ROOT)
        if stage["state"] == "complete":
            try:
                data = json.loads((point / "mock-forward.json").read_text())
                if data["state"] != "complete" or data["execution_commit"] != commit or data["rows"] != length:
                    raise ValueError("Result does not match the running source commit or token length")
            except (OSError, ValueError, KeyError) as error:
                stage.update(state="failed", error=str(error))
        if sweep:
            write(point / "status.json", {"state": stage["state"], "execution_commit": commit})
        stages.append(dict(stage, tokens=length))
        status.update(stages=stages)
        write(folder / "status.json", status)
        print(f"END tokens={length}, state={stage['state']}", flush=True)
    state = "complete" if all(s["state"] == "complete" for s in stages) else "failed"
    status.update(state=state, stages=stages, finished_at=now())
    write(folder / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
