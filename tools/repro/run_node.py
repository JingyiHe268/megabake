#!/usr/bin/env python3
"""Run only inside an existing Slurm GPU allocation; record failures as evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def capture(argv, timeout=30, cwd=None, env=None):
    try:
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, cwd=cwd, env=env)
        return {"argv": list(map(str, argv)), "returncode": p.returncode, "output": p.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(map(str, argv)), "returncode": None, "error": str(error)}


PROBE = '''import importlib.metadata as m, json, sys
packages = {}
for name in ["torch", "transformers", "sglang", "sglang-kernel", "nvidia-cutlass-dsl", "numpy"]:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
data = {"python": sys.version, "executable": sys.executable, "packages": packages}
try:
    import torch
    data.update(cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if data["cuda_available"]:
        x = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        data.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                    cuda_tensor_ok=float(x.cpu()[0]) == 1.0)
except Exception as error: data["cuda_error"] = repr(error)
print("REPRO_PROBE=" + json.dumps(data))
'''


def probe(python, env):
    result = capture([str(python), "-u", "-c", PROBE], timeout=300, env=env)
    for line in result.get("output", "").splitlines():
        if line.startswith("REPRO_PROBE="):
            result["data"] = json.loads(line.removeprefix("REPRO_PROBE="))
    return result


def tool_environment(base, python, cuda_home, python_path):
    env = base.copy()
    env["PYTHONPATH"] = str(python_path)
    env["CUDA_HOME"] = str(cuda_home)
    # Invoking venv/bin/python does not activate venv/bin for subprocesses (e.g. ninja).
    env["PATH"] = str(Path(python).parent) + ":" + str(cuda_home) + "/bin:" + base.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(cuda_home) + "/lib64:" + base.get("LD_LIBRARY_PATH", "")
    return env


def fingerprint_model(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        return {"ready": False, "reason": f"model config missing: {path}/config.json"}
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights or not all(p.is_file() for p in weights):
        return {"ready": False, "reason": "no complete local model weights"}
    files = {}
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"}:
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            files[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    if (path / "model.safetensors.index.json").is_file():
        index = json.loads((path / "model.safetensors.index.json").read_text())
        missing = set(index["weight_map"].values()) - files.keys()
        if missing:
            return {"ready": False, "reason": f"missing shards: {sorted(missing)}"}
    return {"ready": True, "path": str(path.resolve()), "files": files}


def run_stage(name, argv, folder, env, timeout, cwd):
    started = now()
    print(f"[{started}] {name}: {json.dumps(argv)}", flush=True)
    status = {"name": name, "argv": argv, "started_at": started, "state": "running"}
    write(folder / f"{name}.status.json", status)
    begin = time.monotonic()
    try:
        with (folder / f"{name}.log").open("w") as log:
            with subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        status.update(state="complete" if code == 0 else "failed", returncode=code)
    except (OSError, subprocess.TimeoutExpired) as error:
        status.update(state="failed", error=str(error))
    status.update(finished_at=now(), wall_seconds=time.monotonic() - begin)
    write(folder / f"{name}.status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    # Use the same path validation as the local control script.
    from workflow import validate_id
    validate_id(args.run_id)
    config = json.loads(args.config.read_text())
    folder = ROOT / "results" / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists():
        raise SystemExit("Existing run status found; choose a new run ID")
    status = {"state": "running", "started_at": now(), "run_id": args.run_id,
              "baseline_only": args.baseline_only, "blockers": [], "stages": [],
              "comparison": "not_available"}
    write(folder / "status.json", status)
    try:
        write(folder / "config.json", config)
        job = os.environ.get("SLURM_JOB_ID")
        if job != str(config["job_id"]) or not os.environ.get("SLURM_STEP_ID"):
            raise RuntimeError("Must execute under srun inside the configured allocation")
        commit = capture(["git", "rev-parse", "HEAD"], cwd=ROOT)
        dirty = capture(["git", "status", "--porcelain"], cwd=ROOT)
        if dirty.get("returncode") != 0 or dirty.get("output", "").strip():
            raise RuntimeError("Dirty MegaBake worktree; deploy a committed version")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Local checkpoints only. No implicit download from a floating revision.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["OMP_NUM_THREADS"] = "4"
        env["PYTHONPATH"] = str(ROOT / "src")
        metadata = {"captured_at": now(), "hostname": platform.node(), "platform": platform.platform(),
                    "job_id": job, "step_id": os.environ.get("SLURM_STEP_ID"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "megabake_commit": commit,
                    "source_revision": config["source_revision"],
                    "gpu": capture(["nvidia-smi"]),
                    "gpu_details": capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,mig.mode.current,power.limit", "--format=csv"]),
                    "slurm_job": capture(["scontrol", "show", "job", job])}
        sglang = Path(config["sglang_repo"])
        metadata["sglang_commit"] = capture(["git", "rev-parse", "HEAD"], cwd=sglang)
        metadata["sglang_status"] = capture(["git", "status", "--short"], cwd=sglang)
        patch = capture(["git", "diff", "--binary", "HEAD", "--", "*.py"], cwd=sglang)
        (folder / "sglang-tracked-python.patch").write_text(patch.get("output", ""))
        # Include hashes for untracked/ignored benchmark imports without copying unrelated work.
        hashes = {}
        for relative in ("python/sglang/benchmark/one_batch.py", "python/sglang/bench_one_batch.py"):
            p = sglang / relative
            if p.is_file():
                hashes[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        metadata["sglang_benchmark_sha256"] = hashes
        baseline_env = tool_environment(
            env, config["sglang_python"], Path(config["remote_base"]) / "toolchains/cuda-12.9", sglang / "python")
        metadata["sglang_probe"] = probe(config["sglang_python"], baseline_env)
        megabake_env = tool_environment(env, config["megabake_python"], config["cuda_home"], ROOT / "src")
        metadata["megabake_probe"] = probe(config["megabake_python"], megabake_env)
        metadata["tool_paths"] = {
            "sglang_ninja": shutil.which("ninja", path=baseline_env["PATH"]),
            "sglang_cuda_home": baseline_env["CUDA_HOME"],
            "megabake_cuda_home": megabake_env["CUDA_HOME"],
        }
        metadata["nvcc"] = capture([config["cuda_home"] + "/bin/nvcc", "--version"])
        metadata["model"] = fingerprint_model(config["model_path"])
        write(folder / "environment.json", metadata)
        (folder / "gpu.txt").write_text(metadata["gpu"].get("output", ""))
        for label, python in [("sglang", config["sglang_python"]), ("megabake", config["megabake_python"])]:
            if Path(python).exists():
                # uv-created environments need not contain pip; query installed metadata directly.
                freeze = capture([python, "-c", "import importlib.metadata as m; "
                                  "print('\\n'.join(sorted(f\"{d.metadata['Name']}=={d.version}\" "
                                  "for d in m.distributions())))"], timeout=120)
                (folder / f"{label}-installed-distributions.txt").write_text(
                    freeze.get("output", freeze.get("error", "")))
        blockers = status["blockers"]
        if not metadata["model"]["ready"]:
            blockers.append(metadata["model"]["reason"])
        baseline_data = metadata["sglang_probe"].get("data", {})
        baseline_ready = baseline_data.get("cuda_tensor_ok", False) and metadata["model"]["ready"]
        if not baseline_data.get("cuda_tensor_ok"):
            blockers.append("Existing SGLang Python cannot execute a CUDA tensor")
        mb_data = metadata["megabake_probe"].get("data", {})
        packages = mb_data.get("packages", {})
        mb_reasons = []
        if not Path(config["megabake_python"]).is_file():
            mb_reasons.append("Isolated MegaBake Python environment is not installed")
        if not str(packages.get("torch", "")).startswith("2.14."):
            mb_reasons.append("MegaBake requires PyTorch 2.14.x")
        if packages.get("nvidia-cutlass-dsl") != "4.8.0":
            mb_reasons.append("MegaBake requires CuTe DSL 4.8.0")
        if not mb_data.get("cuda_tensor_ok") or mb_data.get("cuda_runtime") != "13.0":
            mb_reasons.append("MegaBake CUDA 13 runtime/tensor execution has not passed")
        if metadata["nvcc"].get("returncode") != 0 or "release 13.0" not in metadata["nvcc"].get("output", ""):
            mb_reasons.append("CUDA 13.0 nvcc is missing")
        if not args.baseline_only:
            blockers.extend(mb_reasons)
        status["megabake_environment_blockers"] = mb_reasons
        write(folder / "status.json", status)
        if baseline_ready:
            for graph in ("on", "off"):
                for repeat in range(config["repeats"]):
                    name = f"sglang-cudagraph-{graph}-{repeat + 1}"
                    raw = folder / f"{name}.jsonl"
                    command = [config["sglang_python"], "-m", "sglang.bench_one_batch",
                               "--model-path", config["model_path"], "--dtype", "bfloat16",
                               "--tp-size", "1", "--batch-size", str(config["batch_size"]),
                               "--input-len", str(config["prompt_tokens"]),
                               "--output-len", str(config["output_tokens"]),
                               "--mem-fraction-static", "0.5", "--context-length", str(config["capacity"]),
                               "--random-seed", "42", "--result-filename", str(raw), "--run-name", args.run_id]
                    if graph == "off":
                        command.append("--disable-cuda-graph")
                    stage = run_stage(name, command, folder, baseline_env, config["timeout_seconds"], sglang)
                    if stage["state"] == "complete" and (not raw.exists() or not raw.read_text().strip()):
                        stage.update(state="failed", error="benchmark returned no measurements")
                        write(folder / f"{name}.status.json", stage)
                    status["stages"].append(stage)
                    write(folder / "status.json", status)
                    if stage["state"] != "complete":
                        break
                if status["stages"][-1]["state"] != "complete":
                    break
        if not args.baseline_only and not mb_reasons and metadata["model"]["ready"]:
            name = "megabake-upstream"
            command = [config["megabake_python"], str(ROOT / "benchmarks/full_model.py"),
                       "--model", config["model_path"], "--revision", config["model_revision"],
                       "--variants", "eager", "inductor", "inductor_cudagraphs", "megabake",
                       "--attention", "megabake", "--samples", str(config["samples"]),
                       "--warmup", str(config["warmup"]), "--prompt-tokens", str(config["prompt_tokens"]),
                       "--capacity", str(config["capacity"]), "--out", str(folder / name)]
            stage = run_stage(name, command, folder, megabake_env, config["timeout_seconds"], ROOT)
            report_path = folder / name / "report.json"
            if stage["state"] == "complete":
                try:
                    report = json.loads(report_path.read_text())
                    for mode in ("prefill", "decode"):
                        for variant in ("eager", "inductor", "inductor_cudagraphs", "megabake"):
                            row = report[mode][variant]
                            if "error" in row or "complete_call_us" not in row:
                                raise ValueError(f"{mode}/{variant}: {row}")
                except (OSError, ValueError, KeyError) as error:
                    stage.update(state="failed", error=f"Incomplete measurements: {error}")
                    write(folder / f"{name}.status.json", stage)
            status["stages"].append(stage)
        status["comparison"] = "not_comparable_across_harnesses"
        # SGLang samples/schedules logits; upstream MegaBake times model forward.
        # Retain both raw results, but never automatically report their ratio as a speedup.
        status["comparison_note"] = (
            "Same model and BF16 are required, but SGLang bench_one_batch includes engine/sampling work "
            "and advancing contexts; MegaBake upstream times forward calls with different cache setup. "
            "No cross-framework speedup is established by this run. A matched harness is required."
        )
        status["state"] = ("failed" if any(s["state"] != "complete" for s in status["stages"])
                           else "blocked" if blockers else "complete")
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        (folder / "error.log").write_text(traceback.format_exc())
    status["finished_at"] = now()
    write(folder / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if status["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
