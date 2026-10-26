#!/usr/bin/env python3
"""Local Git -> offline SSH bundle -> Slurm -> local results -> GitHub.

No third-party Python packages required. GitHub authentication stays local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from publication import export

ROOT = Path(__file__).resolve().parents[2]


def call(args, **kwargs):
    print("+", shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def output(args):
    return subprocess.check_output(list(map(str, args)), text=True, cwd=ROOT).strip()


def remote(config, command, **kwargs):
    return call(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 config["ssh_host"], command], **kwargs)


def validate_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("run ID must contain only letters, numbers, dot, dash, underscore")
    return value


def deploy(config):
    if output(["git", "status", "--porcelain"]):
        raise SystemExit("Commit local changes before deployment; the bundle must represent an exact commit.")
    branch = output(["git", "branch", "--show-current"])
    if branch != config["branch"]:
        raise SystemExit(f"Expected branch {config['branch']}, found {branch}")
    revision = output(["git", "rev-parse", "HEAD"])
    staging = ROOT / ".repro"
    staging.mkdir(exist_ok=True)
    bundle = staging / f"megabake-{revision}.bundle"
    call(["git", "bundle", "create", bundle, f"refs/heads/{branch}"], cwd=ROOT)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    destination = f"{config['remote_base']}/bundles/{bundle.name}"
    remote(config, f"mkdir -p {shlex.quote(str(Path(destination).parent))}")
    call(["scp", "-o", "BatchMode=yes", bundle, f"{config['ssh_host']}:{destination}"])
    # Shell code is sent as a positional argument to bash, with every value quoted.
    script = """set -euo pipefail
bundle=$1; repo=$2; branch=$3; revision=$4; digest=$5; origin=$6
test "$(sha256sum "$bundle" | cut -d' ' -f1)" = "$digest"
if [ ! -e "$repo" ]; then
  git clone --branch "$branch" "$bundle" "$repo"
else
  test -d "$repo/.git"
  test -z "$(git -C "$repo" status --porcelain)"
  test "$(git -C "$repo" branch --show-current)" = "$branch"
  git -C "$repo" fetch "$bundle" "refs/heads/$branch"
  git -C "$repo" merge --ff-only FETCH_HEAD
fi
git -C "$repo" config user.name JingyiHe268
git -C "$repo" config user.email hejy268@mail2.sysu.edu.cn
git -C "$repo" remote set-url origin "$origin"
test "$(git -C "$repo" rev-parse HEAD)" = "$revision"
echo "DEPLOYED_COMMIT=$revision"
"""
    remote(config, shlex.join(["bash", "-c", script, "repro-deploy", destination,
                               config["remote_repo"], branch, revision, digest, config["repository"]]))
    (staging / "deployment.json").write_text(json.dumps(
        {"commit": revision, "bundle_sha256": digest, "remote_repo": config["remote_repo"]}, indent=2) + "\n")


def launch(config, config_path, run_id, baseline_only=False, mock=None, layers=2, tokens=None, rounds=1, generation=False):
    run_id = validate_id(run_id)
    if mock and (baseline_only or config.get("profile") != "cuda12-mock" or layers < 1):
        raise SystemExit("Mock runs require cuda12-mock config, positive layers, and no --baseline-only")
    if generation and (mock or baseline_only or config.get("profile") != "cuda12-generation"):
        raise SystemExit("Generation requires the shared-checkpoint cuda12-generation profile")
    if rounds < 1 or (tokens and ((not mock and not generation) or min(tokens) < 1 or len(set(tokens)) != len(tokens))):
        raise SystemExit("Use unique positive --tokens with --mock and positive --rounds")
    try:
        relative_config = config_path.resolve().relative_to(ROOT)
    except ValueError:
        raise SystemExit("The config must be committed inside this repository")
    expected = output(["git", "rev-parse", "HEAD"])
    if output(["git", "status", "--porcelain"]):
        raise SystemExit("Commit changes before launching a reproducible run")
    repo = config["remote_repo"]
    result = f"{repo}/results/{run_id}"
    runner = "run_generation_node.py" if generation else ("run_mock_node.py" if mock else "run_node.py")
    python = config["megabake_python"] if mock or generation else "python3"
    argv = ["srun", f"--jobid={config['job_id']}", "--overlap", "--exact", "-N1", "-n1", "-c4",
            "--mem=24G", python, f"{repo}/tools/repro/{runner}", "--config",
            f"{repo}/{relative_config}", "--run-id", run_id]
    if baseline_only:
        argv.append("--baseline-only")
    if mock:
        argv.extend(["--workload", mock, "--layers", str(layers), "--rounds", str(rounds)])
        if tokens:
            argv.extend(["--tokens", *map(str, tokens)])
    if generation:
        argv.extend(["--rounds", str(rounds)])
        if tokens:
            argv.extend(["--lengths", *map(str, tokens)])
    # Reserve the run directory atomically, detach on the login node; work executes under srun.
    script = f"""set -euo pipefail
test "$(git -C {shlex.quote(repo)} rev-parse HEAD)" = {shlex.quote(expected)}
test -z "$(git -C {shlex.quote(repo)} status --porcelain)"
mkdir {shlex.quote(result)}
nohup {shlex.join(argv)} > {shlex.quote(result + '/launcher.log')} 2>&1 < /dev/null &
echo $! > {shlex.quote(result + '/launcher.pid')}
echo {shlex.quote('STARTED ' + run_id + '; use status and collect commands')}
"""
    remote(config, shlex.join(["bash", "-c", script]))


def collect(config, run_id):
    validate_id(run_id)
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    remote_path = f"{config['remote_repo']}/results/{run_id}/"
    # No --delete: never delete results or local screenshots on either side.
    call(["rsync", "-az", "--exclude=*.nsys-rep", "--exclude=*.sqlite", "--exclude=*-trace.json", "--exclude=generated/",
          "-e", "ssh -o BatchMode=yes", f"{config['ssh_host']}:{remote_path}", str(destination) + "/"])
    call([sys.executable, ROOT / "tools/repro/report.py", destination])


def publish(config, run_id):
    validate_id(run_id)
    directory = ROOT / "results" / run_id
    status = json.loads((directory / "status.json").read_text())
    if status.get("state") not in {"complete", "blocked", "failed"}:
        raise SystemExit("Run is not terminal; collect again after completion")
    try:
        files = export(directory)
    except ValueError as error:
        raise SystemExit(str(error))
    if output(["git", "diff", "--cached", "--name-only"]):
        raise SystemExit("Unrelated staged changes exist; commit them separately before publish")
    remote_url = output(["git", "remote", "get-url", "origin"])
    if not re.search(r"[:/]JingyiHe268/megabake(?:\.git)?$", remote_url):
        raise SystemExit(f"Unexpected origin: {remote_url}")
    call(["git", "add", "-f", "--", *[str(p.relative_to(ROOT)) for p in files]], cwd=ROOT)
    if output(["git", "diff", "--cached", "--name-only"]):
        call(["git", "commit", "-m", f"Publish performance tables and charts: {run_id}"], cwd=ROOT)
    call(["git", "push", "-u", "origin", f"HEAD:refs/heads/{config['branch']}"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools/repro/ln301.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy")
    for name in ("launch", "status", "collect", "publish"):
        p = sub.add_parser(name)
        p.add_argument("run_id", type=validate_id)
        if name == "launch":
            p.add_argument("--baseline-only", action="store_true")
            p.add_argument("--mock", choices=("mlp", "smollm", "smollm-decode"))
            p.add_argument("--layers", type=int, default=2)
            p.add_argument("--tokens", type=int, nargs="+")
            p.add_argument("--rounds", type=int, default=1)
            p.add_argument("--generation", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "deploy":
        deploy(config)
    elif args.command == "launch":
        launch(config, args.config, args.run_id, args.baseline_only, args.mock, args.layers, args.tokens, args.rounds, args.generation)
    elif args.command == "collect":
        collect(config, args.run_id)
    elif args.command == "publish":
        publish(config, args.run_id)
    else:
        folder = f"{config['remote_repo']}/results/{args.run_id}"
        remote(config, f"cat {shlex.quote(folder + '/status.json')}; tail -20 {shlex.quote(folder + '/launcher.log')}")


if __name__ == "__main__":
    main()
