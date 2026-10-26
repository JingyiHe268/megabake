"""Standard-library tests for transport guards and evidence interpretation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report
import run_node
import workflow
import publication


class WorkflowTests(unittest.TestCase):
    def test_run_ids_reject_paths_and_shell(self):
        for name in ("../escape", "a/b", ";id", "$(id)", "", "a b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                workflow.validate_id(name)

    def test_run_id_accepts_dated_name(self):
        self.assertEqual(workflow.validate_id("20261025-h100-preflight"), "20261025-h100-preflight")

    def test_missing_model_is_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(run_node.fingerprint_model(temporary)["ready"])

    def test_missing_weight_shard_is_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text("{}")
            (folder / "model-1.safetensors").write_bytes(b"test")
            (folder / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
            self.assertFalse(run_node.fingerprint_model(folder)["ready"])

    def test_blocked_report_never_invents_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status.json").write_text(json.dumps({"state": "blocked", "blockers": ["<missing model>"]}))
            path = report.render(folder)
            text = path.read_text()
            self.assertIn("No successful SGLang measurements", text)
            self.assertIn("&lt;missing model&gt;", text)
            self.assertNotIn("<missing model>", text)
            self.assertTrue((folder / "checksums.json").is_file())

    def test_manifest_only_contains_publishable_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status.json").write_text('{"state":"blocked"}')
            (folder / "launcher.pid").write_text("1234")
            (folder / "trace.nsys-rep").write_text("large trace")
            report.render(folder)
            hashes = json.loads((folder / "checksums.json").read_text())
            self.assertNotIn("launcher.pid", hashes)
            self.assertNotIn("trace.nsys-rep", hashes)
            self.assertIn("status.json", hashes)

    def test_report_converts_seconds_to_milliseconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status.json").write_text('{"state":"complete"}')
            (folder / "sglang-cudagraph-on-1.jsonl").write_text(
                '{"prefill_latency":0.001,"median_decode_latency":0.002}\n')
            (folder / "sglang-cudagraph-on-1.status.json").write_text('{"state":"complete"}')
            text = report.render(folder).read_text()
            self.assertIn("1.0000", text)
            self.assertIn("2.0000", text)

    def test_deploy_refuses_dirty_source(self):
        with patch.object(workflow, "output", return_value=" M file.py"), self.assertRaises(SystemExit):
            workflow.deploy({})

    def test_mock_launch_requires_an_explicit_compatibility_profile(self):
        with self.assertRaises(SystemExit):
            workflow.launch({}, Path("unused"), "mock-test", mock="mlp")

    def test_subprocess_tools_use_the_selected_venv(self):
        base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/existing"}
        env = run_node.tool_environment(base, "/env/bin/python", "/cuda", "/source")
        self.assertEqual(env["PATH"], "/env/bin:/cuda/bin:/usr/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/cuda/lib64:/existing")
        self.assertEqual(base["PATH"], "/usr/bin")

    def test_publication_excludes_screenshots_logs_and_failed_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{"model_id":"test/model","model_revision":"abc"}')
            for name, state in [("on", "complete"), ("off", "complete"), ("bad", "failed")]:
                stem = f"sglang-cudagraph-{name}-1"
                (folder / f"{stem}.jsonl").write_text(json.dumps({
                    "batch_size": 1, "input_len": 4, "output_len": 32,
                    "prefill_latency": .001, "median_decode_latency": .002}) + "\n")
                (folder / f"{stem}.status.json").write_text(json.dumps({"state": state}))
            for name in ("report.png", "blockers.png", "error.log"):
                (folder / name).write_text("private diagnostic")
            files = publication.export(folder)
            self.assertEqual({p.suffix for p in files}, {".csv", ".md", ".svg"})
            self.assertTrue(all(p.parent.name == "published" for p in files))
            self.assertNotIn("bad", (folder / "published/performance.csv").read_text())
            self.assertNotIn("private diagnostic", ''.join(p.read_text() for p in files))

    def test_no_measurements_refuses_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{}')
            (folder / "blockers.png").write_text("failure screenshot")
            with self.assertRaises(ValueError):
                publication.export(folder)
            self.assertFalse((folder / "published").exists())

    def test_nonfinite_measurements_are_never_plotted(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{}')
            (folder / "sglang-cudagraph-on-1.status.json").write_text('{"state":"complete"}')
            (folder / "sglang-cudagraph-on-1.jsonl").write_text(
                '{"prefill_latency":NaN,"median_decode_latency":0.001}\n')
            with self.assertRaises(ValueError):
                publication.export(folder)

    def test_successful_upstream_measurements_export_as_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text(json.dumps({
                "model_id": "test/model", "model_revision": "abc", "batch_size": 1, "prompt_tokens": 4}))
            (folder / "megabake-upstream").mkdir()
            (folder / "megabake-upstream.status.json").write_text('{"state":"complete"}')
            (folder / "megabake-upstream/report.json").write_text(json.dumps({
                "prefill": {"megabake": {"complete_call_us": 2000}},
                "decode": {"megabake": {"complete_call_us": 1000}}}))
            files = publication.export(folder)
            self.assertEqual({p.suffix for p in files}, {".csv", ".md"})
            text = (folder / "published/performance.md").read_text()
            self.assertIn("2.0000", text)
            self.assertIn("1.0000", text)

    def test_mock_export_requires_correctness_and_complete_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status.json").write_text('{"state":"complete"}')
            data = {"state": "complete", "mock_weights": True, "workload": "mlp", "rows": 4,
                    "layers": None, "samples": 2, "dtype": "bfloat16", "seed": 1,
                    "relative_l2": .001, "inductor_relative_l2": .001, "workload_kernel_count": 1,
                    "torch": "test", "cute": "test", "gpu": "test", "profile": "test",
                    "execution_commit": "test", "measurements": {
                        name: {"median_call_us": 2, "samples_us": [1, 3]}
                        for name in ("eager", "eager_cuda_graph", "inductor", "megabake")}}
            path = folder / "mock-forward.json"
            path.write_text(json.dumps(data))
            files = publication.export(folder)
            self.assertEqual({p.suffix for p in files}, {".csv", ".md", ".svg"})
            self.assertIn("Random weights", (folder / "published/performance.md").read_text())
            self.assertNotIn("No MegaBake comparison", (folder / "published/mock_forward_ms.svg").read_text())
            data["relative_l2"] = .2
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                publication.export(folder)




class SweepPublicationTests(unittest.TestCase):
    def make_point(self, root, tokens, commit="abc"):
        point = root / f"tokens-{tokens}"
        point.mkdir()
        data = dict(state="complete", mock_weights=True, workload="smollm", layers=30,
                    rows=tokens, dtype="bfloat16", seed=1, torch="test", cute="test", gpu="test",
                    profile="experimental", timing_scope="forward", execution_commit=commit,
                    rounds=3, samples_per_round=2, samples=6, relative_l2=0.0,
                    inductor_relative_l2=0.01, workload_kernel_count=1,
                    measurements={name: dict(samples_us=[1., 3.] * 3, round_medians_us=[2.] * 3,
                                             median_call_us=2.)
                                  for name in ("eager", "eager_cuda_graph", "inductor", "megabake")})
        (point / "mock-forward.json").write_text(json.dumps(data))
        (point / "status.json").write_text('{"state":"complete"}')
        return point

    def test_sweep_excludes_failed_points_and_rejects_mixed_source(self):
        from sweep_publication import load_points
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sweep.json").write_text('{"tokens":[4,8,16],"rounds":3,"execution_commit":"abc"}')
            self.make_point(root, 4)
            self.make_point(root, 8)
            point = self.make_point(root, 16)
            (point / "status.json").write_text('{"state":"failed"}')
            self.assertEqual([p["rows"] for p in load_points(root)], [4, 8])
            (point / "status.json").write_text('{"state":"complete"}')
            data = json.loads((point / "mock-forward.json").read_text())
            data["execution_commit"] = "other"
            (point / "mock-forward.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_points(root)

    def test_sweep_rejects_summary_inconsistent_with_raw_samples(self):
        from sweep_publication import load_points
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sweep.json").write_text('{"tokens":[4,8],"rounds":3,"execution_commit":"abc"}')
            self.make_point(root, 4)
            point = self.make_point(root, 8)
            data = json.loads((point / "mock-forward.json").read_text())
            data["measurements"]["megabake"]["round_medians_us"] = [1., 1., 1.]
            (point / "mock-forward.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_points(root)


    def test_decode_publication_requires_validated_cache_and_exact_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            point = self.make_point(root, 128)
            data = json.loads((point / "mock-forward.json").read_text())
            data.update(mode="decode", workload="smollm-decode", batch_size=1, output_tokens=1,
                        context_length=128, cache_capacity=256, cache_state_validation=True,
                        cache_relative_l2={name: 0.01 for name in data["measurements"]})
            (point / "mock-forward.json").write_text(json.dumps(data))
            self.assertEqual(len(publication.export_mock(point)), 3)
            for field, value in (("cache_state_validation", False), ("cache_capacity", 128)):
                invalid = dict(data, **{field: value})
                (point / "mock-forward.json").write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    publication.export_mock(point)


if __name__ == "__main__":
    unittest.main()
