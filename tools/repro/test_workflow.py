"""Standard-library tests for transport guards and evidence interpretation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report
import run_node
import workflow


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

    def test_subprocess_tools_use_the_selected_venv(self):
        base = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/existing"}
        env = run_node.tool_environment(base, "/env/bin/python", "/cuda", "/source")
        self.assertEqual(env["PATH"], "/env/bin:/cuda/bin:/usr/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/cuda/lib64:/existing")
        self.assertEqual(base["PATH"], "/usr/bin")


if __name__ == "__main__":
    unittest.main()
