from __future__ import annotations

from types import SimpleNamespace

import torch

from megabake import prepare_fx_graph
from megabake.main import main


def test_transformers_cli_dumps_all_graphs(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, value, weight, bias):
            return torch.nn.functional.linear(value, weight, bias)

    model = Model().eval()
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    input_ids = torch.randn(2, 4)
    exported = torch.export.export(model, (input_ids, weight, bias))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (input_ids, weight, bias),
        output_tree_spec=exported.call_spec.out_spec,
    )
    capture = SimpleNamespace(
        graph=prepared.graph,
        context=prepared.context,
        run=lambda value: model(value.float(), weight, bias),
    )
    monkeypatch.setattr(
        "megabake.main.make_backend", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "megabake.main.capture_transformer",
        lambda *args, **kwargs: capture,
    )

    exit_code = main(
        [
            "transformers",
            "--model-name",
            "fake-model",
            "--device",
            "cpu",
            "--dump-dir",
            str(tmp_path),
            "--dump-compute-graph",
            "--dump-lowering-graph",
            "--dump-region-graph",
        ]
    )

    output_directory = tmp_path / "fake-model"
    assert exit_code == 0
    assert (output_directory / "compute_graph.txt").is_file()
    assert (output_directory / "lowering_graph.txt").is_file()
    assert (output_directory / "region_graph.txt").is_file()
