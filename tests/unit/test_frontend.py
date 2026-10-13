import pytest
import torch
from torch.fx import GraphModule

from megabake import make_backend, prepare_fx_graph


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, value):
        return torch.nn.functional.gelu(self.linear(value))


def test_make_backend_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported mode"):
        make_backend(mode="training")


def test_make_backend_rejects_invalid_scheduler():
    with pytest.raises(ValueError, match="scheduler must be"):
        make_backend(scheduler="tile-events")


def test_make_backend_defaults_to_measured_scheduler_selection():
    backend = make_backend()
    assert backend.megabake_options["scheduler"] == "auto"


def test_make_backend_rejects_training_inputs():
    model = Model().eval()
    value = torch.randn(2, 4, requires_grad=True)
    exported = torch.export.export(model, (value,))
    backend = make_backend()
    with pytest.raises(RuntimeError, match="inference only"):
        backend(exported.graph_module, (value,))


def test_prepare_fx_graph_remains_fx_and_skips_inductor_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inductor compilation must not be invoked")

    monkeypatch.setattr(
        torch._inductor.compile_fx, "compile_fx_forward", forbidden
    )

    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value) * 2

    model = Model()

    value = torch.randn(3, 4)
    exported = torch.export.export(model, (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        output_tree_spec=exported.call_spec.out_spec,
    )
    assert isinstance(prepared.graph, GraphModule)
    assert prepared.context.phase == "post_grad"
    with torch.no_grad():
        torch.testing.assert_close(prepared.forward(value), model(value))


def test_prepare_fx_graph_dumps_every_stage(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value) * 2

    value = torch.randn(3, 4)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        dump_dir=tmp_path,
        model_name="test-model",
        output_tree_spec=exported.call_spec.out_spec,
    )
    expected = {
        "phase_0_input_fx.txt",
        "phase_1_normalized.txt",
        "phase_2_pre_grad.txt",
        "phase_3_aot_inference.txt",
        "phase_4_prepared.txt",
        "phase_5_cache_check.txt",
        "phase_6_post_grad.txt",
    }
    dump_directory = tmp_path / "test-model"
    assert {path.name for path in dump_directory.iterdir()} == expected
    assert prepared.context.phase == "post_grad"


def test_torch_compile_returns_fx_reference_and_follows_parameters():
    torch.manual_seed(7)
    model = Model().eval()
    backend = make_backend()
    optimized = torch.compile(model, backend=backend, fullgraph=True)
    first = torch.randn(5, 4)
    second = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.allclose(optimized(first), model(first))
        assert torch.allclose(optimized(second), model(second))
        model.linear.weight.add_(1)
        assert torch.allclose(optimized(second), model(second))
    assert backend.megabake_contexts[-1].phase == "post_grad"
