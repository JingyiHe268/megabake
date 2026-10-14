from __future__ import annotations

import pytest
import torch

from megabake import CompilerError, make_backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 23 requires an SM90 CUDA device",
)


class VisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(6, 2, 3, 3, device="cuda")
        )
        self.bias = torch.nn.Parameter(torch.randn(6, device="cuda"))

    def forward(self, source, residual, index):
        conv = torch.nn.functional.conv2d(
            source,
            self.weight,
            self.bias,
            stride=(2, 1),
            padding=(1, 2),
            dilation=(2, 1),
            groups=2,
        )
        conv = torch.relu(conv)
        pooled = torch.nn.functional.avg_pool2d(
            conv, (2, 2), stride=(1, 1), padding=(1, 1)
        )
        return torch.gather(pooled, 1, index) + residual


def test_task23_compiles_vision_block_with_generic_bodies(tmp_path):
    torch.manual_seed(23)
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        model = VisionBlock().eval()
        source = torch.randn(2, 4, 9, 10, device="cuda")
        residual = torch.randn(2, 6, 5, 13, device="cuda")
        index = torch.randint(0, 6, (2, 6, 5, 13), device="cuda")
        backend = make_backend(
            compile_kernel=True,
            cache=False,
            dump_dir=tmp_path,
            scheduler="phase",
        )
        optimized = torch.compile(model, backend=backend, fullgraph=True)

        with torch.no_grad():
            actual = optimized(source, residual, index)
            expected = model(source, residual, index)

            invalid_index = index.clone()
            invalid_index[0, 0, 0, 0] = 6
            with pytest.raises(CompilerError, match="outside \\[0, 6\\)"):
                optimized(source, residual, invalid_index)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        context = backend.megabake_contexts[-1]
        coverage = context.requirements["generic_coverage"]
        assert context.requirements["execution"] == "megakernel"
        assert coverage["semantic_support"]["unresolved"] == 0
        assert "generic/generic_contraction" in coverage["body_names"]
        assert "generic/generic_reduction" in coverage["body_names"]
        assert coverage["workspace_bytes"] >= 0
        assert any(value is not None for value in coverage["performance"].values())
    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled
