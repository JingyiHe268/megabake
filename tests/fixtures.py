from __future__ import annotations

import torch


class CompleteMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.gate_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        self.up_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        self.down_weight = torch.nn.Parameter(
            torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
        )

    def forward(self, source: torch.Tensor):
        gate = torch.nn.functional.linear(source, self.gate_weight)
        up = torch.nn.functional.linear(source, self.up_weight)
        activated = torch.nn.functional.silu(gate)
        fused = activated * up
        output = torch.nn.functional.linear(fused, self.down_weight)
        return output, gate, activated, fused
