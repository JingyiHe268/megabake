from __future__ import annotations

import torch


def mlp(source, gate_weight, up_weight, down_weight):
    return (
        torch.nn.functional.silu(source @ gate_weight.T)
        * (source @ up_weight.T)
    ) @ down_weight.T
