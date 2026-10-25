# 20261025-h100-smollm-preflight-v2

Status: **blocked**

Execution commit: `0ba8b015bae17a026b0fd1390f0b5900d7cee3e3`

Comparison: `not_comparable_across_harnesses`. No cross-framework speedup claimed.

- model config missing: /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/models/SmolLM-135M/config.json
- MegaBake requires PyTorch 2.14.x
- MegaBake requires CuTe DSL 4.8.0
- MegaBake CUDA 13 runtime/tensor execution has not passed

See [report](report.html), [status](status.json), [environment](environment.json).
