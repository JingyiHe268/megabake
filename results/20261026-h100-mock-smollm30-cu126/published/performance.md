# MOCK smollm forward performance

Random weights, seed 20261026, rows 4, layers 30, bfloat16, 20 samples per variant.

Profile: experimental CUDA 12; not the upstream cu130 environment. Torch 2.14.0+cu126; CuTe 4.8.0; GPU NVIDIA H100 80GB HBM3.

Timing scope: synchronized model-forward calls; SmolLM uses a prebuilt causal mask; no SGLang engine comparison. No SGLang engine comparison or pretrained-model accuracy claim.

Correctness relative L2: MegaBake 0, Inductor 0.0162892. MegaBake workload kernels: 1. Execution commit: `dc5d8dc38f9d24b2f59a63981e0411d2cdf8febb`.

| Variant | Median forward latency (ms) |
| --- | ---: |
| eager | 20.066410 |
| eager_cuda_graph | 3.601586 |
| inductor | 5.483957 |
| megabake | 2.342902 |
