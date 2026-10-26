# MOCK mlp forward performance

Random weights, seed 20261026, rows 4, layers None, bfloat16, 20 samples per variant.

Profile: experimental CUDA 12; not the upstream cu130 environment. Torch 2.14.0+cu126; CuTe 4.8.0; GPU NVIDIA H100 80GB HBM3.

Timing scope: synchronized model-forward calls; no SGLang engine comparison. No SGLang engine comparison or pretrained-model accuracy claim.

Correctness relative L2: MegaBake 0, Inductor 0.0033916. MegaBake workload kernels: 1. Execution commit: `ab5cba082b33855841fc0787cf5fdac5347f2749`.

| Variant | Median forward latency (ms) |
| --- | ---: |
| eager | 0.193251 |
| eager_cuda_graph | 0.031309 |
| inductor | 0.328079 |
| megabake | 1.521561 |
