# MOCK smollm forward performance

Random weights, seed 20261026, rows 4, layers 2, bfloat16, 20 samples per variant.

Profile: experimental CUDA 12; not the upstream cu130 environment. Torch 2.14.0+cu126; CuTe 4.8.0; GPU NVIDIA H100 80GB HBM3.

Timing scope: synchronized model-forward calls; SmolLM uses a prebuilt causal mask; no SGLang engine comparison. No SGLang engine comparison or pretrained-model accuracy claim.

Correctness relative L2: MegaBake 0, Inductor 0.00721429. MegaBake workload kernels: 1. Execution commit: `7e5cb1ba06101bd3432b9ab7d5d4cc0d819bcb39`.

| Variant | Median forward latency (ms) |
| --- | ---: |
| eager | 2.230044 |
| eager_cuda_graph | 0.315759 |
| inductor | 0.532445 |
| megabake | 2.027297 |
