# G0 workload and measurement record

The first workload is the BF16 `HuggingFaceTB/SmolLM-135M` model at revision `1d461723eec654e65efdc40cf49301c89c0c92f4`. The post-grad graph captured from token IDs `[1, 2, 3, 4]` has SHA-256 `e691f1592bcebcc3c28fa8fdfd4ab3527dd07d939e7e517fa34a748f248213d5`. The capture path is [verify_smollm_fx.py](../../src/eager/verify_smollm_fx.py).

The target is an NVIDIA H200 `3g.71gb` MIG device, SM90, 60 SMs, 71,424 MiB, and 29 MiB L2. Cooperative launch is supported. The physical UUID is `GPU-9dea0b6d-ef2b-c5f3-8f4d-f2e6b5a59394`; the MIG UUID is `MIG-ad4a7a4d-729f-5f3e-93e0-90db9e483d39`.

Software pins are PyTorch `2.14.0+cu130` (`08187d9e0fba026dc8217405802ab5381dc88d90`), CUDA runtime 13.0, driver 595.58.03, NVCC 12.8.93, Transformers 5.18.0, CUTLASS Python package 4.2.0.0, and CuTe DSL 4.8.0. NVCC and the PyTorch runtime use different CUDA minor versions; the recorded CuTe JIT probes compiled and ran on this setup.

The model has hidden width 576, MLP width 1536, 30 layers, 9 query heads, 3 key/value heads, head width 64, and vocabulary size 49,152. Model inputs and projection outputs use BF16; the standalone GEMM probes use FP32 accumulation. Input and output matrices are contiguous row-major. Stored projection weights have shape `[N,K]`, stride `[K,1]`; the GEMM output has shape `[M,N]`, stride `[N,1]`.

| Projection | N | K | Weight size | Weight bytes |
| --- | ---: | ---: | --- | ---: |
| Query | 576 | 576 | `[576,576]` | 663,552 |
| Key | 192 | 576 | `[192,576]` | 221,184 |
| Value | 192 | 576 | `[192,576]` | 221,184 |
| Attention output | 576 | 576 | `[576,576]` | 663,552 |
| MLP gate | 1536 | 576 | `[1536,576]` | 1,769,472 |
| MLP up | 1536 | 576 | `[1536,576]` | 1,769,472 |
| MLP down | 576 | 1536 | `[576,1536]` | 1,769,472 |
| Vocabulary head | 49,152 | 576 | `[49152,576]` | 56,623,104 |

The first prefill fixture has M=4 and logits `[1,4,49152]`. Its stateful cache fixture has key and value tensors `[1,3,4,64]`, BF16, stride `(768,256,64,1)`. One decode token uses M=1 and advances both cache tensors to `[1,3,5,64]`. The larger prefill measurement cases use M=256 and M=1024. M=1 and M=4 exercise partial M tiles. The listed K and N widths are multiples of 64. Individual projection weights fit in L2, while the vocabulary matrix and the rotating-weight pool do not. The M=4 stream test rotates 38 MLP weight buffers (about 67.2 MB).

Standalone GEMM correctness requires finite BF16 output and relative L2 error below 0.005 against the FP32 accumulation reference. The complete MLP baseline requires relative L2 below 0.01. The mixed-body fixture checks every intermediate and output at relative L2 below 0.01. Model-output validation is declared at relative L2 below 0.01; cache shape, stride, and advancement are checked separately.

The baseline command is:

```sh
.venv/bin/python experiments/design_2026_10_08/mlp_baseline.py --output experiments/g0_2026_10_09/mlp_baseline.json
```

The GEMM probes use `probe.py --quick --single-m 1|4|256 --resources` for the up-projection shape, `--shape 4,576,1536 --resources` for down, and `--quick --streaming --single-m 4 --resources` for rotating weights. Each record includes setup/compile time, first-call CUDA-event span and host time, raw warm GPU samples, synchronized host-call samples, numerical error, launch count, and compiled resources when requested. Event spans include host submission gaps; the graph-replay samples measure steady GPU time. The MLP report also records input/reference setup, the compile-plus-first-call host span, the first post-compile event span and host call, warm GPU samples, synchronized host-call samples, error, and profiler launch counts.

The mixed-body latency samples include 100,000 deliberate dependent global reads by one CTA to stress delayed-producer synchronization; they are not a performance claim for an undelayed mixed kernel.

Saved evidence is in this directory: [mlp_baseline.json](mlp_baseline.json), the `gemm_*.json` and `down_m4.json` files, [grid_barrier.json](grid_barrier.json), [resident_range.json](resident_range.json), [mixed_body.json](mixed_body.json), [mixed_shape_m1.json](mixed_shape_m1.json), [mixed_shape_m256.json](mixed_shape_m256.json), [generated_typed_bodies.py](generated_typed_bodies.py), and [mixed_kernel_trace.json](mixed_kernel_trace.json). The final mixed-body source passed memcheck and synccheck with zero errors and racecheck with zero hazards at 60 workers; the repeated M=1 and M=256 shape-rebind memcheck runs are also clean. [g0_status.json](g0_status.json) records the completed G0 gates and the graph-replay limitation.
