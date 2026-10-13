# MegaBake Task 20 performance story

This report consolidates the final Task 20 measurement in `artifacts/task20/mlp.json`, the Chrome traces under `artifacts/task20/traces/`, and the historical Task 16–19 artifacts. All times are microseconds unless explicitly marked otherwise.

> **Correction, 2026-10-13.** A later profile of the warm call showed that the time outside the kernel is Python binding work, not launch or scheduling. The real launch takes about 30 us. The affected statements below are corrected in place. See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for the attribution and next steps.
>
> No warm number in this document includes compilation. The complete-call numbers do include per-call Python binding, so they describe the wrapper as well as the kernel. Judge the kernel by launch plus compute only, measured with `benchmarks/isolated.py`.

## 1. Executive summary

MegaBake now compiles the complete gated MLP to **one cooperative CUDA kernel**, preserves the eager output within a measured relative L2 error of **0.0**, and automatically chooses between the phase and region-event schedules by measured synchronized complete-call latency.

| Batch | Best phase candidate | Best event candidate | Event improvement | Final 32-sample MegaBake complete call |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 352.01 | 297.78 | -15.4% | 367.67 |
| 256 | 661.08 | 494.54 | -25.2% | 536.34 |

The event schedule is a real improvement, but G1 remains **hold** because the complete MegaBake call is still much slower than the strongest baseline:

| Batch | CUDA Graph | Eager | Inductor | MegaBake | MegaBake / CUDA Graph |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 29.39 | 70.04 | 122.72 | 367.67 | 12.51x |
| 256 | 35.68 | 84.84 | 136.76 | 536.34 | 15.04x |

The short version of the attribution is:

1. **Phase restart/barrier loss was real.** Region events remove full-grid barriers between locally independent regions and improve candidate medians by 15.4% and 25.2%.
2. **Batch 4 is still dominated by per-call Python binding work.** The MegaBake kernel executes for only 20.48 us, but the synchronized complete call is 367.67 us.
3. **Batch 256 is dominated by mixed-body throughput after overlap.** The one kernel executes for 176.92 us, while CUDA Graph's five kernels sum to only 15.74 us.
4. **CUDA Graph is currently the strongest baseline** because replay removes almost all host launch cost and preserves the well-tuned native GEMM/elementwise mixture.
5. **MegaBake cannot yet be captured in a CUDA Graph.** The phase and event runtimes explicitly reject stream capture for cooperative launches, so there is no “MegaBake + CUDA Graph” measurement in this report.

## 2. Workload

The benchmark uses `benchmarks/workloads.py::mlp`:

```python
silu(source @ gate_weight.T) * (source @ up_weight.T) @ down_weight.T
```

Concrete dimensions:

| Tensor | Shape |
| --- | --- |
| `source` | `[batch, 576]` |
| `gate_weight` | `[1536, 576]` |
| `up_weight` | `[1536, 576]` |
| `down_weight` | `[576, 1536]` |
| intermediate gate/up values | `[batch, 1536]` |
| final output | `[batch, 576]` |

The benchmark function returns one final tensor. This is narrower than the integration-test `CompleteMLP` fixture, which also returns gate, activated, and fused intermediates.

Inputs and weights are `bfloat16` CUDA tensors. The seed is 16. The source is divided by `sqrt(576)`, and each weight is divided by the square root of its input dimension. Gradients are disabled. The eager result is the numerical reference.

Measured relative L2 errors:

| Batch | MegaBake | CUDA Graph | Inductor |
| ---: | ---: | ---: | ---: |
| 4 | 0.0 | 0.0 | 0.003555 |
| 256 | 0.0 | 0.003580* | 0.003580 |

*The CUDA Graph and Inductor errors happened to be identical at batch 256 in the retained report. CUDA Graph is bit-exact with eager at batch 4 and is expected to execute the same captured eager kernels; the tiny Inductor difference comes from its fused/generated kernels.

## 3. Hardware and software

- GPU: **NVIDIA H200 MIG 3g.71gb**
- Compute capability: **SM90 / 9.0**
- Multiprocessors: **60**
- Visible memory: **71,424 MiB**
- L2 cache: **29 MiB**
- GPU UUID: `de3ecfb1-ba9b-9ed1-49f3-5010912e5f4c`
- PyTorch: **2.14.0+cu130**
- MegaBake target: `sm90a`

The device exposes 60 SMs. The selected cooperative kernel can have at most 60 co-resident workers because it uses one resident CTA per SM.

## 4. What each mode means

| Mode | Description | CUDA Graph? | Kernel count |
| --- | --- | --- | ---: |
| Eager | Direct PyTorch execution of the MLP | No | 5 |
| CUDA Graph | The eager MLP captured and replayed as one graph | Yes | 5 |
| Inductor | `torch.compile(..., fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")` | No | 4 |
| MegaBake | Complete FX graph compiled to one generated CuTe cooperative kernel | No | 1 |

Important details:

- The CUDA Graph baseline captures the **eager** function. It uses the same five native kernels but removes per-kernel host launch work during replay.
- Inductor is deliberately configured with `max-autotune-no-cudagraphs`, so it is a non-CUDA-Graph compiled reference.
- MegaBake is not CUDA-Graph-captured because `execute_phase_plan` and `execute_event_plan` reject active stream capture for cooperative kernels.
- The CUDA Graph trace reports `launch_event_count = 0`; graph replay does not expose the five individual `LaunchKernel` runtime calls, but the profiler still records the five GPU kernels.

## 5. Timing definitions

### Warm synchronized complete call

`benchmarks/measure.py::event_samples` records:

1. Host `perf_counter` start.
2. CUDA start event.
3. One function invocation.
4. CUDA end event.
5. Synchronization on the end event.
6. Host `perf_counter` stop.

The reported **complete-call time** includes host argument handling, launch work, device execution, and synchronization. This is the primary G1 metric.

### Device event span

The same function records CUDA events before and after invocation. This span includes device-visible idle time caused by late host launch preparation; it is not only kernel execution.

### Kernel duration

For each mode, one warmed invocation is profiled with `torch.profiler`. `kernel_duration_us` is the sum of profiler-reported kernel durations.

### Kernel span and launch-to-last-kernel span

- `kernel_span_us` is from the first GPU kernel start to the last GPU kernel end.
- `first_launch_to_last_kernel_us` is from the first profiler launch/runtime event to the last GPU kernel end.

These trace-derived spans are useful for structure, but the profiled call is not identical to the unprofiled CUDA-event medians. Profiling can add host launch delay, especially for eager and Inductor. Kernel durations remain the most reliable trace quantities.

### Sample counts

Each final warm table has **32 complete-call samples** and **32 CUDA-event samples**. The report retains every raw sample in `mlp.json`.

Plan-search candidates use the default search timing sample count of 8, followed by one verification invocation. This explains why a candidate median can differ from the later independent 32-sample final measurement.

## 6. First-call and compilation costs

These costs are not comparable to the warm table. The MegaBake first call includes FX capture, reference execution, plan search, compilation, candidate correctness checks, and candidate timing.

### Batch 4

| Cost | Time |
| --- | ---: |
| Eager first call | 82.008 ms |
| CUDA Graph capture | 5.581 ms |
| CUDA Graph first replay | 132.844 us |
| Inductor first call | 1.488 s |
| MegaBake backend construction | 107.561 us |
| Dynamo wrapper construction | 424.020 us |
| MegaBake search/compile | 23.762 s |
| MegaBake first host call | 23.902 s |

### Batch 256

| Cost | Time |
| --- | ---: |
| Eager first call | 4.554 ms |
| CUDA Graph capture | 2.221 ms |
| CUDA Graph first replay | 92.918 us |
| Inductor first call | 44.938 ms |
| MegaBake backend construction | 33.505 us |
| Dynamo wrapper construction | 349.624 us |
| MegaBake search/compile | 57.296 s |
| MegaBake first host call | 57.415 s |

The backend and Dynamo wrapper construction are tiny. The dominant first-call cost is MegaBake's measured search and CuTe compilation. Automatic scheduler selection searches phase and event variants; each search has the default compilation budget.

## 7. Warm complete-call results

### Batch 4

| Mode | Complete call | CUDA event span | Kernel sum | Kernel span | Launch-to-last-kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 70.04 | 58.56 | 13.18 | 1,459.89 | 1,487.14 |
| CUDA Graph | 29.39 | 17.89 | 13.21 | 14.33 | 14.33 |
| Inductor | 122.72 | 109.17 | 11.26 | 222.37 | 246.55 |
| MegaBake | 367.67 | 353.76 | 20.48 | 20.48 | 302.10 |

The eager and Inductor trace spans are much larger than their unprofiled CUDA-event spans because profiler instrumentation changes host launch pacing. They should not be read as warm eager latency.

### Batch 256

| Mode | Complete call | CUDA event span | Kernel sum | Kernel span | Launch-to-last-kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 84.84 | 71.68 | 16.09 | 569.09 | 938.84 |
| CUDA Graph | 35.68 | 22.90 | 15.74 | 16.89 | 16.89 |
| Inductor | 136.76 | 121.49 | 13.69 | 228.58 | 563.78 |
| MegaBake | 536.34 | 514.96 | 176.92 | 176.92 | 411.70 |

CUDA Graph is the only mode whose profiled kernel span is nearly equal to its kernel-duration sum. Replay enqueues the captured work together, so the GPU does not wait for five separate host launches.

## 8. Sample variability

### Batch 4 complete-call samples

| Mode | P10 | Median | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 68.54 | 70.04 | 84.60 | 67.93 | 253.71 |
| CUDA Graph | 28.95 | 29.39 | 35.81 | 28.86 | 57.65 |
| Inductor | 116.33 | 122.72 | 143.04 | 114.63 | 414.38 |
| MegaBake | 345.19 | 367.67 | 405.46 | 335.62 | 602.47 |

### Batch 4 CUDA-event-span samples

| Mode | P10 | Median | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 57.02 | 58.56 | 69.09 | 56.35 | 207.84 |
| CUDA Graph | 17.66 | 17.89 | 19.49 | 17.63 | 44.19 |
| Inductor | 103.17 | 109.17 | 129.73 | 101.73 | 390.05 |
| MegaBake | 330.53 | 353.76 | 388.83 | 320.77 | 588.48 |

### Batch 256 complete-call samples

| Mode | P10 | Median | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 81.33 | 84.84 | 97.82 | 80.45 | 275.18 |
| CUDA Graph | 35.36 | 35.68 | 37.93 | 34.81 | 71.04 |
| Inductor | 131.37 | 136.76 | 159.08 | 129.32 | 389.42 |
| MegaBake | 516.01 | 536.34 | 574.41 | 505.94 | 778.74 |

### Batch 256 CUDA-event-span samples

| Mode | P10 | Median | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 68.77 | 71.68 | 84.29 | 67.78 | 233.63 |
| CUDA Graph | 22.53 | 22.90 | 24.70 | 22.43 | 57.15 |
| Inductor | 116.51 | 121.49 | 144.26 | 114.78 | 374.82 |
| MegaBake | 500.67 | 514.96 | 557.98 | 490.05 | 762.50 |

The event-vs-phase improvements of 15.4% and 25.2% are larger than the observed run-to-run spread. The remaining gap to CUDA Graph is also far larger than that spread.

## 9. Kernel composition and profiler traces

### Batch 4 kernel durations

| Mode | Kernel sequence and durations |
| --- | --- |
| Eager | `nvjet` gate 3.551; PyTorch SiLU 1.312; `nvjet` up 3.167; PyTorch multiply 0.896; `nvjet` down 4.256 |
| CUDA Graph | `nvjet` gate 3.230; PyTorch SiLU 1.439; `nvjet` up 3.231; PyTorch multiply 0.992; `nvjet` down 4.319 |
| Inductor | `nvjet` gate 3.136; `nvjet` up 3.136; fused Triton multiply/SiLU 0.960; `nvjet` down 4.032 |
| MegaBake | one generated CuTe cooperative kernel 20.477 |

### Batch 256 kernel durations

| Mode | Kernel sequence and durations |
| --- | --- |
| Eager | `nvjet` gate 3.999; PyTorch SiLU 2.143; `nvjet` up 3.583; PyTorch multiply 1.568; `nvjet` down 4.799 |
| CUDA Graph | `nvjet` gate 3.551; PyTorch SiLU 2.271; `nvjet` up 3.583; PyTorch multiply 1.535; `nvjet` down 4.798 |
| Inductor | `nvjet` gate 3.552; `nvjet` up 3.551; fused Triton multiply/SiLU 2.079; `nvjet` down 4.511 |
| MegaBake | one generated CuTe cooperative kernel 176.921 |

### Kernel names

- Eager and CUDA Graph use two upstream `nvjet_sm90_tst_*` GEMM kernels for gate/up, a native vectorized SiLU kernel, a native vectorized multiply kernel, and another `nvjet` GEMM for the down projection.
- Inductor uses two `nvjet` GEMMs, one `triton_poi_fused_mul_silu_0` kernel, and one down-projection `nvjet` GEMM.
- MegaBake uses one generated kernel named `kernel_cutlass_megabake_kernel_tensorptr...`.

### Trace files

| Batch | Mode | Trace |
| ---: | --- | --- |
| 4 | Eager | `artifacts/task20/traces/m4-eager-trace.json` |
| 4 | CUDA Graph | `artifacts/task20/traces/m4-cuda_graph-trace.json` |
| 4 | Inductor | `artifacts/task20/traces/m4-inductor-trace.json` |
| 4 | MegaBake | `artifacts/task20/traces/m4-megabake-trace.json` |
| 256 | Eager | `artifacts/task20/traces/m256-eager-trace.json` |
| 256 | CUDA Graph | `artifacts/task20/traces/m256-cuda_graph-trace.json` |
| 256 | Inductor | `artifacts/task20/traces/m256-inductor-trace.json` |
| 256 | MegaBake | `artifacts/task20/traces/m256-megabake-trace.json` |

## 10. Scheduler search and the Task 20 change

The compiler measures the same legal body/worker candidates with two schedules:

1. **Phase schedule:** ordered regions with full-grid barriers at true data boundaries.
2. **Region-event schedule:** GPU-scope release/acquire event counters that let locally independent regions overlap instead of restarting the whole grid at every phase boundary.

The public backend defaults to `scheduler="auto"`, which selects the lower median synchronized complete call. Explicit `phase` and `region-events` modes are also available.

### Batch 4 candidate comparison

| Schedule | Best workers | Median complete call | Verification call | Registers/thread | Dynamic shared memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase | 12 | 352.01 | 340.10 | 93 | 214,016 |
| Region events | 24 | 297.78 | 275.87 | 91 | 214,016 |

Selected: **region events with 24 workers**.

### Batch 256 candidate comparison

| Schedule | Best workers | Median complete call | Verification call | Registers/thread | Dynamic shared memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase | 60 | 661.08 | 638.71 | 93 | 214,016 |
| Region events | 60 | 494.54 | 467.09 | 91 | 214,016 |

Selected: **region events with 60 workers**.

Both schedules have:

- 0 bytes static shared memory reported by the compiled kernel
- 0 bytes local memory per thread
- one active block per SM
- a maximum of 60 cooperative workers

The retained `mlp.json` was generated immediately before scheduler labels were added to candidate descriptions. In that file, the first candidate group is phase and the second is region-event. Later benchmark runs label descriptions with `scheduler=phase;` or `scheduler=region-events;`.

Compiler-owned event counters skip redundant per-event host validation on warm calls. The public `execute_event_plan` API still validates externally supplied counters; this is a compiler-internal trust boundary because the compiler owns and epochs the counter tensor.

## 11. Selected MegaBake body mixture

The selected plan uses the `g0_range_64x128` provider for all three GEMMs:

1. gate projection
2. up projection
3. down projection

The elementwise chain remains explicit generic bodies:

- BF16/FP32 conversion
- negation
- exponent
- addition
- division
- another type conversion
- final multiply

The region matcher also records a paired-GEMM candidate, but it is rejected because the paired provider and required multi-store epilogue are not implemented. The report retains that rejection in `provider_rejections`.

This matters for interpretation: MegaBake currently fuses scheduling into one kernel but does **not** fuse the complete gate/up/activation/multiply chain into a paired GEMM epilogue.

## 12. Where the MegaBake time goes

### Batch 4

| Quantity | Value |
| --- | ---: |
| Complete call | 367.67 |
| CUDA event span | 353.76 |
| Kernel duration | 20.48 |
| Complete call minus kernel duration | 347.19 |
| Event span minus kernel duration | 333.28 |
| Host complete call minus event span | 13.91 |
| Profiler launch-to-last-kernel minus kernel duration | 281.62 |

The kernel itself is only 20.48 us. The dominant loss is approximately 333 us of host-side argument binding before the launch is enqueued, not arithmetic. The CUDA event span includes it because the start event completes at once.

### Batch 256

| Quantity | Value |
| --- | ---: |
| Complete call | 536.34 |
| CUDA event span | 514.96 |
| Kernel duration | 176.92 |
| Complete call minus kernel duration | 359.41 |
| Event span minus kernel duration | 338.04 |
| Host complete call minus event span | 21.38 |
| Profiler launch-to-last-kernel minus kernel duration | 234.78 |

Batch 256 has both problems:

- roughly 338 us of host-side argument binding before the launch;
- a much larger 176.92 us mixed-body kernel, compared with 15.74 us of total CUDA Graph kernel work.

Therefore, reducing launch overhead alone would not make batch 256 competitive. The full body mixture also needs better throughput, likely through paired GEMM epilogues, fewer elementwise phases, and better work partitioning.

## 13. Why CUDA Graph is so strong

| Batch | CUDA Graph complete call | Event span | Kernel sum | Complete call minus kernel sum |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 29.39 | 17.89 | 13.21 | 16.18 |
| 256 | 35.68 | 22.90 | 15.74 | 19.94 |

CUDA Graph keeps the highly tuned native GEMM kernels and removes almost all host launch overhead. Its five kernels execute back-to-back:

- batch-4 kernel span: 14.33 us for 13.21 us of kernel work;
- batch-256 kernel span: 16.89 us for 15.74 us of kernel work.

This is the correct strongest baseline for the current G1 question. A megakernel is not useful merely because it has one launch; it must beat the complete graph-replayed execution of tuned native kernels.

## 14. Historical MegaBake progression

| Artifact / stage | Batch 4 complete call | Batch 4 kernel | Batch 256 complete call | Batch 256 kernel | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Task 16 | 414.42 | 83.98 | 727.71 | 348.06 | First complete one-kernel MLP |
| Task 17 | 509.91 | 138.91 | 787.02 | 333.87 | Fusion-candidate era; batch-4 regression |
| Task 18 | 399.92 | 61.41 | 741.83 | 334.66 | Measured plan search |
| Task 19 default | 417.64 | 130.46 | 732.69 | 376.81 | Event primitives not wired into public backend |
| Task 19 phase rerun | 356.82 | 56.12 | 717.62 | 361.70 | Phase reference |
| Task 19 temporary events | 380.31 | 23.43 | 572.61 | 175.94 | Opt-in temporary wiring |
| Task 20 phase reproduction | 397.53 | 61.47 | 756.76 | 356.07 | Immediately before final scheduler integration |
| Task 20 final | 367.67 | 20.48 | 536.34 | 176.92 | Automatic measured phase/event selection |

Compared with the Task 20 phase reproduction, the final 32-sample complete calls improve by:

- **7.5%** at batch 4;
- **29.1%** at batch 256.

The same-run controlled candidate comparison is the cleaner scheduler measurement: **15.4%** and **25.2%**.

## 15. Artifacts and reproduction

Primary command:

```sh
.venv/bin/python benchmarks/run.py \
  --output artifacts/task20/mlp.json \
  --traces artifacts/task20/traces \
  --batches 4 256 \
  --scheduler auto
```

Primary files:

- Full report and raw samples: `artifacts/task20/mlp.json`
- Baseline-only report: `artifacts/task20/mlp_baseline.json`
- Chrome traces: `artifacts/task20/traces/`
- Short evidence summary: `artifacts/task20/README.md`
- Historical phase reproduction: `artifacts/task20/repro.json`

The benchmark was run on the H200 MIG instance described above. Later in the session, NVML stopped exposing the GPU, so the final labeled-report rerun could not be completed; the retained successful `mlp.json` above is the source for every number in this document.

## 16. Decision and next work

**G1 decision: hold for both regimes.**

The event scheduler is a real, measured improvement and is now the default automatic choice, but the complete correct one-kernel MLP is still 12.51x slower than CUDA Graph replay at batch 4 and 15.04x slower at batch 256.

Highest-value next steps:

1. Remove the roughly 333–338 us of per-call binding work: build launch arguments once and reuse them.
2. Implement a paired gate/up GEMM body with a multi-store epilogue, while preserving secondary outputs.
3. Fuse the activation/multiply chain into the GEMM epilogue instead of executing separate generic elementwise phases.
4. Fuse the SiLU chain into one elementwise region before attempting the paired GEMM; it is the cheaper change and removes most of the batch-256 kernel time.
5. Only after those changes, consider CUDA Graph capture support for the MegaBake cooperative launch.
