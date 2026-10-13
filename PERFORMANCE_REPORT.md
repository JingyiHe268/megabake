# MegaBake performance report

**Date: 2026-10-13. Branch: `v4-impl` at `2d6e1db`. Workload: the gated MLP in `benchmarks/workloads.py`.**

This report states how the current one-kernel MLP performs, where the time goes, what is wrong, and what to do next. It supersedes the attribution in [performance_story.md](performance_story.md), which keeps the raw Task 20 tables.

All times are microseconds. Each number is marked by where it came from:

- **Task 20** — `artifacts/task20/mlp.json`, 32 samples, batches 4 and 256.
- **Diag run** — a `benchmarks/run.py` run on the same GPU, batches 1 and 256.
- **cProfile** — 200 warm calls of the compiled MLP at batch 1 under `cProfile`.
- **Estimate** — reasoning from the code or from other numbers; not measured.

GPU: NVIDIA H200 MIG `3g.71gb`, SM90, 60 SMs, 29 MiB L2. PyTorch 2.14.0+cu130, CuTe DSL 4.8.0.

## 1. Verdict

**The complete MegaBake call is 12.5x slower than CUDA Graph replay at batch 4 and 15.0x slower at batch 256. G1 is on hold.**

The loss has two independent parts:

| Part | Batch 4 | Batch 256 | Cause |
| --- | ---: | ---: | --- |
| Outside the kernel | ~347 | ~359 | Python binding code that runs on every call |
| Kernel above native kernel work | ~7 | ~161 | Seven unfused elementwise phases with a scalar body |

Neither part comes from the architecture's central idea. The GEMM body is close to native, the launch itself is cheap, and the event schedule works. Both parts are implementation.

One finding does change the project's premise. Against CUDA Graph replay, removing launches is worth about 1 µs on this MLP (section 6). The megakernel can win only through concurrency at small M and through fusion.

## 2. Three costs that must not be mixed

A MegaBake call has three separate costs. Only the third describes the kernel.

| Cost | When it is paid | Size | Included in |
| --- | --- | ---: | --- |
| Compilation: plan search and `cute.compile` | Once | 24–57 s | The first-call table only |
| Binding: Python that rebuilds and converts launch arguments | Every call | ~350 µs | Complete-call numbers |
| Launch plus compute: the driver launch and the kernel running | Every call | ~30 + 20 µs at batch 1 | Complete-call and isolated numbers |

Binding is not compilation. The kernel is already compiled; CuTe and the MegaBake wrapper repack pointers and shapes on each call.

**Compilation time is in no warm number in this report.** Every warm figure is taken after compilation and warm-up.

Use the numbers as follows:

- **Kernel verdict:** launch plus compute only. This decides whether the megakernel approach works. `benchmarks/isolated.py` measures it for every mode (section 8).
- **Wrapper to-do:** complete-call time. The 12.5x and 15.0x ratios describe what a caller gets today. They are dominated by binding, which has a known fix and says nothing about the kernel.
- **Kernel quality without launch:** profiler kernel duration. On this measure MegaBake is 1.55x slower at batch 4 and 11.24x slower at batch 256.

One asymmetry remains in the isolated comparison. The baselines launch through CUDA Graph replay and MegaBake launches through a direct CuTe call. Their fixed launch overheads can differ by a few microseconds, so the script reports kernel duration alongside launch plus compute.

## 3. Measured results

### Complete call, warm (Task 20)

These include binding, launch, compute, and the wait. They exclude compilation.

| Batch | CUDA Graph | Eager | Inductor | MegaBake | MegaBake / CUDA Graph |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 29.39 | 70.04 | 122.72 | 367.67 | 12.51x |
| 256 | 35.68 | 84.84 | 136.76 | 536.34 | 15.04x |

### Kernel work only (Task 20, profiler kernel durations)

These are compute only. No launch or binding is included.

| Batch | CUDA Graph (5 kernels) | Inductor (4 kernels) | MegaBake (1 kernel) | MegaBake / CUDA Graph |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 13.21 | 11.26 | 20.48 | 1.55x |
| 256 | 15.74 | 13.69 | 176.92 | 11.24x |

### Batch 1 (diag run)

| Mode | Complete call | Kernels | Kernel work |
| --- | ---: | ---: | ---: |
| CUDA Graph | 30.22 | 5 | not recorded |
| Eager | 74.08 | 5 | 13.8 |
| Inductor | 80.75 | 2 | 7.1 |
| MegaBake | 376.17 | 1 | 20.0 |

### Native kernel breakdown (Task 20)

| Kernel | Batch 4 | Batch 256 |
| --- | ---: | ---: |
| Gate GEMM (`nvjet`) | 3.23 | 3.55 |
| Up GEMM (`nvjet`) | 3.23 | 3.58 |
| SiLU | 1.44 | 2.27 |
| Multiply | 0.99 | 1.54 |
| Down GEMM (`nvjet`) | 4.32 | 4.80 |
| Inductor fused SiLU·multiply | 0.96 | 2.08 |

### First call (Task 20)

MegaBake search and compilation takes 23.8 s at batch 4 and 57.3 s at batch 256. Inductor's first call takes 1.5 s and 45 ms.

## 4. Where the time outside the kernel goes

**It is Python, not the launch.** The cProfile run shows the following per call. cProfile roughly doubles Python-heavy code (800 µs profiled against 376 µs unprofiled), so read the column as proportions.

| Where | Profiled µs per call | Share |
| --- | ---: | ---: |
| `_RuntimeState.arguments()` | ~300 | 37% |
| CuTe `generate_execution_args` | ~80 | 10% |
| `_refresh_values` / `rebind` | ~70 | 9% |
| Rest of `forward` and `execute_event_plan` | ~200 | 25% |
| Dynamo and AOT wrappers | ~110 | 14% |
| CuTe `run_compiled_program` (the real launch) | ~30 | 4% |

Details from the same profile:

- `from_dlpack` runs 4 times per call. One GEMM view group and one ordinary view miss their cache on every call. The likely cause is the output tensor: it is reallocated each call while the previous one is still referenced, so its address changes. This cause is inferred from the code, not confirmed.
- `_tensor_identity` runs 33 times and `_cached_view` 24 times per call. They re-check weights, workspace, work iterators, and the ready buffer, none of which change.
- CuTe converts all launch arguments on every call. Only the input and output addresses can change.

The cooperative launch is not the cost. `run_compiled_program` includes the host launch function, the TMA descriptor setup, and the driver launch, and it takes about 30 µs under the profiler.

The Task 20 report calls this a "device-visible launch/scheduling gap". The gap is device-visible only because the start event completes at once and the kernel is enqueued late.

## 5. Where the kernel time goes

### The selected plan

Both batches select region events with `g0_range_64x128` for all three GEMMs and seven separate generic elementwise regions:

```text
mm → convert → neg → exp → add → div → convert → (mm_1) → mul → mm_2
```

The six regions between `mm` and `mul` are one SiLU. Each is a full pass over the `[batch, 1536]` intermediate through global memory, with its own event.

### Batch 256: the elementwise body

MegaBake spends about 160 µs on work that native kernels do in 3.8 µs and Inductor's fused kernel does in 2.1 µs. That is roughly 40x.

Two causes compound:

- **Seven passes instead of one.** No fused candidate reaches the search. The chain builder in `regions.py` follows only single-use linear chains, and SiLU's FP32 input has two users (`neg` and `div`).
- **The body is scalar.** `emit_map_body` handles one BF16 element per thread per tile. Each element costs an int64 modulo and division and a global load of `work_iterator[tile]`, which is an `arange`.

The kernel is throughput-bound, not synchronization-bound: the best event candidate slows from 504 to 592 to 899 µs as workers drop from 60 to 30 to 15 (diag run).

### Batch 4: phase count

The kernel takes 20.5 µs against 13.2 µs of native kernel work. The three GEMMs account for about 10.8 µs natively, so the GEMM body is near parity and about 7 µs is spread over the seven extra phases and their events. This split is an estimate.

### The phase barrier

The phase schedule's barrier costs time linear in worker count. The Task 20 phase-to-event change shows it:

| Batch | Workers | Phase kernel | Event kernel | Barriers | Per barrier |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 12–24 | 61.47 | 20.48 | 9 | ~4.5 |
| 256 | 60 | 356.07 | 176.92 | 9 | ~20 |

In `grid_barrier`, one thread of CTA 0 issues one atomic compare-and-swap per worker in sequence. The selected plan avoids it by using events. The phase schedule remains the reference path and still carries the cost.

## 6. What the megakernel is actually competing on

CUDA Graph replay runs the five native kernels in a 14.33 µs span for 13.21 µs of kernel work at batch 4, and 16.89 µs for 15.74 µs at batch 256 (Task 20). The gap between kernels is about 0.25 µs each.

**So one launch instead of five saves about 1 µs here.** Launch removal was the first term in the architecture's savings model. Against the strongest baseline it is close to zero.

What remains:

| Source | Available on this MLP | Basis |
| --- | --- | --- |
| Concurrency at small M | Gate and up GEMMs at once; later work overlapped. Perhaps 13 µs → 7–10 µs | Estimate. A batch-4 GEMM takes ~3.2 µs and uses 12 of 60 SMs in the selected tiling |
| Fusion | SiLU·multiply in the GEMM epilogue saves 2–4 µs of passes | Native elementwise kernel times |
| Fixed overhead | CUDA Graph pays ~16 µs outside kernels per replay; one launch should match it | Task 20 |

The realistic prize is about 1.3–2x on kernel time at small M and near 1.0x at large M. It matters across a whole model, where the fixed per-call overhead is paid once and kernel time dominates. It is not a large win on a single MLP.

## 7. Everything wrong, by severity

### Blocking

1. **Per-call Python binding costs about 350 µs.** It is 12x the entire CUDA Graph call.
2. **SiLU is not fused.** Seven elementwise phases run where zero are needed.
3. **The generic elementwise body is about 40x slower than native.**
4. **No hand-written target exists.** Nobody has measured what a good one-kernel MLP costs on this GPU, so the compiler has no number to reach.

### Serious

5. **Sequential GEMM variants sum their scratch.** A plan mixing 64×64 and 64×128 tiles is rejected at 444,416 bytes against a 232,448 limit, so tiles cannot be chosen per GEMM. `codegen.py` and `plan.py` both add instead of taking the maximum.
6. **Plan search ranks by complete-call time.** About 350 µs of host time with a 60 µs P10–P90 spread hides kernel differences of 10 µs. Kernel time for the same phase plan varied between 56, 61, and 130 µs across Task 19 and 20 runs.
7. **Search compiles candidates that cannot win.** Plans using `generic_contraction` for a GEMM measured 3–10 ms at batch 256. They consume half the compilation budget and most of the 24–57 s first call.
8. **The phase barrier scans workers serially.** See section 5.
9. **The paired gate/up GEMM is unconditionally rejected** as "not implemented" in `regions.py`.
10. **MegaBake cannot run under CUDA Graph capture.** Both runtimes reject it. The per-call epoch argument is one obstacle; whether a cooperative launch can be captured is untested.

### Risks not yet measured

11. **Register reallocation across bodies.** After the first GEMM, warps 0–3 have executed `setmaxregister_decrease(40)` and then run generic bodies. Whether this is safe was not confirmed.
12. **One CTA per SM.** The 214,016 bytes of GEMM shared memory fix occupancy for every body in the kernel, including elementwise work.
13. **Every call ends in `stream.synchronize()`.** The callable is blocking, unlike eager PyTorch.
14. **The call trace list grows without bound.** `bind_callable` appends a record per call.

### Process

15. **Gate G0 passed without a clean composition number.** The mixed-body probe's timing loop kept its deliberate 100,000-iteration delay, so its 1,326 µs GPU latency is not a kernel cost.
16. **Task 17 regressed batch 4** from 414 to 510 µs and was kept.
17. **Task 20 is marked complete** although its dominant loss, the host path, was not changed.

## 8. Isolated kernel measurement

`benchmarks/isolated.py` times each mode's final kernels with launch and device execution only. Compilation and binding are outside every sample: baselines replay a captured CUDA Graph, and MegaBake's arguments are built before the timer starts. It was written without access to the GPU and has not been run.

```sh
.venv/bin/python benchmarks/isolated.py --batches 1 4 256 --output artifacts/isolated/mlp.json
```

| Mode | What is launched |
| --- | --- |
| `eager_graph` | CUDA Graph replay of the five eager kernels |
| `inductor_graph` | CUDA Graph replay of the Inductor kernels |
| `megabake_fast` | Prebuilt CuTe execution arguments through `run_compiled_program` |
| `megabake_call` | Prebuilt CuTe arguments through the normal CuTe call |
| `megabake_bound` | The public compiled callable, for reference |

It reports three numbers per mode:

- **kernel** — profiler kernel duration; computation only.
- **burst/launch** — 32 back-to-back launches and one wait, divided by 32. This is launch plus computation without the per-call wake-up.
- **single+wait** — one launch and one wait.

It also copies the generated CuTe module, its source map, and the kernel's resources to `artifacts/isolated/m<batch>-megabake-kernel/`.

Things to check on the first run:

- `megabake_fast` relies on `generate_execution_args` and `run_compiled_program`, whose names come from the cProfile output. Their signatures were not read. If the fast mode is missing or fails, `megabake_call` still gives a valid isolated number.
- Each mode prints its relative L2 error against eager. A MegaBake error above 0.01 means the isolated epoch handling is wrong and the timing should be discarded.
- The script calls two private names, `_RuntimeState` and `_input_tensors`, from `megabake.compiler`.

Expected outcome, as an estimate: `megabake_fast` near 20 µs kernel plus 10–30 µs launch at batch 1. If so, the host path fix alone brings the complete call from 376 µs to roughly 50–60 µs.

## 9. Next steps

In order. Each has a number that decides whether it worked.

| # | Step | Where | Done when |
| ---: | --- | --- | --- |
| 1 | Run `benchmarks/isolated.py` | — | Launch-plus-compute numbers exist for all modes |
| 2 | Hand-write the ideal MLP kernel: gate and up in one phase, SiLU·multiply in the epilogue, one barrier, down GEMM | `experiments/` | Kernel time under 13.2 µs at batch 4 and under 15.7 µs at batch 256 |
| 3 | Build launch arguments once; per call, check two addresses and reuse cached execution arguments | `compiler.py` | Complete call minus kernel at or under ~16 µs |
| 4 | Let scalar fusion accept multi-use values inside the chain and intermediate dtype changes | `regions.py` | The plan has one elementwise region, then none |
| 5 | Remove synchronization between independent regions; use one counter for the phase barrier | `schedule.py`, `runtime.py` | Phase and event kernels within 2 µs of each other |
| 6 | Take the maximum scratch across sequential bodies | `codegen.py`, `plan.py` | Mixed-tile plans are admitted |
| 7 | Rank candidates by kernel time; drop generic contraction when a tuned GEMM is legal | `plan.py` | First call under 10 s; stable selection across runs |
| 8 | Vectorize the generic elementwise body for work that survives fusion | `bodies/generic.py` | Within 2x of the native elementwise kernel |

Step 2 decides the project. If a hand-written kernel cannot beat 13.2 µs of native kernel work at batch 4, compiler work will not either.

**Hold until steps 1–3 pass:** Tasks 21 onward (plan caching, fallback, generic coverage, attention, KV state, full block) and further event-scheduler work.

## 10. Gates

| Gate | Target | Baseline it comes from |
| --- | --- | --- |
| Isolated launch plus compute | Below the `eager_graph` and `inductor_graph` values from `isolated.py` | Not yet measured |
| Host path | Complete call minus kernel ≤ ~16 µs | CUDA Graph: 16.2 at batch 4, 19.9 at batch 256 |
| Kernel, batch 4 | < 13.2 µs, ideally ~10 | CUDA Graph kernel work |
| Kernel, batch 256 | < 15.7 µs | CUDA Graph kernel work |
| Complete call | < 29.4 µs at batch 4, < 35.7 µs at batch 256 | CUDA Graph complete call |
| Compiler against hand-written | Within ~20% | Step 2 |
