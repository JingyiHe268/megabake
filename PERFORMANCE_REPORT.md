# MegaBake performance report

Two reports, newest first:

- **Full model (2026-10-16).** The pinned SmolLM-135M checkpoint compiled into one kernel in prefill and decode.
- **Gated MLP (2026-10-13), sections 1–10.** The first one-kernel MLP and its attribution.

## Full model, 2026-10-16

**Branch `v4-impl`: `a538d68` (round 1) plus the round-2 working-tree changes below. Model: `HuggingFaceTB/SmolLM-135M@1d461723`, bf16. GPU: H200 MIG `3g.71gb`, 60 SMs, shared with other jobs.** Reproduce with `benchmarks/full_model.py`: medians of 40 warm calls, decode KV capacity 128. Eager and Inductor also vary by 5–10% between runs on this machine.

### Verdict

MegaBake beats eager and default `torch.compile` end to end in both modes. It does **not** yet beat CUDA Graph replay: Inductor with CUDA graphs runs prefill in 2.0 ms against MegaBake's 3.0 ms. MegaBake's kernel now takes 1.4x Inductor's kernel time in prefill and 1.6x in decode, down from 2.8–3x after round 1.

| ms | Prefill, 4 tokens | Decode, 1 token |
| --- | ---: | ---: |
| Eager, complete call | 19.4 | 32.8 |
| Inductor, complete call | 7.4 | 13.0 |
| Inductor + CUDA graphs, complete call | **2.0** | not available* |
| MegaBake, complete call | 3.0 | **3.7** |
| Inductor GPU kernel time | 1.07 | 1.37 |
| MegaBake GPU kernel time | 1.48 | 2.22 |
| MegaBake first call (compile) | 127 s | 79 s |

\* `mode="reduce-overhead"` skips CUDA graphs because the 90 KV-cache inputs are mutated, so that run is plain Inductor. The `kv_attention` reference also reads the cache length on the host, which blocks manual graph capture. There is no CUDA-graph decode baseline yet.

Correctness: decode logits and every KV cache match eager exactly across advancing calls. Prefill matches exactly after the first call, which is 1.8% from eager with the same argmax, inside the 2% full-model tolerance. On the two-layer model, every intermediate value matches eager FX execution bit for bit (`tools/diff_intermediates.py --tolerance 0`).

| Complete call, ms | Task 31 | Round 1 | Round 2 |
| --- | ---: | ---: | ---: |
| Prefill | 64.2 | 4.5 | 3.0 |
| Decode | 100.1 | 5.2 | 3.7 |

### Round 1: make the kernel use real bodies (`a538d68`)

Task 31 compiled the model correctly, but the kernel took 52 ms in prefill and about 100 ms in decode. These changes each had a measured cause:

| Change | Cause found | Effect |
| --- | --- | --- |
| Tuned TMA GEMM inside table launches (`codegen.py`, `bodies/gemm.py`) | Above 1,000 graph ops every projection fell back to a one-thread-per-output scalar loop with uncoalesced loads | Kernel writes per-phase TMA descriptors in its prologue from the address table; all 211 projections use the tuned body |
| Counter grid barrier (`runtime.py`) | Block 0 polled every worker's flag serially: 3.8 µs per barrier at 15 workers, 13.3 µs at 60 | About 1.1 µs per barrier at any worker count |
| Full-grid worker count (`plan.py`) | Decode sorted worker counts ascending and a one-compile budget kept the smallest unmeasured: 15 of 60 SMs | Starts from the full grid when the budget cannot measure every count |
| Warp-per-row attention (`bodies/attention.py`) | One thread per query head recomputed every Q·K dot product per output dimension over the whole KV capacity: 723 µs per layer in prefill | Prefill about 4.5 µs per layer |
| Host-mapped per-call addresses and root/offset rebinding (`compiler.py`) | About 1,200 per-call addresses (mostly weights and their views) went through as separate kernel parameters and were re-derived in Python every call: about 10 ms per call | Host overhead about 1.0–1.5 ms per call |
| TMA store completion and proxy fences (`bodies/gemm.py`) | The GEMM returned once its stores had *read* shared memory, not once the global writes landed; the faster barrier exposed a wrong result | Store warp waits for completion and fences before the grid barrier |

### Round 2: fewer phases, fewer barriers, better spread

After round 1, the kernel ran about 1,170 phases (2.3x Inductor's kernel count), each ending in a grid barrier.

| Change | Cause found | Effect |
| --- | --- | --- |
| RMSNorm matcher accepts Hugging Face's order (`regions.py`) | `weight * hidden.to(dtype)` puts the weight multiply after the output cast; the matcher also kept a non-matching "residual" operation, so it never fired on the real model | Each norm is one region instead of 9 phases |
| `rotate_half` RoPE matcher (`regions.py`) | Only the decomposed `cat(x1·cos − x2·sin, …)` form was matched, not `x·cos + cat(−x2, x1)·sin` | Each RoPE is one region instead of 5 phases |
| Row-mode reductions (`bodies/generic.py`) | The generic reduction gave each output element its own row reduction: 576x redundant work for a norm | One reduction per row, then all threads write the row |
| Dependency-level phase schedule (`schedule.py`, `memory.py`) | Every phase ended in a grid barrier, even between independent projections | One barrier per dependency level: 364 per prefill call, 333 per decode call. Workspace lifetimes end at the interval's barrier |
| CTA rotation (`codegen.py`, `bodies/gemm.py`) | Every body started its tiles on CTA 0, so concurrent phases queued on the same CTAs | Each phase starts where the previous phase in its interval ended |
| Multi-CTA decode attention with deferred length publish (`bodies/attention.py`, `codegen.py`) | The body ran on one CTA because it published the new KV length itself | Rows spread across CTAs; the length is published after the closing grid barrier |

Round 2 also fixed four bugs that its changes exposed:

- **Missing parentheses.** A select nested in an expression was emitted without enclosing parentheses, so `a if c else b * x` multiplied only one branch.
- **Contracted half-precision arithmetic.** bf16 and fp16 arithmetic now runs in fp32 with explicit rounding, PyTorch's opmath. Without it, the compiler contracted multiply-add chains and fused RoPE drifted 0.1% from eager.
- **DSL recursion.** Several bounds-checked loads in one statement crashed the CuTe DSL preprocessor at rank 4. They are now bound as separate statements.
- **Single-phase attention.** A stateful attention with no closing barrier keeps the single-CTA, self-publishing body.

### Where the kernel time goes

Measured with global-timer stamps at every phase start (decode at about 40 cached tokens). Phases in one barrier interval overlap, so the attribution is approximate:

| Phase type | Prefill | Decode |
| --- | ---: | ---: |
| 211 tuned GEMM phases | 0.77 ms | 0.98 ms |
| 30 attention phases | 0.14 ms | 0.66 ms |
| 61 fused RMSNorm reductions | 0.21 ms | 0.21 ms |
| ~200–300 other elementwise and copy phases | 0.36 ms | 0.34 ms |

- **GEMMs** are now the largest item: 3.6–4.6 µs per phase, close to Inductor's entire GPU time. Small-N projections leave most workers idle (K and V fill 3 of 60 CTAs), and every phase pays its pipeline start-up. Weight streaming alone sets a floor near 0.13 ms for the whole model at this slice's bandwidth.
- **Decode attention** still grows with context. Rows now spread over CTAs, but each row walks every cached key.
- **Host work** is about 1.5 ms per prefill call and 1.3 ms per decode call. CUDA-graph replay spends about 0.9 ms outside the kernel, including the same Dynamo and Hugging Face wrapper overhead.

### Next steps

1. Split K, or use stream-K, for GEMMs with fewer output tiles than workers, and overlap GEMM pipeline start-up across phases.
2. Split decode attention across the context, with a softmax merge.
3. Fuse residual adds into the following norm, and elementwise work into GEMM epilogues.
4. Cut per-call host work toward CUDA-graph replay's.
5. Build a decode CUDA-graph baseline, which needs a capture-safe reference attention.
6. Find why prefill compilation grew from 80 s to 127 s, then bring both modes toward Inductor's 3–6 s.

---

# Gated MLP report, 2026-10-13

**Branch: `v4-impl` at `2d6e1db`. Workload: the gated MLP in `benchmarks/workloads.py`.**

This report states how the one-kernel MLP performed, where the time went, what was wrong, and what to do next. It supersedes the attribution in [performance_story.md](performance_story.md), which keeps the raw Task 20 tables.

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
