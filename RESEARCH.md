# MegaBake design research

**Date: 2026-10-08. Target: the existing `v4-impl` checkout and H200 MIG device.**

The result of this work is a revised [architecture](ARCHITECTURE.md) and [implementation plan](IMPLEMENTATION_PLAN.md). The scope in [north-star.md](north-star.md) is unchanged.

**Follow-up, 2026-10-09:** [SCHEDULER_REUSE.md](SCHEDULER_REUSE.md) extends the source review across Mirage's compiler and runtime paths. It defines reuse boundaries, generic operation coverage, generated variants, and explicit external fallback. It also records a passing host allocation check and an attempted GPU check that could not run because no CUDA device was available that day. The GPU results below remain the measurements from 2026-10-08.

The central decision is to preserve tuned device pipelines through composition. Use a small catalog of measured body configurations, a simple phase schedule first, and finer tile dependencies only when they repay their cost. Prove a mixed-body kernel before implementing the full compiler.

## What the repository establishes

The inspected `v4-impl` head is `2750a5a298b7c9f0918e0ba4fba1147bc670a42e`. It contains a capture/reference path and an empty MegaBake package entry. The proposed compiler is not implemented. The saved SmolLM metadata describes BF16, four input tokens, full logits, and `use_cache=False`. That is useful workload evidence, but it does not prove stateful decode.

I also inspected GEMM code on `main` at `5a18b8226ac39a35c91492825310dc2679ce83a1` and the CuTe/CUTLASS probes on `v3-impl` at `f579337ded2e4d6e90df09e40499cbb88cda2ff0`. The older code includes special matrix paths and body probes. These source reads do not establish those branches' current performance. No branch was checked out or changed.

The old design correctly separated semantics, dependencies, and execution. It also required source coverage and synchronization proofs. Its build order still allowed correct but slow bodies to advance through the complete-model stages. The revised plan makes body composition and region performance early gates.

## What to reuse

### Mirage and MPK

The local MPK checkout is clean at `f9eb70c254acefc9f3667b2a973d0dcf25471fce`. Its general transpiler maps tensor computations into GPU code. MPK maps registered device work into tasks and events.

| Inspected code | What it actually supplies | MegaBake decision |
| --- | --- | --- |
| [`get_dtensor_tile_layout`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/transpiler_tb_hopper.cc#L207) | A CuTe layout built from the selected shared-tensor dimensions and device-tensor strides, with a dimension permutation | Adapt this small mapping where useful. It does not choose an optimal tile. |
| [`sched_tb_graph.cc`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/sched_tb_graph.cc) | Copy-alignment checks, operation chains, accumulator placement rules and synchronization insertion | Reuse the checks and test cases that match our body contract. |
| [`annotated_graph.cc`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/annotated_graph.cc) | Mapped dependency analysis, fork/join handling and event-domain construction | Port a bounded algorithm with its assumptions and differential tests. Do not treat its accepted graph forms as all FX graphs. |
| [`runtime.cc`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/runtime.cc#L1915) | Task graph construction, descriptors, event counts and successor ordering | Use it as the main reference for a later event plan. Keep an explicit dependency representation before compressing descriptors. |
| [`persistent_kernel.cuh`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/persistent_kernel.cuh#L1955) | Worker/scheduler execution, plus a preparation launch and an optional split-launch path | Adapt execution ideas. Its launch wrapper does not directly satisfy MegaBake's recurring one-kernel contract. |

MPK's paper explains event fusion, task normalization, hybrid scheduling, and cross-task pipelining. These mechanisms address real overheads, but they need not all be present in a first runtime. Its reported performance is evidence for its tested systems, not a performance guarantee for this frontend and GPU partition. [MPK paper](https://arxiv.org/html/2512.22219v2)

The mapping between accesses and dependencies is valuable, as are tested device implementation details. Prove the body interface before adapting a large scheduler. Check the scheduler's assumptions against MegaBake's requirements.

For any source port, retain the upstream revision and applicable license/notice files. The local Mirage and TPU projects use Apache-2.0. No source from either was copied into the compiler during this work.

### Inferact TPU megakernels

The local checkout is clean at `4048f0820aa4ff8787f707ca9d99b2bada9751aa`. In the Qwen implementation, `make_prefill` uses a separate JAX path. `transformer_stack` implements fused decode with model and layout constraints; its inner functions include scheduled weight fetch and GEMV work. This is not an automatic CUDA compiler hidden behind a mode switch. [Inspected implementation](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/qwen/decode_megakernel.py)

Keep the separation of workload regimes, the explicit state contract, and the idea of planned weight movement. CUDA needs its own participant, memory, and synchronization contracts. A single MegaBake compiler can choose different plans for these regimes while preserving one frontend and one semantic representation.

### CUTLASS and CuTe DSL

This is the closest initial source for the required backend. The inspected Hopper examples expose TMA copies, WGMMA, shared layouts, epilogues, and ordinary or persistent work assignment. They are complete kernels. A device-body adapter must still supply ownership, entry/exit, and scratch contracts. [Pinned examples](https://github.com/NVIDIA/cutlass/tree/0b55a2f691d69981583568fd9eb69687b1f0de8a/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm)

The experiments below show that useful performance is available from this source with a small tile search. They also show why fixing one tile or one thread count for all regions is a bad premise.

The current DSL includes an experimental API for explicit warp schedules and resource protocols. Consider it for checking a body's internal pipeline. It does not replace the model's cross-CTA dependency plan. [CuTe task scheduling](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html)

### DeepGEMM

The inspected revision is `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`. It includes an SM90 BF16 GEMM; it is not only an FP8 project. The source has explicit tile/stage parameters, separate TMA and math roles, and a GEMM scheduler. Its entry is a CUDA global kernel. [BF16 implementation](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh)

Use it to investigate a specific remaining gap in a CuTe body: stage policy, warp roles, work assignment, or small-M handling. Extracting its implementation into CuTe is real porting work. Calling its Python GEMM API would add a kernel launch. The public interface also separates some input conversion/layout preparation from the GEMM; that work must be counted if needed. [DeepGEMM interface](https://github.com/deepseek-ai/DeepGEMM#interfaces)

Decision: a targeted source and comparison candidate, not the initial compiler dependency. It was reviewed, not built or benchmarked here.

### FBGEMM

The inspected revision is `8978112d1283ca988ce9a533de03e87ef8b9663e`. The project contains separate CPU, GPU, and GenAI components. For this task, the relevant part is GenAI GPU code. [Project organization](https://github.com/pytorch/FBGEMM#the-fbgemm-project)

Its BF16 fast GEMV has an explicit small-M implementation and shape-specific block heuristics. The inspected header limits M to four. The wrapper launches a kernel. This is a useful source for vector/reduction work and a comparison for small-M projections, not a general body selector for MegaBake. [Wrapper](https://github.com/pytorch/FBGEMM/blob/8978112d1283ca988ce9a533de03e87ef8b9663e/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/bf16_fast_gemv.cu), [device implementation](https://github.com/pytorch/FBGEMM/blob/8978112d1283ca988ce9a533de03e87ef8b9663e/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/include/fast_gemv.cuh).

Decision: test a port only where measured small-M WGMMA or warp-MMA choices leave a gap. The measurements below show that WGMMA can already be competitive at M=1 for one large projection. Do not select SIMT from the mode name alone. FBGEMM was reviewed, not built or benchmarked here.

## Is reverse-engineering cuBLAS worthwhile?

The ACCU article studies Ampere A5000 kernels. It improves async loading, buffering, and instruction overlap after comparing generated assembly with cuBLAS. Its hardware and experiment are narrower than a general recipe for Hopper kernels. [Fabian Schuetze's article](https://accu.org/journals/overload/32/181/schuetze/)

Use this method for diagnosis: identify one measured gap, inspect the selected library kernel and our generated code, form a hypothesis, and test a change. Hopper adds different async mechanisms and warp-group MMA rules, so the instruction schedule must fit the target. [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

The bounded procedure is:

1. Match shape, strides, precision, output policy, and cache regime.
2. Check whether the loss comes from work count, memory traffic, parallelism, or pipeline stalls.
3. Change one body or scheduling choice with a predicted effect.
4. Compare the original launch, the resident form, and the complete region.
5. Keep the change only if the complete target benefits.

Decision: inspect assembly when the simpler source-level choices leave an unexplained gap. Do not make a reconstruction of the cuBLAS kernel catalog a prerequisite. The larger-tile experiment below reached parity on a tested large shape without that work. SASS similarity would not by itself prove composability or full-model speed.

## Lessons from compiler design

The MLC book separates tensor computation, scheduling transformations, graph rewrites, and empirical search. Adopt that separation and the small measured search space. A named graph fusion does not guarantee a good GPU implementation. [Tensor abstractions](https://book.mlc.ai/chapter_tensor_program/index.html), [schedule search](https://book.mlc.ai/chapter_auto_program_optimization/index.html), [graph rewrites](https://book.mlc.ai/chapter_graph_optimization/index.html).

Linalg makes parallel/reduction structure and operand indexing explicit. Those facts let a planner reason about which producer tiles a consumer needs. MegaBake can express the needed maps in a small representation; adopting the whole MLIR stack is not required. [Linalg design](https://mlir.llvm.org/docs/Dialects/Linalg/)

CUTLASS separates the GEMM mainloop, epilogue, and kernel-level organization. Keep these boundaries visible in a device provider. The outer compiler must not replace a carefully scheduled mainloop with a generic scalar implementation. [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)

The Colfax implementation explains why output-tile count and wave balance matter, and why a persistent loop alone does not remove wave quantization. Split K and Stream-K change the work decomposition and add reduction obligations. Use them only when that trade pays. [Persistent kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)

Attention needs the same care as GEMM. FlashAttention-3's authors describe overlapping memory movement, matrix work, and softmax on Hopper. Replacing that pipeline with independent generic operators can lose the benefit of a megakernel before outer scheduling starts. [FlashAttention-3](https://tridao.me/blog/2024/flash3/)

## GPU experiments

### Environment and controls

The device is an H200 MIG `3g.71gb` partition with 60 SMs and approximately 29 MiB L2. PyTorch is `2.14.0+cu130`; CuTe DSL is `4.8.0`; the driver is `595.58.03`. The system `nvcc` is 12.8. The experiments used the DSL compilation path; they do not establish a CUDA-13 C++ build environment. Cooperative launch support was queried and returned true. No megakernel grid barrier was implemented or tested.

All GEMM inputs/results are BF16. Accumulation is FP32. TF32 and PyTorch BF16 reduced-precision reductions were disabled. The reference uses the quantized BF16 input values widened to FP32, then rounds the result to BF16.

Each retained timing is a median of 11 CUDA-event batch averages. Each batch replays a CUDA Graph. This removes Python launch gaps. The harness uses an explicit capture stream and checks that replay overwrites a NaN-filled output. Compile/setup time is separate. This is GPU timing, not complete host-call latency.

The first sweep uses repeated buffers. The second rotates weights through at least 64 MiB. Inputs and outputs are reused in the rotating test, so this is a weight-streaming test, not a cold-cache test for every tensor. The large weight already exceeds L2. Buffer rotation is needed to avoid making small cached matrices stand in for a model's weight stream. [NVIDIA measurement guidance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)

There was no clock lock, thermal sweep, randomized provider order, or exhaustive cuBLASLt search. Treat close numbers as parity, not a proved win. The JSON files retain raw batch means, numerical errors, seeds, source hashes, and versions.

### GEMM providers

`probe.py` compares `torch.mm` with two pinned NVIDIA examples and one experimental adapter:

| Name in artifacts | Execution |
| --- | --- |
| `torch_mm` | PyTorch library-backed GEMM with preallocated output |
| `dense` | Original ordinary-grid Hopper CuTe example |
| `persistent` | Original Hopper CuTe example with its own persistent pipeline and tile scheduler |
| `restart` | The dense example changed to a device body; at most 60 workers iterate over tiles and restart the body for each tile |

The adapter preserves the dense example's tile swizzle, moves scratch allocation to the worker, drains stores, and synchronizes before reuse. It supports batch dimension one and cluster `(1,1)` only. It is not a mixed-body megakernel and has no cross-CTA readiness protocol.

The example CLI's dtype validator rejects BF16. The harness calls the kernel class directly, which uses the BF16-capable Hopper helper, and checks its results. Explicit leading layout dimensions are required for the single-row case. These details are captured in the harness rather than assumed from the example's documentation.

### GEMM results

The table uses the best candidate median from five CTA tile shapes: `64×64`, `64×128`, `128×128`, `64×256`, and `128×256`. The two extra tiles are in a separate sweep. The library column uses the best of the corresponding two library measurements. These are selected results from a small search, not universal provider rankings.

All times are microseconds. Dimensions are listed as `M, K, N`.

| M, K, N | PyTorch GEMM | Dense CuTe | Persistent CuTe | Restarted dense body |
| --- | ---: | ---: | ---: | ---: |
| 1, 576, 1536 | 3.384 | 3.313 | 3.219 | 3.469 |
| 4, 576, 1536 | 3.268 | 3.191 | 2.971 | 3.323 |
| 256, 576, 1536 | 3.541 | 4.167 | 3.918 | 4.306 |
| 1024, 576, 1536 | 7.211 | 8.944 | 8.694 | 8.908 |
| 1, 4096, 11008 | 42.578 | 43.038 | 42.719 | 43.042 |
| 256, 4096, 11008 | 61.671 | 71.856 | 70.799 | 73.795 |
| 1024, 4096, 11008 | 232.826 | 241.817 | 232.012 | 239.861 |

Sources: [initial sweep](experiments/design_2026_10_08/hot.json), [larger tiles](experiments/design_2026_10_08/large_tiles.json).

The larger tile changed the conclusion. On the last shape, the best persistent result from the initial three choices was 249.499 µs. The `128×256` configuration reduced it to 232.012 µs. A separate repeat measured 231.380 µs. This supports approximate standalone parity for this shape. It does not prove full-model parity. [Repeat and resources](experiments/design_2026_10_08/large_repeat.json)

At M=1 with the larger weight, `64×256` also reached approximate parity. Thus padding a WGMMA tile is not enough evidence to reject it. Its actual throughput and memory behavior determine the result.

Some gaps remain: the M=256 large projection is about 15% slower, and the M=1024 small projection is about 21% slower in this search. There is still useful body work before claiming broad coverage.

The restart cost depends on the configuration. At `128×128` on the last shape, dense took 272.222 µs and its resident adapter took 302.307 µs. At `128×256`, the adapter was close to the original dense kernel. The dedicated persistent example uses different warp roles and pipeline organization. Barrier restart alone therefore cannot explain its timing difference from the adapter. Measure the complete schedule; this experiment does not establish a universal restart penalty.

The weight-streaming sweep covered the initial three tile shapes. For the small M=4 projection, library GEMM changed from 3.348 to 3.645 µs. Persistent CuTe changed from 2.971 to 3.348 µs. Cache behavior matters at this scale. The extra large tiles were not repeated in this streaming sweep. [Rotating-weight results](experiments/design_2026_10_08/rotating_weights.json)

### Resource and correctness checks

All retained GEMM timing candidates passed the fixed diagnostic relative-L2 threshold of 0.005 and a finite-output check. Observed errors were much smaller; exact maxima are in each artifact. This random-input diagnostic is not a model numerical contract. It does not cover every tail, mask, stride, or extreme value.

At M=256, the tested `64×64` and `128×128` dense bodies reserved 230,400 bytes of shared storage. The persistent `128×128` body used 214,016 bytes. The driver occupancy query admitted one block per SM for the tested configurations. No local-memory allocation was reported. [Resource records](experiments/design_2026_10_08/resources.json)

The larger `128×256` persistent configuration used 384 threads, 230,400 bytes of shared storage, and 168 driver-reported registers per thread. It also admitted one block per SM and reported zero local bytes. This is the concrete cost of the near-parity result. A mixed kernel must fit that envelope and preserve its warp roles. The data does not show what happens after adding norm and attention bodies.

Compute Sanitizer memcheck reported zero memory errors for the M=256 validation fixture, including repeated calls with changed input contents and resident scratch reuse. API-error reporting was disabled because the installed tool reported optional `cuGetProcAddress_v2` lookups as errors in the CUDA Python binding. All explicit CUDA API results used by the harness were checked. This was a memory check, not a race or synchronization proof. [Validation log](experiments/design_2026_10_08/memcheck.log)

### Actual Inductor MLP baseline

The MLP computes gate and up projections, SiLU, their product, and the down projection. Inputs and boundary results are BF16. Both eager and compiled paths use the same data and externally captured GPU timing.

`max-autotune-no-cudagraphs` was requested, but the installed Inductor code requires at least 68 SMs for its `is_big_gpu` GEMM path. This partition has 60. The warning was observed. These results describe the effective generated kernels, not a successful exhaustive max-autotune run.

| M, hidden, intermediate | Eager graph, µs | Inductor graph, µs | Separate library GEMM sum, µs | Inductor kernels |
| --- | ---: | ---: | ---: | ---: |
| 1, 576, 1536 | 13.570 | 7.405 | 11.447 | 2 |
| 256, 576, 1536 | 16.421 | 14.160 | 12.514 | 4 |
| 1024, 576, 1536 | 30.712 | 27.422 | 23.058 | 4 |
| 1, 4096, 11008 | 136.993 | 174.095 | 133.238 | 2 |
| 256, 4096, 11008 | 200.032 | 196.478 | 185.095 | 4 |
| 1024, 4096, 11008 | 746.797 | 735.849 | 696.373 | 4 |

Source: [MLP results and kernel names](experiments/design_2026_10_08/mlp.json).

At M=1, Inductor selected two fused Triton reduction kernels. At the larger M values, the trace shows three library GEMMs and a fused pointwise kernel. The compiled small MLP is faster than the sum of three separate GEMMs, so that sum is not a physical lower bound. On the larger M=1 MLP, eager graph execution is stronger than the measured compiled path. A fair MegaBake report must show both.

For the last row, the difference between the Inductor MLP and isolated GEMM sum is about 39.5 µs. This estimate includes effects of different cache state and execution context. It is useful evidence that a large GEMM regression can exhaust the available savings. It is not a forecast of a megakernel's exact latency.

## Decisions and remaining uncertainty

| Question | Decision | Evidence still needed |
| --- | --- | --- |
| Replace all existing design concepts? | Keep source semantics, maps, effects and explicit execution planning. Simplify the permanent representations. | Working frontend-to-plan integration |
| Start by reconstructing cuBLAS? | No. Start from measured CuTe families; use assembly inspection for a specific unresolved gap. | More real projection shapes and strong library tuning |
| Can good bodies already exist? | Yes for some measured standalone shapes. The large-tile result changes the initial pessimistic conclusion. | Heterogeneous composition under the same resource envelope |
| One schedule for both modes? | One compiler with separate policies and state validation; choose bodies from actual dimensions. | Complete prefill and advancing decode fixtures |
| Copy MPK scheduling wholesale? | Port selected maps, checks and dependency algorithms when required. | Correctness under our aliases, effects, body lifetimes and launch rule |
| Require a task event for every tile now? | Start with phases and pipeline-preserving ranges. Refine useful boundaries. | A trace showing that finer readiness repays its cost |
| Is a full-model speedup established? | No. The design has early rejection gates to test that risk. | Mixed MLP, attention, full block, full model and complete-call timing |

The next implementation should be a small mixed-body CuTe experiment with a legal cooperative barrier and the measured candidate families. A standalone near-cuBLAS GEMM makes this worth testing. It does not justify assuming the final megakernel will inherit that speed.

## Reproduce the retained experiments

Use the repository's activated environment. The harness downloads the two official example files at a fixed revision into `/tmp/megabake-design-research`. It records their hashes and preserves the upstream license in the generated adapter. Network access is needed only if those files are absent.

```sh
.venv/bin/python experiments/design_2026_10_08/probe.py --output /tmp/hot.json
.venv/bin/python experiments/design_2026_10_08/probe.py --large-tiles --output /tmp/large-tiles.json
.venv/bin/python experiments/design_2026_10_08/probe.py --streaming --output /tmp/rotating-weights.json
.venv/bin/python experiments/design_2026_10_08/probe.py --quick --single-m 1024 --large-shape --large-tiles --resources --output /tmp/large-repeat.json
.venv/bin/python experiments/design_2026_10_08/mlp_baseline.py --output /tmp/mlp.json
compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 .venv/bin/python experiments/design_2026_10_08/probe.py --quick --single-m 256 --resources --validate-only --output /tmp/resources.json
```

Run GPU experiments sequentially. Use a new output path for each run. Do not compare sanitizer timings with the uninstrumented results.
