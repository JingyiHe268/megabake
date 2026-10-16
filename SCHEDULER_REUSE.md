# Operation coverage and implementation reuse

**Date: 2026-10-09. Status: source review and one isolated host check; no production port.**

This report extends [RESEARCH.md](RESEARCH.md). The resulting design is in [ARCHITECTURE.md](ARCHITECTURE.md). The scope in [north-star.md](north-star.md) is unchanged.

## 1. Decisions

1. Compile ordinary tensor programs through generic device generators. Recognizing a named norm, attention variant, or model is an optimization.
2. Generate variations around tuned schedules. Keep specialized bodies for cases where they help and their semantics match.
3. Use Mirage as the main reference architecture. Inherit its compatible algorithms, contracts, implementation details, and tests across the compiler and runtime. Keep the assumptions with each adaptation.
4. Preserve arbitrary dependency sets in the plan. A compact event encoding must not reduce model coverage.
5. Expose external fallback as a distinct execution result. An external launch cannot fill a gap inside a megakernel.

These are design commitments. Generic lowering, expression hooks, and the proposed fallback option still need implementation.

### How this guides implementation

For each subsystem, begin with the matching Mirage code path. Trace its callers, data structures, required conditions, and tests. Reproduce its relevant behavior on a small fixture. Then adapt its interface for MegaBake. Record the source reference, the reason for the adaptation, and the comparison result.

Prefer an existing compatible algorithm over a new one. When porting from C++ to Python/CuTe DSL, preserve the algorithm and check that its behavior remains equivalent. When the contract changes, keep the Mirage test cases and add a case that checks the required difference. Compare the relevant offsets, dependency sets, storage overlap, outputs, or timings.

Follow the complete path through these stages:

```text
tensor meaning -> tile maps -> local schedules and layouts
               -> tasks and dependencies -> persistent execution
```

A helper can depend on conditions established by earlier passes. Later passes can depend on its results. Preserve these relationships when extracting code.

The proposed phase schedule will provide a small executable reference. It does not settle the final scheduling policy. If phase waits consume the performance budget, test the relevant MPK event and work-assignment mechanisms during that gate. Complete any required scheduling work before expanding to a full model.

## 2. Review boundary and source pins

The Mirage checkout is `f9eb70c254acefc9f3667b2a973d0dcf25471fce`. It has 563 tracked files under `src`, `include`, `python`, and `tests`, including 121 task files. I traced both compiler paths from Python entry to emitted code and launch. The review covered:

- Graph and tensor representations, search, and verification.
- Layout and memory passes, and local scheduling.
- MPK graph analysis, task registration, event emission, and worker dispatch.
- Selected Hopper bodies and relevant tests.

This is an implementation reuse review. It is not a claim that every architecture-specific body, serving path, or device instruction was audited. Ampere and Blackwell variants, distributed serving, and speculative decoding were inspected at their interfaces where needed to determine reuse scope. The deeper body review targeted the SM90 design. Existing test files are evidence of useful fixtures, not evidence that their tests passed here.

| Source | Pin or installed revision | Review role |
| --- | --- | --- |
| Mirage/MPK | `f9eb70c254acefc9f3667b2a973d0dcf25471fce` | Main compiler and runtime review |
| Mirage's CUTLASS submodule | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` | Exact headers downloaded to `/tmp` for the attempted body check |
| Inferact TPU megakernels | `4048f0820aa4ff8787f707ca9d99b2bada9751aa` | Qwen packing, schedule, prefill/decode and fetch loop; Kimi interfaces and scratch aliasing |
| Installed PyTorch | `2.14.0+cu130`, commit `08187d9e0fba026dc8217405802ab5381dc88d90` | Decomposition registry and external fallback boundary |
| CuTe, DeepGEMM, FBGEMM | Pins in [RESEARCH.md](RESEARCH.md) | Prior body review and CuTe measurements; targeted sources for future ports |

The local research clones were not modified. The host experiment extracts an unchanged allocation core into a temporary build. No Mirage implementation was installed as a MegaBake compiler dependency.

## 3. Mirage contains two different compilation paths

### Graph search and transpiler

```text
Python KNGraph / TBGraph
    -> optional graph search and candidate verification
    -> Transpiler constructor: graph normalization
    -> device-tensor metadata
    -> local fusion chains
    -> tensor layout selection
    -> global workspace allocation
    -> local schedule, swizzle and shared-memory allocation
    -> CUDA source, compilation and host launch wrapper
```

The pass order is explicit in [`Transpiler::generate_code`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/transpiler/transpiler.h). The threadblock code generator uses reusable device primitives and fused epilogues. This is relevant to generating a new combination of operations without writing a kernel for each combination.

This path has a finite operator vocabulary and unsupported branches. Its kernel-level GEMM helper calls `cublasGemmStridedBatchedEx`. Thus its general host execution path is not automatically a one-kernel device fallback. [GEMM helper](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/transpiler/runtime/kernel/matmul.h)

### MPK task compiler

```text
Model builder / PersistentKernel layer methods
    -> tensor bindings and explicit grid/block maps
    -> registered task kind and generated variant
    -> graph annotation and event domains
    -> task descriptors, event counts and successor ranges
    -> generated device dispatch over registered variants
    -> preparation and persistent execution
```

The layer methods supply layouts and task types. For example, `linear_layer` selects Hopper task variants and receives an explicit grid. `compile` calls `generate_task_graph`. This is a useful device task system, but does not import arbitrary FX semantics. [Python entry points](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/python/mirage/mpk/persistent_kernel.py#L2390)

MegaBake can combine ideas from these two paths: generic tensor-program generation for coverage, tuned body families for speed, and an execution plan for composition.

## 4. Reusable components

“Port” below means adapt a bounded source component into MegaBake with its tests and attribution. “Reference” means retain the mechanism while using CuTe DSL or a simpler representation. Neither means the port is already complete.

| Component and source | What it does | Decision and required adaptation |
| --- | --- | --- |
| [Tile layout adapter](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/transpiler_tb_hopper.cc#L207) | Combines selected tile dimensions, global strides and dimension order | Port the mapping if needed. Return CuTe DSL layouts rather than C++ source strings. Test offsets, unit dimensions and noncontiguous inputs. |
| [Copy checks and local schedule](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/sched_tb_graph.cc) | Checks 16-byte copy alignment, builds pre-loop/loop/post-loop schedules, groups chains and inserts barriers | Port legality checks. Add actual pointer alignment, tails, active thread groups and async completion. Its accumulator register budget is a heuristic, not a compiled resource bound. |
| [Fusion chains](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/resolve_tb_fusion.cc) | Fuses selected single-consumer unary and accumulator chains with exclusions around inputs and reductions | Use as a simple first local fusion pass. Preserve explicit casts, all live outputs and effect ordering. Do not restrict all fusion to unary chains. |
| [Primitive bodies and epilogues](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/transpiler/runtime/threadblock) | Reuses indexed input/output, scalar operations, reductions, MMA and chained epilogues | Adapt this composition pattern to typed expressions and CuTe generators. Check each scalar operation's numerical meaning; upstream fixed clamp/activation policies are not every ATen overload. |
| [Shared-memory allocation core](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/plan_stensor_memory.cc#L15) | Places finite live intervals using first/best/worst fit, coalesces free ranges and chooses the smallest peak | Strong direct reuse candidate. Supply verified lifetimes and padded sizes. The isolated check below covers the unchanged core. Start with one policy if that is enough for our fixtures. |
| [Shared-memory lifetime analysis](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/plan_stensor_memory.cc#L256) | Finds uses around loop phases, extends lifetimes to barriers and reserves pipeline storage | Reference. Derive our lifetimes from actual completion and release points. Its local timeline is not a cross-CTA lifetime proof. |
| [Global workspace allocation](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/plan_dtensor_memory.cc) | Monotonic 128-byte aligned allocation for intermediate tensors | Small enough to port directly. It provides no global lifetime reuse algorithm; do not attribute one to it. |
| [Tensor layout solver](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/resolve_tensor_layout.cc) | Uses Z3 to choose innermost/swizzled dimensions with constraints and fixed cost penalties | Reuse constraints and stride/padding rules. Keep a finite set of body-compatible layouts first. The penalty constants are not measured performance on our target. |
| [Hopper swizzle planning](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/transpiler/plan_tb_swizzle_hopper.cc) | Selects layout atoms and preserves copy/MMA chunk constraints | Reference when implementing capability checks. Prefer existing CuTe atoms. Recheck element size and instruction requirements; the source includes half-sized operand assumptions. |
| [Dimension candidate search](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/search/dim_strategy.cc) | Enumerates grid shapes and input/output/loop maps, prunes divisibility and replication choices | Port candidate enumeration rules that fit the selected body family. Replace fixed grid/block heuristics with legal provider choices and measured target data. |
| [Graph annotation](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/annotated_graph.cc) | Builds producer edges, handles view windows, orders branches, forms GCD event grids and LCM fork/join groups | Reuse regular-partition event grouping after checking its preconditions. Keep explicit prerequisite sets and a phase fallback. See restrictions below. |
| [Task emission and simulation](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/runtime.cc) | Emits task metadata, producer counts, contiguous successor ranges and a host event simulation | Port descriptor compaction and simulation concepts. Validate all expected tasks, exact producer membership, termination, and worker-order edges. Source simulation is not a CUDA progress proof. |
| [Task variant registration](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/task_register.cc#L36) | Deduplicates generated variant strings and binds task arguments to device implementations | Port the reuse principle. Key bodies by normalized program/configuration; bind parameters through runtime data. Avoid model-specific task enums in the semantic graph. |
| [Hopper MPK GEMM bodies](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/tasks/cute/hopper) | Exposes device-callable CUTLASS/CuTe mainloop and epilogue work within an MPK worker | Valuable extraction reference. Test repeated entry, drain, scratch and register roles. The inspected `gemm_ws_mpk.cuh` path restricts batch size to at most 16; it is not our general prefill provider. |
| [Hopper norm and attention bodies](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/tasks/hopper) | Contains actual device functions, reductions, pipelined attention, KV access and tail handling | Use as candidate body sources and differential references. Decouple their model metadata and exact norm/RoPE/softmax policies. Device-callable C++ still needs a CuTe DSL port for our output contract. |
| [Worker runtime and atomics](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/persistent_kernel.cuh) | Fetches task descriptors, waits for events, dispatches bodies, publishes completion and schedules work | Reference for a later event runtime. Re-prove progress, publication and initialization under our one-kernel/cooperative launch. Do not copy the serving loop wholesale. |
| [Device profiler](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/profiler.h) | Records timestamped task/worker events; header attributes this component to FlashInfer | Useful for finding waits and imbalance. Add capacity checks and preserve the applicable attribution. Measure final latency with instrumentation disabled. |

## 5. Restrictions that change how we reuse the code

### General FX graphs need a more general dependency representation

`annotated_graph.cc` explicitly rejects its case-2/case-3 fork/join combinations. A task has only one wait-event and one completion-event slot. The repository has [negative tests for these forms](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/tests/runtime_python/test_mode/test_case2_case3_negative_testmode.py).

Keep those graphs legal in MegaBake's compute graph. Initially, topologically ordered phases handle their dependencies. A later event pass may use lists, group events, or retain a coarse boundary. Its compact encoding is an optimization with explicit admission checks.

For regular aligned partitions, MPK uses the GCD of producer and consumer partition counts to form common event groups. Its fork/join LCM passes coarsen groups to fit shared event slots. These are useful bounded algorithms. They do not describe arbitrary gather indices, sliding windows, overlapping writes, or uneven tails. Validate compressed dependencies against a concrete read/write overlap oracle before using them.

The source also strips residual edges using operation-level reachability. Our event optimizer must establish equivalent tile-level readiness before removing an edge. A path through a partially ready operation is not proof that every residual tile is ready.

### Search access ranges are not a synchronization proof

[`propagate_from_dtensor_to_stensor`](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/search/range_propagation/irange.cc#L55) documents that a range crossing partial tiles can return only a subset. That behavior belongs to its search use. Using such a result as a complete dependency domain could omit producers.

Our access analysis must cover every read/write. When exact analysis is unavailable, use a conservative domain, such as the whole referenced storage, and retain more ordering. Port the range representation only after checking the relevant operations and replacing any under-approximation used for dependency planning.

### Buffer identity is not value identity

MPK's graph analysis tracks recent writers because model builders reuse buffers. Its view implementation supports particular contiguous reshape/narrow conventions. This is useful evidence for alias tests. It is not a replacement for PyTorch's complete view, alias and mutation contract. [View implementation](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/view.cc)

Keep values distinct from storage versions in MegaBake. Add write-after-read and write-after-write ordering where state or storage reuse requires them. Prove async readers have finished before reuse. Begin with distinct intermediate slices so physical reuse does not complicate semantic import.

### Some “generic” helpers still encode a specific semantic choice

The MPK Hopper RMSNorm registration emits `1e-6f`, although the device function accepts epsilon. Its body uses a particular FP32 reduction and multiplication order. This makes it a candidate implementation for matching semantics, not a universal norm definition. [Registration](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/kernel/task_register.cc#L1551)

Likewise, copied activation expressions, online softmax rewrites, and fingerprint/formal verification do not prove equivalence for arbitrary source casts, NaNs, RNG or mutation. Keep numerical validation against the original captured graph. Use any imported verifier only within its stated mathematical model. [Verification implementations](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/src/search/verification)

### Runtime reuse must preserve the launch contract

The inspected launch entry calls `prepare_kernel`, then launches either a combined persistent kernel or separate worker and scheduler kernels. Its serving configuration includes global runtime state. This is useful reference code, but it does not directly implement our invocation-private, recurring one-kernel contract. [Launch entry](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/include/mirage/persistent_kernel/persistent_kernel.cuh#L1955)

The first MegaBake runtime remains a cooperative phase loop. Preserve the tuned range pipeline inside a phase. Add MPK-style events and descriptor prefetch only after a measured bottleneck justifies them.

## 6. Operation variants and the fallback boundary

The design uses three related mechanisms:

| Mechanism | Reuse unit | Example |
| --- | --- | --- |
| Semantic import/decomposition | Operator meaning | Convolution becomes an indexed contraction with padding and groups |
| Generated body | Tensor program plus schedule | A reduction schedule emits a new norm's scalar expressions and casts |
| Tuned family | Algorithm with checked expression hooks | Attention emits a supported score bias and mask inside its existing pipeline |

An unfamiliar model assembled from the supported primitive language uses this same path. A model-specific registration is unnecessary. Adding a primitive extends many models at once. Adding a matcher improves eligible regions without changing the meaning of unmatched regions.

FX capture alone cannot establish backend support. An opaque custom call can carry tensor metadata while still lacking a device implementation. A fake/meta implementation supplies metadata, not the operation's computation. Some operations also require explicit rules for effects or data-dependent storage. The compiler needs an importer, a usable decomposition, or a compatible device provider for those cases.

PyTorch already separates decompositions from external execution. In the inspected wheel, `torch/_inductor/decomposition.py` combines and filters decomposition tables; `fallback_handler` in `torch/_inductor/lowering.py` creates `ir.FallbackKernel`. Reuse selected decomposition definitions through the pinned adapter. Inductor lowerings and generated Triton/library kernels are not automatically callable inside CuTe device code. [Pinned decomposition source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/_inductor/decomposition.py), [pinned fallback source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/_inductor/lowering.py#L2754)

For attention variants, compile typed score/mask expressions into an eligible tiled pipeline. FlexAttention is a useful precedent for this structure; it separates score changes from masking so that skipping work can be justified. This is an interface idea to adapt, not a claim that its current launch wrapper can be embedded. [FlexAttention design](https://pytorch.org/blog/flexattention/)

When no tuned family fits, generate the supported primitive program. When no legal device program exists, the proposed `fallback=inductor` can delegate the whole invocation before execution. Reports must identify that delegation. Default strict compilation retains `fallback=error`. Neither option is implemented yet.

## 7. Other sources: concrete reuse boundaries

| Source | Useful part | Boundary |
| --- | --- | --- |
| [Inferact Qwen weight packing](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/qwen/load.py#L266) | `tile_hw`, `tile_matrix`, schedule offsets and matching fetch order | TPU dimensions and divisibility heuristics are target-specific. Port packing/order only if they benefit a CUDA body; preserve parameter invalidation. |
| [Inferact Qwen decode](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/qwen/decode_megakernel.py#L345) | Static weight stream, ring buffers, explicit asynchronous start/wait, distinct prefill/decode | The schedule encodes a model pattern and TPU constraints. Generalize the dependency/lifetime rules, not its layer-number formulas. |
| [Inferact scratch aliasing](https://github.com/Inferact/tpu-megakernels/blob/4048f0820aa4ff8787f707ca9d99b2bada9751aa/pool_alias.py) | Explicit physical storage size and overwrite-before-read discipline | It patches private Mosaic lowering and TPU layouts. Do not port that mechanism into CUDA. |
| [Official CuTe Hopper families](https://github.com/NVIDIA/cutlass/tree/0b55a2f691d69981583568fd9eb69687b1f0de8a/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm) | Tested TMA/WGMMA mechanics and coupled configurations in our output language | First GEMM provider source; still requires a pipeline-preserving composition interface. Prior local performance evidence is in RESEARCH.md. |
| [DeepGEMM SM90 BF16](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) | Additional GEMM schedule and pipeline choices for a measured gap | A specialized global kernel, not a generic FX lowering or pre-existing CuTe DSL body. No local performance claim. |
| [FBGEMM fast GEMV](https://github.com/pytorch/FBGEMM/blob/8978112d1283ca988ce9a533de03e87ef8b9663e/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/bf16_fast_gemv.cu) | Small-M vector/reduction mechanics and selection heuristics | Useful candidate for admitted decode shapes; no automatic composability or local performance claim. |

Use Mirage's search/configuration code as the first reference for candidate generation and pruning. Start with the part applicable to the chosen body families. Its algebraic superoptimizer and Z3 layout optimizer remain references for extending that search when needed. Alternative Triton/NKI backends, serving, and distributed execution address contracts outside the initial FX-to-CuTe, single-GPU path.

## 8. Checks performed

### Actual allocation core: passed

[check_memory_planner.py](experiments/reuse_2026_10_09/check_memory_planner.py) verifies the clone revision and source SHA-256, extracts the unchanged `memory_planner` namespace, and compiles it with a small C++ harness.

The check supplies 2,000 generated lifetime sets with seed `20261009`. Each set has 1–40 allocations with nonzero sizes padded to 128 bytes. It checks first fit, best fit, worst fit, and the upstream selector. An independent pairwise interval oracle checks that simultaneously live values never overlap in storage. The check also verifies alignment, bounds, a live-byte lower bound, and two boundary cases for touching and overlapping lifetimes.

All checks passed with GCC 14.3.1. This establishes that the allocation core can be extracted without the Mirage graph/compiler dependency. It does not validate upstream lifetime derivation, async uses, arbitrary alignment requirements, or global megakernel memory reuse. [Result](experiments/reuse_2026_10_09/memory_planner.json)

Reproduce from the repository root:

```bash
.venv/bin/python experiments/reuse_2026_10_09/check_memory_planner.py \
  --output /tmp/mirage-memory-planner.json
```

### Hopper attention tail check: could not run

I selected Mirage's existing [attention tail test](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/tests/runtime_python/hopper/test_attention_tail_hopper.py). It poisons shared memory before calling the real attention device function and checks active/inactive rows over 24 capacity/context cases. This is a valuable fixture: a masked probability of zero does not prevent `0 * NaN` from a stale value tile.

The clone lacked CUTLASS submodule contents. The exact pinned headers were downloaded into `/tmp`; the clone was left unchanged. The test then stopped before compilation because no CUDA device was available on 2026-10-09. `nvidia-smi -L` reported no devices, and PyTorch reported zero devices. The H200 MIG used for the previous day's research was not available for this run. [Failure log](experiments/reuse_2026_10_09/attention_tail.log), [environment](experiments/reuse_2026_10_09/environment.json)

The attempted command, with the temporary pinned headers present, was:

```bash
CPATH=/tmp/megabake-reuse-2026-10-09/cutlass-f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include \
  .venv/bin/python agent_space/mirage/tests/runtime_python/hopper/test_attention_tail_hopper.py
```

No new GPU correctness or performance result is claimed.

## 9. Port order and required evidence

Follow the G0/G1 feasibility and performance gates first. Use the list below to order reuse within each component when a gate needs it. Build only the parts needed for the first mixed-body experiment, then expand as later gates require.

1. **Baseline coverage:** build typed scalar/map and reduction generators, then indexed contraction support. Use source decompositions and reusable primitive structure. Check unfamiliar graph combinations with semantic matchers disabled.
2. **Layout and allocation:** port the needed tile/stride and copy legality helpers. Reuse the allocation core after lifetime rules exist. Keep CuTe as the physical layout implementation.
3. **Tuned bodies:** retain official CuTe GEMM as the measured starting point. Compare relevant Mirage device bodies and port useful mechanics. Add norm expressions and checked attention hooks rather than a new body for each variant.
4. **General plan:** keep phases and explicit dependencies. Use source-linked diagnostics and variant deduplication from the start.
5. **Event optimization:** when traces justify it, port regular partition/event grouping and compact descriptors. Compare against a full overlap oracle and the phase reference. Include graphs rejected by MPK's compact format as positive MegaBake fixtures.

Useful upstream fixtures include fork/join/diamond graphs, grid sweeps, residual paths, uneven norm widths, poisoned attention tails, changed KV lengths and windowed attention. Adapt their references to MegaBake's numerical contract; do not copy tolerances that are chosen after observing an error. [MPK tests](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/tests/runtime_python/test_mode), [Hopper tests](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce/tests/runtime_python/hopper)

For every actual port, record the upstream revision, files and changes; preserve applicable copyright, license and notice text. Some components carry third-party attribution. Keep these records next to the port. The source review gives us a concrete reuse queue; each device port still needs the composition and performance checks in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
