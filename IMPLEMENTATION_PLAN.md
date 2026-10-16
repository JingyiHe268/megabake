# MegaBake implementation plan

**Status: G0 probes are recorded as complete for their declared target. Production integration and later gates remain open.**

[north-star.md](north-star.md) defines the scope. [ARCHITECTURE.md](ARCHITECTURE.md) defines the design. [RESEARCH.md](RESEARCH.md) records the work completed for this revision.

[TASKS.md](TASKS.md) gives the step-by-step execution order, functions to implement, inputs, outputs, and completion checks. Follow that sequence to build the compiler. Use the gates below to assess the resulting implementation.

[SCHEDULER_REUSE.md](SCHEDULER_REUSE.md) identifies source components to port and the checks each port needs. Generic operation coverage and specialized performance have separate acceptance records.

Build one small working path at a time. Start with the thin FX frontend and Compute Graph in TASKS.md. Connect them to the recorded body and runtime probes. Keep early fixtures small, and pass G1 before expanding into broad operation and model coverage.

The gate numbers identify acceptance criteria. They do not require delaying the first FX handoff until after G1. Existing probe evidence is retained in [TASKS.md](TASKS.md#previously-completed-probe-work).

Use Mirage as the main implementation reference at each gate. Trace the matching Mirage code path and tests before writing a new algorithm. Preserve compatible behavior. Explain each adaptation and check its effect. Start with the component map in [SCHEDULER_REUSE.md](SCHEDULER_REUSE.md).

## Gate map

| Gate | Result | Advance only when |
| --- | --- | --- |
| G0 | Body and runtime feasibility | Real CuTe device work composes; synchronization and complete-kernel resources are valid |
| G1 | Region performance | A representative complete region beats the strongest equivalent measured baseline for its declared regime |
| G2 | Reliable FX handoff and baseline lowering | The proven region and unfamiliar combinations of admitted primitives compile with correct bindings, guards and explicit fallback behavior |
| G3 | Complete transformer block | Attention, norms, residuals and state compose correctly within the budget |
| G4 | Complete model | Requested outputs/effects, repeated stateful calls, one workload launch and measured latency pass |
| G5 | Targeted optimization | Each added mechanism solves a measured remaining bottleneck |

A gate applies to a stated GPU and workload regime. Passing a short prefill does not pass decode. Passing one shape does not pass all shapes. Keep separate records for correctness, launch count, and performance.

## G0: Prove bodies and the persistent runtime

Use a hand-built plan and a small harness. Keep this code disposable until the interfaces pass. Reuse the research scripts for source pins and measurement methods, but do not treat them as production kernel providers.

### Establish the target and baseline

- [ ] Record device UUID, SM count, memory/L2 capacity, cooperative-launch support, clocks, driver, PyTorch commit, DSL version, and compiler versions.
- [ ] Use BF16 inputs/results and a declared accumulation policy. Record reduction precision and any approximation settings on both sides.
- [ ] Capture actual model shapes and operand strides. Include gate/up/down, Q/K/V, output projection, and the vocabulary head. Do not extrapolate from one projection.
- [ ] Cover small M, larger M, partial tiles, and weight sets both below and above L2 capacity.
- [ ] Measure PyTorch library GEMMs and the effective Inductor choices. Record kernel names and whether autotuning ran. The measured 60-SM partition failed the inspected Inductor `is_big_gpu` threshold of 68 SMs.

The research sweep is initial evidence. It does not replace this inventory or an exhaustive cuBLASLt comparison.

### Prove the device-body interface

Build these cases in order:

1. The original CuTe GEMM launch with its normal pipeline.
2. The same device computation inside a worker that processes several tiles.
3. Repeated entry with changed input addresses and contents, reused scratch, and different tile counts.
4. GEMM followed by a generated vector/reduction body in the same kernel.
5. GEMM followed by the next GEMM configuration that the fixture requires.
6. Two CTAs exchanging produced data with the selected visibility protocol.
7. The same cases under the complete cooperative launch and grid barrier.

- [ ] Allocate scratch at worker scope. Verify barrier initialization, phase changes, TMA completion, and register-role transitions.
- [ ] Generate vector/reduction work from typed expressions. Include two expressions with the same schedule and different casts or scalar parameters. This must exercise generated code beyond GEMM.
- [ ] Keep a range entry that preserves the pipeline. Compare it with the simpler tile entry. Do not require every body to restart at every tile.
- [ ] Verify the actual cooperative launch and a real grid barrier in the installed CuTe backend. A launch flag or a conceptual `grid_sync` call does not pass this check.
- [ ] Compile the mixed kernel and query its actual registers, shared memory, local memory, active blocks, and launch limits. Compare with the standalone body.
- [ ] Check repeated execution, changed runtime bindings, delayed producers, and the largest admitted worker grid. Reject an oversized cooperative grid before launch.
- [ ] Trace the complete callable. Include initialization and any wrapper work in the launch audit.
- [ ] Run an appropriate memory/race check on the custom glue. Record tool limitations and keep instrumented runs separate from timing.

**Stop condition:** if body entry, grid synchronization, or the shared resource envelope fails, fix that interface. Do not add a general task scheduler to work around an unproved body contract.

**Required artifact:** a small reproducible executable, correctness results, a launch trace, compiled resource data, and latency for standalone, resident-range, and mixed-body cases.

## G1: Prove the region's performance budget

Use a complete gated MLP as the first region. Keep all source casts. Use real projection dimensions and weight orientations. Test a small-M case and a larger prefill case separately.

- [ ] Measure eager, eager under CUDA Graph replay, effective `torch.compile` configurations, and the one-kernel candidate on equivalent inputs and outputs.
- [ ] Record both warmed GPU latency and synchronized complete host-call latency. Separate compile, first-call, packing, and steady-state costs.
- [ ] Compare separate gate/up ranges, a combined projection where packing is valid, and a paired local body. Include the down projection and all required intermediate work.
- [ ] Start with ordered phases and one barrier at a true data boundary. Keep the normal GEMM pipeline within each phase.
- [ ] Account for transient and packed memory. Charge any recurring conversion/reset/copy to the invocation.
- [ ] Compare the same bodies with and without local epilogue fusion to identify its real benefit.
- [ ] If phase waits consume the available savings, compare an MPK-derived event plan on the same complete region. Include its work assignment. Introduce the required mechanism during this gate. Make this comparison even if the phase reference has not beaten the baseline.
- [ ] Keep raw samples and use alternating measurements when differences are near the noise level. A single best timing is insufficient.

Use the full region measurement as the decision. The difference between an Inductor MLP and a sum of isolated GEMMs is only a diagnostic estimate of headroom. It can even be negative when Inductor changes the algorithm.

**Performance gate:** the complete correct megakernel must improve the strongest equivalent baseline beyond measured run-to-run noise for a declared target. The register/resource cost of the required body mixture must already be included. Do not require an arbitrary percentage of cuBLAS speed for each body; use the measured complete-region budget.

If a regime fails, identify the dominant loss: body throughput, weight traffic, tile padding, insufficient parallelism, pipeline restart, conversion, barrier cost, or occupancy. Work on that loss before expanding the compiler. Keep the failed result visible. A different successful regime does not turn this failure into a pass.

**Required artifact:** one-kernel MLP report for each tested regime, source coverage, numerical errors, baseline kernel names, and an explicit advance/hold decision.

## G2: Connect the proven region to FX

Expand the thin production path around the proven bodies and plan after G1. The initial handoff and graph import are built earlier in TASKS.md. The checks below apply to their complete integration and to broader generic coverage.

- [ ] Pin and check the supported PyTorch build before importing private APIs.
- [ ] Repair the capture/cache handoff. Test cache hit and miss behavior so both supply the intended graph phase when compilation is required.
- [ ] Preserve AOT/Dynamo argument, output, guard, alias, and mutation wrappers. Refresh graph metadata after rewrites.
- [ ] Retain every FX operation and its source location. Import the primitive families needed by the fixture into explicit tensor programs; keep unknown operations opaque with a lowering reason.
- [ ] Use selected version-pinned PyTorch decompositions when their results have known semantics. Keep useful high-level structure available for matching, and preserve all source casts and effects.
- [ ] Normalize transpose/view plus GEMM, broadcasts, casts, and the supported scalar expressions without changing semantics.
- [ ] Recognize the MLP region while preserving every live output and the source body.
- [ ] Obtain legal body configurations from the catalog. Build the same plan tested in G1, then permit a small bounded search.
- [ ] Return a callable through the normal wrapper. Test changed inputs, changed parameter contents, replacement tensors, and guard failures.
- [ ] Check the resulting region against G1 for correctness, launch count, and performance. Frontend integration must not introduce hidden GPU work.

### Establish baseline coverage independently of region matching

- [ ] Generate baseline map, reduction and indexed-contraction bodies. Handle supported strides, broadcasts, masked tails and degenerate dimensions through one common lowering path.
- [ ] Compile a norm with changed epsilon, optional affine/residual expressions, and an explicit intermediate cast. Disable semantic matchers and compare the result with the original graph. A matcher miss must not cause unsupported-model failure.
- [ ] Add a small vision fixture with convolution, activation, pooling and a residual. Implement the required indexed-contraction/reduction import rules, including padding, stride, dilation and groups. Do not register the model as a special case.
- [ ] Compile an unfamiliar composition of admitted primitives and a fork/join graph that MPK's single-event format cannot encode. The phase plan must handle these without relaxing source dependencies.
- [ ] Add a captured opaque custom operation with metadata but no device lowering. Strict compilation must report the exact gap before launching work.
- [ ] Implement the proposed `fallback=error|inductor` policy, defaulting to `error`. Test whole-invocation delegation without recursive MegaBake entry, changed runtime inputs, and preserved output/state wrappers. Delegation must be explicit in the report.
- [ ] Keep semantic coverage, execution kind, correctness, launch count and measured latency as separate fields. A correct slow generic kernel passes coverage; it does not pass G1's performance gate.

Start these fixtures with small guarded shapes. Large workspace use or missing state semantics can still reject a one-kernel plan. Extend primitive coverage by operator meaning, not by adding a matcher for each model.

The first implementation does not partition a stateful invocation between MegaBake and external kernels. Choose complete device compilation, complete external delegation, or a clear compile failure before execution.

Keep the initial production structure small:

| Responsibility | Initial implementation |
| --- | --- |
| PyTorch integration | One version adapter |
| Semantics | Compute graph with tensor programs, selected decompositions and optional exact region matchers |
| Device implementations | Baseline generators and tuned body families with capability checks |
| Planning | Tile maps, phase order, workspace and launch configuration |
| Code generation | The proven CuTe body/range composition |
| Evidence | Source-linked plan dump and a result record |

These are responsibilities, not a request to create a package hierarchy in advance. Add files when a working slice needs them.

## G3: Complete a block in prefill and decode

Extend the proven plan with RMSNorm, RoPE, residuals, and attention. Use the same source graph and plan concepts in both modes.

- [ ] Add exact semantic definitions from real captures, including tuple outputs and any online-softmax primitives.
- [ ] Parameterize norm expressions and attention score/mask hooks. Check each hook's access, reduction and numerical restrictions. Unsupported hooks must retain the baseline primitive path when that path is admissible.
- [ ] Measure a strong attention implementation before extracting its body. Preserve its tiling and async schedule through the same G0 contexts.
- [ ] Preserve causal offsets, grouped heads, explicit casts, and mask behavior. Include fully masked rows and supported partial tiles.
- [ ] Add `--mode prefill|decode` and backend policy configuration. Reject a mode that contradicts the captured state/effect contract.
- [ ] Start decode with a bounded contiguous KV cache. Include cache writes and all requested outputs in the same kernel.
- [ ] Validate at least two advancing decode steps, then repeated runs near a capacity boundary. Compare both outputs and cache contents with the native model path.
- [ ] Measure the complete block, including body transitions and phase barriers, with the resource envelope of all block body types.

Refine a phase boundary into tile events only after a trace shows useful overlap. Derive dependencies from actual mapped reads/writes. Verify fan-in, fan-out, reductions, and the combined data/worker wait graph. Keep a phase schedule as a correctness reference.

For Mirage ports, compare regular GCD/LCM event grouping with a concrete overlap oracle. Preserve general prerequisite sets before descriptor compression. Do not use search range under-approximations or operation-level residual stripping as a tile dependency proof.

**Stop condition:** if attention or the mixed block reduces throughput beyond available savings, return to body/schedule work. More model coverage does not repair a throughput loss.

## G4: Complete the requested model invocation

Use SmolLM as the first fixture, with a pinned checkpoint and explicit output scope. Dimensions and layer count remain specialization data.

- [ ] Cover embedding, every block, final norm, vocabulary projection, masks, outputs, and requested state updates.
- [ ] Reuse device body code across equivalent layers. Bind layer tensors through data; do not emit a new body for each parameter address.
- [ ] Allocate distinct intermediate storage first. Introduce reuse from proved phase lifetimes or explicit release dependencies.
- [ ] Check the complete compiled resource report and code size. Revalidate residency after every new body family.
- [ ] Compare model outputs and state over changing inputs and advancing decode calls. Use diagnostic intermediates to locate failures.
- [ ] Trace the full callable. Confirm one recurring workload kernel with no hidden resets, conversions, copy-backs, or state kernels.
- [ ] Benchmark the equivalent full `torch.compile` invocation. Match its output scope, precision, attention policy, cache behavior, GPU partition, and warmup conditions.
- [ ] Report peak workspace and compile/first-call cost separately. Report complete-call latency and raw timing samples.

Full-model support is established only here. Earlier block, GEMM, and short no-cache results remain local evidence.

## G5: Add only the mechanism that the measurements need

This is a catalog of targeted changes. Apply an item during an earlier gate when that gate establishes the need. Use the corresponding Mirage implementation as the first reference where one exists.

| Measured problem | Candidate next change | Required additional proof |
| --- | --- | --- |
| Too few useful projection tiles | Smaller tiles or split K/Stream-K | Partial storage, reduction order, ownership and progress |
| Phase barriers cause real idle time | Tile readiness and static dependency lists | Exact dependencies, visibility and an acyclic worker wait graph |
| Unpredictable ready-work imbalance | Bounded dynamic task claiming | Unique claims, capacity, fairness and termination |
| Weight-load stalls | Prefetch across compatible ranges/regions | Input readiness, buffer ownership and async completion |
| Uniform scratch wastes capacity | Better scratch reuse or a measured page scheme | Lifetime and launch resource limits |
| Cluster multicast would save traffic | A cluster body family | Collective scope, cluster occupancy and compatible launch |
| Instruction fetch or compilation grows | Fewer body variants and repeated-phase loops | Same bindings and semantics with less code |
| Decode attention lacks parallel work | Split context with an explicit merge | Stable softmax statistics, numerical policy and state ordering |

Defer distributed execution, a universal symbolic scheduler, global ILP layout search, learned cost models, and binary patching of vendor kernels. Revisit them only when an admitted target has a measured need.

## Evidence format

Each gate report must identify the workload, source/model revision, mode, shapes/strides, state format, output scope, precision, GPU resources, and toolchain. Include commands and source pins.

Keep correctness errors, launch traces, resource reports, workspace size, raw timings, rejection reasons, and unresolved questions. Distinguish measured results from estimates and design choices. Mark an unsupported case explicitly. A missing measurement is not a pass.
