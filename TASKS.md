# MegaBake execution plan

Build the compiler in the numbered order below. Each step consumes an earlier result and produces a concrete implementation.

[ARCHITECTURE.md](ARCHITECTURE.md) defines the design. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) defines acceptance checks. [north-star.md](north-star.md) fixes the scope.

## Start from the current checkout

The previous checklist records G0 workload, range-body, and mixed-runtime probes as complete for their declared target. Preserve that work. The completed entries and evidence remain at the end of this file.

The production package still has an empty `src/megabake/__init__.py`. Start at step 01 to connect live FX to the existing device work. Reuse the G0 implementations at steps 05, 09, and 13.

Function names and new file paths below are proposed implementation interfaces. Create each file when its step needs it. Keep two durable program representations: `ComputeGraph` and `KernelPlan`. Capture records, body configurations, and compiled artifacts hold metadata around them.

Unless a different root is shown, place new Python modules under `src/megabake`.

### The path you will build

```text
PyTorch model and live inputs
  -> Dynamo capture and pinned AOT/Inductor preparation
  -> live post-grad FX with its runtime contract
  -> ComputeGraph with typed tensor programs
  -> regions and valid body configurations
  -> tile maps, dependencies, phases, and storage
  -> KernelPlan
  -> generated CuTe DSL module
  -> compiled persistent kernel and runtime bindings
  -> callable returned through PyTorch's wrappers
```

### Working results along the way

| After step | You have |
| --- | --- |
| 01 | A backend that receives the required live FX graph |
| 04 | An imported graph with explicit computation, regions, and source references |
| 09 | Generic generators and a composable fast GEMM provider |
| 12 | A complete plan with layouts, dependencies, and storage |
| 15 | A small FX graph executing as one generated CuTe kernel |
| 16 | A complete MLP compiled through that path |
| 20 | Measured MLP performance after targeted optimization |
| 23 | Generic graph coverage and explicit external fallback |
| 29 | Complete transformer blocks in prefill and stateful decode |
| 31 | The complete captured model executing as one workload kernel |
| 33 | A documented compiler with complete-model correctness and performance evidence |

Use one small GEMM–elementwise–reduction fixture for steps 01–15. Keep a complete gated MLP as the next fixture. Add only the operations needed by those fixtures until step 20.

Use Mirage as the main implementation reference. The reuse table near the end maps steps to source components. Record source revisions, adaptations, comparisons, and required attribution with each port.

## 01. Get live post-grad FX through a real backend

**Input:** a PyTorch callable, example inputs, and the pinned PyTorch build.

**Implement in:** `frontend.py` and `fx_handler/export.py`, under `src/megabake`.

- [x] Implement `make_backend(**options)` as a Dynamo backend factory. Its returned backend accepts the Dynamo graph and example inputs.
- [x] Exercise that backend through `torch.compile(..., fullgraph=True)`.
- [x] Extract the preparation logic from [verify_smollm_fx.py](src/eager/verify_smollm_fx.py) into `fx_handler/export.py`; keep Hugging Face model capture in `fx_handler/transformers.py`.
- [x] Implement a handoff callback that receives the live prepared graph and binds its optimized FX forward without Inductor compilation.
- [x] Repair the cache-hit branch that currently labels an untransformed graph as post-grad. Keep capture-cache and compiled-plan reuse distinct.
- [x] Retain PyTorch's argument, output, guard, alias, and mutation wrappers. Keep Dynamo's guards active in their owning layer.
- [x] Return a reference callable initially. Record its execution as reference execution until step 15 replaces it.

**Output:** live post-grad FX plus a capture context with input order, metadata, output structure, and runtime requirements.

**Check:** run cache-hit and cache-miss cases through the backend. Change inputs and parameters. Both paths must use the intended graph phase and preserve outputs.

## 02. Import FX into the Compute Graph

**Input:** the live graph and capture context from step 01.

**Implement in:** `graph.py`, with `import_fx(graph, context) -> ComputeGraph`.

- [x] Assign stable IDs to source nodes and tensor results. Flatten boundary tuples while retaining the output reconstruction map.
- [x] Import each placeholder in runtime argument order. Represent parameters as live bindings.
- [x] Import every operation, including unknown targets. Keep source locations and original FX nodes for diagnostics.
- [x] Store shapes, dtypes, devices, boundary strides, aliases, and known effects. Keep value identity separate from storage identity.
- [x] Derive producer/use lists. Add a graph verifier for missing operands, invalid outputs, cycles, and inconsistent metadata.

Use these records as the starting schema:

```text
Value:  id, shape, dtype, device, strides, storage relation
Op:     id, source FX IDs, operands, results, attributes,
        tensor program if known, effects, numerical rules
Region: source op IDs, external inputs, live outputs, semantic facts
Graph:  values, ops, regions, inputs, outputs, runtime context
```

**Output:** a `ComputeGraph` that accounts for every captured input, operation, output, and effect.

**Check:** compare the source and imported inventories. A tuple result, reused value, view, and unknown operator must all survive import.

## 03. Give known operations explicit tensor meaning

**Input:** the imported graph.

**Implement in:** `lowering.py`, with `lower_semantics(graph)` and a registry keyed by exact FX targets.

- [ ] Define typed scalar expressions for constants, arithmetic, comparisons, selection, and casts needed by the fixture.
- [ ] Define iteration domains, indexed loads, reduction axes, predicates, and stores within each operation's tensor program.
- [ ] Lower elementwise operations with explicit broadcasting. Compose view coordinates with source strides.
- [ ] Lower GEMM as indexed contraction, with explicit accumulator and result types. Preserve source rounding points.
- [ ] Lower sum and mean reductions. Use selected decompositions for supported composite operations and retain source provenance.
- [ ] Leave an unknown operation opaque with a missing-rule reason. Detect unsupported effects before device planning.

**Output:** executable meaning for known operations, plus explicit unresolved operations.

**Check:** evaluate small tensor programs with a test-only reference evaluator. Compare casts, broadcasts, transposes, and reductions with the original graph.

## 04. Form regions without losing source operations

**Input:** a Compute Graph with known tensor programs.

**Implement in:** `regions.py`, with `form_regions(graph)`.

- [ ] Start with one region per operation. Store source IDs instead of copying operations into another graph.
- [ ] Compute each region's external inputs and every output used outside it, including graph outputs.
- [ ] Normalize supported view-plus-GEMM forms through index maps. Preserve aliases and explicit materialization when needed.
- [ ] Add exact semantic annotations only where a matcher succeeds. Keep unmatched operations available for generic lowering.
- [ ] Verify that region coverage includes every source operation and preserves effect ordering.

**Output:** baseline regions that body providers can inspect.

**Check:** return an intermediate as an extra graph output. Region formation must retain it. An unfamiliar operation combination must remain representable.

## 05. Define the provider and device-body interfaces

**Input:** regions, tensor programs, and target hardware facts.

**Implement in:** `bodies/interface.py` and `bodies/catalog.py`, under `src/megabake`.

- [ ] Implement `get_candidates(region, target, policy) -> configurations or rejection reasons`.
- [ ] Define `BodyConfig` using architecture section 5. Include semantic support, layouts, tiles, participants, stages, scratch, and completion rules.
- [ ] Define the device entry `run(work_iterator, tensor_bindings, scratch, epilogue)`.
- [ ] State which threads participate, when outputs become ready, and when scratch can be reused.
- [ ] Require generic and tuned providers to use this interface. Keep whole-kernel launch wrappers outside device-body selection.

**Output:** one common contract for generated primitives and tuned bodies.

**Check:** describe the existing G0 range and generated bodies with these records. Reject a configuration with incompatible participation or layout requirements.

## 06. Generate elementwise and copy device bodies

**Input:** a map, cast, or copy tensor program and its `BodyConfig`.

**Implement in:** `generic.py`, with `emit_map_body(program, config)`.

- [ ] Assign output coordinates to threads through a concrete tile map.
- [ ] Generate indexed loads, broadcasts, typed expressions, explicit casts, and stores.
- [ ] Predicate out-of-range coordinates. Preserve the required behavior for supported strided inputs and outputs.
- [ ] Generate device-callable CuTe code. Reuse the typed-expression approach in the G0 mixed-body probe.

**Output:** generated map bodies that the future enclosing kernel can call.

**Check:** use a small test launcher for two different expressions with the same schedule. Check partial tiles and a noncontiguous input.

## 07. Generate reduction device bodies

**Input:** a reduction tensor program and generated scalar expressions.

**Implement in:** `generic.py`, with `emit_reduction_body(program, config)`.

- [ ] Start with one CTA owning an output row. Loop over the reduction domain in bounded chunks.
- [ ] Generate expressions before and after reduction. Preserve accumulator types, casts, axes, and epsilon placement.
- [ ] Support reductions larger than a single loaded tile. Keep barriers consistent across participating threads.
- [ ] Add partial-result and merge phases only where the numerical policy allows them. Describe both phases in the plan.

**Output:** reusable reduction schedules that generate different norms and reductions from their source expressions.

**Check:** compile sum, mean, and two norm variants with different epsilon or casts. Compare uneven and large reduction widths.

## 08. Generate a baseline indexed-contraction body

**Input:** a contraction tensor program with output and reduction coordinates.

**Implement in:** `generic.py`, with `emit_contraction_body(program, config)`.

- [ ] Assign disjoint output coordinates to threads. Accumulate indexed products over bounded reduction loops.
- [ ] Generate operand predicates, explicit accumulator types, and final casts from the tensor program.
- [ ] Support the initial GEMM fixture, including supported strides and tails.
- [ ] Keep this provider available when no tuned configuration supports an otherwise known contraction.

**Output:** a correct baseline contraction provider that uses the common device interface.

**Check:** compare small GEMMs, transposed views, and partial dimensions with the source graph. Record this provider as generic in reports.

## 09. Connect the existing fast GEMM range body

**Input:** a compatible GEMM region and the measured G0 range implementation.

**Implement in:** `gemm.py`, with `gemm_candidates(region, target)` and a device range entry.

- [ ] Adapt [resident_range_probe.py](experiments/g0_2026_10_09/resident_range_probe.py) into a maintained provider.
- [ ] Preserve the source license and revision. Replace experimental source rewriting with an explicit implementation before treating it as production code.
- [ ] Retain the pipeline across a range of tiles. Bind scratch, descriptors, and current tensor addresses through the common interface.
- [ ] Return coupled instruction, tile, layout, stage, and warp configurations. Check shape, stride, alignment, and tail restrictions.
- [ ] Keep the original CuTe launch as a comparison. Reuse existing changed-binding and tile-count fixtures.

**Output:** a tuned GEMM provider alongside the baseline contraction provider.

**Check:** reproduce its original, range, and mixed-body behavior. Query resources and measure the port's latency before selecting it.

## 10. Select body configurations and construct tile maps

**Input:** regions and provider candidates.

**Implement in:** `plan.py`, with `select_bodies(graph, target, policy)` and `make_tile_maps(selection)`.

- [ ] Start with one valid measured configuration per supported region. Keep rejected candidates and reasons in the plan report.
- [ ] Map output tiles to logical tensor coordinates. Derive complete input domains, including contraction reduction dimensions.
- [ ] Map logical coordinates to physical layouts through CuTe. Adapt relevant Mirage stride, copy, and swizzle checks.
- [ ] Insert an explicit conversion body when adjacent layouts need data movement. Include its cost and dependencies.
- [ ] Store participants and resource requirements with each chosen body.

**Output:** selected bodies with concrete access maps and physical layout requirements.

**Check:** enumerate small tiles and compare addresses with a coordinate reference. Check unit dimensions, partial tiles, and boundary strides.

## 11. Derive dependencies and form an ordered phase schedule

**Input:** selected body accesses, source dependencies, aliases, and effects.

**Implement in:** `schedule.py`, with `derive_dependencies(selection)` and `make_phase_schedule(selection, dependencies)`.

- [ ] Add producer-to-consumer dependencies for every required value. Include all input tiles needed by each reduction.
- [ ] Add write-after-read and write-after-write ordering for state or reused source storage.
- [ ] For irregular accesses, use a conservative domain when exact analysis is unavailable.
- [ ] Topologically order the work. Start with one body region per phase and a grid barrier at each required boundary.
- [ ] Assign tile ranges to workers. Keep workers without tiles in the barrier protocol.
- [ ] Retain arbitrary prerequisite sets in the plan. Keep operation names independent of runtime event encoding.

**Output:** an executable phase order with explicit work assignments and dependencies.

**Check:** simulate the schedule on small fixtures. Verify a fork/join graph, a view, and a down projection needing several producer tiles.

## 12. Assign storage and finish the KernelPlan

**Input:** the scheduled work and all live values.

**Implement in:** `memory.py`, with `assign_storage(plan)`, and `KernelPlan` in `plan.py`.

- [ ] Assign a distinct global workspace slice to each materialized intermediate. Retain source aliases in boundary bindings.
- [ ] Use aligned allocation sizes required by the selected bodies. Adapt Mirage's simple global allocation approach.
- [ ] Allocate worker scratch for the required body mixture. Specify barrier offsets, descriptors, and phase initialization.
- [ ] Keep outputs and state bindings separate from temporary storage. Preserve the required boundary layout for each output.
- [ ] Attach bodies, tiles, dependencies, phases, storage, bindings, and launch requirements to `KernelPlan`.
- [ ] Implement `verify_plan(plan)` for coverage, ownership, extent, alignment, dependencies, and collective participation.

**Output:** a complete `KernelPlan` whose addresses and work are defined, with physical runtime pointers still unbound.

**Check:** verify that simultaneously live intermediates cannot overlap. Reject a missing output, undersized slice, or incompatible scratch assignment.

## 13. Turn the G0 runtime into a reusable phase runner

**Input:** the phase schedule, worker scratch, and launch requirements.

**Implement in:** `runtime.py`, using [mixed_body_probe.py](experiments/g0_2026_10_09/mixed_body_probe.py).

- [ ] Extract the cooperative worker loop and its release/acquire barrier into reusable device functions.
- [ ] Preserve CTA synchronization around arrival and release. Drain required asynchronous work before publishing completion or reusing scratch.
- [ ] Define per-invocation state ownership and the epoch lifecycle. Prevent stale completion values from satisfying a later phase.
- [ ] Retain the proved eager-call epoch path initially. Check capacity and overflow before launch, and keep concurrent invocations isolated.
- [ ] Account for initial state setup and any recurring reset. Required recurring GPU work must stay inside the workload kernel.
- [ ] Keep CUDA Graph capture unsupported until replay can advance synchronization state correctly. Add an explicit admission check.

**Output:** a runtime that executes any admitted phase plan using the existing cooperative mechanism.

**Check:** port the delayed-producer, idle-worker, changed-binding, and largest-grid cases. Preserve the G0 probe's declared limits.

## 14. Emit a complete CuTe DSL module from KernelPlan

**Input:** a verified `KernelPlan` and the device-body generators.

**Implement in:** `codegen.py`, with `emit_cute(plan) -> generated module and launch metadata`.

- [ ] Emit `@cute.jit` device helpers and one enclosing `@cute.kernel`. Declare shared scratch and tensor bindings from the plan.
- [ ] Emit each phase's tile iterator and device-body call. Keep the tuned range loop inside the body.
- [ ] Emit required barriers and state operations from the plan. Generate conversions and intermediate stores as explicit work.
- [ ] Generate a host launch wrapper for that one kernel. Keep all required device computation in the emitted module.
- [ ] Deduplicate identical body variants by program and configuration. Bind parameter tensors through runtime arguments.
- [ ] Save the generated source and a source-to-plan map for diagnostics.

**Output:** an inspectable CuTe module for the planned graph.

**Check:** compile a manually constructed plan and compare it with the existing G0 executable. Inspect its body calls and barrier placement.

## 15. Compile, admit, bind, and return the generated callable

**Input:** the generated module, plan, and capture context.

**Implement in:** `compiler.py`, with `compile_post_grad`, `compile_plan`, and `bind_callable`.

- [ ] Wire steps 02–14 into the callback installed at step 01.
- [ ] Compile the complete kernel. Query registers, shared memory, local memory, active blocks, and cooperative launch limits.
- [ ] Choose a legal worker count from actual compiled resources. Keep worker-dependent iterators and scratch consistent with that count.
- [ ] Recompile and repeat admission if a changed worker count changes a compile-time layout.
- [ ] Allocate output and workspace buffers. Bind current inputs, parameters, state, descriptors, and the active CUDA stream on each call.
- [ ] Return outputs through the saved PyTorch wrapper and reconstruction map. Preserve aliases and mutation handling.
- [ ] Reject an invalid complete plan before device execution. Include the source operation and failed requirement in diagnostics.

**Output:** the first complete path from `torch.compile` input to one generated CuTe workload kernel.

**Check:** run the small GEMM–elementwise–reduction graph through the backend. Change inputs and parameters. Compare every output and trace the complete call.

## 16. Compile the complete MLP through the compiler

**Input:** the working frontend-to-kernel path.

**Implement in:** the existing importers and providers; add a complete MLP fixture under `tests`.

- [ ] Capture gate/up projections, SiLU, multiplication, and down projection as one invocation.
- [ ] Add only the missing primitive import rules. Keep all casts and externally visible intermediate outputs.
- [ ] Compile the graph through normal region selection and planning. Use the initial phase schedule.
- [ ] Compare small-M and larger-M cases with the native graph. Trace all recurring work.
- [ ] Record a first complete-call performance report using the existing MLP baseline harness.

**Output:** an automatically compiled complete MLP, with correctness and initial latency results.

**Check:** changing model weights must change outputs without stale bindings. A correct slow result remains a performance gap for steps 17–20.

## 17. Add legal local fusion and tuned region candidates

**Input:** the working MLP graph, baseline regions, and body providers.

**Implement in:** `regions.py` and the relevant generators.

- [ ] Adapt Mirage's local fusion-chain rules as the starting point. Check every external use and source cast.
- [ ] Generate compatible scalar work in GEMM epilogues or vector chains. Preserve exact thread/fragment ownership.
- [ ] Add an exact gated-MLP matcher. Return fused candidates while retaining the baseline region expansion.
- [ ] Compare separate gate/up work with valid combined or paired candidates. Count packing and conversion costs.
- [ ] Keep secondary outputs and effects explicit. Reject fusion when its resource or numerical contract fails.

**Output:** optimized candidates for the same source graph.

**Check:** compare fused and unfused executions, including an extra intermediate output. Measure the full MLP after each accepted change.

## 18. Search a small coupled configuration space

**Input:** valid body families, actual shapes, target resources, and complete candidate plans.

**Implement in:** `plan.py`, with `search_plans(graph, target, policy, budget)`.

- [ ] Adapt applicable Mirage dimension and mapping enumeration rules. Use provider configurations instead of fixed block heuristics.
- [ ] Enumerate a small set of tiles, stages, layouts, worker counts, and compatible fusion choices.
- [ ] Prune invalid alignments, unsupported collectives, excessive storage, and insufficient work before compilation.
- [ ] Compile remaining complete kernels and check actual resources. Run correctness before retaining timings.
- [ ] Compare complete-call latency and retain raw samples. Select the fastest correct measured plan within the search budget.
- [ ] Investigate a specific body gap through CuTe, Mirage, DeepGEMM, or FBGEMM source when current candidates remain slow.

**Output:** a measured plan selection with explicit rejected candidates and reasons.

**Check:** rerun the selected candidate independently. A faster isolated GEMM must not displace a faster complete MLP plan.

## 19. Add tile events if phase waits limit the target

**Input:** a trace showing useful producer/consumer overlap and the existing phase reference.

**Implement in:** `schedule.py` and `runtime.py`. Skip this implementation when measurements show no need.

- [ ] Derive exact producer membership for each consumer tile. Include reduction fan-in, aliases, and state effects.
- [ ] Group identical prerequisite sets. Adapt MPK's regular GCD/LCM grouping only where its preconditions hold.
- [ ] Retain general dependency lists before compressing descriptors. Compare compressed events with concrete access overlap.
- [ ] Assign tasks in a common topological order. Include each worker's task order in the progress analysis.
- [ ] Emit release/acquire publication and waits for all producers. Include CTA participation and asynchronous completion rules.
- [ ] Let bodies consume ready ranges without forcing a pipeline restart at every tile.
- [ ] Extend host simulation to require every task to finish. Compare outputs and timing with the phase plan.

**Output:** an alternative event schedule for the same Compute Graph and body interface.

**Check:** test forks, joins, residuals, uneven tails, delayed producers, and idle workers. Keep a phase schedule when event compression cannot represent the graph.

## 20. Close the first region performance gap

**Input:** the compiler-generated MLP and the measured candidate plans.

**Implement in:** the body or schedule component identified by the measurements.

- [ ] Measure equivalent eager, CUDA Graph reference, `torch.compile`, and MegaBake invocations. Match inputs, outputs, precision, and GPU resources.
- [ ] Record warm GPU and complete host-call latency. Separate compilation, setup, packing, and first-call costs.
- [ ] Attribute losses to body throughput, traffic, padding, restart, conversions, waits, or occupancy.
- [ ] Change the component responsible for the dominant loss. Repeat the complete MLP comparison.
- [ ] Record G1 pass or hold for each regime. Require an improvement beyond measurement noise before broad model expansion.

**Output:** a useful compiler-generated MLP on each passing target, with its remaining failures visible.

**Check:** include the resource cost of the full body mixture. Complete G0 integration and G1 evidence before proceeding to wider coverage.

## 21. Cache compiled plans while retaining live bindings

**Input:** measured plans, compiled kernels, and their guarded runtime contracts.

**Implement in:** `cache.py`, with `make_plan_key` and `get_or_compile`.

- [ ] Key plans by graph, guards, shapes, strides, types, semantic constants, numerical policy, mode/state layout, and target resources.
- [ ] Include body revisions and toolchain versions. Keep parameter addresses and contents as runtime bindings unless explicitly folded.
- [ ] Store selected plans, compiled artifacts, resource requirements, and rejection diagnostics.
- [ ] Rebuild descriptors for current bindings where needed. Invalidate packed or folded parameters on replacement or mutation.
- [ ] Manage workspace lifetime across calls and streams. Prevent concurrent calls from sharing mutable execution state.

**Output:** repeated calls that reuse compilation without retaining example tensor addresses.

**Check:** test cache hits, changed shapes, changed strides, parameter replacement, mutation, and overlapping calls with separate state.

## 22. Implement explicit whole-invocation fallback

**Input:** complete-plan success or a structured compile rejection.

**Implement in:** `frontend.py` and `compiler.py`.

- [ ] Add `fallback=error|inductor`, with `error` as the default.
- [ ] In strict mode, report the unresolved operation, source location, or composition failure before device execution.
- [ ] In fallback mode, delegate the whole captured invocation through a pinned Inductor adapter without recursively entering MegaBake.
- [ ] Retain the same runtime wrappers, guards, outputs, aliases, and state behavior.
- [ ] Label the result `inductor_fallback` and record its reason and launch count.
- [ ] Keep runtime device errors separate from compile rejection. Never retry a partly executed stateful invocation through fallback.

**Output:** a deliberate compatibility path when a complete megakernel cannot be produced.

**Check:** capture an opaque operation with no MegaBake lowering. Test strict rejection and supported delegation on changed runtime inputs.

## 23. Extend generic coverage to unfamiliar graphs

**Input:** the working primitive generators and common planner.

**Implement in:** `lowering.py` and `generic.py`.

- [ ] Express convolution as indexed contraction with padding, stride, dilation, and groups. Avoid a required materialized `im2col` tensor.
- [ ] Express pooling as indexed reductions. Add the gather and index rules needed by the selected fixtures.
- [ ] Preserve invalid-index behavior and all required output types. Use conservative dependency domains when indices are data dependent.
- [ ] Compile a vision block containing convolution, activation, pooling, and a residual through the normal compiler.
- [ ] Compile unfamiliar norm variants and fork/join graphs with named matchers disabled.
- [ ] Report generic-body use, workspace, and performance separately from semantic support.

**Output:** new compositions of supported primitives compile without a model-specific registration.

**Check:** compare unfamiliar graphs with native execution. Verify that a missed optimization uses a valid generic plan rather than claiming unsupported semantics.

## 24. Add reusable norm, residual, and RoPE schedules

**Input:** their exact captured tensor programs and the baseline vector/reduction generators.

**Implement in:** `regions.py` and `generic.py`; split files only as implementations grow.

- [ ] Match exact norm and RoPE semantics. Extract axes, epsilon, residual/affine terms, casts, and rotation rules.
- [ ] Generate those expressions around reusable row and vector schedules.
- [ ] Return tuned candidates only when the full source behavior fits. Retain generic candidates for other variants.
- [ ] Fuse compatible projection epilogues or residual work while preserving all outputs and numerical rules.
- [ ] Compose these schedules with the MLP and repeat resource, scratch, and body-transition checks.

**Output:** performant norm and RoPE variants generated from source expressions.

**Check:** use different epsilons, widths, casts, residual terms, and supported strides. Compare the complete region after composition.

## 25. Import and recognize attention precisely

**Input:** real attention captures and the primitive semantic importer.

**Implement in:** `lowering.py` and `regions.py`.

- [ ] Preserve Q/K/V access maps, head grouping, scaling, masks, casts, requested outputs, and cache effects.
- [ ] Represent supported score and mask expressions as typed device expressions.
- [ ] Recognize a tuned attention candidate only when its algorithm and numerical policy match the captured program.
- [ ] Keep the source expansion available for generic execution. Add missing softmax primitives with their exact supported behavior.
- [ ] Define local hook restrictions. Reject a row-dependent transform from a hook that only has local score data.

**Output:** attention regions with explicit semantics and a retained generic expansion.

**Check:** vary causal offsets, grouped heads, bias, soft caps, windows, and extra outputs. A matcher miss must preserve the source graph.

## 26. Implement the composable attention provider

**Input:** attention regions, typed hooks, and target constraints.

**Implement in:** `attention.py`, using the common provider interface.

- [ ] Measure a strong target implementation and inspect applicable Mirage Hopper device bodies.
- [ ] Port its device pipeline with explicit warp roles, tile layouts, scratch, barriers, and completion points.
- [ ] Compile supported score and mask hooks into that pipeline. Preserve softmax state and probability casts.
- [ ] Return prefill and decode body candidates where their access and numerical contracts hold.
- [ ] Keep generic execution available for valid variants outside the tuned family. Check materialized-score workspace before admission.
- [ ] Compose the provider with GEMM and norm bodies. Query the complete kernel's resource use again.

**Output:** attention device work that can execute inside the same persistent kernel.

**Check:** test original launch, resident entry, and mixed-body execution. Include partial tiles, fully masked rows, poisoned scratch, and repeated bindings.

## 27. Add explicit prefill and decode planning policies

**Input:** graph semantics, actual dimensions, mode, state format, and target facts.

**Implement in:** `plan.py` and backend options.

- [ ] Add `mode=prefill|decode` to `make_backend` and the plan key.
- [ ] Validate the mode against the captured state contract. Keep computation and requested outputs defined by the graph.
- [ ] Prioritize suitable small-M or larger-M body families using actual dimensions.
- [ ] Choose attention candidates using query length, KV length, batch/head parallelism, layout, and mask facts.
- [ ] Set bounded search priorities and work ordering for each mode. Keep one common graph and plan representation.

**Output:** one compiler that selects different legal execution plans for prefill and decode.

**Check:** cover a short prefill and a larger decode batch. The mode name alone must not force the wrong GEMM family.

## 28. Lower KV state access and updates

**Input:** a stateful captured invocation and its explicit cache bindings.

**Implement in:** `lowering.py`, `memory.py`, and the attention provider.

- [ ] Start with a bounded contiguous cache. Define capacity, valid length, layout, storage identity, and update ranges.
- [ ] Import cache reads and writes with source ordering. Bind current state through the PyTorch/Hugging Face wrapper.
- [ ] Guard capacities. Keep runtime lengths dynamic within those guards unless their values are valid specialization assumptions.
- [ ] Emit required writes and state updates inside the workload kernel. Retain every externally visible state result.
- [ ] Add dependencies for overlapping reads and writes. Keep cache lifetime separate from temporary workspace lifetime.

**Output:** stateful decode plans with explicit cache ownership and update work.

**Check:** run prefill followed by at least two advancing decode calls. Compare full cache contents and outputs, including calls near capacity.

## 29. Compile a complete transformer block

**Input:** projection, norm, RoPE, attention, residual, MLP, and state support.

**Implement in:** existing compiler components; add a complete-block fixture.

- [ ] Capture the full block through the normal backend in both modes.
- [ ] Compile every source operation and requested output. Include cache work when the captured block requires it.
- [ ] Verify the combined dependencies, scratch lifetimes, participation, and compiled launch resources.
- [ ] Compare outputs and state with the native block over changed inputs and repeated calls.
- [ ] Measure the complete block against equivalent baselines. Return to the responsible body or scheduling step for any dominant loss.

**Output:** a complete prefill block and a complete stateful decode block as generated megakernels.

**Check:** complete G3 for each declared regime. Include body transitions, synchronization, and recurring wrapper work in the result.

## 30. Bound workspace and repeated-layer code

**Input:** the working block plan and full-model graph structure.

**Implement in:** `memory.py`, `plan.py`, and `codegen.py`.

- [ ] Derive value lifetimes from phase barriers or explicit release dependencies. Include asynchronous readers.
- [ ] Reuse workspace slices only for nonoverlapping lifetimes. Adapt Mirage's allocator with these verified intervals.
- [ ] Verify shared scratch lifetimes separately from global value lifetimes.
- [ ] Deduplicate body variants across equivalent layers. Bind each layer's weights and state through data.
- [ ] Use repeated phase loops where they reduce code without changing bindings or computation.
- [ ] Measure peak workspace, compiled code size, and resource use. Revalidate the worker grid after changes.

**Output:** storage and generated code that can scale to the complete declared model.

**Check:** poison released storage and compare changing layer inputs. Check concurrent-call isolation and all output lifetimes.

## 31. Compile the complete captured model

**Input:** the pinned SmolLM model, the working block compiler, and the declared output/state scope.

**Implement in:** existing compiler components; add missing primitive rules through the common importer.

- [ ] Capture embedding, every layer, final norm, vocabulary projection, masks, outputs, and requested state updates.
- [ ] Lower embedding/index operations with exact source semantics. Keep model dimensions and layer count as specialization data.
- [ ] Build one KernelPlan for the complete invocation. Preserve dependencies between all regions and state operations.
- [ ] Emit, compile, and admit the full mixed kernel. Recheck occupancy, code size, workspace, and collective participation.
- [ ] Return the original output structure and state through the normal wrappers.
- [ ] Trace the complete call and account for every recurring GPU operation.

**Output:** the full declared model invocation executing as one generated CuTe workload kernel where legal.

**Check:** compare full outputs and state with the native model on changed inputs and advancing decode. Retain intermediate diagnostics for failures.

## 32. Expose the compiler and a repeatable run command

**Input:** the working full-model compiler.

**Implement in:** `src/megabake/__init__.py` and a small example/benchmark entry point.

- [ ] Export the implemented `make_backend` API and document supported options, software pins, and hardware requirements.
- [ ] Add a runner with model/revision, input shape, `--mode prefill|decode`, and explicit fallback options.
- [ ] Make the runner compile, check, and benchmark the same declared invocation.
- [ ] Write graph/plan dumps, source references, chosen bodies, resources, launch count, correctness, and timing results.
- [ ] Document unsupported cases and generic-body performance gaps. Keep external fallback clearly labeled.

The intended public use is:

```python
# Target API to implement; this is not available in the current checkout.
compiled_model = torch.compile(
    model,
    backend=megabake.make_backend(mode="prefill", fallback="error"),
    fullgraph=True,
)
with torch.no_grad():
    outputs = compiled_model(*inputs)
```

**Output:** a reproducible entry point that a user can run without editing compiler internals.

**Check:** run from a fresh process. Change weights and inputs, trigger a guard miss, and check strict and explicit fallback behavior.

## 33. Complete the model validation and performance report

**Input:** the public path, declared target matrix, and equivalent baselines.

**Implement in:** the regression fixtures and benchmark runner.

- [ ] Run complete prefill and advancing decode cases for the declared shapes and GPU resources.
- [ ] Check outputs, state, aliases, mutations, changed bindings, and numerical tolerances fixed before measurement.
- [ ] Confirm one recurring workload launch for each claimed megakernel result.
- [ ] Measure complete-call and GPU latency against equivalent `torch.compile` and any stronger measured baseline.
- [ ] Report raw samples, launch traces, resources, workspace, compilation, setup, and first-call costs.
- [ ] Require an improvement beyond measurement noise for each claimed performance success. Keep failures visible and fix their dominant cause.
- [ ] Run retained generic-coverage, norm-variant, attention, fallback, cache, and synchronization regressions.
- [ ] Publish the support matrix and commands. Link every completion claim to its saved evidence.

**Output:** the architecture implemented for the declared targets, with reproducible complete-model evidence.

**Check:** full-model correctness establishes support. The performance objective is complete only for targets that pass the complete-model comparison.

## Mirage source map for this sequence

Use the pinned source links and restrictions in [SCHEDULER_REUSE.md](SCHEDULER_REUSE.md). Inspect connected passes before adapting a helper.

| Steps | Starting component | What to carry into MegaBake |
| --- | --- | --- |
| 02–04 | Graph/tensor representations and transpiler pass order | Tensor meaning, source ownership, and transformation checks |
| 06–08, 17, 24 | Threadblock primitives, epilogues, `resolve_tb_fusion.cc` | Generated expressions, reusable schedules, and legal local fusion |
| 09, 26 | Hopper GEMM and attention device bodies | Pipeline structure, scratch ownership, and repeated-entry tests |
| 10 | Tile-layout adapter, `sched_tb_graph.cc`, Hopper swizzle planning | Offset mapping, copy checks, and instruction/layout compatibility |
| 11, 19 | `annotated_graph.cc`, `runtime.cc` | Dependency grouping, descriptors, and complete task simulation |
| 12, 30 | `plan_dtensor_memory.cc`, `plan_stensor_memory.cc` | Allocation algorithms supplied with MegaBake's verified lifetimes |
| 13, 19 | Persistent worker runtime and atomics | Publication, work dispatch, and progress under the one-kernel contract |
| 14, 30 | Task variant registration | Reuse code by program/configuration and bind tensors through data |
| 18 | `dim_strategy.cc`, layout constraints | Bounded candidate generation tied to provider capabilities |
| 20, 29, 33 | Device profiler and runtime tests | Find waits, preserve attribution, and measure final latency without instrumentation |

## Additional changes when a measured target needs them

Apply these changes in the responsible step and repeat its complete-workload check.

| Measured problem | Implement in | Concrete change |
| --- | --- | --- |
| Too few GEMM tiles | Steps 09 and 18 | Add a smaller tile or split reduction with explicit partial storage and numerical checks |
| Unbalanced ready work | Step 19 | Add bounded task claiming with unique ownership, queue limits, fairness, and termination |
| Weight-load stalls | Steps 09, 13, and 19 | Prefetch between compatible ranges with explicit buffer ownership and completion |
| Decode attention lacks parallelism | Steps 26–28 | Split context and emit a merge with stable softmax statistics |
| Cluster multicast can reduce traffic | Steps 05, 09, and 15 | Add a cluster body with explicit collective scope, occupancy, and launch admission |
| CUDA Graph replay is required | Steps 13, 15, and 21 | Make epoch/reset state advance during replay; test repeated replay without host epoch updates |

## Relationship to the acceptance gates

The numbered steps define execution order. Gates record evidence for the resulting implementation.

| Gate | Where its implementation is assembled and checked |
| --- | --- |
| G0 | Existing probes, then production integration in steps 05–15 |
| G1 | Complete MLP construction and improvement in steps 16–20 |
| G2 | Thin frontend in steps 01–04; integrated compiler and general coverage through step 23 |
| G3 | Norm, attention, mode, and state work in steps 24–29 |
| G4 | Model scaling, full invocation, public entry, and evidence in steps 30–33 |
| G5 | Targeted body, event, storage, and search changes at the step that needs them |

Start the thin frontend now. Keep early fixtures small. Use G1 to control expansion into broad operator and model coverage.

## Previously completed probe work

The entries below retain the completed work from the previous checklist. Their task numbers refer to that checklist. They describe probe results; production integration has separate checks above.

### Recorded task 01. Fix the first workload and measurement setup

**Gate: G0. Start here.**

- [x] Select the GPU target and pin the software versions. Record the actual SM count and cooperative-launch support.
- [x] Pin the first model checkpoint. Declare outputs, precision, numerical tolerances, and required state updates.
- [x] Extract actual projection shapes and strides. Include gate/up/down, Q/K/V, output projection, and the vocabulary head.
- [x] Select small-M and larger-M fixtures, partial tiles, and weight sets below and above L2 capacity.
- [x] Record separate prefill and stateful decode fixtures. Keep their shapes and cache requirements available from the start.
- [x] Reproduce library GEMM and complete MLP baselines. Record effective Inductor choices and whether autotuning ran.
- [x] Make one measurement harness report numerical errors, raw timings, launch count, and compiled resource use.
- [x] Separate setup, compilation, first-call, warm GPU, and complete host-call timing. Include a weight-streaming case.

**Deliverable:** complete. See [WORKLOAD.md](experiments/g0_2026_10_09/WORKLOAD.md), [g0_workload.json](experiments/g0_2026_10_09/g0_workload.json), the saved GEMM reports, and [mlp_baseline.json](experiments/g0_2026_10_09/mlp_baseline.json).

### Recorded task 02. Make a fast GEMM body process a range of tiles

**Gate: G0. Requires task 01.**

- [x] Select a measured official CuTe GEMM configuration for each initial regime. Retain its original launch as a reference.
- [x] Extract device work behind the proposed `run(work_iterator, tensor_bindings, scratch, epilogue)` interface.
- [x] Keep the pipeline active across the assigned tile range. Allocate reusable scratch in the enclosing worker.
- [x] State thread participation, warp roles, barrier state, output readiness, and scratch-release conditions.
- [x] Bind current addresses and descriptors on each call. Check repeated entry with changed tensors, contents, and tile counts.
- [x] Compare the original launch, the range entry, and the restarting adapter. Record latency and resources for each.

**Deliverable:** complete. See [resident_range_probe.py](experiments/g0_2026_10_09/resident_range_probe.py) and [resident_range.json](experiments/g0_2026_10_09/resident_range.json) for the range contract, correctness, resource use, and comparison timings.

### Recorded task 03. Run different bodies inside one persistent kernel

**Gate: G0. Requires task 02.**

- [x] Generate a small vector body and a reduction body from typed expressions. Include different casts or scalar parameters.
- [x] Run GEMM, generated work, and the next required GEMM configuration inside one kernel.
- [x] Start with a manually specified phase plan and distinct intermediate buffers.
- [x] Implement and test the actual cooperative launch and grid barrier in CuTe DSL. Include workers with no assigned tiles.
- [x] Derive the worker limit from the compiled complete kernel. Reject an oversized launch before execution.
- [x] Check cross-CTA data visibility, asynchronous completion, scratch reuse, and transitions between warp roles.
- [x] Keep per-call state private. Include required initialization inside the workload kernel.
- [x] Test repeated calls, changed bindings, delayed producers, and the largest valid worker grid.
- [x] Run the relevant memory/race checks. Save the launch trace and complete resource report.

**Deliverable:** complete for the G0 probe. The cooperative M=4 kernel passes repeated changed bindings and delayed producers at the compiled 60-worker limit; repeated M=4→1 and M=4→256 rebinds also pass. A 61-worker request is rejected before the workload launch. The trace contains one kernel, and memcheck, racecheck, and synccheck are clean at 60 workers. CUDA Graph replay is outside this probe; tested calls supply their barrier epochs on the host. See [g0_status.json](experiments/g0_2026_10_09/g0_status.json), [mixed_body.json](experiments/g0_2026_10_09/mixed_body.json), [mixed_shape_m1.json](experiments/g0_2026_10_09/mixed_shape_m1.json), [mixed_shape_m256.json](experiments/g0_2026_10_09/mixed_shape_m256.json), and [mixed_kernel_trace.json](experiments/g0_2026_10_09/mixed_kernel_trace.json).
