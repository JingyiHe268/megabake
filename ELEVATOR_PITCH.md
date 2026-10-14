# MegaBake elevator pitch

## The one-sentence pitch

MegaBake is an open, PyTorch-native compiler that captures an inference graph in FX and compiles supported tensor regions into one planned, composable CUDA megakernel, without asking teams to abandon PyTorch, Hugging Face, their model code, or their deployment interface.

## The 30-second pitch

Inference performance is increasingly decided below the Python graph: kernel selection, fusion legality, memory traffic, synchronization, occupancy, and launch overhead. `torch.compile` and CUDA Graphs solve important parts of that problem, but they still treat much of the model as a schedule of separate kernels and rely on the strongest available library implementations. MegaBake takes the captured, normalized FX graph, recovers explicit tensor semantics, and plans CuTe DSL device bodies that can legally share one persistent CUDA kernel. The goal is not to replace PyTorch; it is to give PyTorch an additional backend that can prove where whole-region fusion and cross-region overlap beat the existing pipeline.

## Why MegaBake matters

- **PyTorch-native:** Dynamo captures the invocation. MegaBake receives the graph after pinned AOT/Inductor preparation and post-grad transformation, then preserves guards, aliases, mutations, outputs, and runtime wrappers. Users keep the `torch.compile(model, backend=megabake.make_backend())` workflow.
- **Composable, not merely fused:** Every device body must expose work that can execute inside the enclosing kernel and satisfy its data ownership, synchronization, and resource constraints. Separate library launches do not become composable just because they were generated with CuTe DSL.
- **Semantics first:** The captured graph is the correctness reference. Candidate regions carry explicit layouts, dependencies, state effects, and numerical requirements. Unsupported work fails clearly or can eventually delegate to Inductor rather than silently changing model behavior.
- **Hardware-extensible:** The graph, planning, and device-body contracts are separated so additional hardware targets can be added behind the same compiler path. The current implementation is CUDA-only, but ROCm and Pallas-style device-body providers are future possibilities rather than redesigns.
- **Real optimization is preserved:** Carefully tuned GEMM, attention, and elementwise pipelines remain device bodies. MegaBake composes them when the contract admits it; it does not flatten a FlashAttention or CUTLASS-style pipeline into generic scalar operations.
- **Evidence-gated:** Performance claims name the workload, shape, precision, GPU, baseline, and measurement method. The project explicitly distinguishes compilation, binding, launch, and kernel work instead of reporting one misleading number.
- **Open and reusable:** MIT-licensed code, recorded design research, source pins, benchmark artifacts, and an execution plan give partners and contributors a path to evaluate and extend the work.

## The opportunity

Modern serving systems spend enormous engineering effort tuning around dispatch, kernel boundaries, memory movement, and small-batch latency. The usual choices are unsatisfying:

- Eager PyTorch is flexible but launches many kernels.
- CUDA Graphs reduce launch overhead but capture a fixed execution topology.
- Handwritten megakernels can be fast but are expensive, model-specific, and difficult to maintain.
- Generic fusion can accidentally destroy the carefully tuned pipeline inside attention or GEMM.

MegaBake proposes a middle path: a compiler that starts from PyTorch's graph representation and emits one workload kernel only when it can prove that the selected bodies, storage, synchronization, and launch plan are legal and useful for the target.

## What exists today

The first frontend-to-kernel path is implemented for supported SM90 workloads. A gated MLP is captured through Dynamo, planned, generated with CuTe DSL, and executed as one cooperative CUDA kernel.

The current results are deliberately honest:

- The complete warm call is **12.5×–15× slower** than CUDA Graph replay for the measured MLP.
- About **350 µs per call** comes from Python binding work, not from the kernel or compilation.
- Kernel-only work is **1.55× slower** at batch 4 and **11.24× slower** at batch 256.
- The batch-256 gap is largely attributable to unfused scalar elementwise phases, for which a targeted fix is identified.
- The GEMM body is close to native, launch cost is low, and the event schedule works.

Those numbers mean MegaBake is not yet a production performance win. They also separate implementation debt from the architecture: the current bottleneck is known and measurable, while the compiler path and one-kernel execution are already working.

## Who should care

### Large model and inference teams

You can evaluate a transparent compiler path for the workloads where CUDA Graph replay and `torch.compile` still leave latency on the table, especially small-batch inference and cross-region fusion. MegaBake can expose exactly why a region is admitted, what memory and synchronization it uses, and what it costs end to end. That evidence can guide future serving stacks even where MegaBake itself is not deployed.

### OSS AI projects

The project needs real model coverage and independent benchmarking. Contributing captured graphs, region matchers, benchmark fixtures, or device bodies turns isolated performance tricks into reusable, tested compiler components. Explicit fallback boundaries make it safe to add support incrementally.

### PyTorch Foundation and compiler community

MegaBake is an external, experimental Dynamo backend that consumes post-grad FX while preserving the PyTorch runtime contract. It can serve as a concrete downstream user of FX, AOT handoff, guard semantics, and backend integration. The useful collaboration is not an immediate merge; it is defining a durable path for experiments like this to test alternatives to Inductor lowering without forking the PyTorch user experience.

### New contributors

The repository has a clear task ladder and evidence gates. Useful contributions do not require rewriting the entire compiler: improve correctness checks, benchmark a workload, port and validate a device body, extend graph import, harden runtime bindings, or document a measured result. Every contribution has a concrete acceptance test.

## Roadmap

1. Remove per-call Python binding overhead so caller-visible latency reflects kernel work.
2. Fuse the measured elementwise chain and close the known batch-256 kernel gap.
3. Add transformer-block support for prefill and bounded stateful decode.
4. Compile and validate a complete small model such as SmolLM.
5. Add explicit whole-graph Inductor fallback and publish reproducible full-model reports.
6. Evaluate additional hardware targets, including ROCm and Pallas-style device bodies, after the CUDA path has complete-model evidence.

## The ask

- **Companies and labs:** share target models, shapes, GPUs, and latency budgets; help benchmark and prioritize the paths that matter in production.
- **OSS projects:** contribute model capture cases, operator semantics, benchmark harnesses, and reusable device bodies.
- **PyTorch Foundation:** work with us on the requirements for a durable FX backend experiment and eventual integration path.
- **Contributors:** pick a gate in [TASKS.md](TASKS.md), reproduce its evidence, and improve one measured result.

## The bottom line

MegaBake is an early but credible attempt to make inference megakernels a compiler output rather than a handcrafted artifact. It keeps PyTorch as the interface, makes correctness and resource legality explicit, and treats performance claims as evidence to be measured rather than marketing. If whole-region fusion and legal cross-region overlap can win for important serving workloads, MegaBake is building the path to prove it.
