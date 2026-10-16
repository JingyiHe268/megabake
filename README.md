<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# MegaBake

MegaBake turns a captured PyTorch inference workload into a planned, composable CUDA megakernel. PyTorch and Hugging Face keep the model and runtime interface. MegaBake owns region fusion, CuTe DSL device bodies, scheduling, and execution planning.

## Design

[north-star.md](north-star.md) defines the scope. [ARCHITECTURE.md](ARCHITECTURE.md) defines the proposed compiler. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) gives the build gates. [RESEARCH.md](RESEARCH.md) records the source review and GPU experiments behind the design.

[TASKS.md](TASKS.md) gives the step-by-step execution plan: get live FX, build the Compute Graph, plan execution, and generate complete CuTe megakernels. Each step names its implementation, inputs, outputs, and checks.

[ELEVATOR_PITCH.md](ELEVATOR_PITCH.md) summarizes the vision, current status, audience value, and contribution asks.

[SCHEDULER_REUSE.md](SCHEDULER_REUSE.md) covers operation variants, generic lowering, explicit fallback, and reusable Mirage components with source references.

Mirage is the main reference architecture for lowering, tile/layout planning, storage, and persistent scheduling. Start from its compatible algorithms and tests. Adapt them for the FX contract, CuTe DSL, and measured target requirements.

The compilation path is:

1. Capture the invocation with Dynamo and preserve the PyTorch runtime contract.
2. Receive live FX after the pinned AOT/Inductor preparation and post-grad passes, before lowering and scheduling.
3. Recover tensor semantics and source-backed regions with explicit layouts, dependencies, effects, and numerical requirements.
4. Generate baseline CuTe bodies from supported tensor primitives and add valid tuned candidates. Plan their execution in one persistent kernel where legal.
5. Check the complete invocation and measure it against the strongest equivalent baseline: `torch.compile` and CUDA Graph replay.

The design preserves the tuned pipeline inside each body. It starts with ordered phases and refines tile readiness where measurements justify it. The frontend accepts `--mode prefill|decode`, validates the captured state contract, and uses mode-specific bounded search priorities while actual dimensions select the body family. Bounded contiguous one-token decode KV state is supported through an explicit cache contract; broader Hugging Face cache formats remain later work. On an SM90 CUDA device, `make_backend()` compiles supported post-grad graphs into one cooperative CuTe workload kernel and rebinds current inputs and parameters on every call. Pass `compile_kernel=False` to retain the FX reference path for CPU-only capture work.

A complete-model request includes all requested outputs and state updates. Named region matchers enable optimizations. Unfamiliar combinations of supported primitives use generic lowering. Unknown operator semantics or an illegal composition produce an explicit failure before any device work. Strict megakernel compilation is the default (`fallback=error`); `fallback=inductor` delegates the whole invocation to Inductor and labels the result `inductor_fallback`.

## Current status

The complete pinned SmolLM-135M checkpoint (`HuggingFaceTB/SmolLM-135M@1d461723`) compiles through the normal backend into one cooperative CuTe kernel in both prefill and stateful decode. The kernel covers the embedding, all 30 layers, the final norm, the tied vocabulary projection, and the KV-cache updates. Bodies by kind:

- **Projections:** every Q/K/V/O, gate/up/down, and vocabulary projection (211 per call) runs the tuned Hopper TMA/WGMMA GEMM body.
- **Norms and RoPE:** each RMSNorm and each RoPE is fused into one generated region.
- **Attention:** warp-per-row CuTe bodies.
- **Everything else:** other elementwise work runs generated generic bodies.

Phases are grouped by dependency level, so independent work shares one grid barrier: 364 barriers per prefill call and 333 per decode call.

Decode logits and all KV caches match eager bf16 exactly over advancing calls. Prefill matches eager exactly after the first call. The first call differs by 1.8% with the same argmax, within the 2% full-model tolerance.

Support is limited to SM90 GPUs, bf16, and models with Llama-style blocks. Decode needs the explicit one-token KV-cache contract (`megabake::kv_attention`), currently captured through the hand-written `StatefulSmolLM` wrapper in `tests/fixtures.py` rather than Hugging Face's own cache classes.

## Current performance

SmolLM-135M on an H200 MIG `3g.71gb` slice (60 SMs), bf16, 40 warm calls each, decode KV capacity 128. Times are medians in milliseconds. "Complete call" is what a caller waits for: host work, launches, and GPU execution. "GPU kernel time" is the sum of kernel durations in one profiled call. The machine is shared, so the eager and Inductor numbers also vary by 5–10% between runs.

| | Prefill, 4 tokens | Decode, 1 token | GPU kernel time (prefill / decode) | Kernels per call | First call (prefill / decode) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 19.4 | 32.8 | 2.54 / 2.67 | 1,446 / 1,780 | <1 s |
| `torch.compile` (Inductor) | 7.4 | 13.0 | **1.07 / 1.37** | 512 / 632 | 6 s / 3 s |
| Inductor with CUDA graphs (`mode="reduce-overhead"`) | **2.0** | n/a* | 1.13 / – | 1 graph | 5 s / 5 s |
| MegaBake | 3.0 | **3.7** | 1.48 / 2.22 | **1** | 127 s / 79 s |

\* Inductor skips CUDA graphs for decode because the KV caches are mutated inputs, so its decode run is plain Inductor. There is no CUDA-graph decode baseline yet.

What this means:

- **End to end:** MegaBake beats eager and default `torch.compile` in both modes, because one launch replaces 500–1,800.
- **Against CUDA Graph replay:** prefill is still 1.5x slower (3.0 ms vs 2.0 ms). Two gaps remain, and both have to close:
  - **GPU work:** 1.48 ms vs 1.13 ms.
  - **Host work:** MegaBake spends about 1.5 ms outside the kernel per call; CUDA-graph replay spends about 0.9 ms.
- **GPU kernel time:** MegaBake's kernel takes 1.4x Inductor's kernel time in prefill and 1.6x in decode, down from 2.8–3x. Its decode attention cost still grows with context length.
- **Compilation** takes 79–127 s per mode.

Where the kernel time goes, from per-phase GPU timestamps. Phases in one barrier interval overlap, so the attribution is approximate.

| Phase type | Prefill | Decode, about 40 cached tokens |
| --- | ---: | ---: |
| 211 tuned GEMM phases | 0.77 ms | 0.98 ms |
| 30 attention phases | 0.14 ms | 0.66 ms |
| 61 fused RMSNorm reductions | 0.21 ms | 0.21 ms |
| ~200–300 other elementwise and copy phases | 0.36 ms | 0.34 ms |

The next fixes, in order:

1. Split K, or use stream-K, for GEMMs with fewer output tiles than workers. The K/V projections fill 3 of 60 CTAs, and every GEMM phase pays its pipeline start-up.
2. Split decode attention across the context, with a softmax merge, so its cost stops growing with the number of cached tokens.
3. Fuse each residual add into the following norm, and the elementwise work around GEMMs into their epilogues.
4. Cut per-call host work, and find why prefill compilation grew to 127 s.

[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) has the full-model analysis, the history of each fix, and the earlier MLP report.

Reproduce on an SM90 device:

```bash
.venv/bin/python benchmarks/full_model.py                               # 40 calls, decode capacity 128; writes artifacts/full_model/report.json
.venv/bin/python tools/dump_megakernels.py --checkpoint                  # correctness check plus generated source and phase tables
```

`benchmarks/run.py --workload mlp|transformer-block` and `benchmarks/isolated.py` measure the smaller fixtures.

## Current frontend

The Dynamo backend runs the pinned AOT, pre-grad, and post-grad FX transformations, retains the result as an FX `GraphModule`, and compiles it on SM90 when the complete plan is supported. It does not invoke Inductor lowering or code generation. It preserves the PyTorch wrappers around arguments, outputs, aliases, and mutations, and separates its capture cache from compiled-plan reuse.

```python
import torch
import megabake

model = model.eval()
compiled = torch.compile(model, backend=megabake.make_backend(), fullgraph=True)
with torch.no_grad():
    output = compiled(example_input)
```

The development CLI also exercises the capture path. CPU commands use the reference forward; CUDA commands follow the same backend policy and require a supported graph:

```bash
python src/main.py demo --device cpu
python src/main.py transformers --model-name HuggingFaceTB/SmolLM-135M
python src/main.py export --callable module:callable --arg '[[1.0, 2.0]]'
```

## Environment setup

The checked-in requirements target Python 3.13, PyTorch with CUDA 13.0, and CuTe DSL's CUDA 13 package. Use an NVIDIA GPU and driver supported by that PyTorch wheel. Kernel compilation needs the CUDA 13.0 development toolkit and its `nvcc`; installing Python requirements does not install `nvcc`. The CUDA version reported by `nvidia-smi` describes driver support, not the installed compiler toolkit.

From the repository root, create and activate a virtual environment, then install the project and its pinned direct dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Point `CUDA_HOME` at the CUDA 13.0 toolkit before compiling kernels, especially if `/usr/local/cuda` selects another version:

```bash
export CUDA_HOME=/path/to/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
```

Check the selected Python, PyTorch CUDA build, GPU visibility, and CUDA compiler:

```bash
python --version
python -c 'import torch; print("torch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU available:", torch.cuda.is_available())'
nvcc --version  # should report release 13.0
```

Activate the environment again in new shells with `source .venv/bin/activate`.

```bash
.venv/bin/python -m megabake.main transformers \
  --model-name HuggingFaceTB/SmolLM-135M \
  --device cuda \
  --dump-dir fx_traces/smollm \
  --dump-compute-graph \
  --dump-lowering-graph
```
