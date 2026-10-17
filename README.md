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

Complete Hugging Face decoder models compile through the normal backend into one cooperative CuTe kernel, in prefill and stateful decode: SmolLM-135M and SmolLM-1.7B (Llama blocks), Qwen2-0.5B (q/k/v biases). Gemma 3 decode compiles through the declared attention op (ROADMAP step 4; not re-measured since). The kernel covers the embedding, every layer, the final norm, the vocabulary projection, and the KV-cache updates.

- **Projections:** the tuned Hopper TMA/WGMMA GEMM. Launches whose GEMMs have at most 64 rows use swapAB (weights on WGMMA's M side, tokens on N = 16/32/64); `addmm` biases are added in the epilogue.
- **Attention:** declared through transformers' `AttentionInterface` (`megabake::attention`, `megabake::kv_attention`). Decode splits each row's context across CTAs.
- **Small operators** (norms, RoPE, residual adds, gated activations) run generated bodies. At decode sizes they fold into the barrier interval of their consumer: every CTA recomputes them, so a layer costs about six barrier intervals instead of eleven.
- **Synchronization:** a grid barrier on a monotonic arrival counter (~0.7 µs at 60 CTAs). Kernel arguments are constant between calls, so a call packs nothing and does not block.

Decode of SmolLM-135M and Qwen2-0.5B matches eager bf16 exactly over advancing calls. SmolLM-1.7B regions match PyTorch on their own inputs up to GEMM accumulation order (≤ 2.6e-4).

Limits: SM90 and bf16 only; kernels have static shapes, and `megabake.bucketed` serves variable prompt and batch lengths through a few compiled buckets, checking that padding cannot change real outputs; prefill is slow above ~64 tokens; linear-attention mixers (Qwen3.5's gated delta rule) are not declared yet. [ROADMAP.md](ROADMAP.md) has the plan and the measured state.

## Current performance

H200 MIG `3g.71gb` slice (60 SMs, ~2.2 TB/s), bf16, 40 warm calls, KV capacity 128. Medians in microseconds. "Call" is a synchronized complete call; "kernel" is GPU time of one profiled call. The slice shares a GPU with other tenants and its clocks move with their load, so single-call kernel times vary by 10–15%.

| Model | Variant | Decode call | Decode kernel | Prefill call, 4 tokens |
| --- | --- | ---: | ---: | ---: |
| SmolLM-135M | Inductor | 11,575 | 1,384 | 1,931 (CUDA graphs) |
| | MegaBake | **2,135** | 1,503 | 2,093 |
| SmolLM-1.7B | Inductor | 9,568 | 2,421 | 2,923 (CUDA graphs) |
| | MegaBake | **3,318** | 2,983 | 3,317 |
| Qwen2-0.5B | Inductor | 10,178 | 1,650 | 2,162 (CUDA graphs) |
| | MegaBake | **2,282** | **1,578** | 2,354 |

Inductor skips CUDA graphs in decode because the KV caches are mutated inputs. Back-to-back decode calls (`pipelined_call_us`) overlap MegaBake's host work with the previous kernel: 1.50 ms per call on 135M, 2.84 ms on 1.7B.

- **Decode end to end:** 3–5x faster than Inductor, since one launch replaces 450–630.
- **Long contexts:** decode stays flat. At 2,000 cached tokens: 135M 2.3 ms (Inductor 10.5 ms), 1.7B 5.5 ms (8.5 ms).
- **Still behind:** decode GPU time on the SmolLM models (1.50 vs 1.38 ms, 2.98 vs 2.42 ms), prefill on 1.7B and at 32-token prompts, and compilation (1–6 minutes, growing with token count).

Reproduce on an SM90 device:

```bash
.venv/bin/python benchmarks/full_model.py --model HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B
.venv/bin/python benchmarks/full_model.py --variants inductor megabake --modes decode   # a subset
MEGABAKE_PHASE_TIMER=1 .venv/bin/python tools/phase_profile.py --mode decode            # time per barrier interval
.venv/bin/python tools/dump_megakernels.py --checkpoint                                   # correctness check plus generated source
```

`benchmarks/run.py --workload mlp|transformer-block` and `benchmarks/isolated.py` measure the smaller fixtures. [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) has the earlier analysis.

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
