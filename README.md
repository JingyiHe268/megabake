<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# MegaBake

MegaBake turns a captured PyTorch inference workload into a planned, composable CUDA megakernel. PyTorch and Hugging Face keep the model and runtime interface. MegaBake owns region fusion, CuTe DSL device bodies, scheduling, and execution planning.

## Design

[north-star.md](north-star.md) defines the scope. [ARCHITECTURE.md](ARCHITECTURE.md) defines the proposed compiler. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) gives the build gates. [RESEARCH.md](RESEARCH.md) records the source review and GPU experiments behind the design.

[TASKS.md](TASKS.md) gives the step-by-step execution plan: get live FX, build the Compute Graph, plan execution, and generate complete CuTe megakernels. Each step names its implementation, inputs, outputs, and checks.

[SCHEDULER_REUSE.md](SCHEDULER_REUSE.md) covers operation variants, generic lowering, explicit fallback, and reusable Mirage components with source references.

Mirage is the main reference architecture for lowering, tile/layout planning, storage, and persistent scheduling. Start from its compatible algorithms and tests. Adapt them for the FX contract, CuTe DSL, and measured target requirements.

The compilation path is:

1. Capture the invocation with Dynamo and preserve the PyTorch runtime contract.
2. Receive live FX after the pinned AOT/Inductor preparation and post-grad passes, before lowering and scheduling.
3. Recover tensor semantics and source-backed regions with explicit layouts, dependencies, effects, and numerical requirements.
4. Generate baseline CuTe bodies from supported tensor primitives and add valid tuned candidates. Plan their execution in one persistent kernel where legal.
5. Check the complete invocation and measure it against an equivalent `torch.compile` baseline.

The design preserves the tuned pipeline inside each body. It starts with ordered phases and refines tile readiness where measurements justify it. The frontend accepts `--mode prefill|decode` and records that policy; mode-specific decode scheduling is not implemented yet. On an SM90 CUDA device, `make_backend()` compiles supported post-grad graphs into one cooperative CuTe workload kernel and rebinds current inputs and parameters on every call. Pass `compile_kernel=False` to retain the FX reference path for CPU-only capture work.

A complete-model request includes all requested outputs and state updates. Named region matchers enable optimizations. Unfamiliar combinations of supported primitives use generic lowering. Unknown operator semantics or an illegal composition produce an explicit failure. The proposed `fallback=inductor` option can delegate the whole invocation and report that result as external fallback. Strict megakernel compilation defaults to `fallback=error`. These paths are not implemented yet.

The first frontend-to-kernel compiler path is implemented for supported SM90 workloads. Complete-model coverage, stateful decode, fallback delegation, and measured optimization work remain open in later TASKS.md steps.

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
