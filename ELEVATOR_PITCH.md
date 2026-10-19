# MegaBake

*Author: Ayush Satyam* 

**A PyTorch native Mega-kernel compiler that compiles any Hugging Face model step into one persistent CUDA Mega-kernel.**

## In one sentence

MegaBake is an open-source (MIT) PyTorch compiler backend. It takes an unmodified PyTorch or Hugging Face model and generates a single persistent CUDA kernel that covers the embedding, every transformer layer, the final norm, the vocabulary projection and the KV-cache update. Teams keep their model code, their `from_pretrained` workflow and the `torch.compile` interface, and nobody has to hand-write CUDA.

## The problem

Autoregressive decoding produces one token per forward pass. At small batch sizes that pass is **memory-bandwidth bound**: the GPU has to stream every weight from HBM once per token, so the theoretical floor is weight bytes ÷ memory bandwidth. Production stacks fall well short of that floor, and the reason is structural.

A decode step in eager PyTorch is **hundreds to thousands of separate kernel launches**. `torch.compile` fuses some of them, and CUDA graphs remove most of the CPU launch overhead. But the GPU still runs a long chain of separate kernels, and every kernel boundary costs something:

- **A full barrier and a memory flush.** Every streaming multiprocessor (SM) waits for the slowest thread block before any downstream work can start, even work whose inputs are already finished.
- **Cold pipelines.** Each matrix multiply refills its load pipeline from empty. The next layer can't start streaming its weights until its kernel launches, and shared memory doesn't survive across kernels.
- **Stranded SMs.** A small projection that occupies half the GPU leaves the other half idle, and that idle capacity can't be handed future work.

The known answer is a **megakernel**: one persistent kernel that runs the whole model, with fine-grained dependencies between pieces of work in place of kernel boundaries. Hand-written megakernels from research groups have shown large decode speedups on H100 and B200. Each one, though, takes specialists weeks to months, is tied to one architecture on one GPU, and goes stale with the next model release.

## What MegaBake does

MegaBake makes the megakernel a compiler output rather than a hand-crafted artifact.

```python
compiled = torch.compile(model, backend=megabake.make_backend())
```

1. **Capture through PyTorch's own stack.** TorchDynamo captures the call, and PyTorch's AOT and post-grad passes normalize the FX graph. MegaBake takes the graph at the point where Inductor would lower it, and preserves guards, aliasing, mutations and outputs.
2. **Recover what the graph computes.** MegaBake partitions the graph into regions with explicit layouts, dependencies, state effects and numeric requirements. Coverage comes from operator families and declared ops (attention is declared through transformers' `AttentionInterface`), not from a list of supported model names. New architectures therefore compile without new hand-written kernels.
3. **Compose tuned device code, without flattening it.** Each region maps to a device body written in NVIDIA's CuTe DSL. Matrix multiplies use a Hopper TMA/WGMMA tensor-core pipeline with a small-batch variant. Decode attention splits long contexts across SMs. Norms, RoPE and activations are generated and folded into their consumers. Tuned pipelines stay intact inside the composed kernel. This also enables scope for seamless integration of future potential new hardware architectures. 
4. **Plan one persistent kernel.** MegaBake decides which tiles each SM computes, when, in which shared or global buffers, and with what synchronization.
5. **Verify before use.** The captured graph is the correctness reference. If a graph isn't supported, MegaBake either fails with a clear message or falls back to standard `torch.compile`. It never changes numerics silently.

## In our experiments

We tested eight complete Hugging Face decoders on an NVIDIA H200 (60-SM partition, bf16) in batch-1 decode, the latency-critical case for chat and agents: SmolLM-135M and 1.7B, Qwen2-0.5B, gemma-2b, Gemma-3-270M, Qwen3.5-0.8B, and two DeepSeek-V3-family draft models.

- **One launch per token instead of about 1,800.** SmolLM-135M issues 1,780 kernel launches per decoded token in eager PyTorch and 632 under `torch.compile`. With MegaBake it issues 1.
- **Up to 14x lower per-token latency than eager PyTorch.** SmolLM-135M dropped from 30.6 ms to 2.1 ms per token. The speedup was 2.5–14x across all eight models.
- **Up to 5.6x lower per-token latency than `torch.compile` in its default mode.** Qwen2-0.5B and the DeepSeek-V3 draft were 4.5x faster, and SmolLM-1.7B was 3.1x faster.
- **GPU time on par with Inductor's generated kernels on Llama- and Qwen-style models.** It is slightly lower on Qwen2-0.5B and the DeepSeek-V3 draft. This is before the megakernel-only optimizations below are switched on.
- **Bit-exact outputs.** On SmolLM-135M and Qwen2-0.5B, decode output matches eager PyTorch bit for bit over successive tokens.

What's still open:

- We are adding a comparison against `torch.compile` with CUDA graphs and max_autotune.
- Long-prompt prefill is not yet competitive, however long decode sequences are pretty competitive and in a lot of cases better than torch.compile.
- Gemma 3 and Qwen3.5 still need tuning; their GPU time is 2–2.8x Inductor's.

## The headroom: optimizations only a megakernel can do

So far MegaBake has mostly removed launch and dispatch cost. The larger gains come from techniques that a chain of separate kernels can't express. These are under active development:

| Optimization | What it changes | Estimated gain |
|---|---|---|
| **Tile-level dataflow** | Each SM waits only on the specific tiles it consumes, using lightweight event counters in place of grid-wide barriers | 5–30% lower GPU time |
| **Cross-layer weight streaming** | Idle or waiting SMs prefetch the next layer's weights while the current layer computes, and GEMM pipelines stay warm across layers | A further 8–15% on larger models |
| **On-device multi-step decoding** | Sampling and the next embedding run inside the kernel, so several tokens are generated per launch | 10–25% per token end to end |
| **In-kernel speculative verification** | At batch 1, verifying 8 draft tokens costs about the same weight traffic as verifying 1 | 2–4x tokens per second |

The gains are engineering estimates, not yet measurements. Bandwidth use is currently about 50% on SmolLM-1.7B, which leaves substantial room for these optimizations.

## Why this matters to the PyTorch ecosystem

- **Peak inference performance without leaving PyTorch.** The fastest decode today often means porting a model to a specialized engine or maintaining hand-written CUDA. MegaBake delivers that class of optimization through `torch.compile`, under PyTorch's semantics.
- **A new class of `torch.compile` backend.** MegaBake plugs in at the same handoff as Inductor and keeps PyTorch's guarantees about guards, aliasing and state. It shows that the Dynamo/FX stack can support whole-model compilation and cross-operator scheduling, not just per-operator code generation.
- **Day-0 performance for new architectures.** Models change every few weeks: new norms, RoPE variants, attention forms and linear-attention layers. Because MegaBake matches by semantics rather than by model name, adding a new model is an extension, not a rewrite.
- **Hardware extensibility.** The model graph, the execution plan and the device bodies are separate layers. Hopper is supported today. Blackwell, Ampere and other accelerators plug in as new body providers behind the same compiler.
- **Open and reproducible.** MegaBake is MIT-licensed. Every performance claim names the model, GPU, precision, baseline and method.

## Hugging Face Hub as the natural distribution channel

A baked megakernel is specialized to one model revision, one shape bucket and one GPU class, much as weights are specialized to a model. It is also expensive to produce: compilation takes minutes, longer for large models, and per-device autotuning adds measurement time. It should therefore be **compiled once and distributed**, the way weights are today.

The Hub already provides what that needs:

- **Revision pinning.** A baked kernel is tied to the exact weights commit it was compiled and verified against, so a weight update can't silently pair with a stale kernel.
- **Hardware-matched builds.** Hugging Face's Kernel Hub already distributes compiled kernels for each GPU architecture, CUDA version and PyTorch version, and picks the matching build at load time. Its build tooling supports NVIDIA's CuTe DSL, which MegaBake is written in.
- **Efficient transfer and visibility.** Downloads are fast and deduplicated, and usage statistics show which models and GPUs matter most.
- **Distribution where users already are.** Someone who picks a model on the Hub could also pull its pre-baked, verified megakernel for their GPU, with no local compilation and no CUDA toolkit.

Our proposal has two parts:

- **Kernel Hub for the reusable layer.** Publish MegaBake's runtime and device-body library (the tensor-core GEMM, split-context attention and generated operator bodies) so other projects can use them directly.
- **Hub repos for baked artifacts.** Publish one verified megakernel per model revision, GPU class and mode, next to the model. They are loaded automatically when they match the machine, and compiled locally only when no match exists.

## Who benefits

- **Developers and researchers** get megakernel-class decode latency from one line of code, in the tools they already use.
- **Teams serving models** get lower latency per user and higher throughput per GPU, without maintaining bespoke CUDA.
- **Model creators** get fast inference for new architectures on release day.
- **Hardware vendors** get one compiler path that brings their accelerator to the PyTorch and Hugging Face model ecosystem.
- **The PyTorch Foundation** gets an open answer to "do I have to leave PyTorch for state-of-the-art inference?"

## Where we are and where we're going

**Working today:**
- Eight complete Hugging Face decoders compile into a single persistent CUDA kernel, for both prefill and stateful decode.
- Outputs are verified against PyTorch.
- It runs on NVIDIA Hopper in bf16.

**Next:**
- Tile-level dataflow and cross-layer weight streaming.
- Per-GPU autotuning driven by a cost model.
- Fused GEMM epilogues and prologues.
- Paged KV cache and batched serving.
- Fast long-prompt prefill.
- Multi-step and speculative decoding.
- FP8.
- Blackwell and Ampere support.

## The ask

- **The PyTorch Foundation and the compiler team:** help turn the FX handoff MegaBake uses into a stable, supported path for backends that replace Inductor lowering.
- **Hugging Face:** partner on a Hub format for baked megakernels, and on publishing MegaBake's device bodies to the Kernel Hub.
- **Companies and labs:** share your target models, GPUs and latency budgets so we prioritize the workloads that matter in production.
- **Contributors:** every roadmap item has a concrete, measurable goal. Pick one, reproduce the baseline, and move a number.

**MegaBake's goal: the fastest way to run a PyTorch model should still be PyTorch.**
