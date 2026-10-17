# MegaBake roadmap: from one model to many

**2026-10-16. Status: evidence and plan. Items marked done are in the working tree.**

MegaBake compiles SmolLM-135M into one kernel that beats `torch.compile` in decode. It does not yet generalize: three other models fail to compile, and the larger SmolLM-1.7B runs its kernel at 40–60% of the GPU's bandwidth. This document records why, what the reference systems do differently, and the order of work.

## 1. What is solid

These parts are general and correct. Keep them.

- **The FX contract.** Capture through Dynamo, AOT and Inductor's post-grad passes, guards, lifted parameters, mutation and output structure. It handled five model families without change.
- **Exact numerics.** On SmolLM-1.7B decode, every non-GEMM region matches PyTorch bit for bit on its own inputs (49 RMSNorms, 48 RoPEs, 48 SiLU·mul, 24 KV attentions, all residual adds). The 169 GEMMs match cuBLAS to a median of 0 and a maximum of 2.6e-4 relative error.
- **The table-driven persistent kernel.** One launch, a device address table, per-phase TMA descriptors, and a 1 µs grid barrier. It scales to 1,000+ phases.
- **Plan search, the memory planner and the body interface.**

## 2. What is wrong

### 2.1 Performance is latency-bound

Measured with `tools/phase_profile.py`, which stamps the global timer after every grid barrier:

| Decode kernel | Barrier intervals | Kernel time | Weight-bandwidth floor at 2.2 TB/s |
| --- | ---: | ---: | ---: |
| SmolLM-135M | 333 | 1.75 ms | about 0.12 ms |
| SmolLM-1.7B | 267 | 4.43 ms | about 1.6 ms |

- **GEMMs were 83% of 1.7B kernel time** at about 0.93 TB/s. The cause was the operand orientation, not idle SMs. With 1–16 tokens padded into a 64-row tile, each pipeline stage moved 8 KB of mostly-padding activation per weight tile. Narrower N tiles made it worse (measured: 0.84 → 0.41 TB/s), because they shrank the weight share of each stage further. **Fixed by swapAB (§4.3).**
- **Every step pays start-up and drain.** A layer runs about 11 serial barrier intervals. A 576-wide RMSNorm costs 3.5 µs, and decode attention over 10 tokens costs 9 µs per layer.
- **Full models cannot use events.** Event scheduling exists only in the per-tensor-parameter launch, which full models exceed (CUDA's 32 KB parameter limit). The table launch, which every full model uses, has only grid barriers.

### 2.2 Coverage is per-op and per-pattern

- **Lowering is hand-written per ATen op.** It covers 67 ops. Inductor covers all of core ATen with about six IR node kinds: pointwise, reduction, scan, scatter, template, extern.
- **Attention is found by pattern matching decomposed ATen.** A stride heuristic misread the size-1 head axis of a single-KV-head model, so no Gemma 3 attention matched and compilation failed downstream.
- **Stateful token mixers other than KV attention cannot be expressed.** Qwen3.5's Gated DeltaNet needs a recurrent state, a convolution window, `cumsum`, a triangular solve, 1-D convolution and scatters.
- **Decode needs a hand-written wrapper per architecture.** These are `StatefulSmolLM` and `benchmarks/stateful_models.py`.

### 2.3 The acceptance check confused rounding with bugs

Plans had to match eager bf16 end to end within 2%. On the first SmolLM-1.7B decode call (position 0, an attention-sink token), the exact plan above ends 4.2% from eager. Equally valid bf16 orderings land between 1.8% and 4.2% apart. A global tolerance cannot separate rounding from small bugs in a deep model. An fp32 reference does not help either: two exact computations sat 2.6% and 6.4% from fp32.

## 3. What the reference systems do

| System | Generality | Performance |
| --- | --- | --- |
| Mirage MPK (`agent_space/mirage`) | **Per model.** Hand-written builders (`mpk/models/qwen3/builder.py`) call layer APIs: `rmsnorm_layer`, `linear_layer`, `paged_attention_layer`, ... | About 6 tasks per layer. Fused QKV and gate/up weights. `linear_swapAB_hopper` puts the weight dimension on WGMMA's M and up to 16 tokens on N. Every linear is split into 64–96 tasks to fill the GPU; split-K where N is small. Tasks wait on event counters, never on grid barriers. |
| Inferact TPU megakernels (`agent_space/tpu-megakernels`) | **Per model.** One fused decode function per architecture. | The whole decode is one weight stream: a ring of `BANK_COUNT` buffers prefetches the next weight tiles across op and layer boundaries, so HBM never idles. Norms, RoPE, attention and the gated delta rule run between tiles. |
| vLLM (`agent_space/vllm`) | Model code plus **declared custom ops** for every stateful mixer: `unified_attention_with_output`, `qwen_gdn_attention_core`, `mamba_mixer2`, `short_conv`, ... | Inductor compiles the generic graph between those ops ("splitting ops"); the ops have hand-tuned kernels. |
| Inductor (`agent_space/pytorch`) | **Every core ATen op**, via decompositions and about six IR kinds, with an extern fallback. | Templates for GEMM and attention, generated code for the rest. |

No system gets both. The fast ones are per model; the general one relies on libraries and templates for hot ops. MegaBake can combine the general parts that already exist (Inductor-style coverage, vLLM-style declared ops) with a Mirage-style runtime and a few tuned bodies shared by all models. Model specificity then lives in a handful of op declarations, not in compiler matchers or per-model wrappers.

## 4. Target design

### 4.1 Coverage

1. **Declared model ops for stateful and complex mixers**, each with a PyTorch reference, a fake implementation and a tuned body:
   - `megabake::attention` (prefill): scale, causal and sliding window, soft cap, GQA/MQA. Route every Hugging Face model to it by registering a `megabake` implementation with transformers' `AttentionInterface`, which all HF attention modules dispatch through.
   - `megabake::kv_attention` (decode, exists).
   - `megabake::gated_delta_rule` and `megabake::causal_conv1d`, which hold the recurrent and convolution state. HF already routes these through named kernel hooks (`chunk_gated_delta_rule`, `causal_conv1d_update`).

   Decode then needs one generic cache adapter instead of per-architecture wrappers. Pattern matching stays as a fallback for code that does not use these interfaces.
2. **Lowering by primitive family**, not by op:
   - pointwise and casts
   - reduction
   - scan (`cumsum`)
   - views (`select`, `squeeze`, `split`, `unsqueeze`)
   - scatter (`select_scatter`, `slice_scatter`, `index_put`, `copy`)
   - padding
   - N-d contraction (including 1-D convolution)

   Use Inductor's decomposition table to reach these families.
3. **Explicit fallback**: `fallback="inductor"` (whole invocation) exists. Report every op that forced it.

### 4.2 Runtime

1. **Events in the table launch.** Each phase gets a completion counter in the device table; a CTA waits only on its phase's producers, not on a grid barrier. This is MPK's `trigger_event`/`dependent_event`, with static assignment first.
2. **Cross-phase weight prefetch.** While a CTA waits, it issues L2 prefetches (`cp.async.bulk.prefetch.L2`) for its tiles of the next GEMM. This is the TPU bank ring adapted to a GPU's L2.
3. **Fewer serial steps per layer.** Fuse the residual add and SiLU·mul into GEMM epilogues, and RMSNorm into the following GEMM's prologue (Mirage's `rmsnorm_linear`, `linear_with_residual`).

### 4.3 Bodies

1. **swapAB small-M GEMM (done).** Table launches whose GEMMs all have at most 16 rows bind the weight as CUTLASS A (64 rows on WGMMA's M side) and the tokens as B (N=16), and write the transposed output. This follows Mirage's `linear_swapAB_hopper` with the existing CUTLASS kernel; only the operand binding, the tile, and the C descriptor's major-ness change. Standalone kernel time on 60 SMs:

   | M × N × K | Before | swapAB | cuBLAS |
   | --- | ---: | ---: | ---: |
   | 4 × 2048 × 2048 | 10.0 µs | 4.7 µs | 4.7 µs |
   | 4 × 2048 × 8192 | 32.7 µs | 16.2 µs | 18.0 µs |
   | 4 × 8192 × 2048 | 32.2 µs | 17.3 µs | 16.0 µs |

2. **Bias epilogue** for `addmm`. Qwen2-style q/k/v biases currently fall back to the generic contraction body; rewriting them as `mm + add` would add a rounding point eager does not have.
3. **Split-K** for projections with fewer tiles than workers, with a deterministic merge.
4. **Split-context decode attention** with a softmax merge.

### 4.4 Verification (done)

Every region of SmolLM-1.7B decode matches PyTorch on its own inputs. GEMMs differ from cuBLAS only in accumulation order, at most 2.6e-4. Every other op, including RMSNorm, RoPE, SiLU·mul, KV attention and residual adds, is bit for bit.

When a plan misses the end-to-end tolerance, **teacher-forced verification** decides:

- Rerun the same compiled kernel with every intermediate in its own workspace slice. Its outputs must match the normal run bit for bit, which catches workspace-reuse and race bugs.
- Every region must reproduce PyTorch, evaluated on the inputs the candidate produced itself, within 1% relative L2. GEMM rounding-order noise is 3e-4; real bugs are much larger.

Table launches read workspace addresses from the device table, so the rerun reuses the compiled kernel. On SmolLM-1.7B decode this accepts the exact plan that the old check rejected.

### 4.5 Hardware

Coverage, verification and the event runtime are target-independent. Body providers are per target: SM90 today, SM100 started in `bodies/blackwell.py`. A TPU backend is out of scope; the TPU kernels are a reference for schedule design, not for code.

## 5. Order of work

Each step has a measured exit criterion. Run `benchmarks/full_model.py` and `tools/phase_profile.py` before and after.

| # | Step | Status | Exit criterion |
| ---: | --- | --- | --- |
| 1 | Phase profiler; teacher-forced verification | done | SmolLM-1.7B decode compiles; 135M GPU tests pass |
| 2 | Single-KV-head attention matching | done | Gemma 3 270M prefill compiles |
| 3 | swapAB GEMM for small-M table launches | done | 1.7B decode GEMMs 3.66 → 1.92 ms (1.78 TB/s); call 5.77 → 4.39 ms |
| 4 | Declared `megabake::attention` via HF `AttentionInterface`; `DecodeStep` for any HF attention model | done | Gemma 3 decode compiles with no per-model wrapper |
| 5 | Primitive families: views, `split`→`slice`, pointwise (`log*`, `exp2`, `erf`, rounding, logical, min/max, clamp), scatters, padding, 1-D convolution, input writes (`copy_`) | done | Qwen3.5 decode lowers completely |
| 6 | Declared gated delta rule (chunked prefill: `cumsum`, triangular solve) | next | Qwen3.5 prefill compiles |
| 7 | Bias epilogue in the TMA GEMM | | Qwen2-style decode GEMMs on the tuned body |
| 8 | Epilogue and prologue fusion | | 1.7B decode ≤ 6 barrier intervals per layer |
| 9 | Cross-phase L2 weight prefetch | | 1.7B decode GEMMs ≥ 1.9 TB/s effective |
| 10 | Event counters in the table launch | | No grid barrier between independent phases |
| 11 | Split-K, split-context attention | | 1.7B decode kernel ≤ 2.0 ms (Inductor's kernels: 2.4 ms) |

After step 3, the SmolLM-1.7B decode kernel spends 1.92 ms in GEMMs. The remaining 0.77 ms goes to 49 RMSNorms (5 µs each), 24 decode attentions (10 µs each) and elementwise phases, nearly all per-step latency across 267 barrier intervals. Steps 8–10 target exactly that.
