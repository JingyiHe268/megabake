# MegaBake roadmap: from one model to many

**2026-10-16, updated 2026-10-17. Status: evidence, plan, and progress. Section 6 has the measured state.**

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
| 6 | Declared gated delta rule (chunked prefill: `cumsum`, triangular solve) | not started | Qwen3.5 prefill compiles |
| 7 | Bias epilogue in the TMA GEMM | done | Qwen2-0.5B decode on the tuned body, bit-exact with eager |
| 8 | Fusion: small intervals recomputed by their consumers | done | 135M decode 11 → ~6 barrier intervals per layer |
| 9 | Cross-phase L2 weight prefetch | not started | 1.7B decode GEMMs ≥ 1.9 TB/s effective |
| 10 | Event counters in the table launch | not started | No grid barrier between independent phases |
| 11 | Split-K, split-context attention | split-context done; split-K built, off | 1.7B decode kernel ≤ 2.0 ms (Inductor's kernels: 2.4 ms) |
| 12 | Variable lengths: size buckets with checked padding; compile time independent of length | done | One compile per bucket serves every length up to it; compile time no longer grows with tokens |

After step 3, the SmolLM-1.7B decode kernel spends 1.92 ms in GEMMs. The remaining 0.77 ms goes to 49 RMSNorms (5 µs each), 24 decode attentions (10 µs each) and elementwise phases, nearly all per-step latency across 267 barrier intervals. Steps 8–10 target exactly that.

## 6. Measured state (2026-10-17)

### Hardware facts that change the numbers above

- The GPU is an **H200 MIG `3g.71gb` slice: 60 SMs and ~2.2 TB/s** of HBM bandwidth (measured with large copies), not a full H200. The 1.7B weight-bandwidth floor is ~1.5 ms; its GEMMs run at ~1.8 TB/s, about 80% of the slice.
- The slice shares a physical GPU with other tenants. SM clocks move between 345 and 1,980 MHz with load, so a profiled single call can be 10–15% off. Compare changes back to back in one process, after a clock ramp (dense matmuls for ~2 s).

### What changed

| Change | Effect |
| --- | --- |
| Constant kernel arguments: device call counter, 64-entry address ring, packed CuTe arguments cached; no per-call stream sync; incremental input checks | MegaBake's own host work per call 634 → 360 µs; Dynamo guards (~290 µs, 365 inputs) are now the largest host cost |
| Split-context decode attention (chunks exchange max/sum through phase scratch; last chunk merges in order) | 135M decode at 2,000 cached tokens 22.7 → 2.3 ms; 1.7B 26.2 → 5.5 ms |
| Counting grid barrier (one release reduction per CTA, acquire polling) | 1.08 → 0.70 µs per barrier |
| Reduction bodies unroll and issue loads before stores | RMSNorm 5.0 → 2.8 µs on 1.7B |
| Interval folding (step 8) and staged map bodies | 135M decode barriers 333 → 183; −5–7% (135M), −1–2% (1.7B) |
| Bias epilogue (step 7) | Qwen2 q/k/v projections on the tuned GEMM |
| swapAB up to 64 rows (16/32/64 token tile) | Prompts and batches up to 64 rows stay on the small-M GEMM |
| Split-K | No gain on the 60-SM slice; off by default (`MEGABAKE_SPLIT_K=1`) |

### Results

`benchmarks/full_model.py`, 40 calls, KV capacity 128 (context < 100 tokens). Decode "call" is a synchronized complete call; "kernel" is profiled GPU time of one call (see the clock caveat).

| Model | Variant | Decode call (µs) | Decode kernel (µs) | Prefill call, 4 tokens (µs) |
| --- | --- | ---: | ---: | ---: |
| SmolLM-135M | Inductor (CUDA graphs skipped*) | 11,575 | 1,384 | — |
| | Inductor CUDA graphs | — | — | 1,931 |
| | MegaBake | **2,135** (1,498 pipelined) | 1,503 | 2,093 |
| SmolLM-1.7B | Inductor | 9,568 | 2,421 | — |
| | Inductor CUDA graphs | — | — | 2,923 |
| | MegaBake | **3,318** (2,842 pipelined) | 2,983 | 3,317 |
| Qwen2-0.5B | Inductor | 10,178 | 1,650 | — |
| | Inductor CUDA graphs | — | — | 2,162 |
| | MegaBake | **2,282** (1,574 pipelined) | **1,578** | 2,354 |

\* Inductor skips CUDA graphs in decode because the KV caches are mutated inputs.

Decode against context length (ms per call, clock ramped):

| Cached tokens | 16 | 128 | 512 | 1,024 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 135M MegaBake | 1.43 | 1.58 | 1.70 | 1.90 | 2.31 |
| 135M Inductor | 10.98 | 11.13 | 11.18 | 11.09 | 10.47 |
| 1.7B MegaBake | 2.66 | 3.06 | 3.55 | 4.24 | 5.51 |
| 1.7B Inductor | 8.61 | 8.25 | 8.42 | 8.83 | 8.53 |

Correctness: every region of 135M decode and Qwen2-0.5B decode is bit-exact with PyTorch on its own inputs, and both match eager end to end over advancing calls. On 1.7B, GEMMs differ from cuBLAS by accumulation order only (≤ 2.6e-4 per region); attention regions stay within 2e-4 at 1,800 cached tokens.

### Where MegaBake still loses

- **GPU time in decode:** 135M 1.50 ms vs Inductor's 1.38 ms of kernels; 1.7B 2.98 vs 2.42 ms. MegaBake wins end to end because Inductor makes 450–630 launches on a slow host.
- **Prefill:** about even on 135M at 4 tokens, 13% slower on 1.7B, and worse at 32 tokens (135M 2.63 vs 1.81 ms; 1.7B 4.96 vs 3.37 ms).
- **Compilation** is 1–6 minutes at 4 tokens and grows with token count (1.7B, 32-token prefill: 29 minutes), because shapes are static and the planner enumerates every tile in Python.

### Step 12: variable lengths

Kernels keep static shapes; `megabake.bucketed` serves arbitrary lengths through a few compiled sizes, as vLLM does with CUDA-graph capture sizes. A call pads its sequence arguments at the end to the smallest bucket that holds it, runs that bucket's kernel, and cuts the outputs back. On the first call of each bucket, eager calls with two different paddings must agree bit for bit at the real positions, and padded and unpadded eager calls must agree there up to rounding; otherwise the wrapper raises `PaddingDependence` instead of returning wrong results. Causal decoders pass. A model whose outputs read other positions or the padded length does not.

Compile time had grown with the token count because of Python bookkeeping, not GPU code generation (CuTe compilation was ~9 s of 1,409 s):

- The preflight ran a full plan search only to prove a plan exists; it now stops at the first plan.
- Tile maps, body selections, region dependencies, and storage consumers are built once per selection instead of once per use.
- A consumer tile that reads a producer's whole value depends on every producer tile without a pairwise check.
- Reductions get one tile per reduced row, as their bodies run, not one per output element.

| SmolLM-135M compile + first call | Before | After |
| --- | ---: | ---: |
| Decode | 77 s | 26 s |
| Prefill, 32 tokens | 516 s | 29 s |
| Prefill, 128 tokens | — | 37 s |
| Qwen2-0.5B prefill, 512-token bucket | — | 139 s |

SmolLM-1.7B decode compiles in 20 s (was 100 s).

Bucketed prefill against Inductor with CUDA graphs compiled per exact length (µs per call):

| Prompt tokens (bucket) | 5 (16) | 20 (32) | 50 (64) | 100 (128) | 300 (512) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM-135M MegaBake | 2,400 | 2,738 | 3,880 | 6,785 | 54,095 |
| SmolLM-135M Inductor | 1,906 | 1,897 | 1,862 | 2,156 | 2,579 |
| Qwen2-0.5B MegaBake | 2,623 | 3,172 | 4,211 | 7,747 | 67,897 |
| Qwen2-0.5B Inductor | 2,063 | 1,972 | 2,080 | 2,220 | 3,186 |
| SmolLM-1.7B MegaBake | 3,696 | 4,994 | 7,457 | 16,687 | 157,902 |
| SmolLM-1.7B Inductor | 2,889 | 2,954 | 3,081 | 3,767 | 6,673 |

Bucket compile times: 24–40 s up to 128 tokens on the small models, 138–139 s for their 512 bucket; SmolLM-1.7B 24–32 s up to 64 tokens, 181 s at 128, and 41 minutes at 512, so compile time still grows on large buckets of large models. Logits stay within 2.2% of eager at every length.

The buckets work, but **prefill kernels are slow at larger sizes**: 3.1–3.5x Inductor at 128 tokens and 21x at 300 tokens. Prefill still uses the first-generation attention body (one warp per query row, keys in sequence) and has had none of the decode tuning. That is the next gap.

### Gaps found while doing this

- **Large-prompt prefill** (above): the prefill attention body and the >64-row GEMM path need the treatment decode got.
- **Compile time on large buckets of large models**: SmolLM-1.7B's 512 bucket takes 41 minutes; the remaining per-tile work (attention tile maps, dependencies) needs the same treatment.
- Buckets pad, so a 65-token prompt pays for 128; a runtime token count inside the kernel (Mirage's approach) would remove that.
- Reductions over more than one axis (`x.mean()` of a 2-D tensor) do not lower.
- Long chains of elementwise ops exceed the planner's recursion limit.
- `x + mean * c + bias` with a broadcast is not fused into one region.
- `test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid` fails before and after this work; `test_task24_compiles_exact_rope_with_supported_stride` failed once in five runs.
