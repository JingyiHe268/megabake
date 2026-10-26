# Matched decode profiling and compiler/graph ablation

H100 80GB, shared nonzero random checkpoint, BF16, batch=1, prefill=128. Output length D includes the first token from prefill; timed decode steps = D−1. Three continuous, unprofiled runs per length; compilation, warmup and prefill excluded. All engines grow their real prefill KV state. Cross-stack natural greedy histories can differ.

Execution commit: `795ba2638b7b1732c0fc3d14427af9ebe0e1ef54`. Weight SHA256: `459af1fb7c20bdb2982b13330ff8bdab07c3ff88b1d7b506392ddb6de896b2da`.

SGLang uses its existing PyTorch 2.13/cu129 environment; all other variants use PyTorch 2.14/cu126. Inductor options: `emulate_precision_casts=True`; same upstream StatefulSmolLM computation and weights, with its CPU-scalar-extracting declared attention replaced by GPU-only tensor operations. The tensor and SDPA references attend a masked, padded static capacity of 4352 tokens; SGLang and MegaBake can use logical lengths, so this is an explicit attention/layout limitation of the compiler baseline. Masked SDPA is a separate attention implementation ablation, not a guaranteed fast attention dispatch. The optional packed FlashAttention variant uses device `seqused_k` to read the logical KV length, avoiding padded attention arithmetic. Each compiled/graph variant passes same-input/native-reference logits and new-KV relative L2 <0.02 at steps 1,127,1023,4095; old KV remains unchanged.

CUDA Graph variants capture model, greedy argmax, fixed input-buffer update and position increment; output-buffer writes remain in the common outer loop. The current MegaBake table runtime increments a device call counter and expects a matching host-stamped address-ring slot for every launch. Replaying an unchanged captured launch does not maintain this protocol; therefore MegaBake+CUDA Graph is not presented as a valid baseline.

| Engine | Output tokens | Median mean TPOT (ms) | Run range (ms) |
| --- | ---: | ---: | ---: |
| SGLang (CUDA Graph) | 128 | 1.037000 | 1.036939–1.040850 |
| SGLang (CUDA Graph) | 1024 | 1.088903 | 1.088840–1.089181 |
| SGLang (CUDA Graph) | 4096 | 1.108619 | 1.108519–1.108634 |
| MegaBake | 128 | 1.729916 | 1.729858–2.058658 |
| MegaBake | 1024 | 1.766088 | 1.765710–1.766229 |
| MegaBake | 4096 | 2.019929 | 2.019817–2.019987 |
| Eager + CUDA Graph | 128 | 4.471465 | 4.471101–4.472624 |
| Eager + CUDA Graph | 1024 | 4.466972 | 4.466159–4.470626 |
| Eager + CUDA Graph | 4096 | 4.459411 | 4.458116–4.462432 |
| Inductor | 128 | 5.365550 | 5.351650–5.415418 |
| Inductor | 1024 | 5.391591 | 5.369665–5.399269 |
| Inductor | 4096 | 5.368285 | 5.365705–5.501247 |
| Inductor + CUDA Graph | 128 | 1.860825 | 1.860526–1.860833 |
| Inductor + CUDA Graph | 1024 | 1.860574 | 1.860405–1.861072 |
| Inductor + CUDA Graph | 4096 | 1.861413 | 1.861398–1.861739 |
| Inductor + masked SDPA + CUDA Graph | 128 | 8.218524 | 8.215145–8.219332 |
| Inductor + masked SDPA + CUDA Graph | 1024 | 8.216768 | 8.216136–8.218114 |
| Inductor + masked SDPA + CUDA Graph | 4096 | 8.210234 | 8.210013–8.210713 |
| Inductor + FlashAttention + CUDA Graph | 128 | 1.586123 | 1.585500–1.587958 |
| Inductor + FlashAttention + CUDA Graph | 1024 | 1.586366 | 1.585427–1.589538 |
| Inductor + FlashAttention + CUDA Graph | 4096 | 1.596762 | 1.596366–1.597723 |

FlashAttention follow-up execution commit: `acddd00594b8cde0b57807995bc42a69efb40589`; unchanged model/core compiler, same software environment and workload.

Separate CUPTI profiles record 16 consecutive decode steps after 0,2032,4079 prior decode steps. These profiles are not the primary latency measurements. GPU busy is the union of GPU intervals; kernel-category sums may overlap. CPU CUDA API duration overlaps GPU work and must not be added to GPU time. Profiling can enlarge host/GPU gaps.

| Engine | First cached tokens | GPU busy µs/step | GPU gaps µs/step | Kernels/step |
| --- | ---: | ---: | ---: | ---: |
| SGLang (CUDA Graph) | 128 | 1006.39 | 5211.12 | 344.00 |
| SGLang (CUDA Graph) | 2160 | 1111.39 | 103.67 | 344.00 |
| SGLang (CUDA Graph) | 4207 | 1128.52 | 677.89 | 344.00 |
| MegaBake | 128 | 1612.64 | 8332.65 | 1.88 |
| MegaBake | 2160 | 1877.61 | 4.13 | 1.88 |
| MegaBake | 4207 | 2093.31 | 4.04 | 1.81 |
| Eager + CUDA Graph | 128 | 4620.20 | 187.77 | 2023.00 |
| Eager + CUDA Graph | 2160 | 4600.80 | 187.07 | 2022.94 |
| Eager + CUDA Graph | 4207 | 4589.10 | 187.06 | 2022.88 |
| Inductor | 128 | 1997.41 | 10532.77 | 603.12 |
| Inductor | 2160 | 1981.19 | 7466.16 | 603.12 |
| Inductor | 4207 | 1982.42 | 7411.45 | 603.00 |
| Inductor + CUDA Graph | 128 | 1876.35 | 69.74 | 602.81 |
| Inductor + CUDA Graph | 2160 | 1877.32 | 69.57 | 602.81 |
| Inductor + CUDA Graph | 4207 | 1877.39 | 69.87 | 602.75 |
| Inductor + masked SDPA + CUDA Graph | 128 | 8184.86 | 63.19 | 571.75 |
| Inductor + masked SDPA + CUDA Graph | 2160 | 8190.98 | 63.23 | 571.69 |
| Inductor + masked SDPA + CUDA Graph | 4207 | 8172.69 | 63.27 | 571.50 |
| Inductor + FlashAttention + CUDA Graph | 128 | 1597.55 | 66.94 | 575.00 |
| Inductor + FlashAttention + CUDA Graph | 2160 | 1615.58 | 71.88 | 575.00 |
| Inductor + FlashAttention + CUDA Graph | 4207 | 1627.34 | 71.62 | 575.00 |

Internal MegaBake phase stamps come from a **separate instrumented build**. Barrier-to-barrier interval totals include computation and synchronization. Category attribution splits shared intervals in proportion to phase work-item counts and is approximate; it is not isolated operator time or pure barrier overhead. The final trailing phases may not be timed. No raw trace, log or failure screenshot is published.

| First cached tokens | Workers | Phases | Barrier intervals | Timed total µs | Untimed trailing phases |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 132 | 603 | 183 | 1643.81 | 3 |
| 2160 | 132 | 603 | 183 | 1934.69 | 3 |
| 4207 | 132 | 603 | 183 | 2325.12 | 3 |

Separate Nsight Compute counter run (kernel replay can perturb cache/timing): `acddd00594b8cde0b57807995bc42a69efb40589`. Counters describe the complete MegaBake kernel, not individual phases.

| Cached tokens | Metric | Value | Unit |
| ---: | --- | ---: | --- |
| 128 | dram__bytes.sum | 2.74683e+08 | byte |
| 128 | dram__throughput.avg.pct_of_peak_sustained_elapsed | 4.22 | % |
| 128 | gpu__time_duration.sum | 1.94352e+06 | nsecond |
| 128 | sm__throughput.avg.pct_of_peak_sustained_elapsed | 6.79 | % |
| 128 | sm__warps_active.avg.pct_of_peak_sustained_active | 12.5 | % |
| 128 | launch__block_size | 256 |  |
| 128 | launch__grid_size | 132 |  |
| 128 | launch__registers_per_thread | 240 | register/thread |
| 128 | launch__shared_mem_per_block_dynamic | 159744 | byte/block |
| 128 | launch__shared_mem_per_block_static | 0 | byte/block |
| 4207 | dram__bytes.sum | 3.68838e+08 | byte |
| 4207 | dram__throughput.avg.pct_of_peak_sustained_elapsed | 3.99 | % |
| 4207 | gpu__time_duration.sum | 2.75418e+06 | nsecond |
| 4207 | sm__throughput.avg.pct_of_peak_sustained_elapsed | 6.85 | % |
| 4207 | sm__warps_active.avg.pct_of_peak_sustained_active | 12.5 | % |
| 4207 | launch__block_size | 256 |  |
| 4207 | launch__grid_size | 132 |  |
| 4207 | launch__registers_per_thread | 240 | register/thread |
| 4207 | launch__shared_mem_per_block_dynamic | 159744 | byte/block |
| 4207 | launch__shared_mem_per_block_static | 0 | byte/block |
