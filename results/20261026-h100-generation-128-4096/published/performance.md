# Continuous generation: MegaBake vs SGLang

Shared random checkpoint, BF16, batch=1, prefill=128, 3 runs per length. Each output length D includes one token obtained from prefill logits; the timed decode loop has D-1 steps. KV state grows naturally; no reset between decode steps. EOS stopping is disabled by the fixed-length benchmark.

Timing starts after prefill and ends after the continuous decode loop and final GPU synchronization. Includes backend/model-runner calls, KV updates, greedy sampling and output-buffer writes. No per-step global synchronization. Excludes loading, compilation, warmup, prefill, HTTP and tokenization.

SGLang 0.0.0.dev17813+g0948e6ebe, torch 2.13.0+cu129; MegaBake torch 2.14.0+cu126, CuTe 4.8.0. The stacks use different PyTorch/CUDA builds, so this is a backend-stack comparison, not an isolated compiler ablation.

SGLang source `0948e6ebed82c7dd7df8a0c347076e46af84bef6`; existing dirty worktree: True. The existing SGLang environment and modifications were preserved.

MegaBake checked against native PyTorch using identical inputs and KV state at selected steps: max logits L2 0.00757733, max new-KV L2 0.00789304. Cross-stack greedy sequences identical: False; first differing output index (zero-based): 122. Each stack performs its own greedy generation; differing floating-point implementations may yield different token histories. SGLang CUDA Graph replay verified (4095 validation decode calls); MegaBake workload verified as one kernel; sampling and output stores are separate operations.

Weight SHA256: `459af1fb7c20bdb2982b13330ff8bdab07c3ff88b1d7b506392ddb6de896b2da`. Execution commit: `11df6edcdad29c21ff3819e59c107484cf312935`.

| Output tokens | Decode steps | SGLang mean ms/token | MegaBake mean ms/token | MegaBake speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 127 | 1.034033 | 1.729808 | 0.598× |
| 256 | 255 | 1.060801 | 1.740546 | 0.609× |
| 512 | 511 | 1.075818 | 1.747638 | 0.616× |
| 1024 | 1023 | 1.086006 | 1.766213 | 0.615× |
| 2048 | 2047 | 1.095911 | 1.841710 | 0.595× |
| 4096 | 4095 | 1.105513 | 2.020126 | 0.547× |

Table values are medians of per-run mean TPOTs; graph bands are observed run ranges, not confidence intervals. Random weights with the SmolLM-135M structural configuration and an explicit 8192-position synthetic context extension; no pretrained quality claim.
