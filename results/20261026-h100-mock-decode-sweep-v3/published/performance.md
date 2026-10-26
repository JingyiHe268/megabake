# MOCK SmolLM 30-layer decode context sweep

BF16, batch 1, seed 20261026, 3 timing rounds × 50 samples per variant and length. Round order rotates; weights are fixed across lengths; compilation and warmup excluded. The three rounds share one process and compilation, not independent process repeats.

fixed-context one-token decode including KV updates; reset excluded; synchronized host+GPU calls; no SGLang engine comparison. Random weights; no pretrained-model or end-to-end generation claim.

experimental CUDA 12; not the upstream cu130 environment; torch 2.14.0+cu126, CuTe 4.8.0, NVIDIA H100 80GB HBM3. Execution commit `1b3fa9361047f3fd9a8e782bb4f167cf39682998`.

| Cached tokens | Eager ms | CUDA Graph ms | Inductor ms | MegaBake ms | CUDA Graph / MegaBake | Inductor / MegaBake |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 28.414203 | 3.861142 | 5.510993 | 2.256760 | 1.711× | 2.442× |
| 256 | 22.594549 | 3.905907 | 4.361218 | 2.173426 | 1.797× | 2.007× |
| 512 | 22.636298 | 3.985727 | 4.488640 | 2.196452 | 1.815× | 2.044× |
| 1024 | 22.539603 | 4.792398 | 4.380947 | 2.282755 | 2.099× | 1.919× |
| 2048 | 24.005350 | 4.195433 | 4.640771 | 2.471149 | 1.698× | 1.878× |
| 4096 | 22.717340 | 5.189376 | 4.338654 | 2.862399 | 1.813× | 1.516× |

Only validated successful lengths are plotted. CSV includes round ranges, correctness errors and raw performance samples. Ranges describe the observed round medians, not confidence intervals. Speedup >1 means MegaBake is faster; no SGLang engine comparison.

Each call consumes one token at the stated cached context and updates KV state. Capacity is context+128, specialized per length. Logical lengths are restored before each call outside timing. Cached K/V are seeded nonzero random tensors, not computed from a prompt. Eager, CUDA Graph and Inductor use a GPU-only tensor KV reference checked against upstream eager logits and full KV state. MegaBake uses the original declared KV op; existing cache contents and one-token updates are validated.

Inductor enables `emulate_precision_casts=True` to preserve eager BF16 intermediate rounding. This is an explicitly configured baseline, not the default Inductor precision-fusion configuration.
