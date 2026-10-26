# MOCK SmolLM 30-layer token sweep

BF16, batch 1, seed 20261026, 3 timing rounds × 50 samples per variant and length. Round order rotates; weights are fixed across lengths; compilation and warmup excluded. The three rounds share one process and compilation, not independent process repeats.

synchronized model-forward calls; SmolLM uses a prebuilt causal mask; no SGLang engine comparison. Random weights; no pretrained-model or decode claim.

experimental CUDA 12; not the upstream cu130 environment; torch 2.14.0+cu126, CuTe 4.8.0, NVIDIA H100 80GB HBM3. Execution commit `d467692a811ee121daefae5dce369c3b6b650e08`.

| Tokens | Eager ms | CUDA Graph ms | Inductor ms | MegaBake ms | CUDA Graph / MegaBake | Inductor / MegaBake |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 17.900769 | 3.577892 | 4.428692 | 2.018763 | 1.772× | 2.194× |
| 8 | 14.304863 | 3.656404 | 3.550392 | 1.964470 | 1.861× | 1.807× |
| 16 | 17.306825 | 3.779296 | 4.432242 | 2.166577 | 1.744× | 2.046× |
| 32 | 15.053451 | 3.890594 | 3.624739 | 2.232155 | 1.743× | 1.624× |
| 64 | 14.001060 | 3.975736 | 3.579965 | 2.549695 | 1.559× | 1.404× |
| 128 | 17.298004 | 4.092410 | 4.440278 | 4.580336 | 0.893× | 0.969× |

Only validated successful lengths are plotted. CSV includes round ranges, correctness errors and raw performance samples. Ranges describe the observed round medians, not confidence intervals. Speedup >1 means MegaBake is faster; no SGLang engine comparison.
