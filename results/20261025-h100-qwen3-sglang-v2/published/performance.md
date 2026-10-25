# Recorded performance

Successful measurements only. Each row is one trial; no confidence interval. Engine and model-forward scopes differ; no cross-harness speedup is implied.

| Variant | Scope | Model | Batch | Input | Output | Prefill ms | Decode median ms/token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| sglang-cudagraph-off-1 | engine; advancing decode | Qwen/Qwen3-0.6B | 1 | 4 | 32 | 15.5754 | 13.8795 |
| sglang-cudagraph-on-1 | engine; advancing decode | Qwen/Qwen3-0.6B | 1 | 4 | 32 | 4.5699 | 1.6432 |
