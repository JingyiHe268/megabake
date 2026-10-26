# 当前复现进度（2026-10-26）

## 结论与环境

原版 CUDA 13 环境不能在固定 NVIDIA 550.54.15 驱动上直接运行：CUDA 13.0 runtime 实测返回驱动不足。
已另建 CUDA 12 实验环境，完成本地下载 → 校验 → SCP → 服务器校验 → 离线安装。
共 69 个 Linux x86_64 / CPython 3.12 wheels，约 3.84 GB；无需登录节点联网。

| 项目 | 当前实测配置 |
| --- | --- |
| Slurm / GPU | 281111 / hn43 / H100 80GB |
| 驱动 | 550.54.15，保持不变 |
| Python | 3.12.13 |
| PyTorch | 2.14.0+cu126 |
| CuTe / CUTLASS DSL | 4.8.0 |
| CUDA 开发工具链 | 用户目录 CUDA 12.9 |
| 环境目录 | `~/hejy/envs/megabake-cu126` |
| 权重与输入 | 固定种子 20261026，随机初始化，无 checkpoint 下载 |

真实 CUDA 张量、BF16 GEMM、CuTe 生成内核 GPU 单测（1 passed）、MLP、两层与完整 30 层 SmolLM 均已通过。
CuTe 官方 CUDA 12.9 配置要求驱动至少 575.51.03；当前结果仅证明这些工作负载在该机器实测可运行，
不代表 R550 获得官方支持，也不等同于原版 cu130 复现。
参见 [CUDA 13 驱动要求](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
与 [CuTe 兼容要求](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)。

## Mock forward 性能

BF16、4 tokens、batch=1（SmolLM）、每种实现 warmup 5 次并采样 20 次。
记录同步前后包含主机调用开销的 forward 延迟中位数；四种实现使用相同权重和输入。
SmolLM 使用预构造因果 mask，已检查其与原始二维 mask 的 eager 输出一致；计时不含 mask 预处理。
这是单次运行的冒烟测量，没有多轮置信区间，不包含 autoregressive decode 或 SGLang 引擎工作。

| 工作负载 | Eager ms | Eager CUDA Graph ms | Inductor ms | MegaBake ms |
| --- | ---: | ---: | ---: | ---: |
| [MLP](../../results/20261026-h100-mock-mlp-cu126/published/performance.md) | 0.193251 | 0.031309 | 0.328079 | 1.521561 |
| [SmolLM 2 层](../../results/20261026-h100-mock-smollm2-cu126-v2/published/performance.md) | 2.230044 | 0.315759 | 0.532445 | 2.027297 |
| [SmolLM 30 层](../../results/20261026-h100-mock-smollm30-cu126/published/performance.md) | 20.066410 | 3.601586 | 5.483957 | 2.342902 |

MegaBake 相对 eager 的 L2 误差均为 0，三种工作负载均确认只执行 1 个工作内核。
完整 30 层下，MegaBake 比 eager CUDA Graph 约快 1.54 倍，比 Inductor 约快 2.34 倍；
MLP 和两层结果未显示相同优势。上述加速比只适用于这次随机权重、4 tokens 的同步 forward 测量。
完整模型结构已验证可运行；原版 CUDA 13、预训练权重、decode 及共同模型 SGLang 对比仍未完成。

## 六组 token 长度对比

已完成 [4 / 8 / 16 / 32 / 64 / 128 tokens 的完整表格与曲线](../../results/20261026-h100-mock-token-sweep/published/performance.md)。
统一使用 30 层随机权重 SmolLM、BF16、batch=1、seed=20261026；每组独立进程编译，
编译完成后同一进程内重复 3 轮，每轮每种实现 50 个样本，轮换计时顺序。
总计 6 × 4 × 3 × 50 = 3600 个成功性能样本；三轮不是独立重启进程实验。

| Tokens | Eager ms | CUDA Graph ms | Inductor ms | MegaBake ms |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 17.900769 | 3.577892 | 4.428692 | 2.018763 |
| 8 | 14.304863 | 3.656404 | 3.550392 | 1.964470 |
| 16 | 17.306825 | 3.779296 | 4.432242 | 2.166577 |
| 32 | 15.053451 | 3.890594 | 3.624739 | 2.232155 |
| 64 | 14.001060 | 3.975736 | 3.579965 | 2.549695 |
| 128 | 17.298004 | 4.092410 | 4.440278 | 4.580336 |

4–64 tokens 时 MegaBake 快于两个优化基线；128 tokens 时比 CUDA Graph 慢约 11.9%，比 Inductor 慢约 3.2%。
不能从 4 tokens 的优势外推所有长度；当前未对拐点作根因结论，也不能将其直接归因于旧驱动。
各长度均通过数值检查和 1 个工作内核检查；MegaBake / eager 的相对 L2 为 0–0.009645，
Inductor / eager 为 0.013792–0.016289，均低于预设 0.02 阈值。
折线阴影为各轮中位数的观测范围，不是置信区间；汇总、逐轮中位数和全部采样均以 CSV 发布。
实验源码提交 `d467692a811ee121daefae5dce369c3b6b650e08`；编译器源码保持不变。

当前 R550 低于 CuTe CUDA 12.9 的官方驱动要求。六组通过只证明这些路径实测可用；
新驱动 API、新版 PTX JIT 或其他形状仍可能失败。详见 [实际影响](OFFLINE_MOCK.md#低于官方驱动要求的实际影响)。

## SGLang 基线冒烟结果

已有 [Qwen3-0.6B 测量](../../results/20261025-h100-qwen3-sglang-v2/published/performance.md)，
revision `c1899de289a04d12100db370d81485cdf75e47ca`，BF16、TP=1、batch=1、input=4、output=32。
每个配置一轮进程测试，decode 中位数取 31 个推进中的解码步骤；没有重复运行置信区间。

| SGLang 配置 | Prefill ms | Decode median ms/token |
| --- | ---: | ---: |
| CUDA Graph 开 | 4.5699 | 1.6432 |
| CUDA Graph 关 | 15.5754 | 13.8795 |

SGLang 源码 `0948e6ebed82c7dd7df8a0c347076e46af84bef6` 带已有工作区修改，未修改其环境或源码。
SGLang 的 Qwen3 引擎计时与 mock SmolLM forward 的模型、权重和边界不同，不计算二者加速比。
正式比较仍需共同模型、相同 token / KV 上下文、相同计时范围与输出检查。

## 版本与结果工作流

- GitHub：`JingyiHe268/megabake`，分支 `reproduction/h100-sglang`。
- 编译器源码固定上游 `f70b54cec53cc4e46136c7082e0c9e9f774f5ece`，仅新增复现工具。
- Git 作者：`JingyiHe268 <hejy268@mail2.sysu.edu.cn>`，仓库级配置。
- `deploy`：Git bundle + SHA256，服务器快进更新；运行时记录源码提交号。
- `launch --mock`：使用显式 CUDA 12 mock 配置，在现有 GPU 分配执行。
- `collect`：回收原始数据到忽略目录，保留日志供本地与服务器诊断。
- `publish`：只提交成功且有限的性能 CSV / Markdown 表格、柱状图；不上传日志、报告截图或错误图像。

操作见 [README](README.md) 与 [离线 mock 说明](OFFLINE_MOCK.md)。
运行期间不要 deploy 改变服务器源码；运行结束后再同步结果提交。
