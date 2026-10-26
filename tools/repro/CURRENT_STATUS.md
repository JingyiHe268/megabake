# 当前复现进度（2026-10-26）

## 最新：同条件 profiling 与 Inductor/CUDA Graph 消融

已完成 [根因分析](PROFILING_ANALYSIS.md) 与 [成功性能表格、图表](../../results/20261026-h100-decode-profile/published/profiling.md)。
主机计时与 profiler 运行分离；新增逻辑 KV 长度的 FlashAttention+Inductor+CUDA Graph 强基线，以及 MegaBake 阶段计时、NCU 硬件计数器。
中段 GPU 窗口中 MegaBake 模型 kernel 1.863 ms，SGLang 全部 GPU busy 1.111 ms；差距主要已存在于 GPU 内部。
注意力阶段随生成推进约 0.485→0.783→1.170 ms，GEMM 阶段约 0.70 ms 基本稳定；阶段时间包含同步，不是纯算子计时。
4096 输出：SGLang 1.1086、MegaBake 2.0199、Inductor 5.3683、Inductor+Graph 1.8614、Inductor+逻辑长度 FlashAttention+Graph 1.5968 ms/token。
当前 MegaBake 并未战胜强编译器/注意力/图捕获组合。原版核心编译器未改动，下一步候选与对比局限见分析文档。

## 当前主实验：固定 prefill=128，连续输出 128–4096，与 SGLang 比较

**用户所指长度为输出/decode length，不是 KV cache 长度。** 前面的 KV 长度扫描属于另一种微基准，不作为本次需求的性能结论。
[主结果表格与曲线](../../results/20261026-h100-generation-128-4096/published/performance.md)。

共同 checkpoint 为 SmolLM-135M 结构的非零随机权重，固定 seed=20261026；共享 embedding、RMSNorm epsilon=1e-5，
显式将合成测试的最大位置扩到 8192。两个栈加载相同 safetensors（SHA256 已校验），使用同一组 128 个 prompt token IDs。
batch=1、BF16、TP=1；每组真实 prefill 后连续 greedy 生成，KV 自然增长，不在每步恢复状态。
按 SGLang 输出长度定义，D 个输出的第一个来自 prefill logits，计时执行 D-1 次 decode；EOS 不提前终止。
每组重复 3 次，取各次平均 TPOT 的中位数。

| 输出 tokens D | 实际 decode 步数 | SGLang ms/token | MegaBake ms/token | SGLang / MegaBake |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 127 | 1.034033 | 1.729808 | 0.598× |
| 256 | 255 | 1.060801 | 1.740546 | 0.609× |
| 512 | 511 | 1.075818 | 1.747638 | 0.616× |
| 1024 | 1023 | 1.086006 | 1.766213 | 0.615× |
| 2048 | 2047 | 1.095911 | 1.841710 | 0.595× |
| 4096 | 4095 | 1.105513 | 2.020126 | 0.547× |

**当前连续生成口径下 SGLang 更快**：MegaBake 耗时为 SGLang 的约 1.62–1.83 倍。
4096 输出对应 4095 次 timed decode，三轮总耗时中位数约 SGLang 4.527 s、MegaBake 8.272 s。
计时包含后端调用、KV 更新、greedy sampling 与 GPU 输出缓冲写入，只在循环起止同步；
加载、编译、warmup、prefill、HTTP 和分词均不计入。prefill 时间保存在逐轮表中，未加到 decode 指标。

SGLang 已验证长序列的 4095 次 decode 经过 CUDA Graph；MegaBake 模型调用确认只执行 1 个工作内核，
但 greedy sampling 和输出存储是额外操作。MegaBake 在 7 个选定步骤，用同输入、同 KV 状态对照原生 PyTorch，
最大 logits 相对 L2=0.007577、最大新 KV 相对 L2=0.007893，均低于预设 0.02；旧 KV 与长度更新也检查通过。
两栈的自然 greedy 序列在第 123 个输出处开始不同，不声称输出逐位等价。只在相同前缀下检查跨栈数值差异，
不把分叉后的不同前缀 logits 用作正确性判断。

SGLang 保持已有源码及工作区修改、torch 2.13.0+cu129；MegaBake 为 torch 2.14.0+cu126 / CuTe 4.8。
因此是当前两个后端栈的性能比较，不能将差异只归因于 megakernel 编译器。
执行源码提交 `11df6edcdad29c21ff3819e59c107484cf312935`，原 MegaBake 编译器源码不变。

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
完整模型结构已验证可运行；原版 CUDA 13 与预训练权重仍未完成；共同随机模型的 SGLang 连续生成对比见本文顶部。

## 六组 prefill token 长度对比

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

## 历史固定 KV 单步扫描：不是用户指定的输出长度

[完整表格、逐轮数据、3600 个采样及延迟/加速比曲线](../../results/20261026-h100-mock-decode-sweep-v3/published/performance.md)。
30 层随机权重 SmolLM、BF16、seed=20261026、batch=1，每次输入 1 token 并更新 KV。
每个上下文长度独立进程编译，随后同一进程内 3 轮 × 每种实现 50 个样本；不含编译、warmup 或上下文恢复。

| Cached tokens | Eager ms | CUDA Graph ms | Inductor BF16-aligned ms | MegaBake ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 28.414203 | 3.861142 | 5.510993 | 2.256760 |
| 256 | 22.594549 | 3.905907 | 4.361218 | 2.173426 |
| 512 | 22.636298 | 3.985727 | 4.488640 | 2.196452 |
| 1024 | 22.539603 | 4.792398 | 4.380947 | 2.282755 |
| 2048 | 24.005350 | 4.195433 | 4.640771 | 2.471149 |
| 4096 | 22.717340 | 5.189376 | 4.338654 | 2.862399 |

MegaBake 全部六组快于两个优化基线；相对 CUDA Graph 为 1.70–2.10×，相对 BF16 对齐 Inductor 为 1.52–2.44×。
这是固定上下文的同步 host+GPU 单步延迟，不包含采样、分词、引擎调度或持续生成循环。
MegaBake logits 相对 eager 的 L2 最大为 0.001340（2048 上下文）；其他五组为 0。
Inductor logits L2 均为 0；全部实现的新 KV L2 均为 0，旧缓存及未用尾部保持不变，各层长度只增加 1。
所有 MegaBake 组均检查为 1 个工作内核。

KV 内容是非零随机数据，没有用真实 prompt 执行 prefill。各点 cache capacity=context+128，预留编译校准空间；
编译后恢复完整初始状态再验证，每次计时前恢复逻辑长度，恢复不计时。
基线使用 GPU-only tensor KV 实现，已与原声明 op 的 eager logits 和完整 KV 状态检查一致。
Inductor 显式开启 `emulate_precision_casts=True`；图例标记 BF16-aligned，不当作默认 Inductor 配置。
实验源码提交 `1b3fa9361047f3fd9a8e782bb4f167cf39682998`；原编译器源码未修改。

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
