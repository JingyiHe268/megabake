# H100 decode profiling：差距在哪里

本次实测支持：**MegaBake 相对 SGLang 的约 0.7–0.9 ms/token 差距，主要已经存在于 GPU 执行内部；注意力路径随有效 KV 长度增长的成本是重点。** 单一 megakernel 减少了主机发射次数，但当前算子组合、全局同步和统一资源配置尚未达到 SGLang 的执行效率。

数据与图：[完整 profiling 表格](../../results/20261026-h100-decode-profile/published/profiling.md)、[连续 decode 对比](../../results/20261026-h100-decode-profile/published/compiler_graph_comparison.svg)、[编译器和图捕获消融](../../results/20261026-h100-decode-profile/published/cuda_graph_ablation.svg)、[GPU 活动时间](../../results/20261026-h100-decode-profile/published/gpu_timeline.svg)、[内部阶段趋势](../../results/20261026-h100-decode-profile/published/megabake_phase_breakdown.svg)。

## 同条件与计时边界

hn43 的 H100 80GB，batch=1、BF16、TP=1、真实 prefill=128。所有实现加载同一个非零随机 SmolLM-135M checkpoint，权重 SHA256 为 `459af1fb7c20bdb2982b13330ff8bdab07c3ff88b1d7b506392ddb6de896b2da`。模型扩展位置上限为 8192；没有使用预训练权重，不评估生成质量。

输出长度 D 包含来自 prefill logits 的第一个 token，因此连续计时执行 D−1 步 decode。计时包含后端调用、KV 更新、greedy argmax、GPU 输出缓冲写入；仅循环起止全局同步。加载、编译、warmup、prefill、HTTP、分词不计入。每长度三轮，报告各轮平均 TPOT 的中位数。

本轮重测 128、1024、4096 三个代表性输出长度；此前完整六组长度仍见 [原连续生成结果](../../results/20261026-h100-generation-128-4096/published/performance.md)。长度始终指输出长度，profiling 表中的 cached tokens 仅标识窗口所在位置。

SGLang 保留原有脏工作区及 torch 2.13/cu129 环境；MegaBake、Inductor 和 PyTorch 图捕获使用隔离的 torch 2.14/cu126、CuTe 4.8 环境。驱动固定 550.54.15。因此 SGLang 对比包含软件栈及模型实现差异；MegaBake 与 Inductor 使用相同模型包装、权重和环境，注意力替换方式单独标明。

## 连续 decode 结果

单位 ms/token，数值越低越好：

| 实现 | 输出 128 | 输出 1024 | 输出 4096 |
| --- | ---: | ---: | ---: |
| SGLang，已有 CUDA Graph | 1.0370 | 1.0889 | 1.1086 |
| MegaBake | 1.7299 | 1.7661 | 2.0199 |
| Inductor，纯张量注意力 | 5.3655 | 5.3916 | 5.3683 |
| Inductor + CUDA Graph，纯张量注意力 | 1.8608 | 1.8606 | 1.8614 |
| Inductor + FlashAttention + CUDA Graph，逻辑 KV 长度 | 1.5861 | 1.5864 | 1.5968 |
| Eager + CUDA Graph，纯张量注意力 | 4.4715 | 4.4670 | 4.4594 |
| Inductor + masked SDPA + CUDA Graph | 8.2185 | 8.2168 | 8.2102 |

本轮 MegaBake−SGLang 差值为 **0.6929 / 0.6772 / 0.9113 ms/token**。128 组 MegaBake 一轮为 2.0587，另两轮约 1.7299；全部逐轮数据均保留，未删除较慢样本。三轮观测范围不是置信区间，也不是跨进程独立实验。

## GPU 内部已出现大部分差距

用独立 CUPTI 窗口，分别在已执行 0、2032、4079 步 decode 后，再记录 16 步。以下为 GPU 活动时间，不是完整主机调用延迟：

| 窗口起始 cached tokens | SGLang GPU busy，区间并集 | MegaBake 模型单 kernel | 两者差值 |
| ---: | ---: | ---: | ---: |
| 128 | 1.0064 ms | 1.5974 ms | 0.5910 ms |
| 2160 | 1.1114 ms | 1.8627 ms | 0.7514 ms |
| 4207 | 1.1285 ms | 2.0786 ms | 0.9501 ms |

MegaBake 单模型 kernel **尚不包含**另外约 0.013 ms 的 greedy argmax 等操作，其内部时间已明显大于 SGLang 全部 GPU 活动。SGLang 每步观察到 344 个 CUDA kernel，但模型经一次 decode CUDA Graph 执行；kernel 数多不等于逐个 Python 发射。

这组证据足以把优先排查方向放到 GPU 内部。它不能将三组平均 TPOT 的差值精确拆成互不重叠的 CPU/GPU 项：窗口位置和平均整段长度不同，profiler 也会扰动执行。首个 profiler 窗口出现明显主机停顿，**没有用 profiler 放大的空闲间隔归因生产性能**。CUDA API 时间与 GPU 时间重叠，不能直接相加。部分很短的辅助 kernel 在跟踪中少记，主模型 megakernel 的 16 次调用均被记录；完整观测计数以 CSV 为准。

## 内部阶段与代码的对应关系

另外编译启用 global-timer 的 MegaBake，只用于归因，不进入主性能结果。计划共 **603 phases、183 个 barrier 区间、132 workers**。其中共 211 个 GEMM phase，210 个处于计时覆盖范围；最后 3 个 phase 没有被最后一个 barrier stamp 覆盖。

| 近似阶段分组 | 初段 cached=128 | 中段 cached=2160 | 末段 cached=4207 |
| --- | ---: | ---: | ---: |
| 注意力区间，30 层 | 0.4851 ms | 0.7829 ms | 1.1698 ms |
| Tensor-core GEMM 区间 | 0.7061 ms | 0.7020 ms | 0.7036 ms |
| RoPE 区间 | 0.1028 ms | 0.1025 ms | 0.1036 ms |
| 其他 fused elementwise | 0.1820 ms | 0.1813 ms | 0.1815 ms |

注意力区间增加约 0.685 ms，而其他主要类别基本稳定，支持**长序列退化集中在注意力路径**。SGLang 观察到的注意力 kernel 累计约 0.236 / 0.348 / 0.367 ms/step，增长显著更缓。但两种计时机制不同，不能把上述数字逐项相减拼成精确的差距预算。

具体代码线索：

- [StatefulSmolLM](../../tests/fixtures.py) 独立执行 Q/K/V 和 gate/up 投影；MegaBake 当前计划保留每层 7 个 GEMM，共 210 个，再加输出头。SGLang 的 `sglang/srt/models/llama.py` 使用 `QKVParallelLinear` 和 `MergedColumnParallelLinear`，每层 4 个投影 GEMM；trace 为 120 次层内 GEMM 加 1 次输出头。模型投影打包是实际差异，尚未单独消融其收益。
- [attention.py](../../src/megabake/bodies/attention.py) 使用 stateful row/split-context 路径，显式处理 score、softmax、value 求和、chunk 交换及 KV 发布；SGLang trace 使用 Hopper FlashAttention 主 kernel 与 combine kernel。注意力执行策略及中间舍入语义不同，替换时仍须验证数值和 KV 状态。
- [codegen.py](../../src/megabake/codegen.py) 的表驱动 cooperative kernel 仍有跨 phase 的 grid barrier。阶段计时包含计算、交换和同步；按 work-item 比例分摊类别是近似值，**不能声称 183 次 barrier 单独消耗了多少时间**。计划 work-item 数也不等于生成器 split-K/split-context 后的实际设备 tile 数。

## 硬件计数器：统一资源配置有明显代价

独立 Nsight Compute 运行成功，分别采集 cached=128、4207 的完整 MegaBake 模型 kernel：

| 指标 | cached=128 | cached=4207 |
| --- | ---: | ---: |
| Registers/thread | 240 | 240 |
| Dynamic shared memory/block | 159744 B（156 KiB） | 159744 B |
| Grid / block | 132 / 256 threads | 132 / 256 threads |
| Active warps / peak，active cycles | 12.50% | 12.50% |
| DRAM throughput / peak | 4.22% | 3.99% |
| SM throughput / peak | 6.79% | 6.85% |
| DRAM bytes | 274682880 | 368837888 |

在这次采集下，不能解释为全卡显存带宽饱和。统一 kernel 的高寄存器、shared memory 配置及每 SM 一块，限制了混合 body 的驻留选择；低吞吐与同步、低并行度、依赖链等方向相符。**计数器没有独立证明各因素的因果占比**，更不能仅凭 occupancy 判断优化收益。

NCU 使用 kernel replay，测得时间为 1.944 / 2.754 ms，受 replay/cache/采集条件影响，未替代无 profiler 的 TPOT，也未混入 CUPTI 时间作相减。

## Inductor 与 CUDA Graph 的结论

1. **编译与图捕获分别起作用。** 4096 输出时，纯张量 Inductor 5.368 ms，捕获后 1.861 ms，约 2.88 倍；Eager+Graph 仍为 4.459 ms，说明只有去掉主机发射也不够，算子融合与注意力实现同样重要。
2. **MegaBake 对普通 Inductor 的优势会被弱基线放大。** 对纯张量 Inductor+Graph，MegaBake 在 128 输出时快约 7%，4096 时慢约 8.5%。纯张量版本使用固定容量 4352 的掩码注意力，有额外 padded 运算，必须标明此局限。
3. **加入逻辑长度 FlashAttention 后，Inductor+Graph 在三组都更快。** MegaBake 时间分别为其约 1.091 / 1.113 / 1.265 倍。因此当前结果不支持“单 kernel 必然比多个高效 kernel 更快”。
4. **调用 SDPA 不保证落到高效路径。** masked SDPA 组的 trace 为 `fmha_cutlassF_bf16_aligned_64x64_rf_sm80`，注意力约占 6.94 ms/step。该组是派发与布局消融，不作为 H100 最优注意力基线。补充的 FlashAttention 通过 GPU `seqused_k` 读取有效长度，避免 padded attention 运算。
5. **不能直接给当前 MegaBake 外套 CUDA Graph。** [compiler.py](../../src/megabake/compiler.py) 的 `prepare_launch` 每次更新主机地址环及 stamp，设备 launch 计数必须与之对应，复用环槽还会等待 event。只重放已捕获 launch 不维护该协议，可能触发 trap；本次没有把这种调用列为有效结果。要研究 MegaBake+Graph，需要先实现可验证的稳定地址/计数协议，而非只加一个 capture 开关。

所有 Inductor/Graph 组都在步骤 1、127、1023、4095 用同输入、同 KV 状态与各自原生参考核对，logits/new-KV L2 <0.02，旧 KV 不变、长度正确递增。FlashAttention 组本轮最大 logits 和新 KV 误差均为 0；masked SDPA 组最大分别约 0.00811、0.00844。Inductor 显式开启 `emulate_precision_casts=True`。跨不同注意力实现/软件栈不声称 greedy 序列逐位一致，也不评估预训练模型质量。

## 后续优化优先级

以下是由实测提出的候选，尚未实现或验证收益：

1. 优先隔离 stateful attention，保持准确的缓存更新与舍入规则，比较现有 split-context、FlashAttention、不同 chunk 数/资源配置，先解决随长度增长的成本。
2. 加入 QKV、gate/up 权重打包消融，保持共同权重的逻辑映射和正确性，区分模型改写与调度器收益。
3. 降低混合 kernel 的统一资源要求，并对 grid barrier 做空载/有载的独立消融；不能把全部 phase 区间成本算给同步。
4. 最后再改造稳定缓冲与可捕获 runtime，量化地址绑定、Python 包装及 graph replay 的收益。当前证据表明单独消除主机开销无法消除已存在于 GPU 内部的差距。

主实验执行提交 `795ba2638b7b1732c0fc3d14427af9ebe0e1ef54`；逻辑长度 FlashAttention 与 NCU 补充执行提交 `acddd00594b8cde0b57807995bc42a69efb40589`。MegaBake 核心编译器与上游 `f70b54cec53cc4e46136c7082e0c9e9f774f5ece` 一致。只发布成功的性能 CSV/Markdown 表格与数据生成 SVG；原始 trace、诊断日志、错误图像不上传 GitHub。
