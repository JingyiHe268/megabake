# 离线安装与 mock 兼容性测试

本方案使用真实的随机初始化模型与合成输入，**不伪造测量值**，不下载预训练权重。
原版 `requirements.txt` 仍为 CUDA 13；这里是另建环境的 CUDA 12 实验适配。

## 已实测的约束

2026-10-26，在作业 281111 / hn43 / H100 80GB / NVIDIA 550.54.15 上，调用已有 runtime 的
`cudaGetDeviceCount`：CUDA 12.9 返回 1 张卡、错误码 0；CUDA 13.0 返回错误码 35，驱动不足。
因此离线传输原版 cu130 包只能解决下载问题，无法直接解决运行兼容。
这些诊断只保留本地与服务器，不作为性能结果上传。

官方 PyTorch 索引存在 `torch==2.14.0+cu126` 的 CPython 3.12 / Linux x86_64 wheel。
CuTe 4.8 的默认包使用 CUDA 12 库，`[cu13]` extra 另装 CUDA 13 库。
本方案选用 CUDA 12 包，CUDA Python 和 bindings 均固定 12.9.10。
CuTe 的官方驱动要求高于本机驱动，因此只有实际 CuTe 编译和运行通过后才能判定试验可用。

真实验证已完成：PyTorch BF16 GEMM、CuTe GPU 单测、mock MLP、两层及完整 30 层 SmolLM 均通过。
完整模型的 MegaBake / eager 相对 L2 误差为 0，profiler 检查为一个工作内核。
[性能表格](../../results/20261026-h100-mock-smollm30-cu126/published/performance.md)记录了四种实现的实测结果。
这是 CUDA 12 实验配置的可运行证据，不代表 R550 的官方支持或预训练模型复现完成。

## 文件与目录

- `requirements-cu126.in`：试验直接依赖。
- `requirements-cu126-linux.lock`：按 Linux x86_64 / Python 3.12 解析的全部依赖，包含 CUDA 库与 Triton。
- 本地 wheelhouse：`.repro/offline/wheelhouse-cu126/`，不提交 Git。
- 服务器 wheelhouse：`~/hejy/wheelhouse/megabake-cu126/`。
- 新环境：`~/hejy/envs/megabake-cu126/`，与原 MegaBake、SGLang 环境分开。

macOS 的 pip 跨平台下载只选择 Linux wheel 还不够，依赖环境标记也必须按 Linux 解析。
锁文件由 uv 的 `--python-platform x86_64-unknown-linux-gnu --python-version 3.12` 产生；
从锁文件下载时使用 `--no-deps`，或按 PyPI 元数据下载并校验对应 Linux wheel。
torch wheel 的 SHA256 从官方 PyTorch 索引取得，其他 wheel 的 SHA256 从 PyPI 取得。

## 离线安装

传输前生成每个 wheel 的 SHA256 清单；传输后在服务器重新验证。然后从服务器项目根目录执行：

```bash
/HOME/sysu_xwzhang/sysu_xwzhang_3/.local/bin/uv pip install \
  --python /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake-cu126/bin/python \
  --offline --no-index \
  --find-links /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/wheelhouse/megabake-cu126 \
  -r tools/repro/requirements-cu126-linux.lock
```

测试直接通过 `PYTHONPATH` 引用当前源码，不安装原项目 cu13 extra，不改原依赖声明。

## GPU 验证

先测试 torch 的张量运算和 GEMM，再测试 CuTe body，最后执行 MegaBake：

```bash
export CUDA_HOME=/HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/toolchains/cuda-12.9
export PATH="/HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake-cu126/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEGABAKE_RUN_GPU_TESTS=1
srun --jobid=281111 --overlap --exact -N1 -n1 -c4 --mem=24G \
  python tools/repro/mock_smoke.py --workload mlp --rows 4 \
  --output .repro/mock-mlp/report.json
# MLP 通过后，再验证两层随机权重 SmolLM 结构。
srun --jobid=281111 --overlap --exact -N1 -n1 -c4 --mem=24G \
  python tools/repro/mock_smoke.py --workload smollm --layers 2 --rows 4 \
  --output .repro/mock-smollm/report.json
```

脚本只有在数值检查、单工作内核检查和全部计时完成后才写成功报告。
比较范围是同一个随机模型的同步 forward：eager、eager CUDA Graph、Inductor、MegaBake。
不将结果称为预训练模型精度，也不与之前 Qwen3 的 SGLang 引擎延迟直接计算加速比。
失败日志不发布；成功结果仍只导出数据表和性能对比图。

也可从本地通过已集成的工作流运行，显式选择兼容性配置：

```bash
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-cu126.json \
  launch unique-mock-run --mock smollm --layers 30
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-cu126.json status unique-mock-run
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-cu126.json collect unique-mock-run
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-cu126.json publish unique-mock-run
```

此配置不使用模型权重目录；随机权重在 GPU 测试进程内按固定种子生成。
`--mock mlp` 用于小型 MLP 检查，`--mock smollm --layers N` 用于 N 层模型结构测试。
SmolLM 的四种实现均接收相同的预构造因果 mask；脚本先检查它与原始二维 padding mask 的 eager 输出一致。
这样可排除 HF eager mask 构造期间的 CPU 标量复制，允许 CUDA Graph 捕获；计时不包含 mask 预处理。

## 多长度与重复测量

在本地提交并部署后，顺序测试多个长度，每个长度单独编译、检查，再采样。四种实现共用随机权重与输入。
`--rounds` 是同一进程内重复计时，不表示独立重启进程；各轮轮换实现顺序。

```bash
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-sweep.json \
  launch unique-token-sweep --mock smollm --layers 30 --tokens 4 8 16 32 64 128 --rounds 3
python3 tools/repro/workflow.py --config tools/repro/ln301-mock-sweep.json collect unique-token-sweep
# 本地发布器的多长度绘图需要 matplotlib；服务器计算环境无需安装它。
# 首次准备：uv venv .repro/plot-env
# uv pip install --python .repro/plot-env/bin/python matplotlib==3.11.2
.repro/plot-env/bin/python tools/repro/workflow.py --config tools/repro/ln301-mock-sweep.json publish unique-token-sweep
```

多长度发布输出 `performance.csv`、`rounds.csv`、`samples.csv`、Markdown 汇总表和两张 SVG 折线图。
只纳入状态完成、数值检查和单内核检查通过的长度。折线阴影为各轮中位数范围，不是置信区间。
`--tokens` 改变 prefill 输入长度；不把它解释为 decode 的 KV 上下文长度。

## 低于官方驱动要求的实际影响

[CuTe 4.8 文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
要求 CUDA 12.9 对应的驱动至少 575.51.03。
[CUDA 次版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
允许部分 CUDA 12.x 应用在 R525 以上运行，但有功能限制，不能覆盖 CuTe 的更严格要求。

- 依赖新驱动 API 的路径可能报 `cudaErrorCallRequiresNewerDriver`；CUDA 13 runtime 在本机已实测驱动不足。
- 新版 PTX 交给旧驱动 JIT 时可能无法加载；为当前 H100 架构预编译的机器码与驱动 PTX JIT 是不同路径。
- 已通过的工作负载不能证明未测试的算子、形状和编译选项兼容。发生失败时需查看实际错误，不能一律归因于驱动。
- 低版本号本身不说明结果一定错误，也不说明性能固定降低。当前数值检查是特定测试的证据，不能外推到所有输入。
- 同机、同环境内的性能比较可以记录，但不能把该实验配置的数据称为官方 CUDA 13 环境的复现。
