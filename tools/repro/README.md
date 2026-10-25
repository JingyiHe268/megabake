# ln301 单卡复现与版本同步

本工作流固定 MegaBake 源码 `f70b54cec53cc4e46136c7082e0c9e9f774f5ece`，在其上增加实验工具。
当前本地下载目录已与该提交逐文件核对。不要把 `origin/main` 的最新版本当作本次版本。

目标仓库：https://github.com/JingyiHe268/megabake

实验分支：`reproduction/h100-sglang`。保留 `main`；不 force push。
作者身份只配置在本仓库：`JingyiHe268 <hejy268@mail2.sysu.edu.cn>`。
服务器的全局 Git 身份和已有 SGLang 修改均不覆盖。

## 数据流

```text
本地 commit ── git bundle + SHA256 ── SSH/SCP ── ~/hejy/megabake
    │                                              │
    │                                   srun --jobid=281111
    │                                              │
GitHub 实验分支 ← 本地结果 commit ← rsync ← results/<run-id>
```

服务器无须访问 GitHub，也不接收本地私钥或 GitHub token。
`281111` 是 Slurm 作业号，不是进程 PID。当前分配为 hn43 / H100 80GB；后续重新申请后必须更新配置。
`--overlap --exact` 使用这个已有分配；运行脚本仍检查 `SLURM_JOB_ID`，防止在登录节点跑 GPU 测试。
请不要同时在该 GPU 上跑其他性能测试。启动流程不杀已有进程，也不取消分配。

## 日常命令（在本地仓库根目录）

```bash
# 1. 修改源码或配置后先提交；不要提交环境、模型、密钥。
git add tools/repro .gitignore
git commit -m "Update reproduction configuration"

# 2. 传输当前提交；服务器存在未提交源码或分支不一致时拒绝覆盖。
python3 tools/repro/workflow.py deploy

# 3. 后台启动，SSH 断开后继续执行；run ID 不可复用。
python3 tools/repro/workflow.py launch 20261025-h100-smollm-001

# 4. 查询进度；返回状态可能是 running / blocked / failed / complete。
python3 tools/repro/workflow.py status 20261025-h100-smollm-001

# 5. 拉回原始日志、环境、结果；生成 HTML 报告与 SHA256 清单。
python3 tools/repro/workflow.py collect 20261025-h100-smollm-001

# 6. 打开 results/<run-id>/report.html，保存真实浏览器截图为 report.png。
#    截图后重新执行 report.py 更新文件哈希；结果目录可以包含额外 profiler 截图。
python3 tools/repro/report.py results/20261025-h100-smollm-001

# 7. 提交结果并推送实验分支，包含 blocked / failed 的真实诊断结果。
python3 tools/repro/workflow.py publish 20261025-h100-smollm-001

# 8. 下次 deploy 会将结果提交和后续源码一起快进同步到服务器。
```

本机 `origin` 使用现有 SSH 别名 `github_jingyi`，该密钥已验证属于 JingyiHe268。
ChatGPT GitHub connector 绑定的是另一账号，因此本流程使用本地 Git，不使用该 connector 发布。
更换电脑时使用自己的 SSH 配置，避免复制别人的私钥。

`collect` 不删除任一端的文件；大型 generated、Nsight 原始数据库不自动拉取。
`publish` 只提交本次目录内允许的证据文件，单文件上限 10 MiB。
较大的 trace 和 `.nsys-rep` 留在服务器，可单独压缩归档；截图应同时保留对应的原始日志或 trace。
读取到 `running` 时不要发布；`complete` 仅表示所选流程执行成功，不自动代表两套计时口径相同。

## 环境与模型准备

配置文件为 `tools/repro/ln301.json`。已有 SGLang 使用其原 `.venv` 和源码；不安装或升级其中任何包。
MegaBake 必须使用独立环境 `~/hejy/envs/megabake`。

本次实测 SGLang：Python 3.10、PyTorch 2.13.0+cu129、transformers 5.12.1、CuTe 4.6.2。
原版 MegaBake 要求 Python >=3.12、PyTorch 2.14.x、CuTe 4.8.0、CUDA 13.0 开发工具链。
当前没有符合要求的独立 MegaBake 环境，服务器和计算节点也无法解析外部域名。

在已有可用 Python 3.13 和 CUDA 13 工具链、可访问依赖源后，在服务器执行：

```bash
python3.13 -m venv /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake
source /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake/bin/activate
export CUDA_HOME=/HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/toolchains/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
cd /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/megabake
python -m pip install -r requirements.txt
```

无外网时，在同架构 Linux/Python 的联网机器准备完整 wheelhouse，再用 SCP 传输，安装时使用
`pip install --no-index --find-links=/path/to/wheelhouse -r requirements.txt`。
macOS 下载的本机 wheels 不能直接代替 Linux wheels；CUDA toolkit 与 Python 本体也需要分别准备。

当前 NVIDIA 内核驱动为 550.54.15。CUDA 13 通常要求 R580+；数据中心 GPU 可研究 NVIDIA 官方的
[Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
路径，但必须满足兼容表且实际通过 CUDA 张量和 CuTe 编译检查。单纯修改 `CUDA_HOME` 不会升级驱动。
本流程不修改系统驱动，也不把 12.9 环境伪装成原版 CUDA 13 复现。

共同模型默认选择上游已验证的 SmolLM-135M，revision 固定为
`1d461723eec654e65efdc40cf49301c89c0c92f4`。
联网机器可用 `huggingface_hub.snapshot_download` 指定上述 revision 和目标目录，传输时使用 `rsync -aL`
复制实际内容。服务器最终目录应包含 config、tokenizer 和完整 safetensors。
测试始终 offline，从共同本地路径加载；会记录权重和配置 SHA256。
请保留下载时的模型 ID、revision 证明，不能仅通过给目录改名声明模型版本。

## 测试口径

- 基线：已有 SGLang `bench_one_batch`，BF16、TP=1、batch=1、input=4、output=32；CUDA Graph 开和关各重复三次。
- 原版复现：`benchmarks/full_model.py` 中 eager、Inductor、Inductor CUDA Graph、MegaBake；20 samples、5 warmup、KV capacity=128。
- 两者使用同一个本地模型路径，并记录模型文件哈希、完整命令、源码 commit、SGLang 未提交 Python patch、软件包清单、GPU、Slurm 分配和退出码。
- SGLang 包含引擎/采样工作，原版脚本主要计模型 forward，缓存初始化与上下文推进也不同。因此报告并列保留原始数据，**不自动计算 SGLang / MegaBake 加速比**。
- 正式跨框架对比还需统一预填充、固定上下文范围、token 序列、采样开销、输出语义及同步边界，并验证输出正确性。不要把引擎延迟、模型 forward 和 GPU kernel time 混作同一指标。
- 已有 Qwen3.6-35B-A3B-FP8 是 MoE/FP8 工作负载，不能拿来对照本项目 SmolLM/BF16 的结果。

仅检查已有环境的基线，可创建独立配置指定完整的已缓存模型路径，并使用：

```bash
python3 tools/repro/workflow.py --config tools/repro/your-config.json launch unique-baseline-id --baseline-only
```

该结果只能说明该模型基线，不能替代共同模型正式复现。

## 文件与验证

每次运行的 `status.json` 是最终状态，`environment.json` 是环境及版本证据，`*.status.json` 保存阶段退出码，
`*.log` 保存原始输出，`*.jsonl` / `report.json` 保存原始测量，`report.html` 和 `report.png` 用于阅读展示。
阻塞不会填入 0 毫秒或虚构性能；上游脚本返回 0 但内部 variant 失败时，外层仍标记失败。
输入模型缺失时不下载浮动版本，也不会偷偷切换模型。

```bash
python3 -m unittest discover -s tools/repro -p 'test_*.py' -v
```
