# 当前复现进度（2026-10-25）

## 已完成

- GitHub 目标：`JingyiHe268/megabake`，分支 `reproduction/h100-sglang`。
- 本地源码精确匹配上游 `f70b54cec53cc4e46136c7082e0c9e9f774f5ece`；在该版本上添加复现工具，未升级编译器源码。
- Git 作者：`JingyiHe268 <hejy268@mail2.sysu.edu.cn>`，仅仓库级配置。
- 本地 Git bundle → SSH/SCP → `~/hejy/megabake` 已验证，包括 bundle SHA256、提交号一致性和快进更新。
- 在已有 Slurm 作业 `281111`（hn43，H100 80GB）执行；未申请新 GPU，未更改系统驱动，未覆盖 SGLang 修改。
- 结果回收、HTML 报告、浏览器真实截图、校验清单、结果 Git 提交和 GitHub 上传均已验证。
- 使用已有 Python 3.12.13 创建 `~/hejy/envs/megabake`，找到集群 CUDA 13.0 工具链 `/APP/u22/ai_x86/CUDA/13.0`。
- 9 项标准库工作流测试通过。

## SGLang 基线冒烟结果

记录：[20261025-h100-qwen3-sglang-v2](../../results/20261025-h100-qwen3-sglang-v2/SUMMARY.md)。

模型 Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`，BF16，TP=1，batch=1，input=4，output=32。
每个配置一轮进程测试，decode 中位数取 31 个推进中的解码步骤；这是冒烟结果，没有重复运行置信区间。

| SGLang 配置 | Prefill ms | Decode median ms/token |
| --- | ---: | ---: |
| CUDA Graph 开 | 4.5699 | 1.6432 |
| CUDA Graph 关 | 15.5754 | 13.8795 |

SGLang 源码 `0948e6ebed82c7dd7df8a0c347076e46af84bef6` 带已有工作区修改，patch 随结果保存。
执行工具版本为 `8eb076b448a70c1766e7b94a3661b47e1453d31b`。
该次环境完整包清单可同时参阅后续预检 v2 的 `sglang-installed-distributions.txt`；两次之间未改动 SGLang 安装。

最初一轮 smoke 的 PATH 未包含虚拟环境 bin，日志显示 ninja 不可见并触发 JIT fallback；它只作诊断，
不作为这里的最终基线。v2 已修正工具 PATH 并重新运行。

## 完整 MegaBake 复现尚未完成

最新记录：[20261025-h100-smollm-preflight-v2](../../results/20261025-h100-smollm-preflight-v2/SUMMARY.md)，执行工具版本
`0ba8b015bae17a026b0fd1390f0b5900d7cee3e3`。

尚缺 PyTorch 2.14/cu130、CuTe 4.8 等固定依赖，以及共同模型 SmolLM-135M 权重。
已在登录节点从项目根目录实际执行 `uv pip install -r requirements.txt`，下载 PyTorch 时失败：

```text
Temporary failure in name resolution
No appropriate name servers or networks for name found
```

临时指定本地已解析 IP 的 HTTPS 验证也连接超时；用户确认属于当前临时网络问题，平时可直接下载。
安装失败日志为 `dependency-install.log`，网络诊断为 `network-check.log`。

系统驱动固定为 550.54.15。找到 CUDA toolkit 不代表 CUDA 13 runtime 已可运行；依赖下载后仍需在 GPU 节点
验证张量运算、CuTe 编译和完整模型。当前 NVIDIA 官方兼容表未列出 R550，不作兼容承诺。

## 网络恢复后的继续方式

先在登录节点安装固定依赖并下载固定 revision 模型：

```bash
ssh ln301
cd /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/megabake
export CUDA_HOME=/APP/u22/ai_x86/CUDA/13.0
export PATH="$CUDA_HOME/bin:$PATH"
/HOME/sysu_xwzhang/sysu_xwzhang_3/.local/bin/uv pip install \
  --python /HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake/bin/python \
  -r requirements.txt
/HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/envs/megabake/bin/python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("HuggingFaceTB/SmolLM-135M", revision="1d461723eec654e65efdc40cf49301c89c0c92f4", local_dir="/HOME/sysu_xwzhang/sysu_xwzhang_3/hejy/models/SmolLM-135M")'
```

然后在本地按 [README](README.md) 的 `deploy → launch → status → collect → 截图 → publish` 流程继续。
作业号变化时更新 `ln301.json` 并提交后再部署，不复用已有 run ID。

当前脚本能分别收集 SGLang 基线和原版 MegaBake/Inductor 复现，**没有实现统一计时边界的跨框架比较 harness**。
正式 SGLang / MegaBake 加速比还需要统一 token 序列、缓存上下文、采样范围、输出检查、同步边界并重复测量。
当前基线 Qwen3 与目标 SmolLM 也不同，所以没有报告跨框架加速比。
