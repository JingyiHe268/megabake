# 当前复现进度（2026-10-25）

## 已完成

- GitHub 目标：`JingyiHe268/megabake`，分支 `reproduction/h100-sglang`。
- 本地源码精确匹配上游 `f70b54cec53cc4e46136c7082e0c9e9f774f5ece`；在该版本上添加复现工具，未升级编译器源码。
- Git 作者：`JingyiHe268 <hejy268@mail2.sysu.edu.cn>`，仅仓库级配置。
- 本地 Git bundle → SSH/SCP → `~/hejy/megabake` 已验证，包括 bundle SHA256、提交号一致性和快进更新。
- 在已有 Slurm 作业 `281111`（hn43，H100 80GB）执行；未申请新 GPU，未更改系统驱动，未覆盖 SGLang 修改。
- 结果回收与 GitHub 发布已验证；按最新要求，只发布性能表格及数据生成的对比图。
- 使用已有 Python 3.12.13 创建 `~/hejy/envs/megabake`，找到集群 CUDA 13.0 工具链 `/APP/u22/ai_x86/CUDA/13.0`。
- 工作流测试包含发布筛选与无效数据拒绝检查。

## SGLang 基线冒烟结果

记录：[20261025-h100-qwen3-sglang-v2](../../results/20261025-h100-qwen3-sglang-v2/published/performance.md)。

模型 Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`，BF16，TP=1，batch=1，input=4，output=32。
每个配置一轮进程测试，decode 中位数取 31 个推进中的解码步骤；这是冒烟结果，没有重复运行置信区间。

| SGLang 配置 | Prefill ms | Decode median ms/token |
| --- | ---: | ---: |
| CUDA Graph 开 | 4.5699 | 1.6432 |
| CUDA Graph 关 | 15.5754 | 13.8795 |

SGLang 源码 `0948e6ebed82c7dd7df8a0c347076e46af84bef6` 带已有工作区修改，patch 仅在本地与服务器保存。
执行工具版本为 `8eb076b448a70c1766e7b94a3661b47e1453d31b`。
完整 MegaBake 复现仍待依赖与共同模型就绪。环境清单、诊断日志及失败记录只保留本地和服务器。
系统驱动保持 550.54.15；运行兼容性尚待验证。

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

然后在本地按 [README](README.md) 的 `deploy → launch → status → collect → publish（性能表格/对比图）` 流程继续。
作业号变化时更新 `ln301.json` 并提交后再部署，不复用已有 run ID。

当前脚本能分别收集 SGLang 基线和原版 MegaBake/Inductor 复现，**没有实现统一计时边界的跨框架比较 harness**。
正式 SGLang / MegaBake 加速比还需要统一 token 序列、缓存上下文、采样范围、输出检查、同步边界并重复测量。
当前基线 Qwen3 与目标 SmolLM 也不同，所以没有报告跨框架加速比。
