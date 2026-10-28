# A6000 第二部分：按新空卡规则重启

运行编号：20261027-140300。更新：2026-10-28T10:03:04.385472+08:00。

阶段：**GPU_WAITING**。初始运行前已 `git pull --rebase` 并确认 HEAD 为用户指定的 **8b6a767302edd6c00fdba44c5f9d057f2af684db**。使用该提交的独立检出执行任务，发布仓库后续同步不会改变测试源码。不修改源码，不修改或删除其他设备与历史结果目录。

## 本轮范围与环境

只重启 GPU 第二部分，不重复安装依赖或重跑第一部分 CPU selector。上一轮第一部分的 A14B 修复和 CPU 结果见 [上一轮报告](../20261026-161758/README.md)，其测试基线为 03e8d05，不混入本轮结果。

| 版本项 | 当前值 |
|---|---|
| torch | 2.14.1+cu126 |
| cuda | 12.6 |
| torchao | 0.18.0+cu126 |
| diffusers | 0.41.0 |
| transformers | 5.20.2 |
| torchvision | 0.29.1+cu126 |
| sageattention | NOT_INSTALLED |

四版本及 SageAttention 未安装状态复核：**通过**。[版本证据](versions.json)。本轮未安装 flash-attn 或 SageAttention，使用仓库既有后端；未改变系统 CUDA/驱动。GPU preflight 检查 CUDA、HIP、设备、torch.compile、INT8 算子；双卡时也记录 P2P。

## 24 小时自动探查

- 起点：2026-10-27T14:11:34.964789+08:00；等待截止：2026-10-28T14:11:34.964811+08:00。
- 等待期间每 15 分钟检查一次；执行前另做即时复查。24 小时为等待与任务接纳窗口，已经接纳的任务按自身超时收尾并释放资源。
- 空卡条件：**没有计算进程且显存 <1,000,000,000 bytes**。显式查询 `nvidia-smi --query-compute-apps`，同时保守核对 XML 中非 G 进程以防两次查询之间有新任务出现。**Xorg/gnome-shell 等类型 G 的图形进程忽略**；C、C+G 和无法安全分类的非 G 进程仍视为占用。
- 最近检查：2026-10-28T10:00:50.323074+08:00；可用物理 GPU 序号：[0, 2]。完整等待记录见 [gpu-probes.jsonl](gpu-probes.jsonl)，不发布他人的进程名或 GPU UUID。
- **优先单卡三项**：单卡 GPU selector → attention benchmark → INT8 linear benchmark（tokens 14040、32760）。每项执行前重新选择空卡，不要求一直占同一张卡。
- 然后执行**同 NUMA 双卡三项**：NCCL 带宽 → multi_gpu selector → 1.3B Ulysses2 冒烟。双卡只从 0–3 或 4–7 中选择，不跨组。
- 每项结束退出进程并立即释放 GPU，等待/发布期间不占卡。若运行中有别人的计算进程出现，只停止本任务自己的进程，保留中断证据，并在后续探查重试；不干扰 G 图形进程，不 kill 他人的任务。
- 查询失败则不分配卡。出现活动 Slurm 时禁用直接 GPU 执行并记录未完成。保持 MAX_JOBS=4、OMP_NUM_THREADS=8，每条命令设置随机高位端口，torchrun/xdit 显式传 master_port。
- 到期未取得合适卡的任务记录“未完成”；实际测试失败只记录，不改源码。

## 任务结果

计数为 passed / failed / skipped / deselected。跳过与未选择不算通过。卡号为 nvidia-smi 物理序号。

| 任务 | 状态 | 所用 GPU | 随机端口 | 测试计数 |
|---|---|---|---|---|
| pytest-gpu-single | 已执行 | 6 | 54364 | 76 / 0 / 140 / 1872 |
| attention | 已执行 | 6 | 46456 | — / — / — / — |
| int8_linear | 已执行 | 6 | 26807 | — / — / — / — |
| nccl-bandwidth | 已执行 | 0,2 | 25253 | — / — / — / — |
| pytest-gpu-multi | 已执行 | 0,2 | 35390 | 60 / 0 / 42 / 1986 |
| smoke-ulysses2 | 已执行 | 0,2 | 26385 | — / — / — / — |
| preflight-6 | 已执行 | 6 | 22519 | — / — / — / — |
| preflight-7 | 已执行 | 6 | 53478 | — / — / — / — |
| preflight-8 | 已执行 | 6 | 51776 | — / — / — / — |
| preflight-9 | 已执行 | 0,2 | 28609 | — / — / — / — |
| preflight-10 | 已执行 | 0,2 | 56063 | — / — / — / — |
| preflight-11 | 已执行 | 0,2 | 20758 | — / — / — / — |
| preflight-12 | 已执行 | 0,2 | 26546 | — / — / — / — |

失败/部分失败任务：暂无已记录失败任务。完整命令、耗时、卡号、NUMA 组、返回码和失败用例见 [workflow.json](workflow.json)；完整日志在 `logs/`，JUnit 在 `tests/`。

- 单卡 selector：`python -m pytest -m "accelerator and not rocm and not multi_gpu"`。
- Attention：`python benchmarks/wan22/attention_bench.py --json attention.json`；完成后结果位于 `attention.json`。未安装的 FLASH/SAGE 标为 unavailable，不伪造性能数据。
- INT8 linear：`python benchmarks/wan22/int8_linear_bench.py --tokens 14040 32760 --json int8_linear.json`。
- NCCL：步骤 8 的 all_reduce/all_to_all，64/256/1024 MiB，实测在 `logs/nccl-bandwidth.log`；未执行时带宽为**未完成**，不填估算值。
- 多卡 selector：`python -m pytest -m "accelerator and multi_gpu and not rocm"`，只暴露同组两张卡。
- Ulysses2 冒烟：1.3B、480×832、17 帧、4 步、通用 cat prompt、随机端口，不跑 A14B 冒烟。视频副本：smoke/video-01.mp4。产生视频后，发布副本移除容器及流元数据，文件名采用通用编号。

## 发布与脱敏

只写入 `output/a6000/20261027-140300/`。任务结束或等待到期后自动更新结果，推送前 `git pull --rebase`，不强推。执行检出仍固定 8b6a767，结果不自动验证分支后来新增的源码。

主机名、用户名、本地绝对路径、GPU UUID 均替换为逻辑标记。完整日志未截断调用栈；原始数据只保存在私有工作目录。参见 [privacy-audit.json](privacy-audit.json)、[manifest.sha256](manifest.sha256)。
