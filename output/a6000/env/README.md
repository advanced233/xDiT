# A6000 服务器环境回报（脱敏）

整理日期：2026-10-26（UTC+08:00）。按 `docs/wan22_tp_int8/env_setup.md` 步骤 9 编排。

**结论：环境已安装，但验证并非全通过；A14B 权重未完成；NCCL 带宽未完成。** 旧的“后台下载中”状态已更新：下载进程于 2026-10-25 03:44:40 +08:00 以退出码 1 结束，不再运行。当前不进行 GPU 测试或重启下载。

## 证据时间与源码基线

- 本次先执行了 `git pull --rebase`，整理时同步到 `a0fb2b9232b88def1ed13eebd17a109917353cde`；本提交只添加本目录下的回报与脱敏证据，不修改源码。
- 下述测试和算子检查发生在 2026-10-25，基于 **`256cd1941973d8678b1b21ae033627d163a89984`**，不能视为同步后源码的验证。同步后源码的重新验证：**未完成**，本次是历史证据整理，并未重跑这些 selector。
- 硬件/内存/占用快照采集于 2026-10-26 15:25–15:30 +08:00；下载状态和文件大小在本次重新核对。
- `xfuser` 为 editable 安装：发行包版本仍带旧提交标记，实际源码随 pull 更新，两者不得混淆。

## 机器、内存、CPU 与加载约束

- GPU：**NVIDIA RTX A6000 × 8**，每卡 49140 MiB；驱动 **580.178.04**，驱动支持 CUDA **13.0**，主环境 PyTorch CUDA **12.8**。
- CPU：**2 × Intel Xeon Platinum 8488C，48 核/插槽，合计 96 物理核、192 逻辑 CPU**；2 个 NUMA 节点。
- `nproc --all` = 192；去掉 OMP 限制后 `nproc` = 192；进程 CPU affinity 允许 192 个逻辑 CPU。source 环境后 `nproc` = 8 来自 **OMP_NUM_THREADS=8**，不是机器只有 8 核，也不是已证实的 8 核调度配额。
- `free -g`（GiB，整数显示）：RAM total **251** / used **120** / free **13** / buff/cache **116** / available **71**；Swap total **511** / used **34** / free **477**。原始输出见 [cpu-memory.txt](hardware/cpu-memory.txt)，OMP/affinity 证据见 [cpu-affinity.txt](hardware/cpu-affinity.txt)。这是瞬时共享资源快照。
- TP 加载含义：不能按 251 GiB 总内存假设可用容量；当前 available 约 71 GiB，低于 126.20 GB 的模型文件总量。权重文件字节数不等于加载峰值；dtype 转换、完整 state_dict、分片副本及每个 rank 的重复加载会改变峰值。应采用逐分片读取/低峰值 CPU 切片再上卡的方案，避免各 rank 各加载完整权重副本；实施前重新检查 available。**A14B CPU/TP 加载峰值实测：未完成**（权重不完整，未执行加载）。Swap 不作为满足模型物理内存需求的依据。
- PATH 中未发现 Slurm 的 sbatch/srun/squeue（环境探查记录）；未通过公共环境或 root 安装依赖。

## 工作目录与存储

工作根目录用逻辑标记 `$XDIT_WORK` 表示，真实路径及属主均不发布。该目录属于执行账号。快照显示所在盘可用约 **2.0 TiB**，工作目录已占用约 **176 GiB**（`df -h` / `du -sh` 的舍入显示）；配额查询不可用，**配额上限确认未完成**。[存储输出](hardware/storage.txt)。未把当前磁盘空闲误当成账号独占容量。

## 版本与环境检查

| 项 | 版本/状态 |
|---|---|
| python | 3.10.22 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| diffusers | 0.41.0 |
| transformers | 5.20.2 |
| torchao | 0.17.0+cu128 |
| flash-attn | 未安装 |
| yunchang | 0.6.4 |
| sageattention | 1.0.6 |
| lpips | 0.1.4 |
| xfuser | 0.0.0.dev1+g256cd1941 |
| nvcc | 12.8.93，与 torch CUDA major.minor 一致 |
| Nsight Systems | 2025.3.2.474-253236389321v0，使用系统已有版本 |
| VBench 独立环境 | vbench 0.1.5；torch 2.14.1+cu130；Python 3.10 |

- 主环境 `pip check` 通过；导入记录见 [imports.log](logs/imports.log)，包清单见 [pip-freeze.log](logs/pip-freeze.log)。GPU 被隐藏时 `torch.cuda.is_available()=False` 不表示 CUDA 安装失败；历史 GPU 自检在重新确认空闲的 GPU 3 上执行。
- torchao 自身导入没有 C++ 扩展兼容性告警，`torch._int_mm` 返回 `torch.int32`，基础 `sageattn` 返回预期形状。见 [gpu-operators.log](logs/gpu-operators.log)。这不代表所有后端路径都通过，失败见下表。
- 导入 xfuser/diffusers 有 “Unable to import torchao Tensor objects” 告警；以 torchao Tensor 序列化的检查点兼容性**未完成**验证。
- FlashAttention 安装**未完成（用户明确要求跳过）**。本任务主动停止自己的编译，退出码 137 不是自然编译失败。后续实验显式使用 `--attention_backend SDPA`；cross-attention 默认跟随该设置。PyTorch 自带 SDPA 不依赖单独的 flash-attn 包。
- SageAttention2 **未完成**：Git early EOF，官方归档和镜像均超时；按文档退路安装 SageAttention 1.0.6。[源码下载日志](logs/sage-clone.log)、[官方归档日志](logs/sage-archive-download.log)、[镜像日志](logs/sage-mirror-download.log)。
- VBench 和 decord 导入通过，但 `pip check` 提示 decord 0.6.0 平台标签异常（wheel 内部 cp36 标签）；未改包元数据，未宣称质量评测已经完成。[VBench 检查](logs/vbench-pip-check.log)。

## 模型权重：A14B 是否下载完成

**未完成。** 共 49 个文件，预期 **126,200,628,126 bytes**（126.200628126 GB）。48 个完整 snapshot 文件共 **121,281,162,902 bytes**；还有 **4,261,953,505 bytes** 断点数据，不能作为完整权重使用。

唯一未完成文件：`transformer_2/diffusion_pytorch_model-00011-of-00012.safetensors`。

| 预期完整大小 | 完整 snapshot 大小 | 断点数据大小 | 按长度计算至少还差 |
|---:|---:|---:|---:|
| 4,919,465,224 bytes | 0 bytes | 4,261,953,505 bytes | 657,511,719 bytes |

下载依次尝试 ModelScope、HF 镜像和官方源，退出码分别为 92、18、92（HTTP/2/部分传输错误）；整体任务最终退出码 1。该分片未通过整文件哈希，`refs/main` 未建立，因此 A14B 离线完整加载、TP 分片加载均**未完成**。本次不把“文件部分下载到磁盘”算作完成。

- **各文件精确大小**：[Markdown 清单](download/model-files.md)、[CSV](download/model-files.csv)、[JSON（含预期哈希）](download/model-files.json)。
- [下载最终结果](download/models-result.json)、[结束状态](download/model-download-finished.json)、[逐文件状态](download/model-a14b-file-results.json)、[失败分片完整下载日志](download/failed-shard-full.log)。
- 完整文件的哈希在下载时校验过；本次重新核对存在性和大小，未重新读取全部 126 GB 进行哈希。
- 1.3B：31 个文件、28,935,653,511 bytes，下载校验完成；`HF_HUB_OFFLINE=1` 下 WanPipeline 在 CPU 完整加载通过，耗时约 380.36 秒。仅为加载验证，未进行去噪/视频冒烟。[离线加载记录](logs/models-offline-result.json)。

## 拓扑、NVLink 桥与 P2P

- [nvidia-smi topo -m 原始 stdout](hardware/topology.raw.txt) 保留原始内容及终端格式；[纯文本阅读版](hardware/topology.txt) 只去掉 ANSI 格式。GPU 编号保留，UUID 不发布。
- **未检测到可用 NVLink 桥/链路**：GPU 0–3 内部为 NODE，4–7 内部为 NODE，跨组为 SYS；所有卡对均没有 NV#。以任意两卡（如 GPU 0/1）为例，当前是 NODE/PCIe 路径，不是 NVLink。[nvlink --status](hardware/nvlink-status.txt) 对全部卡报告 all links are inActive。
- 运行时可用 NVLink 结论为“无”；**物理桥是否安装的实物确认未完成**，软件状态不足以确认机箱内是否插接了未激活的桥。
- `nvidia-smi topo -p2p r` 能力矩阵为 OK，见 [P2P 原始能力输出](hardware/p2p-capability.txt)。双卡 CUDA peer 实测**未完成**，能力矩阵不等于实测通信带宽。

## 步骤 8：NCCL all_reduce / all_to_all 带宽

**未完成**。历史验证时仅一张空闲卡；本次快照八张卡均有计算任务占用，不能占用他人的 GPU，也不继续等待。没有运行 NCCL 性能测试，未用估算值代替测量。

| 消息大小 | all_reduce busbw (GB/s) | all_to_all busbw (GB/s) |
|---|---|---|
| 64 MiB | 未完成 | 未完成 |
| 256 MiB | 未完成 | 未完成 |
| 1024 MiB | 未完成 | 未完成 |

历史算子/跳过证据：[gpu-results.json](logs/gpu-results.json)。当前占用仅发布无身份字段的 [GPU 快照](hardware/gpus.txt)。

## 各 selector 测试结果与完整失败日志

以下仅对应旧测试提交，未重跑；skipped/deselected/阻塞不作为通过验证。机器为 RTX A6000，单卡测试仅使用当时确认空闲的 GPU 3。

| selector | passed | failed | skipped | deselected | 阻塞/中断 |
|---|---:|---:|---:|---:|---:|
| unit | 1618 | 0 | 6 | 18 | 0 |
| integration-cpu | 40 | 3 | 5 | 3 | 1 |
| gpu-single | 78 | 5 | 133 | 1850 | 0 |
| gpu-multi | 未完成 | — | — | — | — |

- `unit`：`python -m pytest tests/unit -m "not accelerator"`。[完整日志](tests/pytest-unit.log) / [JUnit](tests/pytest-unit.xml)。
- `integration-cpu`：`python -m pytest tests/integration/cpu -m "not accelerator"`。原执行 10 passed / 3 failed / 3 deselected，另一个用例卡住；随后只执行剩余 35 个未执行用例，得到 30 passed / 5 skipped。合并为 40 passed / 3 failed / 5 skipped / 3 deselected，另 1 个中断，不重复计数。
  - [原始完整失败日志](tests/pytest-integration-cpu.log) / [JUnit](tests/pytest-integration-cpu.xml)。
  - [后续 35 个用例完整日志](tests/pytest-integration-cpu-remainder.log) / [JUnit](tests/pytest-integration-cpu-remainder.xml)。
  - [阻塞记录](tests/cpu-blocked.json)：hybrid-ring4 的 900 秒子进程超时结束后，父进程仍卡在 pipe_read。仅中断本任务 pytest 父进程；未改代码。后续调用增加 660 秒/用例 signal 超时保护，精确命令及未执行节点清单见 [command](tests/cpu-remainder-command.json)、[nodes](tests/cpu-remainder-nodes.json)。原 JUnit 的无名中断占位 testcase 没有计作通过。
- `gpu-single`：`python -m pytest -m "accelerator and not rocm and not multi_gpu"`。[完整失败日志](tests/pytest-gpu-single.log) / [JUnit](tests/pytest-gpu-single.xml)。
- `gpu-multi`：`python -m pytest -m "accelerator and multi_gpu and not rocm"`。**未完成**，没有两张空闲卡；所有计数 N/A，不伪造测试日志。
- [机器可读统计及失败用例名](tests/selector-results.json)。完整日志未截断，调用栈、错误文本和代码相对位置保留，仅做脱敏和终端格式清理。

### 失败用例

CPU 三项均在 `test_usp_sequence_parallel_pad.py::test_usp_matches_attention_over_the_real_tokens`，参数为 `[ring]`、`[hybrid]`、`[ring4]`；错误为 TorchDynamo `InternalTorchDynamoError: AttributeError: 'UserDefinedExceptionObjectVariable' object has no attribute 'call_setattr'`。`[hybrid-ring4]` 为阻塞中断，**未完成**。

单卡 GPU 五项：

1. `test_strided_kv_backends.py::test_strided_kv_gives_what_contiguous_kv_gives[SAGE-pad-dropped-batch1]`
2. `test_strided_kv_backends.py::test_strided_kv_gives_what_contiguous_kv_gives[SAGE-transposed-cross-attention-batch1]`
3. `test_fp8_zero_activation.py::test_an_all_zero_activation_gives_the_bias_rather_than_nan`
4. `test_fp8_zero_activation.py::test_the_floor_is_what_prevents_it`
5. `test_fp8_zero_activation.py::test_ordinary_activations_are_untouched_by_the_floor`

前两项是 `ValueError: not enough values to unpack (expected 2, got 1)`；后三项触发 Float8 动态激活量化的硬件能力门槛断言（错误文本要求 CUDA capability >= 8.9 或 MI300+）。没有为这些失败修改源码或伪造通过结果。

## 冒烟、未完成项与共享服务器决策

- 端到端冒烟 **未完成**：用户明确要求跳过；耗时 N/A，视频输出 N/A。历史列出的 1.3B 单卡、Ulysses2、A14B 都未运行；此处不以 CPU 权重加载代替冒烟成功。
- A14B 权重及完整离线及 TP 加载、NCCL 带宽、双卡测试与 P2P、SageAttention2、FlashAttention、阻塞的 CPU 用例和同步后源码重新验证均按上述原因标记 **未完成**。FA/冒烟是用户主动取消要求；其他项是已记录的阻塞或未执行，不能宣称整体环境全部验证通过。
- 所有安装、临时文件和缓存使用执行账号的工作根目录；不使用 sudo、公共模型副本或 pip --user，不写公共目录，不修改系统 CUDA/驱动或 shell 启动文件。
- 当时无空闲 GPU，按规则每 10 分钟检查一次，最多 60 分钟；最后发现 GPU 3 空闲，执行前重新检查后只使用该卡，没有终止他人的进程。本次只做只读探查与报告发布。
- 运行使用随机高位端口；编译 MAX_JOBS=4，运行限制 OMP_NUM_THREADS=8；用户取消 FA 后只停止本任务自身的编译进程。
- 下载慢时尝试了 BFSU Miniforge、Tsinghua PyPI、SJTU PyTorch wheel、HF 镜像与 ModelScope；模型文件以官方元数据中的哈希为校验基准。A14B 失败如实保留，本次没有重新启动下载。

## 脱敏与证据完整性

主机名、账号名、本地绝对路径、GPU UUID 均已替换为逻辑标记（如 `${HOST}`、`${USER}`、`${REPO}`、`${ENV}`、`${XDIT_WORK}`、`${LOCAL_PATH}`、`${GPU_UUID}`）。不公开标记到真实值的映射。保留 GPU 序号、CPU/NUMA 拓扑、版本、模型仓库相对文件名、测试节点和行号。未包含权重二进制、账号环境脚本或未脱敏的探查记录。

[privacy-audit.json](privacy-audit.json) 记录脱敏规则和检查结果；[evidence-provenance.json](evidence-provenance.json) 记录原日志与发布文本的大小/哈希；[manifest.sha256](manifest.sha256) 校验发布文件。原始日志在本机私有工作目录保留。
