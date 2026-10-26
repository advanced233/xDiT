# A6000 环境更新与自动 GPU 验证

运行编号：20261026-161758。更新：2026-10-27T05:05:55.469296+08:00。

阶段：**FINISHED**。执行源码固定为 **03e8d05118e964e49ad8382de05015340f95e0f5**；初始运行前已 pull 并确认 HEAD 为用户指定的 03e8d05。使用独立的只读源码检出执行测试，后续发布仓库的 pull/rebase 不改变测试基线。不修改源码，不改其他设备或早期结果目录。

## 第一部分：无需 GPU

| 版本项 | 实际值 |
|---|---|
| torch | 2.14.1+cu126 |
| cuda | 12.6 |
| torchao | 0.18.0+cu126 |
| diffusers | 0.41.0 |
| transformers | 5.20.2 |
| torchvision | 0.29.1+cu126 |
| sageattention | NOT_INSTALLED |

四个指定版本及 SageAttention 卸载校验：**通过**。详见 [versions.json](versions.json) 和 [版本检查日志](logs/version-check.log)。保留 diffusers 0.41.0 / transformers 5.20.2，移除 SageAttention，FlashAttention 保持未安装。editable 安装首次因镜像构建依赖索引失败，已改用官方源重试成功；最终 selector 在重试成功后执行，setup-interrupted 前缀的早期记录不计入最终统计。主环境使用 cu126；本轮未重装原有 CUDA 12.8 nvcc，GPU 上的 Triton/torch.compile 实际可用性由后台 preflight 检验，不宣称外部 nvcc 扩展编译版本已经对齐。

A14B 缺失分片修复：**PASS**。从原有 4261953505 bytes 断点继续，curl 固定使用 `--http1.1 -fL -C -` 循环续传；不丢弃已有有效断点。

- 分片：`transformer_2/diffusion_pytorch_model-00011-of-00012.safetensors`
- 要求 SHA256：`11fdb612135d52ac770304261f5e1a846882b72a5e81be15970c051edf20b518`
- 实际 SHA256：`11fdb612135d52ac770304261f5e1a846882b72a5e81be15970c051edf20b518`
- `blobs`、`snapshots`、`refs/main` 已按 HF 缓存格式恢复：是。
- `HF_HUB_OFFLINE=1` 与 `snapshot_download(..., local_files_only=True)`：通过；同时逐一检查 49 个文件存在且大小正确，共 126200628126 bytes。
- 这只是缓存完整性与离线定位检查，不是 A14B 完整模型加载或推理验证。

证据：[model-repair.json](model-repair.json)、[逐文件大小](model-files.json)、[curl 完整日志](logs/model-curl.log)。

## selector、基准与 GPU 使用记录

计数列依次为 passed / failed / skipped / deselected；跳过和未选择不算通过。完整命令、耗时、返回码、失败节点及所用物理 GPU 序号见 [workflow.json](workflow.json)，完整日志在 `logs/`，JUnit 在 `tests/`。

| 任务 | 状态 | 物理 GPU 序号 | 随机端口 | 测试计数 |
|---|---|---|---|---|
| pytest-unit | 已执行 | — | 52838 | 1640 / 0 / 6 / 18 |
| pytest-integration-cpu | 已执行 | — | 22009 | 44 / 0 / 5 / 3 |
| nccl-bandwidth | 未完成 | — | — | — / — / — / — |
| pytest-gpu-multi | 未完成 | — | — | — / — / — / — |
| smoke-ulysses2 | 未完成 | — | — | — / — / — / — |
| pytest-gpu-single | 未完成 | — | — | — / — / — / — |
| attention | 未完成 | — | — | — / — / — / — |
| int8_linear | 未完成 | — | — | — / — / — / — |

失败或部分失败任务：截至本次发布无已记录失败任务。测试失败仅记录，未修代码。

单卡任务为 GPU selector、原样运行 `attention_bench.py --json attention.json`、`int8_linear_bench.py --tokens 14040 32760 --json int8_linear.json`。有结果时 JSON 在本目录根部；基准中 SAGE/FLASH unavailable 与未安装状态一致，不算有该后端实测数据。

双卡任务为 NCCL all_reduce/all_to_all（64/256/1024 MiB）、multi_gpu selector、1.3B Ulysses2 冒烟（480×832、17 帧、4 步、固定通用 prompt）。NCCL 实测见生成后的 `logs/nccl-bandwidth.log`，没有运行时带宽为**未完成**，不填估计值。不运行 A14B 冒烟。

视频副本：未完成；尚无成功冒烟视频。若产生视频，提交副本去除容器/流元数据，文件名使用通用编号。

## 自动探查约定

- 等待起点：2026-10-26T17:05:53.744350+08:00；截止：2026-10-27T05:05:53.744375+08:00。
- 每 15 分钟检查一次，最多 12 小时；执行前再次检查。最近探查：2026-10-27T04:52:23.184544+08:00；符合条件的卡：[]。
- “空卡”严格定义为显存 **<1,000,000,000 bytes** 且 **无任何进程，包括图形及 Xorg**。有常驻 Xorg 的卡不会因为没有计算进程而被当成空卡。探查证据：[gpu-probes.jsonl](gpu-probes.jsonl)。
- 单卡任意选择；双卡只选择同一 NUMA 组（0–3 或 4–7）。每项任务结束即退出其进程并释放资源；等待期间不占卡，不使用占卡程序。
- 运行中发现其他进程加入选中卡时，只停止本任务自身的进程组，绝不终止他人的进程。查询失败时不启动任务。
- 每次命令分配随机高位端口；torchrun/xdit 显式传 master_port。保持 MAX_JOBS=4、OMP_NUM_THREADS=8；CPU、GPU 缓存和临时文件均在本账号工作目录中。
- 已探查无活动 Slurm；若等待期间出现活动调度器，停止直接使用卡并记录未完成，不绕过调度器。
- 到期未取得所需空卡的项目标为“未完成”。GPU 任务失败仅记录，不修改源码。独立检出被修改时拒绝继续执行。

## 发布与脱敏

只发布 `output/a6000/20261026-161758/`。每次提交前后同步主发布检出，推送前执行 `git pull --rebase`，不强推。执行检出仍固定 03e8d05，因此这里的结果不自动验证分支随后新增的源码。

主机名、用户名、本地绝对路径、GPU UUID 均已替换为逻辑标记；公开探查只保留进程数量/类型，不发布其他用户的进程名。完整失败日志保留，未截断调用栈。原始日志仅留在私有工作目录。[manifest.sha256](manifest.sha256) 为发布文件校验表；[privacy-audit.json](privacy-audit.json) 记录检查。
