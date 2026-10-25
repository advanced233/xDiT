# Wan2.2-A14B 多卡并行与 INT8 W8A8 推理优化：实现计划

本文档记录在 xDiT 上为 Wan2.2-T2V-A14B 实现张量并行（TP）、CFG 并行和 INT8 W8A8 推理优化的计划内容与实现顺序。

文中性能数字均为按 FLOPs / 字节数的粗估，只用于定预期、排优先级，实测后逐项替换。

- 基线代码：本仓库 `main`，commit `16c34e7`
- 参考项目：[liuzhitu/xDiT_WAN2.1_TP](https://github.com/liuzhitu/xDiT_WAN2.1_TP)：Wan2.1 T2V 的 TP 与 TP×SP 适配，4×RTX 4090，只实测了 1.3B
- 硬件：
  - 2×RTX A6000 48GB：sm_86，BF16 稠密 154.8 TFLOPS，INT8 稠密 309.7 TOPS
  - 1×A100-80G：BF16 稠密 312 TFLOPS，INT8 稠密 624 TOPS
  - 4×A6000：视资源情况，只用于可选项 M10

路径简写：`runner_models/` 指 `xfuser/model_executor/models/runner_models/`，`pipelines/` 指 `xfuser/model_executor/pipelines/`，`transformers/` 指 `xfuser/model_executor/models/transformers/`，`loading/` 指 `runner_models/loading/`。

## 1. 目标与边界

目标：

1. 2×A6000 上实现 BF16 下双专家常驻的 TP 推理；为 Wan2.2 T2V 补齐 CFG 并行；系统对比 2 卡并行策略，包括 FSDP 基线。
2. 实现 INT8 W8A8（权重 per-channel、激活 per-token 动态）与 TP 的协同，并用融合 kernel 让 GEMM 加速接近理论值。
3. 用 nsys 和 Amdahl 模型解释各项优化的收益，以及它们叠加时的关系；评估高噪声、低噪声专家的量化敏感度。

不做：

- 训练；I2V / TI2V（T2V 完成后 I2V 可以顺带验证）。
- FP8 / FP4：Ampere 没有对应的 Tensor Core。
- 非整除分辨率下的 SP padding 与有效长度 K/V 裁剪：上游已经实现，见第 2 节。

## 2. 现状核对

| 能力 | 现状 | 位置 | 结论 |
|---|---|---|---|
| Wan TP | 无。xDiT 已有 TP 进程组和 `--tensor_parallel_degree`，但 Wan 没有声明这项能力 | `xfuser/core/distributed/parallel_state.py:65-75`、`runner_models/wan.py:639` | 本项目主体 |
| Wan2.2 T2V CFG 并行 | 无。T2V 直接使用 diffusers 的 `WanPipeline` | `runner_models/wan.py:639`、`runner_models/wan.py:785` | 需要补齐 |
| Wan2.2 I2V CFG 并行 | 有，包括按 boundary 切换专家和 `guidance_scale_2` | `pipelines/pipeline_wan_i2v.py:82-86,129-133,173-218` | T2V 照此移植 |
| INT8 W8A8 | 有 `--use_int8_gemms`：torchao `Int8DynamicActivationInt8WeightConfig(PerRow, SYMMETRIC)`，目前只有 Z-Image 启用 | `xfuser/core/utils/runner_utils.py:353-394`、`runner_models/z_image.py:133,151` | 单纯打开它不算自己的工作；自己的工作是 TP 集成和融合 kernel |
| FSDP | 有。Wan2.2 T2V 已声明 `fully_shard_degree`；`fs_degree` 不计入并行维度乘积，可以叠加 Ulysses / CFG；与 TP 互斥 | `xfuser/config/config.py:171-184,237-243`、`runner_models/wan.py:791` | **TP 必须对比的基线** |
| SP padding（有效长度 K/V） | 有 | `transformers/transformer_wan.py:364-371`、`tests/integration/accelerator/model_executor/test_wan_sequence_parallel_pad.py` | 不重复实现 |
| SageAttention | 有 `SAGE` 后端 | `xfuser/core/attention/backends/sage/`、`xfuser/core/attention/spec.py:32` | 接入并做消融 |
| CPU offload | 有 `--enable_model_cpu_offload` 等参数 | `xfuser/config/args.py:659-680` | 用作单卡基线 |
| Wan2.2 T2V 默认 guidance | `guidance_scale=3.5`、`guidance_scale_2=None`，即两个专家都用 3.5。官方设置是高噪声 4.0、低噪声 3.0 | `runner_models/wan.py:634-635` | 顺手修正 |

## 3. 关键数字（粗估）

### 3.1 显存

| 项 | BF16 | INT8 W8A8 |
|---|---:|---:|
| 单个专家（约 14.3B 参数，其中 block 内 Linear 约 14.05B） | ~28.6 GB | ~14.6 GB |
| 双专家 | ~57 GB | ~29 GB |
| umt5-xxl 编码器 | ~11.4 GB | — |
| TP2 / FSDP2 下每卡的双专家权重 | ~29 GB | ~15 GB |

- 单张 A6000 在 BF16 下只放得下一个专家。
- BF16 TP2 每卡约 29 GB，再加上 T5 和激活会很紧张。
- INT8 后单卡放双专家（约 29 GB）可行，但 T5 不能同时常驻。

### 3.2 序列长度与 W8A8 / Sage 的加速上限

Wan VAE 在时间维压缩 4 倍、空间维压缩 8 倍，patch 为 1×2×2，所以 L = ((帧数 − 1)/4 + 1) × (H/16) × (W/16)。

下表中"注意力核心"指 QKᵀ 与 PV 两次矩阵乘，Linear 指所有投影层与 FFN。逐元素算子没有计入，所以实际的上限还要更低一些。

| 配置 | L | Linear 占 FLOPs | W8A8 理想上限（GEMM 2×） | W8A8（GEMM 1.75×） | 叠加 Sage（注意力 1.7×，假设值） |
|---|---:|---:|---:|---:|---:|
| 480p，33 帧 | 14040 | 66.7% | 1.50× | 1.40× | 1.73× |
| 480p，49 帧 | 20280 | 58.4% | 1.41× | 1.33× | 1.73× |
| 480p，81 帧 | 32760 | 46.7% | 1.30× | 1.25× | 1.72× |
| 720p，81 帧 | 75600 | 27.7% | 1.16× | 1.13× | 1.71× |

### 3.3 每次 DiT 前向的通信量（每卡发送量，BF16）

| 策略 | 480p 33 帧 | 480p 81 帧 | 720p 81 帧 | 备注 |
|---|---:|---:|---:|---|
| TP2：每 block 3 次输出 AllReduce，每次 L×d | 17.3 GB | 40.3 GB | 92.9 GB | 在关键路径上；另有 4 次统计量 AllReduce，每次 L 个 fp32，受延迟限制 |
| Ulysses2：每 block 4 次 all-to-all | 5.8 GB | 13.4 GB | 31.0 GB | 约为 TP2 的 1/3 |
| FSDP2：all-gather 半个专家 | ~14.3 GB | ~14.3 GB | ~14.3 GB | 与序列长度无关，可以预取并与计算重叠 |
| CFG2 | MB 级 | MB 级 | MB 级 | 每步只 all-gather 一次噪声预测 |

### 3.4 计算时间（A6000 按实效 100 TFLOPS、A100 按实效 200 TFLOPS 估算）

| 配置 | 单次前向 | 单卡单条视频（40 步 × CFG 两次前向）：A100 | 同左：A6000 |
|---|---:|---:|---:|
| 480p，33 帧 | 0.50 PFLOP | ~3.4 min | ~6.7 min |
| 480p，81 帧 | 1.68 PFLOP | ~11 min | ~22 min |
| 720p，81 帧 | 6.52 PFLOP | ~44 min | ~87 min |

按 boundary 0.875、shift 12、40 步计算，高噪声专家约跑 26 步，低噪声专家约 14 步。单卡模型级 offload 时，一次生成只切换专家约 2 次。所以"双专家常驻"本身省下的只是切换耗时，量级是秒。

### 3.5 由此得到的预期

- 2 卡 BF16 下，CFG2 大概率最快，显存问题交给 FSDP2 或 INT8 解决。TP2 的通信在关键路径上，在 PCIe 和 33 帧下约 1.5×。
- INT8 W8A8 在 480p / 33 帧下的上限是 1.5×。现实中要做到约 1.4×，需要 epilogue 融合（见 M6）；再叠加 Sage 可到约 1.7×。
- TP × INT8：INT8 只缩短计算、不缩短通信。在 33 帧下，INT8 在 TP2 上只能再加速约 1.28×，而在单卡上是约 1.4×。

## 4. 实现内容

每个模块按"做什么 / 涉及位置 / 验收 / 测试"组织。测试分层遵循 `.github/skills/testing/SKILL.md`：

- 纯 CPU 的确定性测试放在 `tests/unit/`。
- Gloo 多进程测试放在 `tests/integration/cpu/`，标记 `gloo`。
- 真卡测试放在 `tests/integration/accelerator/`，标记 `nvidia` / `multi_gpu`。
- 完整模型运行放在 `tests/e2e/`。

### M0 测量与评测基建

- **计时**：runner 已有 `--warmup_steps`、`--warmup_calls`、`--num_iterations`、`--profile*`（`xfuser/config/args.py:481,1082-1135`）。需要补充：
  - 用 CUDA event 按专家分别统计单步 DiT 时间。
  - 专家切换耗时单独记录。
  - 报告预热后若干步的中位数。
- **NVTX**：给 GEMM、注意力、通信、逐元素算子、量化算子分别打 range，供 M9 用 nsys 拆分耗时。
- **显存**：记录每个 rank 的 `torch.cuda.max_memory_allocated`，分"权重常驻"和"去噪阶段峰值"两种口径。
- **环境记录**：
  - 版本：GPU 型号、驱动、CUDA、torch、diffusers、torchao、flash-attn、sageattention。
  - 拓扑：`nvidia-smi topo -m` 的输出，以及是否有 NVLink。
- **画质评测脚本**：
  - 同 seed 对照：LPIPS、PSNR、SSIM，衡量偏离度。
  - VBench 自定义 prompt 模式的子集：subject_consistency、background_consistency、motion_smoothness、dynamic_degree、aesthetic_quality、imaging_quality，衡量画质。
- **产出**：`benchmarks/wan22/` 下的脚本和结果表模板。

> **状态：阶段 1 需要的部分已实现**，用法见 [`phase1_runbook.md`](phase1_runbook.md)。
> - `benchmarks/wan22/bench.py`：加载一次跑多组配置；按专家、按步统计 DiT 时间（`xfuser/core/utils/measurement.py` 的 `ComponentCallTimer`）；记录三种口径的显存和前后两次 `nvidia-smi` 快照（同卡有其他进程的运行会被标记）；输出 `env.json`、`runs.jsonl`、`summary.csv`；`--nsys-capture` 只采集计时的运行。
> - `attention_bench.py`：注意力后端微基准。`compare_videos.py`：LPIPS / PSNR / SSIM。
> - 尚未覆盖：NVTX 目前只到"每个专家的每次调用"这一级，GEMM、注意力、通信的拆分先靠 nsys 的 kernel 名归类（M9 再补）；专家切换耗时只在 offload 基线中才有意义，留到 M8；拓扑由 `env_setup.md` 的自检记录。

### M1 Wan2.2 T2V CFG 并行

- 仿照 `pipelines/pipeline_wan_i2v.py` 新增 T2V 管线，继承 diffusers 的 `WanPipeline`：
  - CFG rank 0 计算 uncond，rank 1 计算 cond；每步用 `get_cfg_group().all_gather` 收集两个噪声预测后再做合成。
  - 按 `boundary_ratio × num_train_timesteps` 切换 `transformer` / `transformer_2`，以及各自的 guidance scale。
  - 每个 rank 只编码自己需要的那条 prompt（正向或 negative），编码完即可释放 T5。
- `xFuserWan22T2VModel` 声明 `use_cfg_parallel=True` 并改用新管线。`_calculate_hybrid_attention_step_multiplier` 已经处理了 CFG 并行的情况（`runner_models/wan.py:602-606`）。
- 修正 Wan2.2 T2V 的默认值：`guidance_scale=4.0`、`guidance_scale_2=3.0`。（已随 M2 完成。）
- 显存：BF16 下每卡需要放双专家（57 GB），必须配合 FSDP2（`--fully_shard_degree 2`）、模型级 offload，或 M2 的 INT8。
- 验收：
  - 同 seed 下 CFG2 与单卡输出在 BF16 舍入误差内一致，因为两者只差一次 all_gather。
  - DiT 加速接近 2×。
- 测试：在 `tests/integration/cpu/` 用随机初始化的小 Wan 配置跑 Gloo 2 进程，验证 CFG2 与单进程的噪声预测一致、专家切换的步数一致。

### M2 单卡 INT8 接入与专家敏感度（A100）

- 为 `xFuserWan22T2VModel` 声明 `use_int8_gemms=True`，并设置 `int8_gemm_module_list=["transformer.blocks", "transformer_2.blocks"]`。
- 支持按专家选择量化范围：none / high-only / low-only / both。
- 对比 eager 与 `--use_torch_compile` 两种模式，区分 epilogue 是否被融合。
- 序列长度扫描：480p 下 33 / 49 / 81 帧，再加 720p 81 帧一组，用实测值对照第 3.2 节的预测。
- 敏感度实验：
  - 高噪声专家约跑 26/40 步，所以只量化它就能拿到约 65% 的加速收益。
  - 预期高噪声专家决定构图和运动，量化后同 seed 的 LPIPS 可能明显升高，但 VBench 不一定下降；低噪声专家主要影响纹理细节。两类指标需要一起解读。
- 可选：逐 block 敏感度分析，得出混合精度策略。可以参考 `runner_models/wan.py:664` 中 FP4 量化时保留首尾 10 个 block 用更高精度的做法。
- 产出：显存、DiT 加速、画质三张表，以及专家敏感度结论。这里得到的是 torchao 基线，供 M6 对比。

> **状态：代码已实现，等 A100 实测。**
> - Wan2.1 T2V（14B / 1.3B）和 Wan2.2 T2V 都声明了 INT8；Wan2.2 的两个专家都是 INT8 目标。
> - 新参数 `--int8_gemm_modules` 可以只选其中一个专家。它要求 `--gemm_quantization int8`，模块名必须和声明的目标完全一致，否则直接报错。实现在 `loading/quantization_plan.py` 的 `apply_int8_module_selection`。
> - 测试在 `tests/unit/model_executor/loading/test_int8_plan.py` 和 `tests/unit/config/test_gemm.py`。
> - 720p 81 帧那一组放在 33 / 49 / 81 帧的结果出来之后再定。

### M3 Wan2.2 双专家 TP

分片布局（P 为 TP 度，heads 由 40 变为 40/P，两个专家用同一套逻辑）：

| 模块 | 切分 | 说明 |
|---|---|---|
| `attn1.to_q/to_k/to_v` | Column：输出维按 head 切 | 关闭 `fused_projections` 路径 |
| `attn1.norm_q/norm_k` | 跨 rank 归约统计量 | Wan 的 QK RMSNorm 作用在完整的 5120 维上（across heads）。做法：本地算 fp32 平方和 → AllReduce → 按全维归一化；affine weight 按通道切分 |
| `attn1.to_out.0` | Row | bias 只在 rank 0 上加一次，之后 AllReduce 求和 |
| `attn2.to_q/to_k/to_v` | Column | K/V 作用在 512 个文本 token 上，保留完整的文本序列 |
| `attn2.norm_q/norm_k` | 同 `attn1` | |
| `attn2.to_out.0` | Row | |
| `ffn.net.0.proj` | Column | GELU（tanh 近似）是逐元素操作，不需要通信 |
| `ffn.net.2` | Row | |
| patch embedding、condition embedder、各 norm、`scale_shift_table`、`proj_out` | 复制 | |

每个 block 的通信为 3 次输出 AllReduce 和 4 次统计量 AllReduce；M4 把后者合并为 2 次。

与参考项目不同、需要重点处理的集成点：

1. **分片时机**：
   - 当前加载顺序：
     1. `_load_model` 填充权重。
     2. `place_pipeline_components`（`loading/meta_load.py:551`）。
     3. `pipe.to(cuda)`（`loading/placement.py:83`）。
     4. INT8 转换（`loading/placement.py:104-105`）。
     5. `initialize_runtime_state`。
     6. `_post_load_and_state_initialization`（`runner_models/base_model.py:478`）。
   - 参考项目在最后一步分片，前提是完整模型已经先放上 GPU。A14B 双专家 BF16 约 57 GB，超过 48 GB，会在加载阶段 OOM。
   - 因此分片必须在 host 侧、`pipe.to(cuda)` 之前完成，或者逐 block 流式分片后再上卡。这样 INT8 转换自然作用在本地分片上，"先分片再量化"由加载顺序直接保证。
   - 需要确认 TP 进程组在分片时已经可用。进程组由 `initialize_model_parallel` 创建（`xfuser/core/distributed/runtime_state.py:153-168`）；FSDP 路径在加载阶段就用到了 `get_fs_group()`（`loading/shard.py`），可以参照它的初始化时机。
2. **本地 heads**：`xFuserWanAttnProcessor` 用 `attn.heads` 做 unflatten（`transformers/transformer_wan.py:115-117,171-173`），TP 下要改成本地 heads 数。
3. **融合的 QK-norm + RoPE**：FlyDSL 融合路径（`transformers/transformer_wan.py:105-117`）只在本地通道上归一化，TP 下必须绕开或改造，统一走带跨 rank 归约的实现。
4. **能力声明与约束**：
   - Wan2.2 T2V 声明 `tensor_parallel_degree=True`。
   - TP 与 FSDP 互斥（`xfuser/config/config.py:181`）。
   - `_splits_weights` 已经把 TP 计入（`loading/contracts.py:276-281`），会跳过 replicated 广播加载。
5. **输出收集**：只由 TP rank 0 提交视频，避免重复输出。

验收：

- TP2 与单卡比较首步 DiT 输出：fp32 下相对误差约 1e-5；bf16 下设定明确的容差，例如相对 L2 < 1e-2。最终视频用 LPIPS / PSNR 对照。
- 2×A6000 BF16 下双专家常驻，记录每卡峰值显存。

测试：

- `tests/integration/cpu/`：用随机初始化的小 Wan 配置（例如 2 层、4 heads）跑 Gloo 2 进程，fp32 下对比 TP2 与单进程的前向输出。覆盖 self-attention、cross-attention、FFN、QK-norm 归约和 Row bias。
- `tests/integration/accelerator/`：BF16 版本，标记 `nvidia`、`multi_gpu`。
- `tests/e2e/`：在 2×A6000 上用 480p / 33 帧完整跑一次，记录命令和结果。

### M4 TP 通信优化

- **合并统计量归约**：Q 和 K 的平方和拼接成一个张量后做一次 AllReduce，每个 block 的 AllReduce 次数从 7 降到 5。cross-attention 的 Q 是 [L]、K 是 [512]，同样可以拼接。
- **计算与通信重叠**：Row 并行层按序列分成 2～4 块，前一块的 AllReduce 异步发出，同时计算下一块的 GEMM。输出投影和 FFN 第二层是主要目标。
- 可选：Megatron 式序列并行（reduce-scatter + all-gather 代替 AllReduce），让 norm 和调制按序列切分。通信量不变，但能降低激活显存和重复计算。
- 验收：
  - nsys 显示暴露在关键路径上的通信时间下降。
  - 输出与 M3 一致：合并归约应逐元素一致；分块重叠后误差在 BF16 舍入范围内。
- 测试：在 M3 的 Gloo 测试上增加优化开关，断言开关前后输出一致。

### M5 INT8 × TP

- 依赖 M3 的分片时机：INT8 转换作用在本地分片的 `nn.Linear` 上。
- 数值语义：
  - Column 并行：先切再量化，与先量化再切等价，因为 per-channel scale 跟随输出通道。
  - Row 并行：per-token scale 在本地那一段输入维上计算，粒度比单卡 INT8 更细，所以数值与单卡 INT8 不同。各 rank 先在本地反量化，再做 BF16 AllReduce。
- 验收：
  - TP2 + INT8 能运行，每卡显存约 15 GB 加激活。
  - 画质与单卡 INT8、BF16 分别做对照。
  - 用 nsys 测出 INT8 在 TP2 上的额外加速，与第 3.5 节的约 1.28× 对照。
- 测试：在 Gloo 测试中加入 INT8 路径。如果 CPU 上 torchao int8 不可用，就放进 accelerator 测试。

### M6 融合 INT8 kernel（Triton）

- **W8A8 GEMM**：int8 × int8 → int32 累加，epilogue 融合 `acc × s_token[:, None] × s_channel[None, :] + bias`；`ffn.net.0` 再额外融合 GELU（tanh 近似）。这样避免 torchao `_int_mm` 路径先写出 int32 结果的额外读写：33 帧下一个 5120×5120 的 GEMM 约多出 288 MB 读写，720p 下 FFN 第一层的 int32 中间结果约 4.2 GB。
- **调制 + 量化融合**：把 adaLN 的 `norm → (1 + scale) × x + shift` 与 per-token absmax 量化合成一个 kernel，直接输出 int8 激活和 scale。
- **基准形状**：
  - M 取 14040 和 32760。
  - (K, N) 取 (5120, 5120)、(5120, 13824)、(13824, 5120)。
  - 另加上 TP2 下 N 或 K 减半后的形状。
- 对比对象：torchao eager、torchao + `torch.compile`、cuBLAS BF16。
- 验收：
  - kernel 输出与 fp32 参考的误差在容差内。
  - microbench 中 GEMM 层面的加速 ≥ 1.7×。
  - 端到端加速相对 M2 的 torchao 路径有提升。
- 测试：kernel 正确性测试放在 `tests/unit/`（标记 `accelerator`）或 `tests/integration/accelerator/`；microbench 只作为基准，不当作测试。

### M7 SageAttention 接入与消融

- 使用 `--attention_backend SAGE`。验证它在 TP 本地 heads 和 Ulysses2 下都能正常工作。
- 确认 sm_86 上走的是 INT8 QK + FP16 PV 路径（FP8 PV 需要 sm_89 以上），实测 kernel 层面的加速，替换第 3.2 节中假设的 1.7×。
- 做 2×2 消融（W8A8 开 / 关 × Sage 开 / 关），分别测速度和画质。

### M8 2 卡并行策略对比（2×A6000）

| 配置 | BF16 | INT8 |
|---|---|---|
| 单卡 + 模型级 offload | 基线 | 无需 offload |
| FSDP2 + Ulysses2 | ✓ | ✓（如果 xDiT 的 FSDP 量化路径支持） |
| FSDP2 + CFG2（依赖 M1） | ✓ | ✓（同上） |
| TP2（M3/M4） | ✓ | ✓（M5） |
| Ulysses2 | 每卡单专家 + 切换 | 双专家常驻 |
| Ring2 | 每卡单专家 + 切换 | 双专家常驻 |
| CFG2（M1） | 每卡单专家 + 切换 | 双专家常驻 |

- 指标：
  - 按专家分别统计的单步 DiT 时间。
  - 端到端时间。
  - 每卡峰值显存。
  - 通信量。
  - 专家切换耗时。
- 配置：480p / 33 帧为主表，另做 480p / 81 帧，720p / 81 帧只跑一组作为 headline。
- 报告中写明硬件拓扑（NVLink 或 PCIe，P2P 是否可用）。
- 预期结论要如实写，即使 TP2 不是最快的配置。TP 的适用场景包括：去掉 CFG 的蒸馏模型（例如 `Wan2.2-Distilled-I2V` 声明了 `use_cfg_parallel=False`，见 `runner_models/wan.py:458`），以及 4 卡以上需要和 CFG / SP 组合的情况。

### M9 性能分析

- 用 nsys 和 M0 的 NVTX range，拆分 GEMM、注意力、通信、逐元素算子和量化开销的时间占比。
- 用实测的时间占比（而不是 FLOPs 占比）代入 Amdahl 模型，得出预测加速比，与实测值对照。
- 解释 TP × INT8 加速比低于两者乘积的原因：INT8 只缩短计算，TP 通信不变；序列越短，计算越以 Linear 为主，通信占比越高。
- 汇总高噪声、低噪声专家的量化敏感度：分别对比偏离度（LPIPS）和画质（VBench）。

### M10（可选，4×A6000）混合并行

- TP2×CFG2、TP2×Ulysses2。
- 在 TP×SP 下验证非整除序列长度时的 padding 与 K/V 裁剪。只有在这种组合下，TP 与 padding 的交互才需要额外处理。

## 5. 实现顺序

```text
阶段 0  M0 测量基建                  任意 GPU
   │
阶段 1  M2 单卡 INT8 + 专家敏感度     A100        ← 无分布式复杂度，最快出数据
        M7 Sage 快速消融（单卡）       A100
   │
阶段 2  M1 Wan2.2 T2V CFG 并行        2×A6000     ← 改动小，解锁 FSDP2+CFG2 / INT8+CFG2
        M8 基线部分：单卡 offload、FSDP2+Ulysses2、FSDP2+CFG2、Ulysses2、Ring2、CFG2
   │
阶段 3  M3 双专家 TP                  2×A6000     ← 工作量最大
        M4 TP 通信优化
   │
阶段 4  M5 INT8 × TP                  2×A6000
        M8 完整对比表
   │
阶段 5  M6 融合 INT8 kernel            A100        ← 可与阶段 3 并行；视进度决定是否纳入
   │
阶段 6  M9 性能分析与总结；M10（如果能拿到 4 卡）
```

排序理由：

1. **先做单卡 INT8（阶段 1）**：不涉及分布式，A100 可以单独使用，很快就能得到"上限预测 vs 实测"以及画质数据，同时确定 33 帧主配置是否可行。
2. **CFG 并行先于 TP（阶段 2）**：改动小，有现成的 I2V 实现可以参照；它同时是 TP 最强的对手。先拿到基线，TP 的结果才有参照。
3. **TP 放在阶段 3**：工作量最大，涉及加载流程。先在 CPU Gloo 测试里把数值对齐，再上真卡。
4. **INT8 × TP 依赖 TP 的分片时机（阶段 4）**：M3 把分片放在上卡之前后，M5 的主要工作就是验证和测量。
5. **融合 kernel 是加分项（阶段 5）**：它独立于分布式，可以在阶段 3 期间用 A100 并行推进；时间不够可以砍掉，只保留 torchao 基线。

每个阶段的完成标准就是其中各模块的"验收"条目，阶段之间的交接物是对应的结果表。

## 6. 实验协议

- **固定参数**：
  - 采样：40 步、guidance 4.0 / 3.0、flow_shift 12、Wan 默认 scheduler。
  - seed：固定一组 seed。
  - 注意力：在 Ampere 上，xDiT 没指定后端时默认用 `CUDNN`（`xfuser/core/distributed/runtime_state.py:96-103`）。M0 先在 Wan 的形状上（B=1、H=40 或 TP2 下的 20、D=128、L=14040 / 32760、bf16）对比 `CUDNN`、`SDPA_FLASH`，以及装了 flash-attn 时的 `FLASH`，选最快的 dense 后端作为基线，此后所有实验固定使用。注意力越快，Linear 的时间占比越高，INT8 的加速比也会越大，所以报告里必须写明基线用的是哪个后端。
- **Prompt**：
  - 速度测试固定 1～3 条。
  - 画质测试用 N ≥ 50 条，在 480p / 33 帧下进行。A100 BF16 单条约 3～4 分钟，BF16 + INT8 两组约 6～7 小时。
- **计时**：先预热一次完整生成；正式计时报告单步 DiT 时间的中位数（按专家分开）和端到端时间。生成流水线时间不包含模型加载和视频编码落盘。
- **显存**：分"权重常驻"和"去噪峰值"两个口径，每个 rank 分别报告。
- **画质**：
  - 偏离度：同 seed 下相对 BF16 的 LPIPS / PSNR / SSIM。
  - 画质：VBench 子集各维度的均值与置信区间，写明"N 个 prompt × M 个维度"。
  - 两类指标不能混用：同 seed 偏离大不等于画质差。
- **记录**：每组结果都附带环境、命令、拓扑和样本数。

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| TP2 在 2 卡上慢于 FSDP2 + CFG2 | 如实报告；故事线改为"TP 用通信换显存，实测给出各策略的适用区间"，同时突出 M4 的通信优化和 TP 在去 CFG 模型、4 卡上的价值 |
| 加载阶段 OOM（完整模型先上卡） | 在 host 侧分片或逐 block 流式分片（M3 第 1 点） |
| torchao INT8 未融合导致 GEMM 加速不足 | 使用 `torch.compile`，或者 M6 的 Triton 融合 kernel |
| 融合 QK-norm + RoPE 路径与 TP 不兼容 | TP 下强制走带归约的实现（M3 第 3 点） |
| PCIe 无 P2P，通信代价变大 | 先跑 `nvidia-smi topo -m` 确认；在报告中写明拓扑 |
| xDiT FSDP 与 INT8 的组合不可用 | 先在 M8 中验证；不可用就把它从 INT8 列中去掉并注明原因 |
| sm_86 上 Sage 版本或功能受限 | 先在 M7 确认可用的 kernel 路径 |
| T5 与 INT8 双专家同时常驻时显存不足 | 先编码再释放 T5，或把 T5 放在 CPU；CFG2 下每卡只编码自己的那条 prompt |
| 画质指标噪声大 | 固定 seed、保证足够的 prompt 数、报告置信区间 |

## 8. 简历要点模板（实测后填数）

> **Wan2.2-A14B 多卡并行与 INT8 W8A8 推理优化（基于 xDiT）**
>
> - **张量并行**：为 Wan2.2 双专家 DiT 实现 TP（Column/Row 分片、跨 rank 归约全维 QK-RMSNorm 统计量），在 host 侧分片以避免加载阶段 OOM；合并 Q/K 统计归约（每 block AllReduce 7→5 次），并分块重叠 Row GEMM 与通信。2×A6000 BF16 下每卡 x GB，双专家常驻。为 Wan2.2 T2V 补齐 CFG 并行。
> - **并行策略对比**：在 2×A6000（注明 NVLink/PCIe）上对比单卡 offload、FSDP2+Ulysses2/CFG2、TP2、Ulysses2、Ring2、CFG2 的单步延迟、显存和通信量。TP2 相对单卡基线 DiT 加速 x×，最优组合 x×。
> - **INT8 W8A8**：权重 per-channel、激活 per-token 动态量化，TP 下先分片再量化；自写 Triton GEMM，融合反量化、bias 和 GELU，相对 torchao 路径再快 x%。INT8 后单张 A6000 无需 offload 即可让双专家常驻。A100 上 480p/33 帧 DiT 加速 x×，叠加 SageAttention 后 x×；N 个 prompt 同 seed 对照 LPIPS 为 x，VBench（M 个维度）BF16 x vs INT8 y。
> - **性能分析**：用 nsys 拆分 GEMM、注意力、通信的耗时占比，用实测时间占比代入 Amdahl 模型，解释加速比随序列长度的变化，以及 TP×INT8 叠加加速低于两者乘积的原因；对比高噪声、低噪声专家分别量化时的偏离度与画质。
