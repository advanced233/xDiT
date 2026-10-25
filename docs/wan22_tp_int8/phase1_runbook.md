# 阶段 1 实验手册：A100 单卡 BF16 / INT8 与专家敏感度

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M0、M2。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节。

这是写进简历的项目，实验只做到能支撑结论为止：
- 一共 4 个进程，每个加载一次模型。
- 不单独做冒烟，也不做预热：第一组正式运行就是冒烟，出错时几分钟内就会报出来。
- 合计约 3.5 小时。

用到的代码：
- `benchmarks/wan22/bench.py`：加载一次模型，循环跑配置里的各个 case；按专家分别统计每步 DiT 时间、显存和 GPU 状态。
- `benchmarks/wan22/configs/phase1.yaml`：两个 case。
  - `480p_33f`：10 个 prompt，完整 40 步，保存视频。
  - `480p_81f`：1 个 prompt，10 步，只用来测速。
- `benchmarks/wan22/compare_videos.py`：与同 seed 的参考视频对比 LPIPS / PSNR / SSIM。
- `--int8_gemm_modules`：只量化 Wan2.2 的某一个专家。`transformer.blocks` 是高噪声专家，`transformer_2.blocks` 是低噪声专家。

## 0. 每次开始前

```bash
source $XDIT_WORK/env-cu126.sh
cd $XDIT_WORK/src/xDiT && git pull && git log --oneline -1
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=1     # 只用空闲的 80G 卡；GPU 2 有他人任务时不要用
R=$XDIT_WORK/results/phase1       # 每组结果一个子目录；bench.py 拒绝写入已有结果的目录
C=benchmarks/wan22/configs/phase1.yaml
```

GPU 任务继续通过你自己的 `scripts/run_gpu_job.py`（或等效方式）执行。`bench.py` 会在每次计时前后记录 `nvidia-smi`。同一张卡上出现别的进程时，那次运行会被标记（`runs.jsonl` 里的 `foreign_processes`），只需重跑被标记的那个 case（`--case` 指定）。

## 1. 生成（4 个进程，约 3 小时）

| 进程 | 命令 | 内容 |
|---|---|---|
| BF16 | `python benchmarks/wan22/bench.py --config $C --output-dir $R/bf16` | 33 帧 10 个视频 + 81 帧测速 |
| INT8，两个专家 | `... --output-dir $R/int8_both --runner-arg gemm_quantization=int8` | 同上 |
| INT8，只高噪声 | `... --output-dir $R/int8_high --case 480p_33f --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks` | 只跑 33 帧 10 个视频 |
| INT8，只低噪声 | `... --output-dir $R/int8_low --case 480p_33f --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer_2.blocks` | 只跑 33 帧 10 个视频 |

（表中的 `...` 即 `python benchmarks/wan22/bench.py --config $C`。）

## 2. 读数

**速度与显存**：读 BF16 和 `int8_both` 两组的 `summary.csv`。
- `transformer_step_ms_median`、`transformer_2_step_ms_median`：两个专家的单步 DiT 时间（含 CFG 两次前向）。33 帧取 10 个视频的中位数，81 帧取 10 步的中位数。INT8 与 BF16 之比就是单步加速比，可以对照 `plan.md` 第 3.2 节的上限（33 帧 1.50×，81 帧 1.30×）。
- `pipeline_s_median`：33 帧 40 步的端到端时间，可以直接用。
- `dit_peak_gib_max`、`run_peak_gib_max`，以及 `env.json` 里的 `load.resident_gib`：显存对比。
- 只量化一个专家时的速度不用单独测：那个专家取 INT8 组的单步时间，另一个取 BF16 组的单步时间，再按 `transformer_steps` / `transformer_2_steps`（40 步时预计约 26 / 14）加权。

**偏离度（LPIPS / PSNR / SSIM，相对 BF16）：**

```bash
V=videos/480p_33f
for c in int8_both int8_high int8_low; do
  python benchmarks/wan22/compare_videos.py --reference $R/bf16/$V --candidate $R/$c/$V \
      --json $R/compare_${c}_vs_bf16.json
done
```

LPIPS 第一次运行会下载 AlexNet 权重，下载位置在 `$XDG_CACHE_HOME`、`$TORCH_HOME` 下。

**画质（VBench，在 `envs/vbench` 环境里运行，4 组 × 6 个维度，约半小时）：**

```bash
# 参数名以所装 VBench 版本的 `vbench evaluate --help` 为准
for c in bf16 int8_both int8_high int8_low; do
  for d in subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality; do
    vbench evaluate --videos_path $R/$c/$V --dimension $d --mode custom_input --output_path $R/vbench_$c/
  done
done
```

解读时要注意：量化高噪声专家可能会改变构图，导致 LPIPS 升高，但 VBench 不一定下降。两类指标要一起看。

## 3. 按需补充（结果出现意外时才做）

| 现象 | 补做 | 耗时 |
|---|---|---|
| INT8 加速明显低于上限 | BF16 和 `int8_both` 两组都加 `--runner-arg use_torch_compile=true` 重跑，看反量化有没有被融合 | 约 1.5 小时 |
| INT8 的 LPIPS 很小，分不清是量化误差还是编码误差 | BF16 加 `--case 480p_33f --repeats 2` 重跑，测噪声底 | 约 1 小时 |
| 想换注意力基线 | `python benchmarks/wan22/attention_bench.py --json $R/attention.json` | 约 5 分钟 |

## 4. 回报

把下面这些文件打包发回（都在 `$R` 下，不进仓库，视频不用发）：
- 每组结果目录的 `env.json`、`summary.csv`、`runs.jsonl`。
- `compare_*_vs_bf16.json`，以及 VBench 的输出目录。
- 被标记为"同卡有其他进程"后重跑的次数；任何失败的完整报错。

不要为了让实验跑通而修改仓库代码。遇到问题时记下现象和报错，发回来由我处理。
