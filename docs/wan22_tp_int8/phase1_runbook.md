# 阶段 1 实验手册：A100 单卡 BF16 / INT8 与专家敏感度

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M0、M2。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节。

这是写进简历的项目，实验只做到能支撑结论为止：速度 2 组、画质 4 组 × 10 个 prompt，合计约 5 小时。

用到的代码：
- `benchmarks/wan22/bench.py`：加载一次模型，循环跑多组配置；按专家分别统计每步 DiT 时间、显存和 GPU 状态。
- `benchmarks/wan22/compare_videos.py`：与同 seed 的参考视频对比 LPIPS / PSNR / SSIM。
- `--int8_gemm_modules`：只量化 Wan2.2 的某一个专家。`transformer.blocks` 是高噪声专家，`transformer_2.blocks` 是低噪声专家。
- Wan2.2 T2V 的默认 guidance 已改为官方值：高噪声专家 4.0，低噪声专家 3.0。

## 0. 每次开始前

```bash
source $XDIT_WORK/env-cu126.sh
cd $XDIT_WORK/src/xDiT && git pull && git log --oneline -1
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=1     # 只用空闲的 80G 卡；GPU 2 有他人任务时不要用
R=$XDIT_WORK/results/phase1       # 每组结果一个子目录；bench.py 拒绝写入已有结果的目录
```

GPU 任务继续通过你自己的 `scripts/run_gpu_job.py`（或等效方式）执行。`bench.py` 会在每次计时前后记录 `nvidia-smi`：同一张卡上出现别的进程时，运行会被标记（`runs.jsonl` 里的 `foreign_processes`）。被标记的那组重跑一次即可。

## 1. 速度与显存（约 1 小时）

`configs/speed_sweep.yaml`：480p，33 帧和 81 帧，每个 case 跑 10 步，1 次预热加 1 次计时，不保存视频。

```bash
CFG=benchmarks/wan22/configs/speed_sweep.yaml
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_bf16
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_int8 --runner-arg gemm_quantization=int8
```

从 `summary.csv` 读：
- `transformer_step_ms_median`、`transformer_2_step_ms_median`：两个专家的单步 DiT 时间（含 CFG 两次前向）。INT8 与 BF16 之比就是单步加速比，可以对照 `plan.md` 第 3.2 节的上限（33 帧 1.50×，81 帧 1.30×）。
- `dit_peak_gib_max`、`run_peak_gib_max`，以及 `env.json` 里的 `load.resident_gib`：显存对比。

"只量化高噪声专家"或"只量化低噪声专家"的速度不用单独测：一个专家取 INT8 组的单步时间，另一个取 BF16 组的单步时间，再按步数加权。40 步的端到端时间同样用单步时间推算：

非 DiT 部分 + 高噪声步数 × 高噪声单步时间 + 低噪声步数 × 低噪声单步时间

其中步数取第 2 节画质运行里报告的 `transformer_steps` / `transformer_2_steps`（40 步时预计约 26 / 14）。

## 2. 画质（约 4 小时）

`configs/quality_480p33.yaml`：10 个 prompt，取自 VBench 的 prompt 集（来源与抽样方式写在配置文件开头）；480p / 33 帧，完整 40 步，每个 prompt 1 个 seed。每组约 35 分钟加一次加载。

```bash
Q=benchmarks/wan22/configs/quality_480p33.yaml
python benchmarks/wan22/bench.py --config $Q --output-dir $R/quality_bf16
python benchmarks/wan22/bench.py --config $Q --output-dir $R/quality_int8_both --runner-arg gemm_quantization=int8
python benchmarks/wan22/bench.py --config $Q --output-dir $R/quality_int8_high \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks
python benchmarks/wan22/bench.py --config $Q --output-dir $R/quality_int8_low \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer_2.blocks
```

**偏离度（LPIPS / PSNR / SSIM，相对 BF16）：**

```bash
V=videos/480p_33f
for c in int8_both int8_high int8_low; do
  python benchmarks/wan22/compare_videos.py --reference $R/quality_bf16/$V \
      --candidate $R/quality_$c/$V --json $R/compare_${c}_vs_bf16.json
done
```

LPIPS 第一次运行会下载 AlexNet 权重，下载位置在 `$XDG_CACHE_HOME`、`$TORCH_HOME` 下。

**画质（VBench，在 `envs/vbench` 环境里运行，4 组 × 6 个维度，约半小时）：**

```bash
# 参数名以所装 VBench 版本的 `vbench evaluate --help` 为准
for c in bf16 int8_both int8_high int8_low; do
  for d in subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality; do
    vbench evaluate --videos_path $R/quality_$c/$V --dimension $d --mode custom_input --output_path $R/vbench_$c/
  done
done
```

解读时要注意：量化高噪声专家可能会改变构图，导致 LPIPS 升高，但 VBench 不一定下降。两类指标要一起看。

## 3. 按需补充（结果出现意外时才做）

| 现象 | 补做 | 耗时 |
|---|---|---|
| INT8 加速明显低于上限 | 两组都加 `--runner-arg use_torch_compile=true` 重测速度，看反量化有没有被融合 | 约 1 小时 |
| INT8 的 LPIPS 很小，分不清是量化误差还是编码误差 | BF16 用 `--repeats 2` 跑 2～3 个 prompt，测噪声底 | 约 15 分钟 |
| 想换注意力基线 | `python benchmarks/wan22/attention_bench.py --json $R/attention.json` | 约 5 分钟 |

## 4. 回报

把下面这些文件打包发回（都在 `$R` 下，不进仓库，视频不用发）：
- 每组结果目录的 `env.json`、`summary.csv`、`runs.jsonl`。
- `compare_*_vs_bf16.json`，以及 VBench 的输出目录。
- 被标记为"同卡有其他进程"后重跑的次数；任何失败的完整报错。

不要为了让实验跑通而修改仓库代码。遇到问题时记下现象和报错，发回来由我处理。
