# 阶段 1 实验手册：A100 单卡 BF16 / INT8 与专家敏感度

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M0、M2、M7。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节。

用到的代码：
- `benchmarks/wan22/bench.py`：加载一次模型，循环跑多组配置；按专家分别统计每步 DiT 时间、显存和 GPU 状态。
- `benchmarks/wan22/attention_bench.py`：在 Wan 的形状上对比各个注意力后端。
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

GPU 任务继续通过你自己的 `scripts/run_gpu_job.py`（或等效方式）执行，以落实空闲检查和设备选择。`bench.py` 本身也会在每次计时前后记录 `nvidia-smi`：同一张卡上出现别的进程时，那次运行会被标记，`summary.csv` 的中位数只用干净的运行计算（`clean_repeats` 列）。

## 1. 选注意力基线（约 5 分钟）

```bash
python benchmarks/wan22/attention_bench.py --json $R/attention.json
```

默认对比 `CUDNN`、`SDPA_FLASH`、`FLASH`、`SAGE`，形状为 L = 14040 / 32760、H = 40 / 20、D = 128、bf16。flash-attn 没装时，`FLASH` 会显示 unavailable，这是正常的。

- 在 `CUDNN` 和 `SDPA_FLASH` 中选更快的那个作为基线。如果不是 `CUDNN`，把两份配置里的 `attention_backend` 改掉，或者每次运行都加上 `--runner-arg attention_backend=SDPA_FLASH`。
- `SAGE` 的 `relative_l2` 和 `cosine` 一并记下，M7 要用。

## 2. 速度扫描（BF16 与 INT8）

`configs/speed_sweep.yaml`：480p，33 / 49 / 81 帧，40 步，1 个 prompt，每个 case 预热 1 次后重复 3 次。

```bash
CFG=benchmarks/wan22/configs/speed_sweep.yaml
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_bf16
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_int8_both \
    --runner-arg gemm_quantization=int8
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_int8_high \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_int8_low \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer_2.blocks
# epilogue 融合对照：同样的 INT8 两个专家，打开 torch.compile；BF16 也跑一份 compile 作为对照
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_int8_both_compile \
    --runner-arg gemm_quantization=int8 --runner-arg use_torch_compile=true
python benchmarks/wan22/bench.py --config $CFG --output-dir $R/speed_bf16_compile \
    --runner-arg use_torch_compile=true
```

单卡 A100 的耗时粗估：33 帧约 3.4 分钟/条、49 帧约 5.5 分钟/条、81 帧约 11 分钟/条。每个 case 跑 4 次（1 次预热 + 3 次计时），所以一组配置约 80 分钟，再加上每次加载的 10～20 分钟。6 组配置合计约 9～10 小时，可以分几次跑；`--case 480p_33f` 只跑指定的 case，`--repeats N` 可以临时改重复次数。时间紧的话，先跑 `speed_bf16`、`speed_int8_both` 和 `speed_int8_both_compile` 三组。

需要关注的读数（`summary.csv`）：
- `transformer_step_ms_median` / `transformer_2_step_ms_median`：两个专家各自的单步 DiT 时间（含 CFG 两次前向）。用 INT8 与 BF16 之比，对照 `plan.md` 第 3.2 节的预测（33 帧上限 1.50×，81 帧上限 1.30×）。
- `transformer_steps` / `transformer_2_steps`：预计约 26 / 14 步。
- `dit_peak_gib_max`、`run_peak_gib_max`，以及 `env.json` 里的 `load.resident_gib`：显存对比。

## 3. 画质（480p / 33 帧）

### 3.1 准备 prompt 列表

`prompts/dev.txt` 只有 8 条，只用来试跑流程。正式数据用 VBench 的 prompt 集：

```bash
git clone https://github.com/Vchitect/VBench.git $XDIT_WORK/src/VBench   # 已有就跳过
mkdir -p $XDIT_WORK/prompts
python - <<'EOF'
import os, random
src = os.path.expandvars("$XDIT_WORK/src/VBench/prompts/all_dimension.txt")
prompts = [line.strip() for line in open(src) if line.strip()]
random.Random(0).shuffle(prompts)
open(os.path.expandvars("$XDIT_WORK/prompts/quality_v1.txt"), "w").write("\n".join(prompts[:50]) + "\n")
EOF
sha256sum $XDIT_WORK/prompts/quality_v1.txt   # 记进回报，以后所有画质实验都用同一份
```

如果该路径下没有 `all_dimension.txt`，在 VBench 仓库的 `prompts/` 下找全维度的 prompt 列表，并在回报中注明实际使用的文件。

### 3.2 生成视频

每组 50 条 × 约 3.4 分钟 ≈ 3 小时：

```bash
Q=benchmarks/wan22/configs/quality_480p33.yaml
P=$XDIT_WORK/prompts/quality_v1.txt
python benchmarks/wan22/bench.py --config $Q --prompts-file $P --output-dir $R/quality_bf16
python benchmarks/wan22/bench.py --config $Q --prompts-file $P --output-dir $R/quality_bf16_again   # 噪声底与确定性
python benchmarks/wan22/bench.py --config $Q --prompts-file $P --output-dir $R/quality_int8_both \
    --runner-arg gemm_quantization=int8
python benchmarks/wan22/bench.py --config $Q --prompts-file $P --output-dir $R/quality_int8_high \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks
python benchmarks/wan22/bench.py --config $Q --prompts-file $P --output-dir $R/quality_int8_low \
    --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer_2.blocks
```

### 3.3 偏离度：LPIPS / PSNR / SSIM

```bash
V=videos/480p_33f
for c in bf16_again int8_both int8_high int8_low; do
  python benchmarks/wan22/compare_videos.py --reference $R/quality_bf16/$V \
      --candidate $R/quality_$c/$V --json $R/compare_${c}_vs_bf16.json
done
```

`bf16_again` 对 `bf16` 的结果就是噪声底：它反映 MP4 编码误差，以及运行本身是否确定。INT8 的偏离要和它比较才有意义。LPIPS 第一次运行会下载 AlexNet 权重，下载位置在 `$XDG_CACHE_HOME`、`$TORCH_HOME` 下。

### 3.4 画质：VBench 子集（在 `envs/vbench` 环境里运行）

维度：`subject_consistency`、`background_consistency`、`motion_smoothness`、`dynamic_degree`、`aesthetic_quality`、`imaging_quality`。这些维度都支持自定义 prompt 的视频。

```bash
# 参数名以所装 VBench 版本的 `vbench evaluate --help` 为准
for c in bf16 int8_both int8_high int8_low; do
  for d in subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality; do
    vbench evaluate --videos_path $R/quality_$c/$V --dimension $d --mode custom_input \
        --output_path $R/vbench_$c/
  done
done
```

解读时要注意：量化高噪声专家可能会改变构图，导致 LPIPS 升高，但 VBench 不一定下降。两类指标要一起看，`plan.md` 的 M2 有说明。

## 4. 可选：nsys 时间线（M9 的前置）

挑一组配置，例如 33 帧、两个专家都 INT8，只跑一个 case、一次计时：

```bash
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o $XDIT_WORK/profiles/int8_both_480p33 \
    python benchmarks/wan22/bench.py --config $CFG --output-dir $R/nsys_int8_both --nsys-capture \
        --case 480p_33f --repeats 1 --runner-arg gemm_quantization=int8
```

`--nsys-capture` 只把计时的那一次运行纳入采集范围，并给两个专家的每次调用打上 NVTX range（`transformer` / `transformer_2`）。

## 5. 回报

把下面这些文件打包发回（都在 `$R` 下，不进仓库）：
- 每组结果目录的 `env.json`、`summary.csv`、`runs.jsonl`。视频不用发。
- `attention.json`。
- `compare_*_vs_bf16.json`，以及 VBench 的输出目录。
- `quality_v1.txt` 的 sha256。
- 被标记为"同卡有其他进程"的运行次数；任何失败的完整报错。

不要为了让实验跑通而修改仓库代码。遇到问题时记下现象和报错，发回来由我处理。
