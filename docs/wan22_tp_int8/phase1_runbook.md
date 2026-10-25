# 阶段 1 实验手册：A100 单卡 BF16 / INT8 与专家敏感度

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M0、M2。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节。

这是写进简历的项目，实验只做到能支撑结论为止：
- 一共 4 个进程，每个加载一次模型。
- 不单独做冒烟，也不做预热：第一组正式运行就是冒烟，出错时几分钟内就会报出来。
- 合计约 3.5 小时。

用到的代码：
- `benchmarks/wan22/bench.py`：加载一次模型，循环跑配置里的各个 case；按专家分别统计每步 DiT 时间、显存和 GPU 状态。
- `benchmarks/wan22/configs/phase1.yaml`：三个 case，每条命令都用 `--case` 指定要跑哪些。
  - `480p_33f`：10 个 prompt，完整 40 步，保存视频。
  - `480p_33f_speed`：1 个 prompt，10 步，只测速，给第 5 节的 compile 对照用。
  - `480p_81f`：1 个 prompt，10 步，只测速。
- `benchmarks/wan22/int8_linear_bench.py`：Wan block 线性层形状上的 BF16 / INT8 单层计时，见第 5 节。
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
| BF16 | `python benchmarks/wan22/bench.py --config $C --output-dir $R/bf16 --case 480p_33f --case 480p_81f` | 33 帧 10 个视频 + 81 帧测速 |
| INT8，两个专家 | `... --output-dir $R/int8_both --case 480p_33f --case 480p_81f --runner-arg gemm_quantization=int8` | 同上 |
| INT8，只高噪声 | `... --output-dir $R/int8_high --case 480p_33f --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks` | 只跑 33 帧 10 个视频 |
| INT8，只低噪声 | `... --output-dir $R/int8_low --case 480p_33f --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer_2.blocks` | 只跑 33 帧 10 个视频 |

（表中的 `...` 即 `python benchmarks/wan22/bench.py --config $C`。）

状态（2026-10-25）：四行和第 5 节的单层计时、compile 对照都已完成，结果在仓库 `output/20261025-235940/`，结论记在 `plan.md` M2。还剩三件事：
- 第 2 节的偏离度：先按第 2 节下好 LPIPS 权重。
- 第 2 节的 VBench。
- 第 3 节最后一行的 `attention_bench.py`。注意力已经占单步时间约 35%，需要确认 CUDNN 是否是最快的 dense 后端，并测出 SAGE 的单核加速，用来决定 M7 做不做。

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

LPIPS 需要 torchvision 的 AlexNet 权重（约 233 MB）。上次在线下载中途断了，所以先用断点续传下好，再跑上面的循环：

```bash
F=$(python -c "import torch; print(torch.hub.get_dir())")/checkpoints/alexnet-owt-7be5be79.pth
mkdir -p "$(dirname "$F")"
for i in $(seq 20); do curl -fL -C - -o "$F" https://download.pytorch.org/models/alexnet-owt-7be5be79.pth && break; sleep 5; done
sha256sum "$F"    # 前 8 位必须是 7be5be79；不是的话删掉重下
```

如果实在下不下来，先给 `compare_videos.py` 加 `--no-lpips`，只出 PSNR / SSIM，报告里注明。

**画质（VBench，在 `envs/vbench` 环境里运行，4 组 × 6 个维度，约半小时）：**

```bash
# 参数名以所装 VBench 版本的 `vbench evaluate --help` 为准
for c in bf16 int8_both int8_high int8_low; do
  for d in subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality; do
    vbench evaluate --videos_path $R/$c/$V --dimension $d --mode custom_input --output_path $R/vbench_$c/
  done
done
```

每个维度第一次运行时，VBench 会把各自的权重（DINO、CLIP、RAFT、AMT、MUSIQ、LAION 美学头）下载到 `$VBENCH_CACHE_DIR`，来源包括 GitHub、Dropbox、OpenAI 和 HuggingFace。下载中断时，按报错里的 URL 和目标路径，用上面同样的 curl 断点续传循环手动下好，再重跑。某个维度 30 分钟内拿不到权重就跳过，报告里写明。

解读时要注意：量化高噪声专家可能会改变构图，导致 LPIPS 升高，但 VBench 不一定下降。两类指标要一起看。

## 3. 按需补充（结果出现意外时才做）

| 现象 | 补做 | 耗时 |
|---|---|---|
| INT8 加速明显低于上限 | 已触发，见第 5 节 | — |
| INT8 的 LPIPS 很小，分不清是量化误差还是编码误差 | BF16 加 `--case 480p_33f --repeats 2` 重跑，测噪声底 | 约 1 小时 |
| 想换注意力基线 | `python benchmarks/wan22/attention_bench.py --json $R/attention.json` | 约 5 分钟 |

## 4. 回报

把下面这些文件打包发回（都在 `$R` 下，不进仓库，视频不用发）：
- 每组结果目录的 `env.json`、`summary.csv`、`runs.jsonl`。
- `compare_*_vs_bf16.json`、`int8_linear.json`（第 5.1 节），以及 VBench 的输出目录。
- 被标记为"同卡有其他进程"后重跑的次数；任何失败的完整报错。

不要为了让实验跑通而修改仓库代码。遇到问题时记下现象和报错，发回来由我处理。

## 5. INT8 eager 变慢：诊断与补测

首轮结果：33 帧下 INT8 两个专家单步 19.5 s，BF16 8.3 s，慢 2.35×；81 帧下慢 1.9×。按线性层的时间占比反推，两种序列长度下 INT8 线性层都比 BF16 慢约 3 倍，另有每步约 0.5 s 的固定开销。所以问题出在线性层本身，不在加载、VAE 或降频。这台 A100 PCIe 持续负载时会热降频到约 960 MHz，但两组都处在同样的降频稳态，比值仍然可用，报告里注明即可。

原因（读 torchao 0.18 源码得出，待 5.1 实测确认）：xDiT 用 eager 模式跑 torchao W8A8，每个线性层有三处开销。
- **量化和反量化不融合**：激活量化拆成约 8 个独立 kernel，中间结果是 FP32；反量化 epilogue 又是约 5 次整张 M×N 的 FP32 读写。33 帧下一个 5120×5120 的层要多读写约 5 GB，而 BF16 GEMM 的输入输出只有约 0.3 GB。
- **每次调用都格式化张量**：`safe_int_mm` 用 `"FakeTensor" in input.__repr__()` 判断是否在编译。eager 下每次调用都会把 int8 激活格式化成字符串，伴随几十次 `.item()` 同步，GPU 空等。每步有 800 次线性层调用（CFG 两次前向 × 40 个 block × 10 个线性层），那 0.5 s 的固定开销估计来自这里。
- **GEMM 布局**：`safe_int_mm` 把权重 `.contiguous()` 成 (K, N) 交给 cuBLAS，得到 "NN" 布局；cuBLASLt 的 INT8 tensor core 只支持 "TN"，所以 GEMM 很可能没用上 tensor core。

`torch.compile` 走的是 torchao 的编译分支：不调用 `__repr__`，权重以转置视图传入（TN），量化和反量化也被融合成少数几个 kernel。torchao 的 W8A8 本来就是配合 compile 用的，`plan.md` 第 3.2 节的上限也假设量化开销已被融合。

### 5.1 单层计时（约 5 分钟）

```bash
python benchmarks/wan22/int8_linear_bench.py --tokens 14040 32760 --json $R/int8_linear.json
```

覆盖 Wan block 的三种线性层形状和两种序列长度。读 `speedup_vs_bf16`：
- `torchao_eager` 约 0.3～0.5×：复现了端到端的变慢。
- `int_mm_tn` 明显快于 `int_mm_nn`：布局问题坐实。`matches_int_mm_nn` 必须为 true。
- `torchao_compile` 在 14040 下三种形状都 ≥ 1.2×：继续做 5.2。否则停下，把输出发回来。

### 5.2 compile 对照（2 个进程，约 1.5 小时，含编译）

| 进程 | 命令 |
|---|---|
| BF16 + compile | `... --output-dir $R/bf16_compile --case 480p_33f_speed --case 480p_81f --runner-arg use_torch_compile=true` |
| INT8 + compile | `... --output-dir $R/int8_compile --case 480p_33f_speed --case 480p_81f --runner-arg use_torch_compile=true --runner-arg gemm_quantization=int8` |

- 编译预热由 xDiT 在加载后完成（用第一个 case 完整跑 10 步），不计入计时。
- 81 帧换了形状，会触发重新编译。编译耗时落在 81 帧两个专家各自的第一步上；取单步中位数（7 步 / 3 步）时不受影响。
- 读数同第 2 节。简历上的 INT8 加速比用这一组（compile 对 compile），否则 compile 本身带来的融合收益会被算到 INT8 头上。40 步的端到端时间按 26 × 高噪声单步 + 14 × 低噪声单步 + 非 DiT 时间拼出；非 DiT 时间取 bf16 组 33 帧的 `pipeline_s_median − dit_ms_median`，约 2.3 s。

### 5.3 剩余的画质实验

第 1 节后两行（只高噪声、只低噪声）和第 2 节的对比、VBench 照原样用 eager 跑，不加 compile。这样能和已有的 bf16、int8_both 视频保持同一条数值路径：compile 不改变量化算法，但会改变舍入的位置，40 步去噪会把这种微小差异放大成可见的细节差异，混用会干扰专家敏感度的比较。eager INT8 慢，这两组合计约 3 小时，可以放到夜里跑。
