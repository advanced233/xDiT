# 运行环境配置清单（共享服务器）

本文档面向在共享服务器上执行配置的本地 agent，配合 [`plan.md`](plan.md) 使用。

2×A6000 和 A100 如果不在同一台机器上，每台机器各执行一遍。

## 0. 共享服务器规则（全程遵守）

**写入范围**
- 所有文件只写到自己账号名下的工作根目录 `$XDIT_WORK`，默认是 `$HOME/xdit-work`。
- 如果 `$HOME` 配额不够，只能改用已经存在、且属主是自己的大盘目录（用 `stat -c %U <目录>` 确认）。不要自己在公共盘上新建目录。

**禁止的操作**
- 不用 `sudo`，不用 `apt` / `yum`。
- 不写 `/usr`、`/opt`、`/etc`，也不写公共数据盘或公共模型目录。
- 不修改系统级的 CUDA、驱动、conda base 环境和 `/etc/pip.conf`。

**Python 包隔离**
- 不用 `pip install --user`。
- 设置 `PYTHONNOUSERSITE=1`，避免读到 `~/.local` 里的包。

**缓存与临时文件**
- pip、HuggingFace、torch、Triton、inductor、CUDA 的缓存，以及 `TMPDIR`，全部指向 `$XDIT_WORK` 下（见步骤 2）。
- 系统 `/tmp` 是公用的，编译时尤其不要往里写。

**使用 GPU**
- 每次运行前先用 `nvidia-smi` 查看占用情况，用 `CUDA_VISIBLE_DEVICES` 只选空闲的卡。
- 不要 kill 别人的进程，不改 GPU 时钟、persistence mode、MIG 或 ECC。
- 如果服务器有 Slurm 之类的调度器，就通过调度器申请 GPU，不要直接在登录节点上跑。

**避免和他人冲突**
- torchrun / `xdit` 默认端口是 29500，容易和别人撞车，每次显式传一个随机高位端口（`--master_port`）。
- 编译时限制并行度（`MAX_JOBS=4`），运行时限制 `OMP_NUM_THREADS`，避免占满共享 CPU。

**不修改 `~/.bashrc`**：环境变量写进 `$XDIT_WORK/env.sh`，需要时手动 `source`。

**遇到以下情况先停下问用户**：需要 root；磁盘或配额不够；驱动太旧；GPU 全被占用；网络不通；需要写入自己目录以外的位置；发现公共只读模型副本，想读取它。

## 1. 清单

| # | 项 | 要求 / 目标 | 完成标志 |
|---|---|---|---|
| 1 | 工作根目录 | `$XDIT_WORK`，属主是自己；可用空间 ≥ 250 GB（环境约 20 GB，加两套模型，再加输出和 profile） | `df -h $XDIT_WORK` |
| 2 | `env.sh` | 所有缓存、`TMPDIR`、`HF_HOME` 都指向 `$XDIT_WORK` | `source` 后 `env` 可见 |
| 3 | 驱动 / CUDA | 驱动支持的 CUDA ≥ 12.6（`nvidia-smi` 右上角显示）；不需要系统装 nvcc | `nvidia-smi` |
| 4 | Python 环境 | 自己目录下的 Miniforge，环境用 `-p` 建在 `$XDIT_WORK/envs/xdit`，Python 3.10（与 CI 一致） | `which python` 指向该目录 |
| 5 | PyTorch | CUDA 版 wheel，CUDA 版本 ≤ 驱动支持的版本 | `torch.cuda.is_available()`，`torch.version.hip is None` |
| 6 | nvcc（编译用） | 装进 env，与 `torch.version.cuda` 的 major.minor 一致 | `$CONDA_PREFIX/bin/nvcc --version` |
| 7 | xDiT | 克隆到 `$XDIT_WORK/src/xDiT`，分支 `claude/keen-bardeen-ntp7zr`，执行 `pip install -e ".[test,torchao]"` | `import xfuser` |
| 8 | diffusers | ≥ 0.35.2（Wan runner 的最低要求） | 版本号 |
| 9 | torchao | 与 torch 版本匹配（见 torchao README 中的兼容性表） | `import torchao` 无 C++ 扩展告警；`torch._int_mm` 在 GPU 上能跑 |
| 10 | flash-attn | ≥ 2.7.0，优先用预编译 wheel | `import flash_attn` |
| 11 | yunchang | ≥ 0.6.0（作为依赖自动安装） | `import yunchang` |
| 12 | SageAttention | 优先源码编译 SageAttention2（sm_80/86 走 INT8 QK + FP16 PV）；退路是 PyPI 上的 `sageattention==1.0.6` | `sageattn` 在 GPU 上能跑 |
| 13 | Nsight Systems | 优先用系统已有的 `nsys`；没有就报告，不要自己用 root 安装 | `nsys --version` |
| 14 | 画质评测 | 主环境装 `lpips`；VBench 单独建环境 `$XDIT_WORK/envs/vbench`（它的依赖版本会冲突），阶段 1 前完成即可 | `import lpips` |
| 15 | 模型权重 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（开发用）和 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`，都下载到 `$HF_HOME` | 设置 `HF_HUB_OFFLINE=1` 后可以加载 |
| 16 | 拓扑与带宽 | `nvidia-smi topo -m`、P2P 状态，以及 NCCL all_reduce / all_to_all 实测带宽 | 写进回报 |
| 17 | 仓库测试 | CPU 测试与 GPU 测试的各个 selector 都跑一遍（见步骤 8） | 记录 passed / failed / skipped 数 |
| 18 | 端到端冒烟 | 1.3B 单卡和 Ulysses2 各跑一次；A14B 用短配置跑一次 | 生成出视频文件 |

## 2. 执行步骤

### 步骤 1：只读探查（完成后先向用户汇报，确认再继续）

```bash
whoami; echo "$HOME"; hostname
df -h "$HOME"; quota -s 2>/dev/null || true
# 是否已有属于自己的大盘目录（只看，不创建）
for d in /data/$USER /scratch/$USER /mnt/*/$USER; do [ -d "$d" ] && stat -c '%U %n' "$d"; done 2>/dev/null
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
which nvcc && nvcc --version; ls -d /usr/local/cuda* 2>/dev/null
gcc --version | head -1
free -g; nproc
which conda python3 uv 2>/dev/null; type module 2>/dev/null && module avail 2>&1 | head -40
which sbatch srun squeue 2>/dev/null
which nsys ncu 2>/dev/null; cat /proc/sys/kernel/perf_event_paranoid
curl -sI https://huggingface.co | head -1; curl -sI https://pypi.org/simple/ | head -1
curl -sI https://download.pytorch.org/whl/ | head -1; curl -sI https://github.com | head -1
```

需要汇报的内容：
- 准备使用的 `$XDIT_WORK` 及其可用空间。
- 驱动版本和支持的 CUDA 版本。
- GPU 型号、当前占用、拓扑（是否有 NVLink，P2P 是否可用）。
- 是否有调度器。
- 能否访问外网（HuggingFace、PyPI、PyTorch、GitHub）。
- 是否已有 `nsys`。

访问不了 huggingface.co 时，可以设置 `HF_ENDPOINT` 指向镜像，或改用 ModelScope。pip 访问慢时，可以设置 `PIP_INDEX_URL`。这两项都只写进 `env.sh`，不改全局配置。

### 步骤 2：目录与 `env.sh`

```bash
export XDIT_WORK="$HOME/xdit-work"   # 或步骤 1 确认过的、属于自己的大盘目录
mkdir -p "$XDIT_WORK"/{envs,src,scripts,outputs,logs,profiles,tmp,cache}
cat > "$XDIT_WORK/env.sh" <<EOF
export XDIT_WORK="$XDIT_WORK"
EOF
cat >> "$XDIT_WORK/env.sh" <<'EOF'
export HF_HOME="$XDIT_WORK/cache/hf"
export PIP_CACHE_DIR="$XDIT_WORK/cache/pip"
export TORCH_HOME="$XDIT_WORK/cache/torch"
export TRITON_CACHE_DIR="$XDIT_WORK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDIT_WORK/cache/inductor"
export CUDA_CACHE_PATH="$XDIT_WORK/cache/nv"
export XDG_CACHE_HOME="$XDIT_WORK/cache/xdg"
export CONDA_PKGS_DIRS="$XDIT_WORK/cache/conda-pkgs"
export VBENCH_CACHE_DIR="$XDIT_WORK/cache/vbench"
export TMPDIR="$XDIT_WORK/tmp"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MAX_JOBS=4
# export HF_ENDPOINT=...        # 需要镜像时再打开
# export PIP_INDEX_URL=...
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$VBENCH_CACHE_DIR" "$TMPDIR"
if [ -f "$XDIT_WORK/miniforge3/bin/activate" ]; then
  source "$XDIT_WORK/miniforge3/bin/activate"
  [ -d "$XDIT_WORK/envs/xdit" ] && conda activate "$XDIT_WORK/envs/xdit"
  [ -n "$CONDA_PREFIX" ] && export CUDA_HOME="$CONDA_PREFIX"
fi
EOF
source "$XDIT_WORK/env.sh"
```

之后每次开新 shell，都先执行 `source $XDIT_WORK/env.sh`。

### 步骤 3：Python 环境（装在自己的目录里）

```bash
curl -L -o "$TMPDIR/miniforge.sh" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$TMPDIR/miniforge.sh" -b -p "$XDIT_WORK/miniforge3"   # -b：不执行 conda init，不改 ~/.bashrc
source "$XDIT_WORK/miniforge3/bin/activate"
conda create -y -p "$XDIT_WORK/envs/xdit" python=3.10
source "$XDIT_WORK/env.sh"   # 进入 envs/xdit，并设置 CUDA_HOME
which python pip             # 两者都必须指向 $XDIT_WORK/envs/xdit/bin
```

如果系统里已经有 conda，也不要使用它的 base 环境或公共环境。

### 步骤 4：PyTorch

按 `nvidia-smi` 显示的 CUDA Version 选 wheel：≥ 12.8 用 `cu128`，≥ 12.6 用 `cu126`；低于 12.6 就停下来汇报，不要尝试升级驱动。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())"
```

### 步骤 5：仓库与 Python 依赖

```bash
cd "$XDIT_WORK/src"
git clone https://github.com/advanced233/xDiT.git   # 已经有自己的克隆就改为 git fetch
cd xDiT && git checkout claude/keen-bardeen-ntp7zr
pip install -e ".[test,torchao]"
python -c "import diffusers; print(diffusers.__version__)"   # 必须 ≥ 0.35.2，否则执行：pip install -U "diffusers>=0.35.2"
pip install lpips
pip install pre-commit && pre-commit install                 # 可选；hook 只写入本克隆的 .git
```

安装 torchao 后要核对版本是否匹配：查 torchao README 的兼容性表，确认它和当前 torch 版本对应；不对应就安装匹配的版本。

### 步骤 6：需要编译的扩展（flash-attn、SageAttention）

```bash
# nvcc 装进 env，版本与 torch 的 CUDA 一致
CUDA_MM=$(python -c "import torch; print(torch.version.cuda)")
conda install -y -c nvidia "cuda-toolkit=${CUDA_MM}"
nvcc --version   # 应指向 $CONDA_PREFIX/bin/nvcc

# flash-attn：安装时会先尝试下载匹配的预编译 wheel，找不到才在本地编译（受 MAX_JOBS 限制）
pip install "flash-attn>=2.7.0" --no-build-isolation

# SageAttention2：必须在能看到 GPU 的节点上编译
git clone https://github.com/thu-ml/SageAttention.git "$XDIT_WORK/src/SageAttention"
cd "$XDIT_WORK/src/SageAttention" && pip install -e . --no-build-isolation
# 编译失败时的退路（Triton 版 v1，同样走 INT8 QK + FP16 PV）：pip install sageattention==1.0.6
```

编译中间文件都落在 `$TMPDIR` 和 `$XDIT_WORK/src` 下。

### 步骤 7：模型权重

```bash
# 先查大小，与剩余空间比较
python - <<'EOF'
from huggingface_hub import HfApi
for repo in ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"]:
    info = HfApi().model_info(repo, files_metadata=True)
    print(repo, f"{sum(s.size or 0 for s in info.siblings) / 1e9:.1f} GB")
EOF
df -h "$XDIT_WORK"

# 下载到 $HF_HOME/hub。不要加 --local-dir，runner 按 repo id 从缓存加载
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers          # 旧版 CLI 用 huggingface-cli download
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

下载完成后，运行时一律设置 `export HF_HUB_OFFLINE=1`。

### 步骤 8：自检

先选空闲的卡，再生成随机端口：

```bash
export CUDA_VISIBLE_DEVICES=0,1   # 按步骤 1 的占用情况选择
PORT=$((20000 + RANDOM % 20000))
```

**8.1 版本与关键算子**

```bash
python - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", torch.version.hip)
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for m in ["diffusers", "transformers", "torchao", "flash_attn", "yunchang", "sageattention", "lpips", "xfuser"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, e)
a = torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device="cuda")
print("int_mm", torch._int_mm(a, a.t()).dtype)
try:
    from sageattention import sageattn
    q = torch.randn(1, 8, 4096, 128, dtype=torch.float16, device="cuda")
    print("sageattn", sageattn(q, q, q, tensor_layout="HND").shape)
except Exception as e:
    print("sageattn FAILED", type(e).__name__, e)
if torch.cuda.device_count() > 1:
    print("p2p 0<->1", torch.cuda.can_device_access_peer(0, 1))
EOF
nsys --version 2>/dev/null || echo "nsys missing"
```

**8.2 NCCL 带宽**

脚本放在 `$XDIT_WORK/scripts`，不进仓库。实测值用于替换 `plan.md` 第 3.3 节的估算。

```bash
cat > "$XDIT_WORK/scripts/nccl_bw.py" <<'EOF'
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def bench(fn, nbytes, label, factor):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 20
    if rank == 0:
        print(f"{label} {nbytes / 2**20:.0f} MiB: {dt * 1e3:.2f} ms, busbw {nbytes * factor / dt / 1e9:.1f} GB/s")

for mib in (64, 256, 1024):
    x = torch.zeros(mib * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    bench(lambda: dist.all_reduce(x), x.numel() * 2, "all_reduce", 2 * (world - 1) / world)
    bench(lambda: dist.all_to_all_single(y, x), x.numel() * 2, "all_to_all", (world - 1) / world)
dist.destroy_process_group()
EOF
torchrun --nproc_per_node 2 --master_port "$PORT" "$XDIT_WORK/scripts/nccl_bw.py"
```

**8.3 仓库测试**（遵循 `.github/skills/testing/SKILL.md`）

```bash
cd "$XDIT_WORK/src/xDiT"
python -m pytest tests/unit -m "not accelerator"
python -m pytest tests/integration/cpu -m "not accelerator"
CUDA_VISIBLE_DEVICES=0 python -m pytest -m "accelerator and not rocm and not multi_gpu"
python -m pytest -m "accelerator and multi_gpu and not rocm"   # 需要 2 张空闲卡
```

记录每条命令的 passed / failed / skipped / deselected 数。如果有失败，只记录现象并汇报，不要修改代码。

**8.4 端到端冒烟**

```bash
export HF_HUB_OFFLINE=1
OUT="$XDIT_WORK/outputs/smoke"
CUDA_VISIBLE_DEVICES=0 xdit --model Wan2.1-T2V-1.3B --prompt "a cat walking on grass" \
  --height 480 --width 832 --num_frames 17 --num_inference_steps 4 \
  --master_port "$PORT" --output_directory "$OUT"
xdit --model Wan2.1-T2V-1.3B --prompt "a cat walking on grass" \
  --height 480 --width 832 --num_frames 17 --num_inference_steps 4 \
  --ulysses_degree 2 --master_port "$PORT" --output_directory "$OUT"
# Wan2.2-A14B：2×A6000 上用 FSDP2 + Ulysses2，让 BF16 双专家放得下
xdit --model Wan2.2-T2V --prompt "a cat walking on grass" \
  --height 480 --width 832 --num_frames 17 --num_inference_steps 4 \
  --fully_shard_degree 2 --ulysses_degree 2 --master_port "$PORT" --output_directory "$OUT"
# A100 单卡：直接跑；OOM 时加 --enable_model_cpu_offload
```

### 步骤 9：回报

把下面的内容写到 `$XDIT_WORK/env_report.md`，并展示给用户。主机名、用户名和路径不要提交到仓库；如果用户同意，可以只把版本、拓扑和带宽这几项整理进 `plan.md`。

```text
机器：GPU 型号 × 数量 | 驱动 / 支持的 CUDA | 拓扑（NVLink/PCIe，P2P 是否可用）| 是否有调度器
工作目录：$XDIT_WORK | 可用空间 | 已占用空间
版本：python / torch(cuda) / diffusers / transformers / torchao / flash-attn / yunchang / sageattention / nsys
NCCL：all_reduce 与 all_to_all 在 64/256/1024 MiB 下的 busbw
测试：4 条 selector 各自的 passed / failed / skipped / deselected 数（附失败用例名）
冒烟：3 条命令的结果、耗时、输出文件路径
未完成项和原因（例如 nsys 缺失、SageAttention2 编译失败而改用 v1）
```
