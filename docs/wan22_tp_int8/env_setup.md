# 运行环境配置清单（共享服务器）

本文档面向在共享服务器上执行配置的本地 agent，配合 [`plan.md`](plan.md) 使用。

2×A6000 和 A100 如果不在同一台机器上，每台机器各执行一遍。

## 0. 共享服务器规则（全程遵守）

**写入范围**
- 所有文件只写到自己账号名下的工作根目录 `$XDIT_WORK`，默认是 `$HOME/xdit-work`。工具在 `$HOME` 下自动生成的少量配置文件（例如 `~/.conda/`）也属于自己账号，可以接受，不需要为此额外加沙箱。
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
- 空闲卡的定义：这张卡上没有计算进程（`nvidia-smi --query-compute-apps` 里没有它），并且显存占用低于 1 GB。Xorg、gnome-shell 这类图形进程（`nvidia-smi` 进程表里类型为 `G`）只占几十 MB，不算占用。
- 不要 kill 别人的进程，不改 GPU 时钟、persistence mode、MIG 或 ECC。
- 如果服务器有**正在运行**的 Slurm 之类的调度器（`sinfo` 能列出分区），就通过调度器申请 GPU，不要直接在登录节点上跑。只装了客户端、没有配置文件也没有服务进程，不算有调度器，按上面两条直接使用空闲的卡。
- 如果用沙箱（例如 bwrap）包装命令，GPU 相关的步骤必须把 `/dev/nvidia*`（含 `nvidiactl`、`nvidia-uvm`、`nvidia-uvm-tools`）透传进去，否则 CUDA 不可见。多进程运行时所有 rank 要在同一个沙箱里，以便共享 `/dev/shm`（NCCL 在无 P2P 时依赖它）。

**避免和他人冲突**
- torchrun / `xdit` 默认端口是 29500，容易和别人撞车，每次显式传一个随机高位端口（`--master_port`）。
- 编译时限制并行度（`MAX_JOBS=4`），运行时限制 `OMP_NUM_THREADS`，避免占满共享 CPU。

**不修改 `~/.bashrc`**：环境变量写进 `$XDIT_WORK/env.sh`，需要时手动 `source`。

**两台服务器共用一个分支**（`claude/keen-bardeen-ntp7zr`）
- 源码只由 Claude 修改，agent 不改、不提交源码。
- 结果放在 `output/<设备>/<时间戳>/` 下（`a100` 或 `a6000`），只往自己设备的目录里加文件，不改、不删别的目录。`output/` 下已有的无设备前缀的目录是早期的 A100 结果，保持不动。
- 运行前先 `git pull`，确认 `git log -1` 是消息里指定的 commit。推送前先 `git pull --rebase`，不要强推。
- 发布前脱敏：主机名、用户名、本地绝对路径、GPU UUID。
- 多卡运行只用同一个 NUMA 组里的卡（A6000 服务器是 0–3 或 4–7，跨组走 SYS），并在报告里写明用的是哪几张。A6000 服务器的 2 号卡链路只有 Gen4 x8，其他卡都是 x16；多卡实验避开 2 号卡，只在凑不出 x16 的两张卡时才用它，并在报告里注明。

**无人值守执行**：用户不在场，整个流程不需要等确认，从头跑到尾。遇到阻塞时按下表处理，绝不为了绕过阻塞而越过上面的规则：

| 情况 | 处理 |
|---|---|
| 某一步需要 root | 跳过该项，记录原因，继续其余步骤 |
| 空间或配额不够放 A14B 权重 | 只下载 1.3B，记录缺口大小 |
| 驱动版本低于 525.60.13 | 选 PyTorch 提供的、CUDA 版本不高于驱动支持版本的最新 wheel，并在回报中注明（驱动 ≥ 525.60.13 时见步骤 4，不属于阻塞） |
| GPU 全被占用 | 每 10 分钟重新检查一次，最多等 1 小时；仍然没有空闲卡就跳过 GPU 相关的自检，记录原因 |
| 某个源访问不了 | 尝试 `env.sh` 里注释掉的镜像选项；仍然失败就跳过依赖它的项 |
| 需要写入自己目录以外的位置 | 不写，跳过该项并记录 |
| 发现公共的模型副本 | 不使用，照常下载自己的副本，并在回报中提一句 |
| 某个包编译失败 | 按对应步骤里的退路处理；退路也失败就记录完整报错，继续后面的步骤 |

## 1. 清单

| # | 项 | 要求 / 目标 | 完成标志 |
|---|---|---|---|
| 1 | 工作根目录 | `$XDIT_WORK`，属主是自己；可用空间 ≥ 250 GB（环境约 20 GB，加两套模型，再加输出和 profile） | `df -h $XDIT_WORK` |
| 2 | `env.sh` | 所有缓存、`TMPDIR`、`HF_HOME` 都指向 `$XDIT_WORK` | `source` 后 `env` 可见 |
| 3 | 驱动 / CUDA | 驱动 ≥ 525.60.13 即可使用 CUDA 12.x 的 wheel（次版本兼容，见步骤 4）；不需要系统装 nvcc | `nvidia-smi` |
| 4 | Python 环境 | 自己目录下的 Miniforge，环境用 `-p` 建在 `$XDIT_WORK/envs/xdit`，Python 3.10（与 CI 一致） | `which python` 指向该目录 |
| 5 | PyTorch | cu126 或 cu128 的 CUDA 版 wheel | `torch.cuda.is_available()`，`torch.version.hip is None`，`torch.compile` 能在 GPU 上跑 |
| 6 | nvcc（编译用） | 装进 env，与 `torch.version.cuda` 的 major.minor 一致 | `$CONDA_PREFIX/bin/nvcc --version` |
| 7 | xDiT | 克隆到 `$XDIT_WORK/src/xDiT`，分支 `claude/keen-bardeen-ntp7zr`，执行 `pip install -e ".[test,torchao]"` | `import xfuser` |
| 8 | diffusers | ≥ 0.35.2（Wan runner 的最低要求） | 版本号 |
| 9 | torchao | 与 torch 版本匹配（见 torchao README 中的兼容性表） | `import torchao` 无 C++ 扩展告警；`torch._int_mm` 在 GPU 上能跑 |
| 10 | flash-attn（可选） | ≥ 2.7.0，只用预编译 wheel；装不上就跳过（Wan 走 SDPA / cuDNN 后端，不依赖它） | `import flash_attn` 或回报中注明跳过 |
| 11 | yunchang | ≥ 0.6.0（作为依赖自动安装） | `import yunchang` |
| 12 | SageAttention | 优先源码编译 SageAttention2（sm_80/86 走 INT8 QK + FP16 PV）；退路是 PyPI 上的 `sageattention==1.0.6` | `sageattn` 在 GPU 上能跑 |
| 13 | Nsight Systems | 优先用系统已有的 `nsys`；没有就报告，不要自己用 root 安装。`/proc/sys/kernel/perf_event_paranoid` ≥ 2 时 CPU 采样不可用，用 `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none`，GPU 时间线不受影响 | `nsys --version` |
| 14 | 画质评测 | 主环境装 `lpips`；VBench 单独建环境 `$XDIT_WORK/envs/vbench`（它的依赖版本会冲突），阶段 1 前完成即可 | `import lpips` |
| 15 | 模型权重 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（开发用）和 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`，都下载到 `$HF_HOME` | 设置 `HF_HUB_OFFLINE=1` 后可以加载 |
| 16 | 拓扑与带宽 | `nvidia-smi topo -m`、P2P 状态，以及 NCCL all_reduce / all_to_all 实测带宽 | 写进回报 |
| 17 | 仓库测试 | CPU 测试与 GPU 测试的各个 selector 都跑一遍（见步骤 8） | 记录 passed / failed / skipped 数 |
| 18 | 端到端冒烟 | 只跑一次 1.3B：有两张空闲卡就用 Ulysses2，否则单卡；A14B 不单独冒烟，由第一组正式实验验证 | 生成出视频文件 |

## 2. 执行步骤

### 步骤 1：只读探查（结果写进回报，不用等确认，直接继续）

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

`$XDIT_WORK` 按以下规则自动决定：
- `$HOME` 所在盘的可用空间（并且配额允许）≥ 250 GB 时，用 `$HOME/xdit-work`。
- 否则，如果上面找到了属主是自己、空间足够的大盘目录，就用其下的 `xdit-work`。
- 两者都不满足时，仍用 `$HOME/xdit-work`，步骤 7 只下载 1.3B。

记入回报的内容：
- 最终使用的 `$XDIT_WORK`、选择它的理由，以及可用空间。
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

`nvidia-smi` 右上角的 CUDA Version 是驱动原生支持的最高版本，并不是 wheel 的硬上限。CUDA 12 有次版本兼容（minor version compatibility）：驱动 ≥ 525.60.13，就能运行任何用 CUDA 12.x 编译的程序。PyTorch wheel 自带 sm_80 / sm_86 的 SASS，不依赖驱动 JIT。

所以只要驱动 ≥ 525.60.13，就能装 cu126 的 wheel，不要因为驱动显示 12.4 就退回 torch 2.6 + cu124。不要尝试升级驱动。

两台服务器统一用 A100 上验证过的版本：torch 2.14.1（cu126）、torchao 0.18.0、diffusers 0.41.0、transformers 5.20.2。版本不同，INT8 和 torch.compile 的行为就不同，两台机器的结果没法放在一起比。另外，torch 2.11 的 TorchDynamo 有问题：`test_usp_sequence_parallel_pad` 报 `InternalTorchDynamoError`，`hybrid-ring4` 卡死；这些在 2.14.1 上都能通过。

```bash
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())"
# 在 GPU 上验证 Triton 编译链路
python -c "import torch; f = torch.compile(lambda x: (x * 2).sin()); print(f(torch.randn(1024, device='cuda')).sum())"
```

如果 `torch.compile` 或 Triton 报 PTX / toolchain 相关的错误：
1. 在 env 里装一个与驱动 CUDA 版本一致的 ptxas，例如 `conda install -y -c nvidia cuda-nvcc=12.4`，装到单独的 prefix 也可以。
2. 设置 `TRITON_PTXAS_PATH=<该 ptxas 的路径>` 后重试。
3. 仍然失败，就回退到 cu124 的版本组合，并在回报中注明。

### 步骤 5：仓库与 Python 依赖

```bash
cd "$XDIT_WORK/src"
git clone https://github.com/advanced233/xDiT.git   # 已经有自己的克隆就改为 git fetch
cd xDiT && git checkout claude/keen-bardeen-ntp7zr
pip install -e ".[test,torchao]"
pip install torchao==0.18.0 --index-url https://download.pytorch.org/whl/cu126   # 与 A100 一致
python -c "import diffusers; print(diffusers.__version__)"   # 必须 ≥ 0.35.2，否则执行：pip install -U "diffusers>=0.35.2"
pip install lpips
pip install pre-commit && pre-commit install                 # 可选；hook 只写入本克隆的 .git
```

装完核对 `torch`、`torchao`、`diffusers`、`transformers` 四个版本与上面一致，不一致就按上面的版本重装。

### 步骤 6：需要编译的扩展（SageAttention，flash-attn 可选）

flash-attn 不是必需的：
- 在 Ampere 上，xDiT 没指定后端时默认用 PyTorch SDPA 的 `CUDNN` 后端。
- `SDPA_FLASH`（PyTorch 内置的 FlashAttention-2 kernel）、`SDPA_EFFICIENT`、`CUDNN` 都支持 Ring（见 `xfuser/core/attention/backends/sdpa/spec.py`），所以 Wan 的 Ulysses / Ring 都不依赖 flash-attn 包。
- flash-attn 只在显式指定 `--attention_backend FLASH`，或者走 yunchang 旧路径的其他模型时才需要。

所以：能直接装上预编译 wheel 就装，装不上不必从源码编译，在回报中注明即可。

```bash
# nvcc 装进 env，版本与 torch 的 CUDA 一致
CUDA_MM=$(python -c "import torch; print(torch.version.cuda)")
conda install -y -c nvidia "cuda-nvcc=${CUDA_MM}" "cuda-cudart-dev=${CUDA_MM}" "cuda-cccl=${CUDA_MM}"
nvcc --version   # 应指向 $CONDA_PREFIX/bin/nvcc

# flash-attn（可选）：只尝试预编译 wheel。找不到匹配的 wheel 时 pip 会转入本地编译，用 timeout 截断；
# 超时，或者导入时报 GLIBC_2.xx not found，就卸载并跳过，不要从源码编译
timeout 900 pip install "flash-attn>=2.7.0" --no-build-isolation || echo "flash-attn skipped"
# 编译 SageAttention2 时如果缺 CUDA 头文件，再补装：conda install -y -c nvidia "cuda-libraries-dev=${CUDA_MM}"

# SageAttention2：只有 A100 需要（M7），A6000 不装。必须在能看到 GPU 的节点上编译
git clone https://github.com/thu-ml/SageAttention.git "$XDIT_WORK/src/SageAttention"
cd "$XDIT_WORK/src/SageAttention" && pip install -e . --no-build-isolation
# 编译失败就跳过，不要退回 sageattention 1.0.6：它不支持 return_lse，xDiT 的 SAGE 后端会报错
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

下载 A14B 前，确认剩余空间 ≥ 查询到的大小再加 50 GB 余量；不够就只保留 1.3B，并记录缺口大小。

下载完成后，运行时一律设置 `export HF_HUB_OFFLINE=1`。

### 步骤 8：自检

先选空闲的卡，再生成随机端口：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 重新确认占用
export CUDA_VISIBLE_DEVICES=0,1   # 只选当前空闲的卡；没有空闲卡时按第 0 节的规则等待或跳过
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

带宽偏低时（A6000 服务器实测约 7 GB/s），再用同样两张卡花约 5 分钟查一下原因：

```bash
# PCIe 链路的代数和宽度（只读，有负载时 current 才准；不支持的字段删掉即可）
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
# NCCL 实际走的传输方式：日志里找 "via P2P/..." 或 "via SHM"
NCCL_DEBUG=INFO torchrun --nproc_per_node 2 --master_port "$PORT" "$XDIT_WORK/scripts/nccl_bw.py" 2>&1 | grep -E "via|busbw"
# 关掉 P2P、改走主机内存，对比带宽
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 2 --master_port "$PORT" "$XDIT_WORK/scripts/nccl_bw.py"
# 在 Intel CPU 上，NCCL 默认只在 PXB 以内（经过 PCIe 交换机）用 P2P；经过 CPU 根复合体（nvidia-smi 显示 NODE）
# 时会退回 SHM。把门槛放宽到 PHB，强制走 P2P，再对比一次
NCCL_P2P_LEVEL=PHB NCCL_DEBUG=INFO torchrun --nproc_per_node 2 --master_port "$PORT" "$XDIT_WORK/scripts/nccl_bw.py" 2>&1 | grep -E "via|busbw"
# 裸拷贝带宽：卡间 P2P，以及卡和锁页内存之间
python - <<'EOF'
import time, torch
def bw(src, dst, n=1 << 30):
    make = lambda dev: torch.empty(n, dtype=torch.uint8).pin_memory() if dev == "cpu" else torch.empty(n, dtype=torch.uint8, device=dev)
    x, y = make(src), make(dst)
    for _ in range(3):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(10):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(); print(f"{src} -> {dst}: {10 * n / (time.perf_counter() - start) / 1e9:.1f} GB/s")
for src, dst in [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0"), ("cpu", "cuda:0"), ("cuda:0", "cpu")]:
    bw(src, dst)
EOF
```

**8.3 仓库测试**（遵循 `.github/skills/testing/SKILL.md`）

```bash
cd "$XDIT_WORK/src/xDiT"
python -m pytest tests/unit -m "not accelerator"
python -m pytest tests/integration/cpu -m "not accelerator"
CUDA_VISIBLE_DEVICES=0 python -m pytest -m "accelerator and not rocm and not multi_gpu"
python -m pytest -m "accelerator and multi_gpu and not rocm"   # 需要 2 张空闲卡
```

记录每条命令的 passed / failed / skipped / deselected 数。如果有失败，只把现象记进回报，不要修改代码。

**8.4 端到端冒烟（只跑一次）**

用 1.3B 小模型跑一条短视频，同时覆盖模型加载、推理和视频输出。有两张空闲卡时加 `--ulysses_degree 2`，顺带验证多卡通信。A14B 不单独冒烟：它的加载就要 10～20 分钟，第一组正式实验会顺带验证它。

```bash
export HF_HUB_OFFLINE=1
xdit --model Wan2.1-T2V-1.3B --prompt "a cat walking on grass" \
  --height 480 --width 832 --num_frames 17 --num_inference_steps 4 \
  --ulysses_degree 2 --master_port "$PORT" --output_directory "$XDIT_WORK/outputs/smoke"
# 只有一张空闲卡时：去掉 --ulysses_degree 2，并设置 CUDA_VISIBLE_DEVICES 为那张卡
```

### 步骤 9：回报

把下面的内容写到 `$XDIT_WORK/env_report.md`，用户回来后展示给他们看。主机名、用户名和路径不要提交到仓库；如果用户同意，可以只把版本、拓扑和带宽这几项整理进 `plan.md`。

```text
机器：GPU 型号 × 数量 | 驱动 / 支持的 CUDA | 拓扑（NVLink/PCIe，P2P 是否可用）| 是否有调度器
工作目录：$XDIT_WORK | 可用空间 | 已占用空间
版本：python / torch(cuda) / diffusers / transformers / torchao / flash-attn / yunchang / sageattention / nsys
NCCL：all_reduce 与 all_to_all 在 64/256/1024 MiB 下的 busbw
测试：4 条 selector 各自的 passed / failed / skipped / deselected 数（附失败用例名）
冒烟：那 1 条命令的结果、耗时、输出文件路径
未完成项和原因（例如 nsys 缺失、SageAttention2 编译失败而改用 v1）
按第 0 节规则自行做的决定（例如 $XDIT_WORK 的选择、跳过的项、等待 GPU 的情况）
```
