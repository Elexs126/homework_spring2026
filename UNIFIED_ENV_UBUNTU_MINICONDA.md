# Ubuntu + Miniconda：一键复刻“全仓库单环境”

目标：把本仓库在 **一个** Conda 环境里装齐依赖（含 GPU Torch、ogbench、Gym、MuJoCo），并通过关键脚本的 `--help` smoke test；对方只需要照抄命令，不需要自己试探缺什么包。

> 说明：本方案使用 `uv` 来安装/解析 Python 依赖，但虚拟环境本身由 Miniconda 创建。

## 0) 系统前置（Ubuntu）

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl build-essential unzip ffmpeg \
  libgl1 libglib2.0-0 libxrender1 libsm6 libxext6
```

如果你要用 GPU Torch：确保 `nvidia-smi` 可用（已装好 NVIDIA driver）。

## 1) Clone 仓库

```bash
git clone <YOUR_REPO_URL>
cd "CS 285 HW"  # 或你的仓库目录
```

## 2) 创建并激活 Conda 环境（单环境）

推荐用 Python 3.11（与当前统一环境一致）：

```bash
conda create -y -n cs285 python=3.11 pip
conda activate cs285
```

## 3) 安装 uv（只用于安装 uv 本体）

```bash
python -m pip install -U uv
```

## 4) 运行一键脚本（安装 + 打补丁 + 快速验收）

```bash
bash homework_spring2026/scripts/setup_unified_env_ubuntu_miniconda.sh
```

脚本会：
- 用 `uv sync` 把 `hw1` 的锁文件依赖（含 `torch==...+cu126`）装进当前 conda env
- 补齐跨作业需要的核心包（gym/numpy<2/ogbench/transformers/peft/modal/matplotlib/...）
- 自动修复 Gym 0.25 的 Hopper/Walker2d MuJoCo 模型在 `mujoco>=3.x` 下的兼容性问题（会写 `.orig` 备份）
- 运行一组非常快的 sanity checks（torch CUDA、几个 Gym 环境可创建、ogbench 可导入、若干 `--help` 入口）

## 5) （可选）安装 Atari + ROM（仅 hw3 的 MsPacman 需要）

HW3 的 `experiments/dqn/mspacman.yaml` 需要 Atari 依赖和 ROM。

```bash
export PY="$CONDA_PREFIX/bin/python"
export UV_HTTP_TIMEOUT=600

uv pip install --python "$PY" "gym[atari,accept-rom-license]==0.25.2"

# 下载并安装 ROM（AutoROM 在上一步会被安装）
AutoROM --accept-license

# 快速验证
python -c "import gym; env=gym.make('MsPacmanNoFrameskip-v4'); env.reset(); env.close(); print('MsPacman OK')"
```

如果你不需要跑 MsPacman，可以跳过本节。
