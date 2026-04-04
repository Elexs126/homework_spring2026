#!/usr/bin/env bash
set -euo pipefail

# One-shot unified environment setup for Ubuntu + Miniconda.
# - Assumes: repo is already cloned; conda env is activated.
# - Installs dependencies into the ACTIVE conda environment.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: No active conda env detected. Run: conda activate cs285" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$CONDA_PREFIX/bin/python"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

echo "[info] repo_root=$REPO_ROOT"
echo "[info] python=$PY"

# Ensure uv is available inside the conda env (bootstrap only).
if ! command -v uv >/dev/null 2>&1; then
  echo "[step] install uv"
  "$PY" -m pip install -U uv
fi

# 1) Base install from HW1 lock (includes GPU torch index/source via pyproject + uv.lock)
echo "[step] uv sync (hw1) -> active conda env"
(
  cd "$REPO_ROOT/homework_spring2026/hw1"
  uv sync --active --frozen --inexact --refresh-package torch
)

# 2) Union of cross-homework deps (pinned to the set validated in the unified env)
#    Keep gym at 0.25.2 and numpy<2 for compatibility.
echo "[step] install unified deps"
uv pip install --python "$PY" \
  "numpy==1.26.4" \
  "gym==0.25.2" \
  "absl-py==2.4.0" \
  "ml-collections==1.1.0" \
  "wandb==0.23.1" \
  "mujoco==3.1.6" \
  "ogbench==1.2.1" \
  "transformers==4.56.2" \
  "peft==0.18.1" \
  "modal==1.4.1" \
  "matplotlib==3.10.8" \
  "gradescope-utils==0.5.0" \
  "opencv-python==4.11.0.86" \
  "box2d==2.3.10"

# 3) Patch Gym MuJoCo assets (Hopper/Walker2d) for mujoco>=3.x compatibility.
echo "[step] patch gym mujoco assets"
"$PY" "$REPO_ROOT/homework_spring2026/scripts/patch_gym_mujoco_global_coords.py"

# 4) Quick sanity checks (no heavy training runs)
echo "[check] torch cuda"
"$PY" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda)"

echo "[check] gym envs"
"$PY" -c "import gym; env=gym.make('CartPole-v1'); env.reset(); env.close(); print('CartPole-v1 OK')"
"$PY" -c "import gym; env=gym.make('LunarLander-v2'); env.reset(); env.close(); print('LunarLander-v2 OK')"
"$PY" -c "import gym; env=gym.make('HalfCheetah-v4'); env.reset(); env.close(); print('HalfCheetah-v4 OK')"
"$PY" -c "import gym; env=gym.make('Hopper-v4'); env.reset(); env.close(); print('Hopper-v4 OK')"
"$PY" -c "import gym; env=gym.make('InvertedPendulum-v4'); env.reset(); env.close(); print('InvertedPendulum-v4 OK')"
"$PY" -c "import gym; env=gym.make('Walker2d-v4'); env.reset(); env.close(); print('Walker2d-v4 OK')"

echo "[check] ogbench import"
"$PY" -c "import ogbench; print('ogbench_import OK')"

echo "[check] entrypoints --help"
# Avoid writing __pycache__ during checks
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$REPO_ROOT/homework_spring2026/hw1/src/hw1_imitation/train.py" --help >/dev/null
(
  cd "$REPO_ROOT/homework_spring2026/hw2"
  PYTHONPATH=src "$PY" src/scripts/run.py --help >/dev/null
)
(
  cd "$REPO_ROOT/homework_spring2026/hw3"
  PYTHONPATH=src "$PY" src/scripts/run_dqn.py --help >/dev/null
  PYTHONPATH=src "$PY" src/scripts/run_sac.py --help >/dev/null
)
(
  cd "$REPO_ROOT/homework_spring2026/hw4"
  "$PY" -m hw4.train --help >/dev/null
)
(
  cd "$REPO_ROOT/homework_spring2026/hw5"
  PYTHONPATH=src "$PY" src/scripts/run.py --help >/dev/null
)
(
  cd "$REPO_ROOT/homework_spring2026/final_project_offline_online/problem"
  PYTHONPATH=src "$PY" src/scripts/run.py --help >/dev/null
)
(
  cd "$REPO_ROOT/homework_spring2026/final_project_llm_rl"
  "$PY" -m llm_rl_final_proj.train --help >/dev/null
  "$PY" -m llm_rl_final_proj.reward_model.train --help >/dev/null
  "$PY" -m llm_rl_final_proj.online.train_rm_grpo --help >/dev/null
)

echo "[done] unified env ready"
