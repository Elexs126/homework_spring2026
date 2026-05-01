# Homework 1: Imitation Learning

## Setup

This project uses the conda environment `hw1-imitation`. Conda owns the environment and Python runtime.
Use `uv pip` only as the package installer inside that conda environment.

### Activate the environment

```powershell
conda activate hw1-imitation
```

If you need to add another package, install it into the active conda environment with:

```powershell
uv pip install --python "$env:CONDA_PREFIX\python.exe" <package-name>
```

Run scripts with the conda environment's Python:

```powershell
python src/hw1_imitation/train.py --help
```

## Weights & Biases (wandb) login

These assignments use [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. Before running a training script, you will need to log in to WandB using your API key.

```bash
wandb login
```

Follow the prompt to paste your API key.

## Using Modal

**Note that Modal is likely not necessary for this assignment. In testing, training was much faster on a local laptop CPU than on Modal. However, you may need to use Modal in future assignments, so if you want to get set up, here are the instructions:**

First, create a Modal account. You should recieve $30 in free credits, which will be plenty for this assignment. Then, you can train on Modal with the following command:

```bash
modal run src/hw1_imitation/modal_train.py
```

This will build a Modal container and launch training remotely. You can pass the same flags as the local training script. If you are logged into WandB locally, your API key will be automatically forwarded to the Modal container.

Logs and checkpoints will be saved to a Modal volume called `hw1-imitation-volume`. To inspect the logs, you can use:

```bash
modal volume ls hw1-imitation-volume exp
```

Then, you can download the logs and checkpoints to your local machine using a command like the following:

```bash
modal volume get hw1-imitation-volume exp/<experiment_name>
```
