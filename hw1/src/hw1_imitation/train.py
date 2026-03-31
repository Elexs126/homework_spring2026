"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 512
    lr: float = 1.2e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 10
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    # 1. 定义优化器：负责根据 Loss 更新模型的参数
    # 使用 Adam 优化器，它是深度学习中最通用的“自动驾驶”优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    def lr_lambda(current_epoch):
        if current_epoch < config.warmup_epochs:
            # 在预热期内，学习率从极小值线性增加到 config.lr
            return float(current_epoch) / float(max(1, config.warmup_epochs))
        # 预热结束后保持 config.lr (或者你也可以在这里加后续的 Decay)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # -----------------------------
    global_step = 0
    # 开始漫长的训练季节 (Epochs)
    for epoch in range(config.num_epochs):
        model.train()  # 开启训练模式
        epoch_losses = []

        # 2. 从流水线 (DataLoader) 拿出一批批数据
        for batch in loader:
            # --- 把列表里的东西按顺序掏出来 ---
            state, action = batch
            
            # 搬运工：把数据从 CPU 搬到显卡 GPU 上
            state = state.to(device)
            action = action.to(device)

            # --- 核心三步走 ---
            # (1) 算分：调用你刚才在 model.py 写的 compute_loss
            loss = model.compute_loss(state, action)

            # (2) 清空：擦掉上一次留下的梯度残余
            optimizer.zero_grad()

            # (3) 进化：反向传播计算梯度，并更新参数
            loss.backward()
            optimizer.step()
            # ----------------

            epoch_losses.append(loss.item())
            global_step += 1

            # 定期向 WandB 汇报进度
            if global_step % config.log_interval == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train/learning_rate": current_lr, "epoch": epoch}, step=global_step)

        # 3. 定期“大考”：运行评估脚本看看机器人推得怎么样
        if epoch % 50 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch}: Mean Loss = {np.mean(epoch_losses):.4f}")
            # 调用框架自带的评估函数，这会生成推 T 型块的视频
            from hw1_imitation.evaluation import evaluate_policy
            evaluate_policy(
                model=model,                       # 你的模型大脑
                normalizer=normalizer,             # 在 line 94 定义的标准化器
                device=device,                     # 计算设备
                chunk_size=config.chunk_size,      # 动作块大小
                video_size=config.video_size,      # 视频尺寸 (从 config 拿)
                num_video_episodes=config.num_video_episodes, # 测试集数
                flow_num_steps=config.flow_num_steps, # 流匹配步数
                step=global_step,                  # 当前训练到了第几步
                logger=logger                      # 在 line 124 定义的日志器
            )
            

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
