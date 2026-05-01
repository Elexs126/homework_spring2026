"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:# 返回值注解：指明该函数执行完毕后产出的数据类型，即明确该功能的“输出规格”或数学上的“值域”。
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),# 隐藏层架构：默认 2 层 128 宽；数字个数 = 层数，数字大小 = 该层宽度。
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):# 处理中间隐藏层，响应可变架构需求
            layers += [nn.Linear(hidden_dims[i - 1], hidden_dims[i]), nn.ReLU()]
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        self.MLP = nn.Sequential(*layers) # 将上述层按顺序封装为完整的 MLP 策略网络


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions = self.MLP(state)#上方的初始化会使用MLP根据输入给出预测呃出。
        target_actions = action_chunk.flatten(start_dim=1)#为了计算loss对齐,将目标动作展平为与预测动作相同的形状.
        return nn.functional.mse_loss(pred_actions, target_actions)
        
       

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
      pred_actions = self.MLP(state)
      return pred_actions.reshape(-1, self.chunk_size, self.action_dim)#将预测动作重新调整为(batch, chunk_size, action_dim)的形状，以符合输出规格。
    



class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        combined_input_dim = state_dim + (chunk_size * action_dim) + 1
        layers = [nn.Linear(combined_input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers += [nn.Linear(hidden_dims[i - 1], hidden_dims[i]), nn.ReLU()]
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        self.MLP = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x1 = action_chunk.flatten(start_dim=1)  # (batch, chunk_size * action_dim).
        x0 = torch.randn_like(x1)
        t  = torch.rand(batch_size, 1, device=state.device)   
        xt = (1 - t)*x0 + t * x1
        model_input = torch.cat([state, xt, t], dim=1)
        pred_vt = self.MLP(model_input)
        target_velocity = x1 - x0
        return nn.functional.mse_loss(pred_vt, target_velocity)
        

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.randn(batch_size, self.chunk_size * self.action_dim, device=state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,1), i * dt, device=state.device)
            model_input = torch.cat([state, x, t], dim=1)       
            pred_velocity = self.MLP(model_input)
            x = x + pred_velocity * dt
        return x.reshape(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )#关键字参数，防呆设计：明确每个参数的含义，避免传参时顺序错误导致的bug。
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
