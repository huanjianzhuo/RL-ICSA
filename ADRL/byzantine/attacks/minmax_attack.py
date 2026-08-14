# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/minmax_attack.py
# Description:  MinMax攻击
# ===========================================================================

import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from .base import CollusionAttackBase

logger = logging.getLogger(__name__)


class MinMaxAttack(CollusionAttackBase):
    """
    AGR-agnostic Min-Max 攻击（Attack-1）
    最小化最大距离攻击：确保恶意梯度到任意良性梯度的最大距离不超过良性梯度之间的最大距离
    
    优化目标：
    argmax_γ max_{i∈[n]} ||∇^m - ∇_i||_2 ≤ max_{i,j∈[n]} ||∇_i - ∇_j||_2
    
    策略：
    1. 收集所有良性客户端的梯度
    2. 计算良性梯度的平均值 g_b
    3. 选择扰动方向 p = -g_b (反向攻击)
    4. 对每个良性梯度 g_j，计算成对最大距离约束
    5. 求解二次不等式得到 gamma 上界
    6. 选择所有约束的最小值作为最终 gamma
    7. 生成恶意梯度：∇^m = g_b + γ * p
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.safety_margin = config.get('minmax_safety_margin', 0.95)  # 安全边界（避免边界条件）
    
    def _get_params_vector(self, model: torch.nn.Module) -> torch.Tensor:
        """将模型参数展平为一维向量"""
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    def _set_params_vector(self, model: torch.nn.Module, vec: torch.Tensor) -> None:
        """从一维向量恢复模型参数"""
        pointer = 0
        for p in model.parameters():
            n = p.numel()
            p.data = vec[pointer:pointer+n].view_as(p.data)
            pointer += n
    
    def _get_global_vector(self) -> torch.Tensor:
        """获取全局模型参数向量"""
        if hasattr(self.runner, 'server') and self.runner.server is not None:
            return torch.cat([p.data.view(-1) for p in self.runner.server.model.parameters()])
        return self._get_params_vector(self.clients[0].model)
    
    def _collect_benign_gradients(self) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        收集良性客户端的梯度信息
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, list]: 
                - old_vec: 全局模型参数向量
                - g_b: 良性梯度平均值
                - grads: 良性梯度列表
        """
        old_vec = self._get_global_vector().detach().cpu()
        grads = []
        
        for c in self.benign_clients:
            if getattr(c, 'model', None) is None:
                continue
            w = self._get_params_vector(c.model).detach().cpu()
            grads.append(w - old_vec)
        
        if not grads:
            return old_vec, None, []
        
        G = torch.stack(grads, dim=0)
        g_b = G.mean(dim=0)
        
        return old_vec, g_b, grads
    
    def _generate_attack_params(self, client, agent_id: int) -> torch.Tensor:
        """
        生成 Min-Max 攻击参数
        
        根据论文 Attack-1 (Min-Max):
        目标: argmax_γ max_{i∈[n]} ||∇^m - ∇_i||_2 ≤ max_{i,j∈[n]} ||∇_i - ∇_j||_2
        生成: ∇^m = f_avg(∇_{i∈[n]}) + γ∇^p
        
        核心思路：
        1. 计算良性梯度平均值 f_avg(∇_{i∈[n]}) = g_b
        2. 选择扰动方向 ∇^p = -g_b（反向攻击，最大化破坏）
        3. 对每个良性客户端 j，计算约束：
           ||(g_b + γ∇^p) - ∇_j||^2 ≤ D_j_max
           其中 D_j_max = max_i ||∇_i - ∇_j||^2
        4. 求解二次不等式得到 gamma 上界
        5. 选择所有约束的最小值（确保满足所有约束）
        6. 生成恶意梯度: ∇^m = g_b + γ∇^p
        """
        old_vec, g_b, grads = self._collect_benign_gradients()
        
        if g_b is None:
            return None
        
        # 扰动方向 ∇^p：选择负的良性平均梯度（反向攻击）
        # 这样可以最大化对模型收敛的破坏
        perturbation = -g_b
        p_norm2 = torch.dot(perturbation, perturbation).item() + 1e-12
        
        # 计算每个良性客户端的成对最大距离 D_j_max
        G = torch.stack(grads, dim=0)  # [m, d] 其中 m 是良性客户端数量
        D_j_max = []
        for j in range(G.size(0)):
            # 计算 max_i ||∇_i - ∇_j||^2
            diff = G - G[j]  # [m, d]
            D_j_max.append(torch.max((diff*diff).sum(dim=1)).item())
        
        # 对每个良性客户端 j 求解二次约束
        # 约束: ||(g_b + γ∇^p) - ∇_j||^2 ≤ D_j_max
        # 展开: ||g_b - ∇_j||^2 + 2γ(g_b - ∇_j)·∇^p + γ^2||∇^p||^2 ≤ D_j_max
        # 整理: a*γ^2 + b*γ + c ≤ 0
        gamma_bounds = []
        for j in range(G.size(0)):
            gb_minus_gj = (g_b - G[j]).detach()
            
            # 二次方程系数
            a = p_norm2  # ||∇^p||^2
            b = 2.0 * torch.dot(perturbation, gb_minus_gj).item()  # 2(g_b - ∇_j)·∇^p
            c = float(torch.dot(gb_minus_gj, gb_minus_gj).item() - D_j_max[j])  # ||g_b - ∇_j||^2 - D_j_max
            
            # 求解判别式
            disc = b*b - 4*a*c
            if disc <= 0 or a <= 0:
                gamma_bounds.append(0.0)
                continue
            
            # 求两个根，选择较大的作为上界
            root1 = (-b - np.sqrt(disc)) / (2*a)
            root2 = (-b + np.sqrt(disc)) / (2*a)
            upper = max(root1, root2)
            gamma_bounds.append(max(0.0, upper))
        
        if not gamma_bounds:
            return None
        
        # 选择最小的 gamma（满足所有约束）
        gamma = max(0.0, min(gamma_bounds))
        gamma = float(gamma * self.safety_margin)  # 应用安全边界，避免数值问题
        
        # 生成恶意梯度: ∇^m = f_avg(∇_{i∈[n]}) + γ∇^p
        malicious_gradient = g_b + gamma * perturbation
        
        # 最终恶意参数: w^m = w_old + ∇^m
        crafted = old_vec + malicious_gradient
        
        return crafted



