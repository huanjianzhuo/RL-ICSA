# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/momentum_screening_defence.py
# Description:  Momentum Screening 防御方法
# ===========================================================================

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class MomentumScreeningDefence(ByzantineDefence):
    """
    Momentum Screening防御：基于动量筛选的拜占庭鲁棒聚合
    
    论文: "SIMPLE MINIMAX OPTIMAL BYZANTINE ROBUST ALGORITHM FOR NONCONVEX 
           OBJECTIVES WITH UNIFORM GRADIENT HETEROGENEITY"
    
    核心算法：
    1. 每个客户端维护动量：m_i^t = (1-α)m_i^{t-1} + αg_i^t
    2. 服务器筛选客户端：保留至少与一半客户端距离≤τ的客户端
    3. 仅聚合筛选后的客户端动量
    
    关键特性：
    - 基于动量的鲁棒性：平滑梯度噪声
    - 自适应筛选：自动检测异常客户端
    - 理论保证：在异构数据下达到最优统计率
    
    配置参数：
    - momentum_alpha (float): 动量系数 α ∈ [0,1]，默认0.9
    - screening_tau (float): 筛选阈值 τ，默认自动计算
    - auto_tune_tau (bool): 是否自动调整τ，默认True
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        
        # 动量系数 (论文中的 α)
        self.alpha = config.get('momentum_alpha', 0.9)
        
        # 筛选阈值 (论文中的 τ)
        self.tau = config.get('screening_tau', None)
        self.auto_tune_tau = config.get('auto_tune_tau', True)
        
        # 拜占庭客户端数量 (论文中的 δn)
        self.byzantine_ratio = config.get('byzantine_ratio', 0.3)
        
        # 动量存储：每个客户端的动量参数
        self.client_momentums = {}  # Dict[int, OrderedDict]
        
        # 全局模型参数（用于计算梯度）
        self.previous_global_state = None
        
        # 当前轮次
        self.round_num = 0
        
        logger.info(f"初始化Momentum Screening防御: alpha={self.alpha}, "
                   f"tau={'auto' if self.auto_tune_tau else self.tau}, "
                   f"byzantine_ratio={self.byzantine_ratio}")
    
    def _get_model_update(self, current_model: torch.nn.Module, 
                         previous_state: OrderedDict) -> OrderedDict:
        """
        计算模型更新（梯度）：Δ = current - previous
        
        Args:
            current_model: 当前模型
            previous_state: 上一轮的全局模型状态
            
        Returns:
            模型更新的状态字典
        """
        update = OrderedDict()
        current_state = current_model.state_dict()
        
        for key in current_state.keys():
            if current_state[key].dtype.is_floating_point:
                # 确保两个张量在同一设备上（修复设备不匹配问题）
                current_param = current_state[key]
                previous_param = previous_state[key].to(device=current_param.device, dtype=current_param.dtype)
                # 计算梯度：Δ = x_current - x_previous
                update[key] = current_param - previous_param
            else:
                # 非浮点参数（如 BatchNorm 的 num_batches_tracked）直接复制
                update[key] = current_state[key]
        
        return update
    
    def _update_momentum(self, client_idx: int, gradient: OrderedDict) -> OrderedDict:
        """
        更新客户端动量：m_i^t = (1-α)m_i^{t-1} + αg_i^t
        
        Args:
            client_idx: 客户端索引
            gradient: 当前梯度
            
        Returns:
            更新后的动量
        """
        if client_idx not in self.client_momentums:
            # 第一轮：m_i^0 = g_i^0
            self.client_momentums[client_idx] = gradient
            return gradient
        
        # 后续轮次：m_i^t = (1-α)m_i^{t-1} + αg_i^t
        momentum = OrderedDict()
        prev_momentum = self.client_momentums[client_idx]
        
        for key in gradient.keys():
            if gradient[key].dtype.is_floating_point:
                # 确保动量和梯度在同一设备上
                grad = gradient[key]
                prev_mom = prev_momentum[key].to(device=grad.device, dtype=grad.dtype)
                momentum[key] = (1 - self.alpha) * prev_mom + self.alpha * grad
            else:
                momentum[key] = gradient[key]
        
        self.client_momentums[client_idx] = momentum
        return momentum
    
    def _flatten_state(self, state_dict: OrderedDict) -> torch.Tensor:
        """
        将状态字典展平为一维向量
        
        Args:
            state_dict: 模型状态字典
            
        Returns:
            展平后的向量
        """
        flat_params = []
        target_device = None
        
        for key, param in state_dict.items():
            if param.dtype.is_floating_point:
                # 确定目标设备（使用第一个参数的设备）
                if target_device is None:
                    target_device = param.device
                # 确保所有参数在同一设备上
                flat_params.append(param.flatten().to(device=target_device))
        
        if not flat_params:
            # 如果没有浮点参数，返回空张量
            return torch.tensor([], device=target_device if target_device else 'cpu')
        
        return torch.cat(flat_params)
    
    def _compute_pairwise_distances(self, momentums: List[OrderedDict]) -> torch.Tensor:
        """
        计算所有动量对之间的L2距离
        
        Args:
            momentums: 客户端动量列表
            
        Returns:
            距离矩阵 D，其中 D[i,j] = ||m_i - m_j||_2
        """
        n = len(momentums)
        
        # 展平所有动量
        flat_momentums = [self._flatten_state(m) for m in momentums]
        
        # 确定目标设备（使用第一个动量的设备）
        target_device = flat_momentums[0].device if flat_momentums else 'cpu'
        
        # 确保所有展平的动量在同一设备上
        flat_momentums = [fm.to(device=target_device) for fm in flat_momentums]
        
        # 堆叠成矩阵
        M = torch.stack(flat_momentums, dim=0)  # Shape: (n, d)
        
        # 计算成对距离：||m_i - m_j||_2
        # 使用 torch.cdist 高效计算
        distances = torch.cdist(M, M, p=2)  # Shape: (n, n)
        
        return distances
    
    def _auto_compute_tau(self, distances: torch.Tensor) -> float:
        """
        自动计算筛选阈值 τ
        
        根据论文，τ 应该能够区分正常客户端和拜占庭客户端。
        策略：使用距离矩阵的中位数或基于拜占庭比例的分位数。
        
        Args:
            distances: 距离矩阵
            
        Returns:
            自动计算的阈值
        """
        # 提取上三角距离（排除对角线和重复）
        n = distances.size(0)
        triu_indices = torch.triu_indices(n, n, offset=1)
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]
        
        # 根据拜占庭比例计算分位数
        # 如果有30%拜占庭客户端，我们希望τ能够容纳70%的正常距离
        quantile = 1.0 - self.byzantine_ratio
        tau = torch.quantile(pairwise_dists, quantile).item()
        
        logger.debug(f"自动计算τ = {tau:.6f} (分位数 {quantile:.2f})")
        
        return tau
    
    def _screen_clients(self, momentums: List[OrderedDict], 
                       distances: torch.Tensor, tau: float) -> List[int]:
        """
        执行Screen算法：筛选出可信客户端
        
        算法 2 (Screen):
        G̃ = {i ∈ [n] : |{j ∈ [n] : ||m_i - m_j|| ≤ τ}| ≥ 0.5n}
        
        Args:
            momentums: 客户端动量列表
            distances: 距离矩阵
            tau: 筛选阈值
            
        Returns:
            筛选后的客户端索引列表
        """
        n = len(momentums)
        threshold_count = n * 0.5  # 至少一半客户端
        
        screened_indices = []
        
        for i in range(n):
            # 计算与客户端 i 距离≤τ的客户端数量
            close_count = (distances[i] <= tau).sum().item()
            
            # 如果至少一半客户端与 i 接近，则保留客户端 i
            if close_count >= threshold_count:
                screened_indices.append(i)
        
        logger.debug(f"筛选结果: {len(screened_indices)}/{n} 客户端通过 (τ={tau:.6f})")
        
        # 如果筛选后客户端太少，放宽条件
        if len(screened_indices) < n * 0.3:
            logger.warning(f"筛选后客户端数量过少 ({len(screened_indices)}/{n})，"
                          f"使用所有客户端")
            return list(range(n))
        
        return screened_indices
    
    def _aggregate_momentums(self, momentums: List[OrderedDict], 
                            screened_indices: List[int]) -> OrderedDict:
        """
        聚合筛选后的客户端动量
        
        m = (1/|G̃|) Σ_{i∈G̃} m_i
        
        Args:
            momentums: 所有客户端动量
            screened_indices: 筛选后的客户端索引
            
        Returns:
            聚合后的动量
        """
        if not screened_indices:
            raise ValueError("筛选后没有可用的客户端")
        
        # 选择筛选后的动量
        selected_momentums = [momentums[i] for i in screened_indices]
        
        # 计算平均
        aggregated = OrderedDict()
        template = selected_momentums[0]
        
        for key in template.keys():
            if template[key].dtype.is_floating_point:
                # 确定目标设备（使用第一个动量的设备）
                target_device = template[key].device
                # 确保所有动量在同一设备上，然后堆叠并计算平均
                momentum_params = [m[key].to(device=target_device) for m in selected_momentums]
                stacked = torch.stack(momentum_params, dim=0)
                aggregated[key] = torch.mean(stacked, dim=0)
            else:
                # 非浮点参数，使用第一个
                aggregated[key] = template[key]
        
        return aggregated
    
    def _apply_momentum_to_model(self, base_state: OrderedDict, 
                                 momentum: OrderedDict, 
                                 learning_rate: float = 1.0) -> OrderedDict:
        """
        将动量应用到模型：x^{t+1} = x^t - η * m^t
        
        Args:
            base_state: 当前全局模型状态
            momentum: 聚合后的动量
            learning_rate: 学习率 η
            
        Returns:
            更新后的模型状态
        """
        new_state = OrderedDict()
        
        for key in base_state.keys():
            if base_state[key].dtype.is_floating_point and key in momentum:
                # 梯度下降更新：x = x - η * m
                new_state[key] = base_state[key] - learning_rate * momentum[key]
            else:
                new_state[key] = base_state[key]
        
        return new_state
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> OrderedDict:
        """
        使用Momentum Screening聚合客户端模型
        
        算法流程：
        1. 获取上一轮的全局模型状态
        2. 计算每个客户端的模型更新（梯度）
        3. 更新每个客户端的动量
        4. 计算所有动量对之间的距离
        5. 执行Screen算法筛选客户端
        6. 聚合筛选后的客户端动量
        7. 应用动量更新全局模型
        
        Args:
            client_models: 客户端模型列表
            
        Returns:
            聚合后的模型状态字典
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        n_clients = len(client_models)
        self.round_num += 1
        
        logger.info(f"=== Momentum Screening 第 {self.round_num} 轮聚合 ===")
        logger.info(f"客户端数量: {n_clients}")
        
        # 1. 获取上一轮的全局模型状态（作为基线）
        if self.previous_global_state is None:
            # 第一轮：使用第一个客户端模型作为基线
            self.previous_global_state = copy.deepcopy(client_models[0].state_dict())
            logger.info("初始化全局模型状态")
        
        # 2. 计算每个客户端的模型更新（梯度）
        gradients = []
        for i, model in enumerate(client_models):
            gradient = self._get_model_update(model, self.previous_global_state)
            gradients.append(gradient)
        
        logger.debug(f"计算了 {len(gradients)} 个客户端的梯度")
        
        # 3. 更新每个客户端的动量
        momentums = []
        for i, gradient in enumerate(gradients):
            momentum = self._update_momentum(i, gradient)
            momentums.append(momentum)
        
        logger.debug(f"更新了 {len(momentums)} 个客户端的动量")
        
        # 4. 计算所有动量对之间的距离
        distances = self._compute_pairwise_distances(momentums)
        
        # 5. 确定筛选阈值 τ
        if self.auto_tune_tau or self.tau is None:
            tau = self._auto_compute_tau(distances)
        else:
            tau = self.tau
        
        # 6. 执行Screen算法筛选客户端
        screened_indices = self._screen_clients(momentums, distances, tau)
        
        logger.info(f"筛选结果: {len(screened_indices)}/{n_clients} 客户端通过筛选")
        logger.info(f"筛选客户端索引: {screened_indices}")
        
        # 7. 聚合筛选后的客户端动量
        aggregated_momentum = self._aggregate_momentums(momentums, screened_indices)
        
        # 8. 应用动量更新全局模型
        # 注意：这里假设客户端模型已经是 x + η*Δ 的形式
        # 所以我们需要取平均而不是应用动量
        # 实际上，根据算法1，服务器应该维护全局参数并应用聚合后的动量
        # 但在联邦学习框架中，通常是聚合客户端模型而不是梯度
        # 因此这里我们采用折中方案：直接聚合筛选后的客户端模型
        
        aggregated_state = OrderedDict()
        selected_models = [client_models[i] for i in screened_indices]
        
        template_state = selected_models[0].state_dict()
        for key in template_state.keys():
            if template_state[key].dtype.is_floating_point:
                # 确定目标设备（使用第一个模型的设备）
                target_device = template_state[key].device
                # 确保所有模型参数在同一设备上，然后堆叠并计算平均
                params = [m.state_dict()[key].to(device=target_device) for m in selected_models]
                stacked = torch.stack(params, dim=0)
                aggregated_state[key] = torch.mean(stacked, dim=0)
            else:
                aggregated_state[key] = template_state[key]
        
        # 更新全局模型状态
        self.previous_global_state = copy.deepcopy(aggregated_state)
        
        logger.info(f"=== 聚合完成 ===")
        
        return aggregated_state
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Momentum Screening聚合预测
        
        由于预测不是序列数据，这里简化为基于距离的筛选
        
        Args:
            client_predictions: 客户端预测列表
            
        Returns:
            (聚合后的预测, 异常分数)
        """
        if not client_predictions:
            raise ValueError("客户端预测列表为空")
        
        n = len(client_predictions)
        
        # 确定目标设备
        target_device = client_predictions[0].device if client_predictions else 'cpu'
        
        # 确保所有预测在同一设备上
        client_predictions = [p.to(device=target_device) for p in client_predictions]
        
        # 堆叠预测
        stacked_preds = torch.stack(client_predictions)  # Shape: (n, ...)
        
        # 展平预测以计算距离
        flat_preds = [p.flatten() for p in client_predictions]
        M = torch.stack(flat_preds, dim=0)  # Shape: (n, d)
        
        # 计算成对距离
        distances = torch.cdist(M, M, p=2)
        
        # 自动计算阈值
        triu_indices = torch.triu_indices(n, n, offset=1)
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]
        tau = torch.quantile(pairwise_dists, 0.7).item()
        
        # 筛选客户端
        threshold_count = n * 0.5
        screened_indices = []
        outlier_scores = torch.zeros(n)
        
        for i in range(n):
            close_count = (distances[i] <= tau).sum().item()
            if close_count >= threshold_count:
                screened_indices.append(i)
                outlier_scores[i] = 0.0
            else:
                outlier_scores[i] = 1.0
        
        # 如果筛选后太少，使用所有客户端
        if len(screened_indices) < n * 0.3:
            screened_indices = list(range(n))
            outlier_scores = torch.zeros(n)
        
        # 聚合筛选后的预测
        selected_preds = stacked_preds[screened_indices]
        aggregated_pred = torch.mean(selected_preds, dim=0)
        
        return aggregated_pred, outlier_scores

