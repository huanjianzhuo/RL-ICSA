# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/krum_defence.py
# Description:  Krum防御
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class KrumDefence(ByzantineDefence):
    """
    Krum防御：选择最相似的模型
    
    论文: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    
    核心算法：
    1. 对每个工作节点 i，计算分数 s(i) = Σ ||V_i - V_j||²
       其中求和在 n-f-2 个最接近 V_i 的向量上
    2. 选择分数最小的模型：KR(V_1,...,V_n) = V_{i*}
       其中 i* 满足 s(i*) ≤ s(i) 对所有 i
    
    拜占庭弹性：当 2f + 2 ≤ n 时，Krum 满足 (α,f)-拜占庭弹性条件
    时间复杂度：O(n² · d)，其中 n 是向量数量，d 是向量维度
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.f = config.get('f', 1)  # 最多容忍f个拜占庭客户端
        
        # 验证配置是否满足 Krum 的要求
        n_clients = len(clients)
        if n_clients < 2 * self.f + 3:
            logger.warning(f"Krum要求至少 2f+3 个客户端，当前有 {n_clients} 个客户端，f={self.f}")
        
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        使用Krum选择模型
        
        根据论文算法：
        1. 计算所有模型对之间的平方距离矩阵
        2. 对每个模型 i，计算其与最近的 n-f-2 个模型的平方距离和
        3. 选择分数最小的模型作为聚合结果
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        n_models = len(client_models)
        if n_models < 2 * self.f + 3:
            # 如果客户端数量不满足 Krum 要求，使用简单平均
            logger.warning(f"Krum要求至少 {2*self.f+3} 个客户端，当前只有 {n_models} 个，回退到简单平均")
            from utilities import Utilities as Utils
            return Utils.average_client_models(client_models)
        
        # 计算每个模型与其他模型的平方距离
        distances = self._compute_distances(client_models)
        
        # 为每个模型计算Krum分数
        # 根据论文: s(i) = Σ_{j≠i} ||V_i - V_j||² 其中求和在 n-f-2 个最近的向量上
        krum_scores = []
        for i in range(n_models):
            # 选择最近的 n-f-2 个距离（排除自己，即跳过距离为0的第一个元素）
            sorted_distances = torch.sort(distances[i])[0]
            # 跳过索引0（与自己的距离），选择接下来的 n-f-2 个
            krum_score = torch.sum(sorted_distances[1:n_models - self.f - 1])
            krum_scores.append(krum_score)
        
        # 选择Krum分数最小的模型
        best_model_idx = torch.argmin(torch.stack(krum_scores))
        return client_models[best_model_idx].state_dict()
    
    def _compute_distances(self, client_models: List[torch.nn.Module]) -> torch.Tensor:
        """
        计算模型之间的平方距离矩阵
        
        返回 n×n 距离矩阵，其中 distances[i,j] = ||V_i - V_j||²
        """
        n_models = len(client_models)
        
        # 确定目标设备（使用第一个模型的设备）
        first_param = next(client_models[0].parameters())
        device = first_param.device
        
        # 初始化距离矩阵（对角线为0）
        distances = torch.zeros(n_models, n_models, device=device)
        
        # 计算上三角矩阵，然后对称复制
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # 计算平方距离
                dist = self._model_distance(client_models[i], client_models[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def _model_distance(self, model1: torch.nn.Module, model2: torch.nn.Module) -> torch.Tensor:
        """
        计算两个模型之间的平方欧氏距离
        
        根据论文: s(i) = Σ ||V_i - V_j||²
        将所有参数展平为一个向量，然后计算平方距离
        """
        # 获取设备
        first_param = next(model1.parameters())
        device = first_param.device
        
        # 展平所有参数为一个向量
        params1 = []
        params2 = []
        
        for (param1, param2) in zip(model1.parameters(), model2.parameters()):
            # 确保两个参数在同一设备上
            param1 = param1.to(device)
            param2 = param2.to(device)
            params1.append(param1.flatten())
            params2.append(param2.flatten())
        
        # 连接所有参数
        vec1 = torch.cat(params1)
        vec2 = torch.cat(params2)
        
        # 计算平方欧氏距离: ||v1 - v2||²
        squared_distance = torch.sum((vec1 - vec2) ** 2)
        
        return squared_distance
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Krum选择预测
        
        根据论文算法，计算每个预测的分数并选择最优预测
        """
        n_predictions = len(client_predictions)
        if n_predictions < 2 * self.f + 3:
            # 如果预测数量不满足 Krum 要求，使用简单平均
            logger.warning(f"Krum要求至少 {2*self.f+3} 个预测，当前只有 {n_predictions} 个，回退到简单平均")
            avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
            outlier_scores = torch.zeros(n_predictions)
            return avg_prediction, outlier_scores
        
        # 计算预测之间的平方距离
        stacked_predictions = torch.stack(client_predictions)
        # torch.cdist 默认计算欧氏距离，需要平方得到平方距离
        distances = torch.cdist(stacked_predictions, stacked_predictions, p=2).pow(2)
        
        # 为每个预测计算Krum分数
        # 根据论文: s(i) = Σ_{j≠i} ||V_i - V_j||² 其中求和在 n-f-2 个最近的向量上
        krum_scores = []
        for i in range(n_predictions):
            # 选择最近的 n-f-2 个距离（排除自己，即跳过距离为0的第一个元素）
            sorted_distances = torch.sort(distances[i])[0]
            # 跳过索引0（与自己的距离），选择接下来的 n-f-2 个
            krum_score = torch.sum(sorted_distances[1:n_predictions - self.f - 1])
            krum_scores.append(krum_score)
        
        # 选择Krum分数最小的预测
        best_prediction_idx = torch.argmin(torch.tensor(krum_scores))
        best_prediction = client_predictions[best_prediction_idx]
        
        # 计算异常分数
        outlier_scores = torch.tensor(krum_scores)
        outlier_scores = outlier_scores / torch.max(outlier_scores)
        
        return best_prediction, outlier_scores

