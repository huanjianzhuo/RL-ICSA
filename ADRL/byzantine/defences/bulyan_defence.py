# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/bulyan_defence.py
# Description:  Bulyan防御
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class BulyanDefence(ByzantineDefence):
    """
    Bulyan防御：拜占庭弹性聚合
    
    论文: "The Hidden Vulnerability of Distributed Learning in Byzantium"
    
    核心算法：
    1. 要求 n ≥ 4f + 3 个客户端
    2. 递归地使用拜占庭弹性聚合规则 A（如 Krum）选择 θ = n - 2f 个梯度：
       - 步骤1：用 A 选择最接近 A 输出的向量
       - 步骤2：从接收集合中移除该向量，加入选中集合 S
       - 步骤3：循环直到 |S| = θ
    3. 对选中的 θ 个梯度，计算每个坐标的中位数（coordinate-wise median）
    
    关键特性：
    - 确保每个坐标都被大多数非拜占庭向量所认可
    - (α,f)-拜占庭弹性且收敛
    - 时间复杂度：O(n² · d)（当 A 是 Krum 或 GeoMed 时）
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.f = config.get('f', 1)
        self.aggregation_rule = config.get('bulyan_rule', 'krum')  # 使用的拜占庭弹性聚合规则
        
        # 验证配置是否满足 Bulyan 的要求
        n_clients = len(clients)
        if n_clients < 4 * self.f + 3:
            logger.warning(f"Bulyan要求至少 4f+3 = {4*self.f+3} 个客户端，当前有 {n_clients} 个客户端，f={self.f}")
        
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        使用Bulyan聚合模型
        
        根据论文算法：
        1. 递归地使用 Krum 选择 θ = n - 2f 个模型
        2. 对选中的模型，计算每个参数坐标的中位数
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        n_models = len(client_models)
        theta = n_models - 2 * self.f  # 选择的模型数量
        
        # 验证是否满足 Bulyan 要求
        if n_models < 4 * self.f + 3:
            logger.warning(f"Bulyan要求至少 {4*self.f+3} 个客户端，当前只有 {n_models} 个，回退到简单平均")
            from utilities import Utilities as Utils
            return Utils.average_client_models(client_models)
        
        # 步骤1：递归地使用 Krum 选择 θ = n - 2f 个模型
        selected_models = self._bulyan_select(client_models, theta)
        
        # 步骤2：对选中的 θ 个模型，计算每个参数坐标的中位数
        aggregated_state = self._coordinate_wise_median(selected_models)
        
        return aggregated_state
    
    def _bulyan_select(self, client_models: List[torch.nn.Module], theta: int) -> List[torch.nn.Module]:
        """
        使用 Bulyan 递归选择算法选择 θ 个模型
        
        根据论文算法：
        1. 初始化：接收集合 = 所有模型，选中集合 S = 空
        2. 循环 θ 次：
           a. 用 Krum 在当前接收集合上计算最优模型
           b. 找到最接近 Krum 输出的模型（可以是 Krum 自己选的）
           c. 将该模型从接收集合移到选中集合
        3. 返回选中的 θ 个模型
        """
        remaining_models = list(client_models)  # 接收集合
        selected_models = []  # 选中集合 S
        
        # 循环选择 θ 个模型
        for iteration in range(theta):
            if len(remaining_models) == 0:
                break
            
            # 使用 Krum 在当前接收集合上选择一个模型
            selected_model = self._krum_select_one(remaining_models)
            
            # 将选中的模型加入选中集合
            selected_models.append(selected_model)
            
            # 从接收集合中移除该模型
            remaining_models.remove(selected_model)
        
        return selected_models
    
    def _krum_select_one(self, candidate_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        使用 Krum 从候选模型中选择一个最优模型
        
        计算每个模型的 Krum 分数，返回分数最小的模型
        """
        n_models = len(candidate_models)
        
        if n_models == 1:
            return candidate_models[0]
        
        # 计算距离矩阵
        distances = self._compute_distances(candidate_models)
        
        # 计算每个模型的 Krum 分数
        krum_scores = []
        for i in range(n_models):
            # 选择最近的 n-f-2 个距离（排除自己）
            sorted_distances = torch.sort(distances[i])[0]
            # 确保不会超出索引范围
            n_neighbors = min(n_models - self.f - 2, n_models - 1)
            if n_neighbors > 0:
                krum_score = torch.sum(sorted_distances[1:n_neighbors + 1])
            else:
                krum_score = torch.sum(sorted_distances[1:])
            krum_scores.append(krum_score)
        
        # 选择 Krum 分数最小的模型
        best_idx = torch.argmin(torch.stack(krum_scores))
        return candidate_models[best_idx]
    
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
    
    def _coordinate_wise_median(self, selected_models: List[torch.nn.Module]) -> OrderedDict:
        """
        对选中的 θ 个模型计算每个参数坐标的中位数
        
        根据论文：G[i] = median({G[j][i] for j in selected models})
        其中 median[i] = argmin_{m∈M[i]} (Σ |Z[i] - m|)
        
        简单来说：对每个参数的每个坐标，计算所有选中模型在该坐标上的中位数
        
        Args:
            selected_models: 选中的 θ 个模型
        
        Returns:
            aggregated_state: 聚合后的模型参数（每个坐标是中位数）
        """
        if not selected_models:
            raise ValueError("选中的模型列表为空")
        
        # 获取第一个模型作为模板，并确定目标设备
        template_model = selected_models[0]
        first_param = next(template_model.parameters())
        device = first_param.device
        
        aggregated_state = OrderedDict()
        
        # 对每个参数进行坐标中位数聚合
        for param_name in template_model.state_dict().keys():
            param_values = []
            for model in selected_models:
                # 确保参数在同一设备上
                param = model.state_dict()[param_name].to(device)
                param_values.append(param.unsqueeze(0))
            
            # 堆叠所有参数值：shape = [θ, param_shape...]
            stacked_params = torch.cat(param_values, dim=0)
            
            # 保存原始数据类型
            original_dtype = stacked_params.dtype
            
            # 如果是整数类型，先转换为浮点数
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                stacked_params = stacked_params.float()
            
            # 计算坐标中位数：沿第0维（模型维度）计算中位数
            median_params = torch.median(stacked_params, dim=0)[0]
            
            # 如果原始参数是整数类型，转换回整数
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                median_params = median_params.round().to(original_dtype)
            
            aggregated_state[param_name] = median_params
        
        return aggregated_state
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Bulyan聚合预测
        
        根据论文算法：
        1. 递归地使用 Krum 选择 θ = n - 2f 个预测
        2. 对选中的预测，计算每个坐标的中位数
        """
        n_predictions = len(client_predictions)
        theta = n_predictions - 2 * self.f
        
        if n_predictions < 4 * self.f + 3:
            logger.warning(f"Bulyan要求至少 {4*self.f+3} 个预测，当前只有 {n_predictions} 个，回退到简单平均")
            avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
            outlier_scores = torch.zeros(n_predictions)
            return avg_prediction, outlier_scores
        
        # 步骤1：递归地使用 Krum 选择 θ = n - 2f 个预测
        selected_predictions = self._bulyan_select_predictions(client_predictions, theta)
        
        # 步骤2：计算所有预测的坐标中位数
        stacked_selected = torch.stack(selected_predictions)
        median_prediction = torch.median(stacked_selected, dim=0)[0]
        
        # 计算异常分数：基于每个预测与中位数的距离
        stacked_all = torch.stack(client_predictions)
        distances = torch.norm(stacked_all - median_prediction.unsqueeze(0), dim=-1)
        outlier_scores = distances / (torch.max(distances) + 1e-10)
        
        return median_prediction, outlier_scores
    
    def _bulyan_select_predictions(self, client_predictions: List[torch.Tensor], theta: int) -> List[torch.Tensor]:
        """
        使用 Bulyan 递归选择算法选择 θ 个预测
        
        与模型选择类似，但操作对象是预测向量
        """
        remaining_predictions = list(client_predictions)  # 接收集合
        selected_predictions = []  # 选中集合 S
        
        # 循环选择 θ 个预测
        for iteration in range(theta):
            if len(remaining_predictions) == 0:
                break
            
            # 使用 Krum 在当前接收集合上选择一个预测
            selected_prediction = self._krum_select_one_prediction(remaining_predictions)
            
            # 将选中的预测加入选中集合
            selected_predictions.append(selected_prediction)
            
            # 从接收集合中移除该预测
            remaining_predictions.remove(selected_prediction)
        
        return selected_predictions
    
    def _krum_select_one_prediction(self, candidate_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        使用 Krum 从候选预测中选择一个最优预测
        """
        n_predictions = len(candidate_predictions)
        
        if n_predictions == 1:
            return candidate_predictions[0]
        
        # 计算预测之间的平方距离矩阵
        stacked_predictions = torch.stack(candidate_predictions)
        distances = torch.cdist(stacked_predictions, stacked_predictions, p=2).pow(2)
        
        # 计算每个预测的 Krum 分数
        krum_scores = []
        for i in range(n_predictions):
            sorted_distances = torch.sort(distances[i])[0]
            # 确保不会超出索引范围
            n_neighbors = min(n_predictions - self.f - 2, n_predictions - 1)
            if n_neighbors > 0:
                krum_score = torch.sum(sorted_distances[1:n_neighbors + 1])
            else:
                krum_score = torch.sum(sorted_distances[1:])
            krum_scores.append(krum_score)
        
        # 选择 Krum 分数最小的预测
        best_idx = torch.argmin(torch.stack(krum_scores))
        return candidate_predictions[best_idx]

