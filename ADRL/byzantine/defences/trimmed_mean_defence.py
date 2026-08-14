# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/trimmed_mean_defence.py
# Description:  修剪均值防御
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class TrimmedMeanDefence(ByzantineDefence):
    """
    修剪均值防御：移除异常值后计算均值
    
    基于 PoisonedFL 实现，添加了以下改进：
    1. NaN/Inf 异常值处理
    2. 支持基于比例或固定数量的修剪策略
    
    参考 PoisonedFL trim 方法：
    - b = nfake（两端各修剪 b 个）
    - m = n - b*2（保留中间 m 个）
    - 对排序后的数组取 [b:b+m] 计算均值
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.trim_ratio = config.get('trim_ratio', 0.2)  # 修剪比例（默认两端各修剪20%）
        self.trim_count = config.get('trim_count', None)  # 固定修剪数量（如果指定）
        
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        使用修剪均值聚合模型
        
        参考 PoisonedFL 的 trim 方法：
        - 处理 NaN 和 Inf 异常值
        - 排序后移除两端的极端值
        - 对中间值计算均值
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        n_models = len(client_models)
        
        # 计算修剪数量（参考 PoisonedFL 的 b = nfake）
        if self.trim_count is not None:
            # 使用固定数量（与 PoisonedFL 一致）
            b = self.trim_count
        else:
            # 使用比例
            b = int(n_models * self.trim_ratio)
        
        # 保留的数量 m = n - b*2（参考 PoisonedFL 第112行）
        m = n_models - b * 2
        if m <= 0:
            logger.warning(f"修剪后没有剩余模型 (n={n_models}, b={b}, m={m})，使用简单平均")
            from utilities import Utilities as Utils
            return Utils.average_client_models(client_models)
        
        # 获取第一个模型作为模板，并确定目标设备
        template_model = client_models[0]
        first_param = next(template_model.parameters())
        device = first_param.device
        
        aggregated_state = OrderedDict()
        
        # 对每个参数进行修剪均值聚合
        for param_name in template_model.state_dict().keys():
            param_values = []
            for model in client_models:
                # 确保参数在同一设备上
                param = model.state_dict()[param_name].to(device)
                param_values.append(param.unsqueeze(0))
            
            # 堆叠所有参数值
            stacked_params = torch.cat(param_values, dim=0)
            
            # 保存原始数据类型
            original_dtype = stacked_params.dtype
            
            # 如果是整数类型，先转换为浮点数
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                stacked_params = stacked_params.float()
            
            # 处理 NaN 和 Inf 值（参考 PoisonedFL 第113-115行）
            nan_mask = torch.isnan(stacked_params)
            inf_mask = torch.isinf(stacked_params)
            outlier_mask = nan_mask | inf_mask
            if outlier_mask.any():
                logger.warning(f"参数 {param_name} 中检测到 NaN 或 Inf 值，已替换为大数")
                stacked_params = torch.where(outlier_mask, 
                                            torch.ones_like(stacked_params) * 1e8, 
                                            stacked_params)
            
            # 排序并取中间部分（参考 PoisonedFL 第129-130行）
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            # 取 [b:b+m] 即 [b:n-b] 部分
            trimmed_params = sorted_params[b:b+m]
            mean_params = torch.mean(trimmed_params, dim=0)
            
            # 如果原始参数是整数类型，转换回整数
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                mean_params = mean_params.round().to(original_dtype)
            
            aggregated_state[param_name] = mean_params
            
        return aggregated_state
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用修剪均值聚合预测
        
        参考 PoisonedFL trim 方法处理异常值并修剪极端值
        """
        n_predictions = len(client_predictions)
        
        # 计算修剪数量（参考 PoisonedFL 的 b = nfake）
        if self.trim_count is not None:
            b = self.trim_count
        else:
            b = int(n_predictions * self.trim_ratio)
        
        # 保留的数量 m = n - b*2
        m = n_predictions - b * 2
        if m <= 0:
            logger.warning(f"修剪后没有剩余预测 (n={n_predictions}, b={b}, m={m})，使用简单平均")
            avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
            outlier_scores = torch.zeros(n_predictions)
            return avg_prediction, outlier_scores
        
        stacked_predictions = torch.stack(client_predictions)
        
        # 处理 NaN 和 Inf 值
        nan_mask = torch.isnan(stacked_predictions)
        inf_mask = torch.isinf(stacked_predictions)
        outlier_mask = nan_mask | inf_mask
        if outlier_mask.any():
            logger.warning(f"预测中检测到 NaN 或 Inf 值，已替换为大数")
            stacked_predictions = torch.where(outlier_mask, 
                                             torch.ones_like(stacked_predictions) * 1e8, 
                                             stacked_predictions)
        
        # 排序并取中间部分
        sorted_predictions, _ = torch.sort(stacked_predictions, dim=0)
        # 取 [b:b+m] 即 [b:n-b] 部分
        trimmed_predictions = sorted_predictions[b:b+m]
        mean_prediction = torch.mean(trimmed_predictions, dim=0)
        
        # 计算异常分数
        distances = torch.norm(stacked_predictions - mean_prediction.unsqueeze(0), dim=-1)
        outlier_scores = distances / (torch.max(distances) + 1e-10)
        
        return mean_prediction, outlier_scores

