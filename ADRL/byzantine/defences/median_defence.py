# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/median_defence.py
# Description:  中位数防御
# ===========================================================================

import torch
from typing import List, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class MedianDefence(ByzantineDefence):
    """
    中位数防御：使用中位数聚合
    
    基于 PoisonedFL 实现，添加了以下改进：
    1. NaN/Inf 异常值处理
    2. 正确的中位数计算（偶数情况取中间两个的平均）
    """
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        使用中位数聚合模型
        
        参考 PoisonedFL 的 median 方法：
        - 处理 NaN 和 Inf 异常值
        - 对每个坐标计算中位数
        - 偶数个模型时取中间两个的平均值
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        n_models = len(client_models)
        
        # 获取第一个模型作为模板，并确定目标设备
        template_model = client_models[0]
        first_param = next(template_model.parameters())
        device = first_param.device
        
        aggregated_state = OrderedDict()
        
        # 对每个参数进行中位数聚合
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
            
            # 处理 NaN 和 Inf 值（参考 PoisonedFL 第31-33行）
            # 将 NaN 和 Inf 替换为大数，使它们在排序时被排除
            nan_mask = torch.isnan(stacked_params)
            inf_mask = torch.isinf(stacked_params)
            outlier_mask = nan_mask | inf_mask
            if outlier_mask.any():
                logger.warning(f"参数 {param_name} 中检测到 NaN 或 Inf 值，已替换为大数")
                stacked_params = torch.where(outlier_mask, 
                                            torch.ones_like(stacked_params) * 1e8, 
                                            stacked_params)
            
            # 计算中位数（参考 PoisonedFL 第42-47行）
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            if n_models % 2 == 1:
                # 奇数个模型：取中间值
                median_params = sorted_params[n_models // 2]
            else:
                # 偶数个模型：取中间两个的平均值
                median_params = (sorted_params[n_models // 2 - 1] + sorted_params[n_models // 2]) / 2
            
            # 如果原始参数是整数类型，转换回整数
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                median_params = median_params.round().to(original_dtype)
            
            aggregated_state[param_name] = median_params
            
        return aggregated_state
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用中位数聚合预测
        
        参考 PoisonedFL 实现，处理异常值并正确计算中位数
        """
        n_predictions = len(client_predictions)
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
        
        # 计算中位数（参考 PoisonedFL）
        sorted_predictions, _ = torch.sort(stacked_predictions, dim=0)
        if n_predictions % 2 == 1:
            # 奇数个预测：取中间值
            median_prediction = sorted_predictions[n_predictions // 2]
        else:
            # 偶数个预测：取中间两个的平均值
            median_prediction = (sorted_predictions[n_predictions // 2 - 1] + sorted_predictions[n_predictions // 2]) / 2
        
        # 计算异常分数（基于与中位数的距离）
        distances = torch.norm(stacked_predictions - median_prediction.unsqueeze(0), dim=-1)
        outlier_scores = distances / (torch.max(distances) + 1e-10)
        
        return median_prediction, outlier_scores

