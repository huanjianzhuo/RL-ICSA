# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/base.py
# Description:  拜占庭防御基类
# ===========================================================================

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ByzantineDefence(ABC):
    """拜占庭防御基类"""
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        self.clients = clients
        self.config = config
        self.runner = runner_instance
        self.byzantine_clients = [client for client in clients if client.is_byzantine]
        
    @abstractmethod
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """聚合客户端模型"""
        pass
    
    @abstractmethod
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """聚合客户端预测"""
        pass


class NoDefence(ByzantineDefence):
    """无防御（基线）"""
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """简单平均聚合"""
        from utilities import Utilities as Utils
        return Utils.average_client_models(client_models)
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """简单平均预测"""
        avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
        outlier_scores = torch.zeros(len(client_predictions))
        return avg_prediction, outlier_scores

