# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/base.py
# Description:  拜占庭攻击基类
# ===========================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
import copy
from typing import List, Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NoAttack(ABC):
    """无攻击（基线）"""
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        self.clients = clients
        self.config = config
        self.runner = runner_instance
    
    def get_perturbed_client_models(self) -> List[torch.nn.Module]:
        """返回原始客户端模型"""
        return [client.model for client in self.clients]
    
    def get_perturbed_client_predictions(self) -> List[torch.Tensor]:
        """返回原始客户端预测"""
        return [client.model for client in self.clients]


class CollusionAttackBase(NoAttack):
    """
    共谋攻击基类
    
    核心功能：
    1. 内部通讯器(Communicator): 用于拜占庭客户端之间共享信息
    2. 协调器(Coordinator): 确保拜占庭客户端的攻击目标一致
    
    子类需要实现：
    - _generate_attack_params(): 生成具体的攻击参数
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        self.clients = clients
        self.config = config
        self.runner = runner_instance
        self.byzantine_clients = [client for client in clients if client.is_byzantine]
        self.benign_clients = [client for client in clients if not client.is_byzantine]
        
        # 共谋配置
        self.enable_collusion = config.get('enable_collusion', True)
        self.collusion_strength = config.get('collusion_strength', 1.0)
        self.current_round = 0
        
        # 创建内部通讯器
        self.communicator = self._create_communicator()
        
        # 创建协调器
        self.coordinator = self._create_coordinator()
        
        if self.enable_collusion and len(self.byzantine_clients) > 1:
            logger.info(f"共谋攻击基类初始化：{len(self.byzantine_clients)}个拜占庭客户端")
            logger.info(f"  - 通讯器：已创建")
            logger.info(f"  - 协调器：已创建")
    
    def _create_communicator(self):
        """
        创建内部通讯器，用于拜占庭客户端之间共享信息
        
        Returns:
            Communicator: 通讯器实例
        """
        class Communicator:
            """内部通讯器：负责拜占庭客户端之间的信息共享"""
            
            def __init__(self, byzantine_clients, benign_clients, config):
                self.byzantine_clients = byzantine_clients
                self.benign_clients = benign_clients
                self.config = config
                self.n_byzantine = len(byzantine_clients)
                
                # 共享信息存储
                self.shared_info = {
                    'gradients': {},           # {agent_id: gradient_tensor}
                    'model_updates': {},       # {agent_id: update_tensor}
                    'attack_targets': {},      # {agent_id: target_info}
                    'benign_statistics': None, # 良性客户端统计信息
                }
                
                logger.debug(f"通讯器初始化: {self.n_byzantine}个拜占庭客户端")
            
            def share_gradient(self, agent_id: int, gradient: torch.Tensor):
                """共享梯度信息"""
                self.shared_info['gradients'][agent_id] = gradient.detach().clone()
            
            def share_model_update(self, agent_id: int, update: torch.Tensor):
                """共享模型更新信息"""
                self.shared_info['model_updates'][agent_id] = update.detach().clone()
            
            def share_attack_target(self, agent_id: int, target_info: Dict[str, Any]):
                """共享攻击目标信息"""
                self.shared_info['attack_targets'][agent_id] = target_info
            
            def get_shared_gradients(self, exclude_agent_id: Optional[int] = None) -> List[torch.Tensor]:
                """获取其他拜占庭客户端共享的梯度"""
                gradients = []
                for agent_id, gradient in self.shared_info['gradients'].items():
                    if exclude_agent_id is None or agent_id != exclude_agent_id:
                        gradients.append(gradient)
                return gradients
            
            def get_shared_updates(self, exclude_agent_id: Optional[int] = None) -> List[torch.Tensor]:
                """获取其他拜占庭客户端共享的模型更新"""
                updates = []
                for agent_id, update in self.shared_info['model_updates'].items():
                    if exclude_agent_id is None or agent_id != exclude_agent_id:
                        updates.append(update)
                return updates
            
            def analyze_benign_clients(self):
                """分析良性客户端，提取统计信息"""
                if not self.benign_clients:
                    return None
                
                benign_params_list = []
                for client in self.benign_clients:
                    if hasattr(client, 'model') and client.model is not None:
                        params = []
                        for param in client.model.parameters():
                            params.append(param.data.flatten())
                        benign_params_list.append(torch.cat(params))
                
                if not benign_params_list:
                    return None
                
                # 计算统计信息
                benign_params_tensor = torch.stack(benign_params_list)
                
                self.shared_info['benign_statistics'] = {
                    'mean': benign_params_tensor.mean(dim=0),
                    'std': benign_params_tensor.std(dim=0),
                    'median': benign_params_tensor.median(dim=0)[0],
                    'count': len(benign_params_list),
                }
                
                logger.debug(f"良性客户端分析完成: {len(benign_params_list)}个客户端")
                return self.shared_info['benign_statistics']
            
            def get_benign_statistics(self) -> Optional[Dict[str, torch.Tensor]]:
                """获取良性客户端统计信息"""
                return self.shared_info['benign_statistics']
            
            def clear(self):
                """清空共享信息（新一轮开始时调用）"""
                self.shared_info['gradients'].clear()
                self.shared_info['model_updates'].clear()
                self.shared_info['attack_targets'].clear()
        
        return Communicator(self.byzantine_clients, self.benign_clients, self.config)
    
    def _create_coordinator(self):
        """
        创建协调器，确保拜占庭客户端的攻击目标一致
        
        Returns:
            Coordinator: 协调器实例
        """
        class Coordinator:
            """协调器：确保拜占庭客户端的攻击目标一致"""
            
            def __init__(self, byzantine_clients, communicator, config):
                self.byzantine_clients = byzantine_clients
                self.communicator = communicator
                self.config = config
                self.n_byzantine = len(byzantine_clients)
                
                # 统一的攻击目标
                self.unified_target = {
                    'attack_direction': None,  # 统一的攻击方向
                    'attack_strength': config.get('collusion_strength', 1.0),
                    'target_params': None,     # 目标参数
                }
                
                logger.debug(f"协调器初始化: {self.n_byzantine}个拜占庭客户端")
            
            def coordinate_attack_direction(self) -> torch.Tensor:
                """
                协调攻击方向，确保所有拜占庭客户端朝同一方向攻击
                
                Returns:
                    torch.Tensor: 统一的攻击方向
                """
                # 获取良性客户端统计信息
                benign_stats = self.communicator.get_benign_statistics()
                
                if benign_stats is None:
                    # 如果没有良性统计，使用共享的梯度计算方向
                    shared_gradients = self.communicator.get_shared_gradients()
                    if shared_gradients:
                        # 使用中位数方向
                        gradients_stack = torch.stack(shared_gradients)
                        direction = torch.median(gradients_stack, dim=0)[0]
                    else:
                        # 默认方向
                        return torch.zeros(1)
                else:
                    # 使用良性均值的反方向作为攻击方向
                    benign_mean = benign_stats['mean']
                    direction = -benign_mean
                
                # 归一化
                direction = direction / (torch.norm(direction) + 1e-8)
                self.unified_target['attack_direction'] = direction
                
                logger.debug(f"攻击方向已协调，范数: {torch.norm(direction).item():.4f}")
                return direction
            
            def coordinate_attack_strength(self) -> float:
                """
                协调攻击强度，确保所有拜占庭客户端使用相同的攻击强度
                
                Returns:
                    float: 统一的攻击强度
                """
                return self.unified_target['attack_strength']
            
            def coordinate_target_params(self) -> Optional[torch.Tensor]:
                """
                协调目标参数，确保所有拜占庭客户端针对相同的目标
                
                Returns:
                    torch.Tensor: 统一的目标参数
                """
                benign_stats = self.communicator.get_benign_statistics()
                
                if benign_stats is not None:
                    # 使用良性均值作为目标参数
                    self.unified_target['target_params'] = benign_stats['mean']
                    return benign_stats['mean']
                
                return None
            
            def get_unified_target(self) -> Dict[str, Any]:
                """获取统一的攻击目标"""
                return self.unified_target
        
        return Coordinator(self.byzantine_clients, self.communicator, self.config)
    
    @abstractmethod
    def _generate_attack_params(self, client, agent_id: int) -> torch.Tensor:
        """
        生成攻击参数（子类必须实现）
        
        子类可以通过以下方式使用通讯器和协调器：
        - self.communicator.share_gradient(agent_id, gradient)
        - self.communicator.get_shared_gradients(agent_id)
        - self.coordinator.coordinate_attack_direction()
        - self.coordinator.coordinate_attack_strength()
        
        Args:
            client: 客户端
            agent_id: 拜占庭客户端ID
            
        Returns:
            torch.Tensor: 恶意参数（一维张量）
        """
        pass
    
    def get_perturbed_client_models(self) -> List[torch.nn.Module]:
        """
        获取被攻击的客户端模型
        
        Returns:
            List[torch.nn.Module]: 模型列表
        """
        # 清空上一轮的共享信息
        self.communicator.clear()
        
        # 分析良性客户端
        if self.enable_collusion:
            self.communicator.analyze_benign_clients()
            self.coordinator.coordinate_attack_direction()
        
        models = []
        byzantine_idx = 0
        
        # 为每个客户端生成模型
        for client in self.clients:
            if client.is_byzantine:
                # 生成攻击参数
                malicious_params = self._generate_attack_params(client, byzantine_idx)
                
                if malicious_params is not None:
                    # 创建恶意模型
                    malicious_model = self._create_malicious_model(client, malicious_params)
                    models.append(malicious_model)
                else:
                    models.append(client.model)
                
                byzantine_idx += 1
            else:
                models.append(client.model)
        
        # 增加轮次
        self.current_round += 1
        
        return models
    
    def _create_malicious_model(self, client, malicious_params: torch.Tensor):
        """
        创建恶意模型（辅助方法）
        
        Args:
            client: 客户端
            malicious_params: 恶意参数（一维张量）
            
        Returns:
            torch.nn.Module: 恶意模型
        """
        if client.model is None:
            return None
        
        # 使用 deepcopy 创建模型副本（更安全、更通用）
        model = copy.deepcopy(client.model)
        
        # 将恶意参数分配回模型
        param_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data = malicious_params[param_idx:param_idx + param_size].reshape(param.shape)
            param_idx += param_size
        
        return model

