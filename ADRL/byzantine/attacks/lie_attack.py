# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/lie_attack.py
# Description:  LIE攻击
# ===========================================================================

import torch
from typing import List, Dict, Any
from .base import CollusionAttackBase


class LIEAttack(CollusionAttackBase):
    """
    A Little Is Enough (LIE) 攻击（支持共谋）
    基于论文: "A Little Is Enough: Circumventing Defenses For Distributed Learning"

    共谋特点：
    - 所有拜占庭客户端使用相同的LIE攻击方法
    - 通过通讯器共享信息
    - 通过协调器确保攻击方向一致
    """

    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.z_max = config.get('lie_z_max', 3.0)  # 最大z-score，控制攻击强度
        self.attack_direction = config.get('lie_direction', 'negative')  # 'negative' or 'positive'
    
    def _generate_attack_params(self, client, agent_id: int) -> torch.Tensor:
        """
        生成LIE攻击参数
        
        使用通讯器获取良性统计信息，使用协调器确保攻击方向一致
        策略：mean + z_max * std * direction
        """
        # 从通讯器获取良性客户端统计信息
        benign_stats = self.communicator.get_benign_statistics()
        
        if benign_stats is None:
            return None
        
        mean = benign_stats['mean']
        std = benign_stats['std']
        
        # 避免除以零
        std = torch.clamp(std, min=1e-6)
        
        # 确定攻击方向
        direction = -1.0 if self.attack_direction == 'negative' else 1.0
        
        # 获取协调的攻击强度
        attack_strength = self.coordinator.coordinate_attack_strength()
        
        # 计算恶意参数：mean + z_max * std * direction
        z_score = self.z_max * attack_strength
        malicious_params = mean + direction * z_score * std
        
        # 通过通讯器共享梯度信息（供其他拜占庭客户端参考）
        self.communicator.share_gradient(agent_id, malicious_params)
        
        return malicious_params

