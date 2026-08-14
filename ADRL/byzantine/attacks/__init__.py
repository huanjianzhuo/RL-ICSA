# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/__init__.py
# Description:  攻击模块初始化文件
# ===========================================================================

from typing import List, Dict, Any
from .base import NoAttack, CollusionAttackBase
from .gaussian_attack import GaussianAttack
from .lie_attack import LIEAttack
from .qlearning_attack import QLearningAttack
from .poisonedfl_attack import PoisonedFLAttack
from .fang_attack import FangAttack
from .minmax_attack import MinMaxAttack
from .minsum_attack import MinSumAttack
from .fedsa_attack import FedSAAttack
from .secondhighestconfidence_attack import SecondHighestConfidenceAttack
from .RLFL import RLFLAttack


def get_attack(attack_name: str, clients: List, config: Dict[str, Any], runner_instance):
    """
    获取指定的攻击方法
    
    Args:
        attack_name: 攻击名称
        clients: 客户端列表
        config: 配置字典
        runner_instance: 运行器实例
        
    Returns:
        攻击实例
    """
    attacks = {
        'none': NoAttack,
        'lie': LIEAttack,
        'qlearning': QLearningAttack,
        'minmax': MinMaxAttack,
        'minsum': MinSumAttack,
        'second': ICSA,
    }
    
    attack_name = attack_name.lower()
    if attack_name not in attacks:
        raise ValueError(f"未知的攻击方法: {attack_name}")
    
    return attacks[attack_name](clients, config, runner_instance)


__all__ = [
    'NoAttack',
    'CollusionAttackBase',
    'LIEAttack',
    'QLearningAttack',
    'MinMaxAttack',
    'MinSumAttack',
    'ICSA',
    'get_attack',
]

