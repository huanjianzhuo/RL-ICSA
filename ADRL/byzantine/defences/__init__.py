# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/__init__.py
# Description:  拜占庭防御方法模块
# ===========================================================================

# 从各个文件导入防御类
from .base import ByzantineDefence, NoDefence
from .median_defence import MedianDefence
from .trimmed_mean_defence import TrimmedMeanDefence
from .krum_defence import KrumDefence
from .bulyan_defence import BulyanDefence
from .fltrust_defence import FLTrustDefence
from .foolsgold_defence import FoolsGoldDefence
from .ada_defence import AdaAggRLDefence
from .momentum_screening_defence import MomentumScreeningDefence

from typing import List, Dict, Any

def get_defence(defence_name: str, clients: List, config: Dict[str, Any], runner_instance) -> ByzantineDefence:
    """获取指定的防御方法"""
    defences = {
        'none': NoDefence,
        'median': MedianDefence,
        'trimmed_mean': TrimmedMeanDefence,
        'krum': KrumDefence,
        'bulyan': BulyanDefence,
        'fltrust': FLTrustDefence,
        'fool': FoolsGoldDefence,
        'ada': AdaAggRLDefence,
        'mom': MomentumScreeningDefence,
        'momentum': MomentumScreeningDefence,
    }
    
    defence_class = defences.get(defence_name.lower())
    if defence_class is None:
        raise ValueError(f"未知的防御方法: {defence_name}")
    
    return defence_class(clients, config, runner_instance)

__all__ = [
    'ByzantineDefence',
    'NoDefence',
    'MedianDefence',
    'TrimmedMeanDefence',
    'KrumDefence',
    'BulyanDefence',
    'FLTrustDefence',
    'FoolsGoldDefence',
    'AdaAggRLDefence',
    'MomentumScreeningDefence',
    'get_defence',
]

