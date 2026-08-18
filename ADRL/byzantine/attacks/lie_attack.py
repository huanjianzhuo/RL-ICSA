# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/lie_attack.py
# Description:  LIE Attack
# ===========================================================================

import torch
from typing import List, Dict, Any
from .base import CollusionAttackBase


class LIEAttack(CollusionAttackBase):
    """
    A Little Is Enough (LIE) Attack (Collusion Supported)
    Based on paper: "A Little Is Enough: Circumventing Defenses For Distributed Learning"

    Collusion Features:
    - All Byzantine clients use the same LIE attack method
    - Information shared via communicator
    - Direction consistency ensured via coordinator
    """

    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.z_max = config.get('lie_z_max', 3.0)  # Maximum z-score controlling attack strength
        self.attack_direction = config.get('lie_direction', 'negative')  # 'negative' or 'positive'
    
    def _generate_attack_params(self, client, agent_id: int) -> torch.Tensor:
        """
        Generates LIE attack parameters.
        
        Retrieves benign statistics from communicator and uses coordinator to ensure consistent attack direction.
        Strategy: mean + z_max * std * direction
        """
        # Retrieve benign client statistics from communicator
        benign_stats = self.communicator.get_benign_statistics()
        
        if benign_stats is None:
            return None
        
        mean = benign_stats['mean']
        std = benign_stats['std']
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-6)
        
        # Determine attack direction
        direction = -1.0 if self.attack_direction == 'negative' else 1.0
        
        # Calculate LIE perturbation
        lie_params = mean + direction * self.z_max * std
        
        return lie_params
