# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/minmax_attack.py
# Description:  MinMax Attack
# ===========================================================================

import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from .base import CollusionAttackBase

logger = logging.getLogger(__name__)


class MinMaxAttack(CollusionAttackBase):
    """
    AGR-agnostic Min-Max Attack (Attack-1)
    Minimizes maximum distance: ensures the maximum distance between a malicious
    gradient and any benign gradient does not exceed the maximum distance among benign gradients.
    
    Optimization Objective:
    argmax_γ max_{i∈[n]} ||∇^m - ∇_i||_2 ≤ max_{i,j∈[n]} ||∇_i - ∇_j||_2
    
    Strategy:
    1. Collect gradients from all benign clients.
    2. Compute the mean of benign gradients g_b.
    3. Choose perturbation direction p = -g_b (reverse direction attack).
    4. For each benign gradient g_j, compute pairwise maximum distance constraints.
    5. Solve quadratic inequality to obtain the upper bound for gamma.
    6. Select the minimum across all constraints as the final gamma.
    7. Generate malicious gradient: ∇^m = g_b + γ * p
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.safety_margin = config.get('minmax_safety_margin', 0.99)  # Safety factor to prevent numerical overflow
        self.dev_type = config.get('minmax_dev_type', 'unit_vec')  # Perturbation direction: 'unit_vec' (opposite direction) or 'sign'
        
    def _generate_attack_params(self, client, agent_id: int) -> torch.Tensor:
        """
        Generates MinMax attack parameters.
        
        Calculates optimal scaling factor gamma based on benign gradients collected via communicator.
        
        Returns:
            torch.Tensor: Malicious local update parameters (vector).
        """
        # Get flattened local updates from all benign clients from communicator
        benign_updates = self.communicator.get_all_benign_flat_updates()
        
        if len(benign_updates) == 0:
            logger.warning("No benign updates available, cannot perform MinMax attack.")
            return None
            
        # Stack into matrix [m, d], where m is the number of benign clients and d is the parameter dimension
        G = torch.stack(benign_updates).to(self.device)  # [m, d]
        
        # 1. Compute mean of benign gradients g_b
        g_b = torch.mean(G, dim=0)  # [d]
        
        # 2. Determine perturbation direction
        if self.dev_type == 'unit_vec':
            # Reverse direction: -g_b normalized or directly using -g_b
            perturbation = -g_b
        elif self.dev_type == 'sign':
            # -sign(g_b)
            perturbation = -torch.sign(g_b)
        elif self.dev_type == 'std':
            # -std(G)
            perturbation = -torch.std(G, dim=0)
        else:
            perturbation = -g_b
            
        # Normalize perturbation direction or keep original scale
        p_norm2 = float(torch.dot(perturbation, perturbation).item())
        if p_norm2 < 1e-12:
            logger.warning("Perturbation vector norm is too small, returning benign mean.")
            return g_b
            
        # 3. Compute pairwise maximum distance among benign gradients D_j_max
        # D_j_max[j] = max_k ||∇_j - ∇_k||^2
        D_j_max = []
        for j in range(G.size(0)):
            diff = G - G[j]  # [m, d]
            D_j_max.append(torch.max((diff * diff).sum(dim=1)).item())
        
        # Solve quadratic constraint for each benign client j
        # Constraint: ||(g_b + γ∇^p) - ∇_j||^2 ≤ D_j_max
        # Expansion: ||g_b - ∇_j||^2 + 2γ(g_b - ∇_j)·∇^p + γ^2||∇^p||^2 ≤ D_j_max
        # Formulate: a*γ^2 + b*γ + c ≤ 0
        gamma_bounds = []
        for j in range(G.size(0)):
            gb_minus_gj = (g_b - G[j]).detach()
            
            # Quadratic coefficients
            a = p_norm2  # ||∇^p||^2
            b = 2.0 * torch.dot(perturbation, gb_minus_gj).item()  # 2(g_b - ∇_j)·∇^p
            c = float(torch.dot(gb_minus_gj, gb_minus_gj).item() - D_j_max[j])  # ||g_b - ∇_j||^2 - D_j_max
            
            # Solve discriminant
            disc = b * b - 4 * a * c
            if disc <= 0 or a <= 0:
                gamma_bounds.append(0.0)
                continue
            
            # Find both roots and select the larger positive root as upper bound
            gamma2 = (-b + np.sqrt(disc)) / (2.0 * a)
            if gamma2 > 0:
                gamma_bounds.append(gamma2)
            else:
                gamma_bounds.append(0.0
