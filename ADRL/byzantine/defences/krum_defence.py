# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/krum_defence.py
# Description:  Krum Defense
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class KrumDefence(ByzantineDefence):
    """
    Krum Defense: Selects the most representative model.
    
    Paper: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    
    Core Algorithm:
    1. For each worker i, calculate score s(i) = Σ ||V_i - V_j||^2
       where sum is taken over the n - f - 2 closest vectors to V_i.
    2. Select model with minimal score: KR(V_1, ..., V_n) = V_{i*}
       where i* satisfies s(i*) <= s(i) for all i.
    
    Byzantine Resilience: Satisfies (alpha, f)-Byzantine resilience when 2f + 2 <= n.
    Time Complexity: O(n^2 * d), where n is vector count and d is vector dimension.
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.f = config.get('f', 1)  # Maximum number of Byzantine clients

    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates Krum score for each prediction and selects the optimal prediction."""
        n_predictions = len(client_predictions)
        if n_predictions < 2 * self.f + 3:
            # Fallback to simple average if prediction count does not satisfy Krum requirement
            logger.warning(f"Krum requires at least {2*self.f+3} predictions, but got {n_predictions}. Falling back to simple average.")
            avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
            outlier_scores = torch.zeros(n_predictions)
            return avg_prediction, outlier_scores
        
        # Compute squared distances between predictions
        stacked_predictions = torch.stack(client_predictions)
        # torch.cdist computes Euclidean distances by default; pow(2) gives squared distance
        distances = torch.cdist(stacked_predictions, stacked_predictions, p=2).pow(2)
        
        # Compute Krum score for each prediction
        # Formula: s(i) = Σ_{j≠i} ||V_i - V_j||^2 over n-f-2 nearest vectors
        krum_scores = []
        for i in range(n_predictions):
            # Select nearest n-f-2 distances (excluding self, i.e., skipping first zero distance element)
            sorted_distances = torch.sort(distances[i])[0]
            # Skip index 0 (distance to self) and sum next n-f-2 distances
            krum_score = torch.sum(sorted_distances[1:n_predictions - self.f - 1])
            krum_scores.append(krum_score.item())
            
        best_idx = torch.argmin(torch.tensor(krum_scores)).item()
        return client_predictions[best_idx], torch.tensor(krum_scores)
