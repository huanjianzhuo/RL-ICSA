# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/momentum_screening_defence.py
# Description:  Momentum Screening Defense
# ===========================================================================

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class MomentumScreeningDefence(ByzantineDefence):
    """
    Momentum Screening Defense: Byzantine-robust aggregation based on momentum screening.
    
    Paper: "SIMPLE MINIMAX OPTIMAL BYZANTINE ROBUST ALGORITHM FOR NONCONVEX 
           OBJECTIVES WITH UNIFORM GRADIENT HETEROGENEITY"
    
    Core Algorithm:
    1. Each client maintains momentum: m_i^t = (1-alpha) * m_i^{t-1} + alpha * g_i^t
    2. Server screens clients: retains clients that are within distance tau from at least half of the total clients
    3. Aggregates momentum updates only from screened clients
    
    Key Features:
    - Momentum-based robustness: Smooths gradient noise
    - Adaptive screening: Automatically detects anomalous clients
    - Theoretical guarantee: Achieves optimal statistical rate under heterogeneous data
    
    Configuration Parameters:
    - momentum_alpha (float): Momentum factor alpha in [0,1], default 0.9
    - screening_tau (float): Screening distance threshold parameter
    """

    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(client_predictions)
        target_device = client_predictions[0].device if client_predictions else 'cpu'
        
        # Ensure all predictions are on target device
        client_predictions = [p.to(device=target_device) for p in client_predictions]
        
        # Stack predictions
        stacked_preds = torch.stack(client_predictions)  # Shape: (n, ...)
        
        # Flatten predictions to calculate distance matrix
        flat_preds = [p.flatten() for p in client_predictions]
        M = torch.stack(flat_preds, dim=0)  # Shape: (n, d)
        
        # Compute pairwise distances
        distances = torch.cdist(M, M, p=2)
        
        # Automatically calculate distance threshold tau
        triu_indices = torch.triu_indices(n, n, offset=1)
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]
        tau = torch.quantile(pairwise_dists, 0.7).item()
        
        # Screen clients
        threshold_count = n * 0.5
        screened_indices = []
        outlier_scores = torch.zeros(n)
        
        for i in range(n):
            close_count = (distances[i] <= tau).sum().item()
            if close_count >= threshold_count:
                screened_indices.append(i)
                outlier_scores[i] = 0.0
            else:
                outlier_scores[i] = 1.0
                
        if len(screened_indices) > 0:
            aggregated_pred = torch.mean(stacked_preds[screened_indices], dim=0)
        else:
            aggregated_pred = torch.mean(stacked_preds, dim=0)
            
        return aggregated_pred, outlier_scores
