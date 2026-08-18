# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/fltrust_defence.py
# Description:  FLTrust Defense
# ===========================================================================

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
import copy
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class FLTrustDefence(ByzantineDefence):
    """
    FLTrust Defense: Trust-bootstrapped Byzantine-robust Federated Learning.
    
    Paper: "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"
    
    Core Algorithm:
    1. Server broadcasts global model to all clients.
    2. Clients train on local data and return model updates delta_i = w_new - w_old.
    3. Server trains on root dataset (clean data) to get root update delta_0.
    4. Compute trust scores and normalization:
       - TS_i = ReLU(cosine_similarity(delta_i, delta_0))
       - norm_i = ||delta_0|| / ||delta_i||
       - TSnorm_i = TS_i * norm_i
    5. Weighted aggregation: delta = (Σ TSnorm_i * delta_i) / (Σ TS_i)
    """

    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates predictions using FLTrust concept.
        Note: Prediction aggregation does not require a root dataset; weighted averaging is used.
        """
        # Compute similarity between predictions as trust scores
        n_predictions = len(client_predictions)
        trust_scores = []
        
        # Use mean of all predictions as "root prediction"
        mean_prediction = torch.mean(torch.stack(client_predictions), dim=0)
        mean_norm = torch.norm(mean_prediction)
        
        for pred in client_predictions:
            pred_norm = torch.norm(pred)
            if pred_norm > 0 and mean_norm > 0:
                cos_sim = torch.dot(pred.flatten(), mean_prediction.flatten()) / (pred_norm * mean_norm)
                ts = max(0.0, cos_sim.item())
            else:
                ts = 0.0
            trust_scores.append(ts)
        
        # Normalize trust scores
        total_ts = sum(trust_scores)
        if total_ts > 0:
            trust_scores = [ts / total_ts for ts in trust_scores]
        else:
            trust_scores = [1.0 / n_predictions] * n_predictions
        
        # Weighted aggregation
        aggregated_pred = torch.zeros_like(client_predictions[0])
        for pred, ts in zip(client_predictions, trust_scores):
            aggregated_pred += ts * pred
            
        return aggregated_pred, torch.tensor(trust_scores)
