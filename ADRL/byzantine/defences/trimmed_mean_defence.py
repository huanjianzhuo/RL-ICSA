# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/trimmed_mean_defence.py
# Description:  Trimmed Mean Defense
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class TrimmedMeanDefence(ByzantineDefence):
    """
    Trimmed Mean Defense: Removes extreme values before computing mean.
    
    Based on PoisonedFL implementation with enhancements:
    1. Handling of NaN and Inf outliers.
    2. Support for percentage-based or fixed-count trimming strategies.
    
    Reference PoisonedFL trim method:
    - b = nfake (trim b elements from both ends)
    - m = n - b * 2 (keep middle m elements)
    - Take slice [b:b+m] on sorted parameters and compute mean
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.trim_ratio = config.get('trim_ratio', 0.2)  # Trim ratio (default 20% trimmed from both ends)
        self.trim_count = config.get('trim_count', None)  # Fixed trim count (if specified)
        
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        n_predictions = len(client_predictions)
        
        # Calculate number of predictions to trim from each side (b)
        if self.trim_count is not None:
            b = self.trim_count
        else:
            b = int(n_predictions * self.trim_ratio)
        
        # Number of kept predictions m = n - b * 2
        m = n_predictions - b * 2
        if m <= 0:
            logger.warning(f"No predictions remaining after trimming (n={n_predictions}, b={b}, m={m}). Falling back to simple average.")
            avg_prediction = torch.mean(torch.stack(client_predictions), dim=0)
            outlier_scores = torch.zeros(n_predictions)
            return avg_prediction, outlier_scores
        
        stacked_predictions = torch.stack(client_predictions)
        
        # Handle NaN and Inf values
        nan_mask = torch.isnan(stacked_predictions)
        inf_mask = torch.isinf(stacked_predictions)
        outlier_mask = nan_mask | inf_mask
        if outlier_mask.any():
            logger.warning("Detected NaN or Inf values in predictions, replacing with large numbers")
            stacked_predictions = torch.where(
                outlier_mask, 
                torch.ones_like(stacked_predictions) * 1e8, 
                stacked_predictions
            )
        
        # Sort and select middle section
        sorted_predictions, _ = torch.sort(stacked_predictions, dim=0)
        # Take slice [b:b+m] (i.e. [b:n-b])
        trimmed_predictions = sorted_predictions[b:b+m]
        mean_prediction = torch.mean(trimmed_predictions, dim=0)
        
        outlier_scores = torch.norm(stacked_predictions - mean_prediction, dim=-1)
        return mean_prediction, outlier_scores
