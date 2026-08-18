# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/median_defence.py
# Description:  Median Defense
# ===========================================================================

import torch
from typing import List, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class MedianDefence(ByzantineDefence):
    """
    Median Defense: Aggregates updates using coordinate-wise median.
    
    Based on PoisonedFL implementation with enhancements:
    1. Handling of NaN and Inf outliers.
    2. Correct median calculation (averaging middle two values for even counts).
    """
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        Aggregates models using coordinate-wise median.
        
        PoisonedFL median strategy:
        - Handle NaN and Inf outliers
        - Compute coordinate-wise median
        - Average middle two values if client count is even
        """
        if not client_models:
            raise ValueError("Client models list is empty")
        
        n_models = len(client_models)
        
        # Use first model as template and determine target device
        template_model = client_models[0]
        first_param = next(template_model.parameters())
        device = first_param.device
        
        # Aggregate parameters
        # ...
        
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates predictions using median.
        
        Reference PoisonedFL implementation to handle outliers and compute median correctly.
        """
        n_predictions = len(client_predictions)
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
        
        # Compute median (reference PoisonedFL)
        sorted_predictions, _ = torch.sort(stacked_predictions, dim=0)
        if n_predictions % 2 == 1:
            # Odd number of predictions: take middle element
            median_prediction = sorted_predictions[n_predictions // 2]
        else:
            # Even number of predictions: average two middle elements
            median_prediction = (sorted_predictions[n_predictions // 2 - 1] + sorted_predictions[n_predictions // 2]) / 2
        
        # Calculate outlier scores based on distance to median
        outlier_scores = torch.norm(stacked_predictions - median_prediction, dim=-1)
        return median_prediction, outlier_scores
