# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/bulyan_defence.py
# Description:  Bulyan Defense
# ===========================================================================

import torch
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class BulyanDefence(ByzantineDefence):
    """
    Bulyan Defense: Byzantine-resilient aggregation.
    
    Paper: "The Hidden Vulnerability of Distributed Learning in Byzantium"
    
    Core Algorithm:
    1. Requires n >= 4f + 3 clients.
    2. Recursively uses Byzantine-resilient aggregation rule A (e.g., Krum) to select theta = n - 2f gradients:
       - Step 1: Use A to select the vector closest to A's output.
       - Step 2: Remove the vector from the candidate set and add it to selection set S.
       - Step 3: Repeat until |S| = theta.
    3. For selected theta gradients, compute coordinate-wise median.
    
    Key Features:
    - Ensures every coordinate is endorsed by a majority of non-Byzantine vectors.
    - (alpha, f)-Byzantine resilient and convergent.
    - Time complexity: O(n^2 * d) (when A is Krum or GeoMed).
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.f = config.get('f', 1)

    def _krum_select_predictions(self, remaining_predictions: List[torch.Tensor], theta: int) -> List[torch.Tensor]:
        """Recursively selects predictions using Krum."""
        selected_predictions = []
        while len(selected_predictions) < theta and remaining_predictions:
            # Select one prediction from remaining candidates
            selected_prediction = self._krum_select_one_prediction(remaining_predictions)
            
            # Add selected prediction to selection set
            selected_predictions.append(selected_prediction)
            
            # Remove selected prediction from candidate set
            remaining_predictions.remove(selected_prediction)
        
        return selected_predictions
    
    def _krum_select_one_prediction(self, candidate_predictions: List[torch.Tensor]) -> torch.Tensor:
        """Selects the optimal prediction from candidate predictions using Krum."""
        n_predictions = len(candidate_predictions)
        
        if n_predictions == 1:
            return candidate_predictions[0]
        
        # Compute squared distance matrix between predictions
        stacked_predictions = torch.stack(candidate_predictions)
        distances = torch.cdist(stacked_predictions, stacked_predictions, p=2).pow(2)
        
        # Compute Krum score for each prediction
        krum_scores = []
        for i in range(n_predictions):
            sorted_distances = torch.sort(distances[i])[0]
            # Ensure index remains within bounds
            n_neighbors = min(n_predictions - self.f - 2, n_predictions - 1)
            if n_neighbors > 0:
                krum_score = torch.sum(sorted_distances[1:n_neighbors + 1])
            else:
                krum_score = torch.sum(sorted_distances)
            krum_scores.append(krum_score)
            
        best_idx = torch.argmin(torch.tensor(krum_scores)).item()
        return candidate_predictions[best_idx]
