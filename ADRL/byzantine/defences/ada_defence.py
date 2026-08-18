# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/ada_defence.py
# Description:  AdaAggRL Adaptive Aggregation Defense
# ===========================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


def create_adaagg_eval_fn(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> Callable:
    """
    Creates evaluation function for AdaAggRL.
    
    Args:
        model: Model used for evaluation (only used for architecture)
        x_val: Validation data
        y_val: Validation labels
    
    Returns:
        Evaluation function fn(weights_list) -> (loss, acc_dict, label_acc, label_loss)
    """
    device = next(model.parameters()).device
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    # Get number of labels
    num_labels = len(torch.unique(y_val))
    
    def eval_fn(weights_list: List[np.ndarray]) -> Tuple[float, Dict, List[float], List[float]]:
        """Evaluates model with given weights."""
        pass
    return eval_fn


class AdaAggRLDefence(ByzantineDefence):
    """AdaAggRL adaptive aggregation defense mechanism."""

    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.no_labels = config.get('n_classes', 10)

    def _eval_client_worker(self, result: Tuple[int, List[np.ndarray]]) -> Tuple[int, float, Dict, List[float], List[float]]:
        """
        Worker function to evaluate a single client's parameters.
        
        Args:
            result: (client_id, params_list)
        
        Returns:
            (client_id, loss, acc_dict, label_acc, label_loss)
        """
        client_id, params = result
        
        try:
            loss, acc_dict, label_acc, label_loss = self.eval_fn(params)
            return client_id, loss, acc_dict, label_acc, label_loss
        except Exception as e:
            logger.warning(f"AdaAggRL: Failed to evaluate client {client_id}: {e}")
            # Return default values
            return client_id, float('inf'), {'accuracy': 0.0}, [0.0] * self.no_labels, [float('inf')] * self.no_labels
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates predictions using AdaAggRL simplified method.
        
        For prediction aggregation, a simplified approach is used:
        1. Compute average of all predictions as benchmark
        2. Calculate outlier scores based on prediction distance to average
        
        Args:
            client_predictions: List of client prediction tensors
        
        Returns:
            (aggregated_prediction, outlier_scores)
        """
        if not client_predictions:
            raise ValueError("Client predictions list is empty")
        
        # Compute mean prediction
        stacked_predictions = torch.stack(client_predictions)
        avg_prediction = torch.mean(stacked_predictions, dim=0)
        
        # Compute distance to mean as outlier score
        outlier_scores = torch.norm(stacked_predictions - avg_prediction, dim=-1)
        return avg_prediction, outlier_scores
