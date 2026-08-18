# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         utilities.py
# Description:  Utility functions
# ===========================================================================

import os
import sys
import time
import json
import logging
from collections import OrderedDict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy as Accuracy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlteredDatasetWrapper(torch.utils.data.Dataset):
    """Dataset wrapper that returns item indices alongside data and labels."""
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple):
            image, label = item
        else:
            image = item
            label = None
        return image, label, idx

    def __len__(self):
        return len(self.dataset)

    @property
    def targets(self):
        """Exposes the targets property of the original dataset."""
        if hasattr(self.dataset, 'targets'):
            return self.dataset.targets
        elif hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        else:
            raise AttributeError(f"Original dataset {type(self.dataset)} has no 'targets' or 'labels' attribute.")


class Utilities:
    """Utility functions class."""

    @staticmethod
    def get_index_dataset(original_dataset):
        """Adds indices to the dataset."""
        return AlteredDatasetWrapper(original_dataset)

    @staticmethod
    def reset_dataset_subset_indices(dataset: torch.utils.data.dataset.Subset):
        """Resets dataset subset indices."""
        dataset.indices = list(range(len(dataset.indices)))

    @staticmethod
    def get_model_communication_cost(model: torch.nn.Module) -> int:
        """Calculates model communication cost (in bytes)."""
        total_bytes = 0
        for param in model.parameters():
            bytes_per_number = param.element_size()
            total_bytes += param.numel() * bytes_per_number
        return int(total_bytes)

    @staticmethod
    def calculate_communication_cost(tensorList: List[torch.Tensor]) -> int:
        """Calculates communication cost for a list of tensors (in bytes)."""
        total_bytes = 0
        for x in tensorList:
            bytes_per_number = x.element_size()
            total_bytes += x.numel() * bytes_per_number
        return int(total_bytes)

    @staticmethod
    def get_client_models(clients):
        return [
            {
                k: v.detach().cpu()
                for k, v in client.model.state_dict().items()
            }
            for client in clients
        ]

    @staticmethod
    @torch.no_grad()
    def average_client_models(client_model_list):
        """Averages and
