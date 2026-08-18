# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         runners/BaseRunner.py
# Description:  Base runner class from which all other runners inherit
# ===========================================================================

import os
import sys
import time
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

from actors import Client, Server, Actor
from config import (datasetDict, n_classesDict, num_workersDict,
                   testTransformDict, trainTransformDict)
from utilities import Utilities as Utils
from strategies import *
logger = logging.getLogger(__name__)


class BaseRunner:
    """Base class for all runners, defining common functionality."""

    def __init__(self, config: Dict[str, Any], tmp_dir: str, debug: bool):
        """Initializes variables, server, and clients.

        Args:
            config: Configuration dictionary
            debug: If True, uses local dataset instead of cluster-specific dataset
        """
        self.history = {
            'round': [],
            'server_val_accuracy': [],
            'server_val_loss': [],
            'server_train_accuracy': [],
            'server_train_loss': [],
            'client_val_accuracies': [],  # Stores validation accuracy for all clients
            'client_train_accuracies': [],  # Stores training accuracy for all clients
            'global_model_accuracy': [],  # Global model accuracy
        }

        self.config = config
        self.tmp_dir = tmp_dir
        self.debug = debug

        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Set AMP (Automatic Mixed Precision)
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            logger.info("Using Automatic Mixed Precision (AMP)")

        # Dataset configurations
        self.dataset_name = config['dataset']
        self.n_classes = n_classesDict[self.dataset_name]
        self.num_workers = num_workersDict[self.dataset_name]

        # Federated learning parameter settings
        self.n_clients = config['n_clients']
        self.n_byzantine = config.get('n_byzantine', 0)
        self.byzantine_client_ids = list(range(self.n_byzantine))

        # Initialize Server and Clients
        self.server = Server(use_amp=self.use_amp, n_classes=self.n_classes,
                             tmp_dir=self.tmp_dir, num_workers=self.num_workers,
                             config=self.config, callbacks=[], device=self.device)

        self.clients = [
            Client(use_amp=self.use_amp, client_id=i, n_classes=self.n_classes,
                   tmp_dir=self.tmp_dir, num_workers=self.num_workers,
                   config=self.config, callbacks=[], device=self.device)
            for i in range(self.n_clients)
        ]

        # Mark Byzantine clients
        for client_id in self.byzantine_client_ids:
            self.clients[client_id].is_byzantine = True

        # Total communicated bytes
        self.total_bytes_communicated = 0

        # Strategy
        self.strategy = get_strategy(config.get('strategy', 'fedavg'), config, self)

    def set_seed(self):
        """Sets random seed for reproducibility."""
        seed = self.config.get('seed', 42)
        Utils.set_seed(seed)
        logger.info(f"Random seed set to: {seed}")

    def assign_dataloaders(self):
        """Assigns dataloaders to clients and server."""
        logger.info("Loading datasets and creating dataloaders...")
        
        dataset_cls = datasetDict[self.dataset_name]
        train_transform = trainTransformDict[self.dataset_name]
        test_transform = testTransformDict[self.dataset_name]

        # Root directory for dataset
        data_root = os.path.join(self.tmp_dir, 'data')

        train_dataset = dataset_cls(root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = dataset_cls(root=data_root, train=False, download=True, transform=test_transform)

        # Wrap dataset to include item indices
        train_dataset = Utils.get_index_dataset(train_dataset)
        test_dataset = Utils.get_index_dataset(test_dataset)

        # Assign dataset subsets to clients (Non-IID or IID partitioning)
        self.partition_and_assign_datasets(train_dataset)

        # Assign test dataset dataloader to server
        self.server.dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.num
