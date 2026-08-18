# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         runners/FedAVGRunner.py
# Description:  FedAVG runner class for running from scratch
# ===========================================================================

import sys
import time
import logging
from typing import Optional

import torch
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from actors import Actor
from runners.BaseRunner import BaseRunner
from utilities import Utilities as Utils
from byzantine.attacks import get_attack
from byzantine.defences import get_defence
import gc
gc.collect()
torch.cuda.empty_cache()
logger = logging.getLogger(__name__)


class FedAVGRunner(BaseRunner):
    """Handles federated training by concurrently training clients and server without pretrained models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_round = None
        self.attack = None
        self.defence = None

    @torch.no_grad()
    def broadcast_server_model_to_clients(self):
        """Broadcasts server model parameters to all clients."""
        logger.info("Broadcasting server model to clients")
        server_state_dict = self.server.model.state_dict()
        for client in self.clients:
            client.model.load_state_dict(server_state_dict)

        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    @torch.no_grad()
    def aggregate_client_models_to_server(self):
        """Aggregates client model updates to update server model using Byzantine defense mechanisms."""
        logger.info("Aggregating client models using defense strategy")

        # Get state dict list of all clients
        client_models = [client.model.state_dict() for client in self.clients]

        # Apply Byzantine attack (if attack mechanism is set)
        if self.attack is not None:
            logger.info("Applying Byzantine attack...")
            client_models = self.attack.apply_attack(
                client_models=client_models,
                server_model=self.server.model.state_dict(),
                byzantine_client_ids=self.byzantine_client_ids
            )

        # Apply Byzantine defense aggregation
        if self.defence is not None:
            logger.info("Applying Byzantine defense aggregation...")
            aggregated_state_dict = self.defence.aggregate(
                client_models=client_models,
                server_model=self.server.model.state_dict()
            )
        else:
            # Standard FedAVG parameter averaging
            aggregated_state_dict = Utils.average_client_models(client_models)

        # Update server model
        self.server.model.load_state_dict(aggregated_state_dict)

        # Track communication cost
        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    def train_single_round(self, current_round: int):
        """Executes a single communication round of training."""
        self.current_round = current_round
        logger.info(f"=== Communication Round {current_round}/{self.config['n_communications']} ===")

        # Before local training (broadcast server model to clients)
        self.strategy.before_local_training()

        # Local training for clients
        phase_length = self.strategy.get_phase_length(current_round)
        logger.info(f"Clients performing {phase_length} local training epochs...")

        for client in self.clients:
            client.reset_averaged_metrics()
            for epoch in range(phase_length):
                self.train_epoch(actor=client, data='train', is_training=True)

        # After local training (client updates aggregated to server)
        self.strategy.after_local_training()

        # Evaluate updated server model
        self.server.reset_averaged_metrics()
        self.evaluate_model(actor=self.server, data='test')
        server_test_metrics = self.server.get_metrics()['test']

        # Log metrics
        logger.info(f"Round {current_round} - Server Test Accuracy: {server_test_metrics['accuracy']:.4f}, Loss: {server_test_metrics['loss']:.4f}")

        # Record history
        self.history['round'].append(current_round)
        self.history['global_model_accuracy'].append(server_test_metrics['accuracy'])
        self.history['server_val_loss'].append(server_test_metrics['loss'])

        # End of round callback
        self.strategy.at_round_end()

    def plot_and_save_accuracy_curve(self, save_dir: str):
        """Plots and saves global model accuracy curve."""
        import os
        import matplotlib.pyplot as plt

        accuracy_list = self.history['global_model_accuracy']
        if not accuracy_list:
            logger.warning("Accuracy history is empty, skipping curve plotting")
            return

        rounds = list(range(1, len(accuracy_list) + 1))

        plt.figure(figsize=(8, 5))

        plt.plot(
            rounds,
            accuracy_list,
            marker='o',
            linewidth=2,
            label='Global Model Accuracy'
        )

        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy")

        plt.title(
            f"{self.config['attack']} + {self.config['defence']}"
        )

        plt.grid(True)
        plt.legend()

        save_path = os.path.join(
            save_dir,
            "global_accuracy.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        logger.info(
            f"Global model accuracy curve saved to: {save_path}"
        )

    def run(self):
        """Executes the main workflow."""
        # Initialize models before setting seed!
        self.set_client_models()  # Each client initializes its own model
        self.server.set_model(reinit=True)  # Server initializes its model
        self.set_seed()  # Set seed

        # Initialize dataloaders after setting seed!
        self.assign_dataloaders()  # Assign dataloaders

        self.set_client_optimizers()
        self.set_server_optimizer(reinit_server=self.config.get('reinit_server', False), first_init=True)

        # Set Byzantine attack and defense mechanisms
        self.attack = get_attack(self.config.get('attack', 'none'), self.config)
        self.defence = get_defence(self.config.get('defence', 'fedavg'), self.config)

        # Main communication loop
        n_communications = self.config['n_communications']
        progress_bar = tqdm(range(1, n_communications + 1), desc="Federated Training")

        for round_idx in progress_bar:
            self.train_single_round(current_round=round_idx)
            current_acc = self.history['global_model_accuracy'][-1]
            progress_bar.set_postfix({'Global Acc': f"{current_acc:.4f}"})

        logger.info("Federated training completed!")
        
        # Save training results and accuracy plot
        results_dir = os.path.join(self.tmp_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        Utils.save_results_to_json(self.history, os.path.join(results_dir, 'history.json'))
        self.plot_and_save_accuracy_curve(results_dir)
