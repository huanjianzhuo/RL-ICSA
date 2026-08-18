# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         strategies.py
# Description:  FedAVG Federated Learning Strategy
# ===========================================================================

import logging

logger = logging.getLogger(__name__)


class FedAVG:
    """Federated Averaging: Server model is updated by averaging client models, then broadcast back to clients."""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.runner = kwargs['runner_instance']

    def do_clients_train_on_public_data(self):
        """Clients train only on private data, not on public data."""
        return False

    def verify_input(self):
        """Verifies strategy inputs."""
        assert self.config['n_total_local_epochs'] is not None, 'Total local epochs must be specified'
        assert self.config['n_total_local_epochs'] >= 0, 'Total local epochs should be positive'

        assert self.config['n_communications'] is not None, 'Number of communications must be specified'
        assert self.config['n_communications'] >= 0, 'Number of communications should be positive'
        assert self.config['n_communications'] <= self.config[
            'n_total_local_epochs'], 'Number of communications must be less than or equal to total local epochs'

    def get_phase_length(self, current_round: int) -> int:
        """Returns the number of local epochs for a given round."""
        n_epochs_total = self.config['n_total_local_epochs']
        n_communications = self.config['n_communications']

        # Evenly distribute total local epochs across communication rounds
        epochs_per_round, remainder = divmod(n_epochs_total, n_communications)
        epochs_per_
