# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         main.py
# Description:  Main execution file
# ===========================================================================

import argparse
import os
import logging
from typing import Dict, Any
from tempfile import TemporaryDirectory as tempdir
from datetime import datetime

import torch

from runners.FedAVGRunner import FedAVGRunner
from utilities import Utilities as Utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration
# Only retain:
#   1. Attack methods in this project: Q-Learning / ICSA
#   2. Parameters required by the Q-Learning action library: LIE / Min-Max / Min-Sum / ICSA
#   3. Defense methods and their parameters
# Removed comparison attacks such as Gaussian, PoisonedFL, Fang, FedSA, and RLFL and their corresponding parameters.
defaults = {
    # System
    'run_id': 1,
    'seed': None,

    # Data and model
    'dataset': 'cifar10',
    'arch': 'resnet18',
    'batch_size': 128,
    'input_channels': 3,

    # Efficiency
    'use_amp': True,

    # Optimizer
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr': 0.01,
    'client_lr': 0.01,
    'server_lr': 0.01,
    'scheduler': 'linear',

    # Federated learning settings
    'n_clients': 20,
    'n_total_local_epochs': 10,
    'n_communications': 5,
    'n_server_epochs_per_round': 1,
    'restart_client_lr': False,
    'reinit_server': False,

    # Device
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    # ------------------------------------------------------------------
    # Basic attack and defense configuration
    # ------------------------------------------------------------------
    # Only expose the attack entry points required by this project:
    #   none       : No attack
    #   qlearning  : Q-Learning adaptive attack, with the action library:
    #                LIE / Min-Max / Min-Sum / ICSA
    #   second     : Run ICSA independently
    'attack': 'none',
    'defence': 'none',
    'n_byzantine_clients': 2,

    # ------------------------------------------------------------------
    # Q-Learning parameters
    # ------------------------------------------------------------------
    'ql_learning_rate': 0.05,
    'ql_discount_factor': 0.9,
    'ql_epsilon': 0.1,
    'ql_epsilon_decay': 0.995,
    'ql_epsilon_min': 0.01,
    'ql_use_nn': True,
    'ql_memory_size': 10000,
    'ql_batch_size': 64,
    'ql_update_target_freq': 100,
    'ql_training_mode': True,
    'ql_save_path': './qlearning_models',

    # ------------------------------------------------------------------
    # Q-Learning action library parameters
    # These are not entry points for independent comparison attacks; they are parameters required internally by the four-action attack library.
    # ------------------------------------------------------------------
    # Action 0: LIE
    'lie_z_max': 3.0,
    'lie_direction': 'negative',

    # Action 1: Min-Max
    'minmax_safety_margin': 0.95,

    # Action 2: Min-Sum
    'minsum_safety_margin': 0.95,

    # Action 3: ICSA
    'tau': 0.1,
    'max_attack_samples': None,
    'lambda_eps': 1e-12,

    # ------------------------------------------------------------------
    # Collusion / advanced Q-Learning parameters
    # ------------------------------------------------------------------
    'enable_collusion': False,
    'collusion_strategy': 'coordinated',
    'collusion_strength': 1.0,
    'ql_coordination_mode': 'diverse',
    'ql_diversity_bonus': 0.1,
    'ql_synergy_bonus': 0.2,
    'use_advanced_collusion': True,
    'reward_accuracy_weight': 10.0,
    'reward_collusion_weight': 2.0,
    'reward_diversity_weight': 0.5,
    'reward_stealth_weight': 1.0,
    'reward_consistency_weight': 1.0,
    'reward_penalty_weight': 10.0,

    # ------------------------------------------------------------------
    # Defense method parameters
    # ------------------------------------------------------------------
    # Standard robust aggregation
    'trim_ratio': 0.2,
    'f': 2,
    'krum_k': 16,

    # FLTrust
    'fltrust_threshold': 0.0,
    'fltrust_norm_clipping': True,
    'fltrust_root_size': 100,
    'fltrust_root_batch_size': 32,

    # Non-IID 数据划分
    'dirichlet_alpha': 0.5,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ADRL - Adversarial Deep Reinforcement Learning')

    # Basic settings
    parser.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['mnist', 'cifar10', 'cifar100', 'fashionmnist'],
        help='Dataset name'
    )
    parser.add_argument(
        '--arch', type=str, default='resnet18',
        choices=['simple_cnn', 'mnist_cnn', 'simple_resnet', 'resnet18'],
        help='Model architecture'
    )

    # Federated learning settings
    parser.add_argument('--n_clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--n_communications', type=int, default=5, help='Number of communication rounds')
    parser.add_argument('--n_total_local_epochs', type=int, default=10, help='Total number of local training epochs')
    parser.add_argument('--n_server_epochs_per_round', type=int, default=1, help='Number of server training epochs per round')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--client_lr', type=float, default=0.01, help='客户端Learning rate')
    parser.add_argument('--server_lr', type=float, default=0.01, help='服务器Learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument(
        '--scheduler', type=str, default='linear',
        choices=['linear', 'cosine', 'step'], help='Learning rate调度器'
    )

    # System设置
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cpu等)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='不Use automatic mixed precision')

    # Other settings
    parser.add_argument('--restart_client_lr', action='store_true', help='重启客户端Learning rate')
    parser.add_argument('--reinit_server', action='store_true', help='Reinitialize server')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--output_dir', type=str, default='./result', help='Output directory')

    # ------------------------------------------------------------------
    # Attack entry points: retain only the attacks in this project
    # ------------------------------------------------------------------
    parser.add_argument(
        '--attack', type=str, default='none',
        choices=['none', 'qlearning', 'second'],
        help='Attack method: none / qlearning / second (ICSA)'
    )
    parser.add_argument(
        '--n_byzantine', type=int, default=2,
        help='拜占庭Number of clients'
    )

    # ------------------------------------------------------------------
    # ICSA parameters
    # ------------------------------------------------------------------
    parser.add_argument('--tau', type=float, default=0.1, help='Target Top-1/Top-2 logit margin for ICSA')
    parser.add_argument(
        '--max_attack_samples', type=int, default=None,
        help='Maximum number of local samples processed by ICSA; if not specified, all local samples are used'
    )
    parser.add_argument('--lambda_eps', type=float, default=1e-12, help='Numerical stability term for ICSA lambda computation')

    # ------------------------------------------------------------------
    # Q-Learning parameters
    # ------------------------------------------------------------------
    parser.add_argument('--ql_learning_rate', type=float, default=0.05, help='Q-LearningLearning rate')
    parser.add_argument('--ql_discount_factor', type=float, default=0.9, help='Q-Learning discount factor')
    parser.add_argument('--ql_epsilon', type=float, default=0.1, help='Q-Learning exploration rate')
    parser.add_argument('--ql_epsilon_decay', type=float, default=0.995, help='Q-Learning exploration rate衰减')
    parser.add_argument('--ql_epsilon_min', type=float, default=0.01, help='Q-Learning minimum exploration rate')
    parser.add_argument('--ql_use_nn', action='store_true', default=True, help='Use a neural network for Q-Learning')
    parser.add_argument('--ql_memory_size', type=int, default=10000, help='Q-Learning experience replay buffer size')
    parser.add_argument('--ql_batch_size', type=int, default=64, help='Q-LearningBatch size')
    parser.add_argument('--ql_update_target_freq', type=int, default=100, help='Q-Learning target network update frequency')
    parser.add_argument('--ql_training_mode', action='store_true', default=True, help='Q-Learning training mode')
    parser.add_argument('--ql_save_path', type=str, default='./qlearning_models', help='Q-Learning model save path')

    # Internal parameters of the four-action Q-Learning library
    parser.add_argument('--lie_z_max', type=float, default=3.0, help='Maximum z-score of LIE in the Q-Learning action library')
    parser.add_argument(
        '--lie_direction', type=str, default='negative',
        choices=['negative', 'positive'], help='Attack direction of LIE in the Q-Learning action library'
    )
    parser.add_argument('--minmax_safety_margin', type=float, default=0.95, help='Safety margin of Min-Max in the Q-Learning action library')
    parser.add_argument('--minsum_safety_margin', type=float, default=0.95, help='Safety margin of Min-Sum in the Q-Learning action library')

    # Collusion / advanced Q-Learning
    parser.add_argument('--enable_collusion', action='store_true', help='Enable collusion among Byzantine clients')
    parser.add_argument(
        '--collusion_strategy', type=str, default='coordinated',
        choices=['coordinated', 'distributed', 'adaptive'],
        help='Collusion strategy'
    )
    parser.add_argument('--collusion_strength', type=float, default=1.0, help='Collusion strength')
    parser.add_argument(
        '--ql_coordination_mode', type=str, default='diverse',
        choices=['diverse', 'focused', 'adaptive'],
        help='Q-Learning coordination mode'
    )
    parser.add_argument('--ql_diversity_bonus', type=float, default=0.1, help='Q-Learning diversity bonus')
    parser.add_argument('--ql_synergy_bonus', type=float, default=0.2, help='Q-Learning synergy bonus')
    parser.add_argument('--use_advanced_collusion', action='store_true', default=True, help='Use advanced collusion attack')
    parser.add_argument('--reward_accuracy_weight', type=float, default=10.0, help='Accuracy reward weight')
    parser.add_argument('--reward_collusion_weight', type=float, default=2.0, help='Collusion reward weight')
    parser.add_argument('--reward_diversity_weight', type=float, default=0.5, help='Diversity reward weight')
    parser.add_argument('--reward_stealth_weight', type=float, default=1.0, help='Stealth reward weight')
    parser.add_argument('--reward_consistency_weight', type=float, default=1.0, help='Consistency reward weight')
    parser.add_argument('--reward_penalty_weight', type=float, default=10.0, help='Reward penalty weight')

    # ------------------------------------------------------------------
    # Defense methods
    # ------------------------------------------------------------------
    parser.add_argument(
        '--defence', type=str, default='none',
        choices=['none', 'median', 'trimmed_mean', 'krum', 'bulyan', 'fltrust', 'fool', 'ada', 'momentum'],
        help='拜占庭Defense methods'
    )
    parser.add_argument('--trim_ratio', type=float, default=0.2, help='Trimmed Mean trimming ratio')
    parser.add_argument('--f', type=int, default=2, help='最多容忍的拜占庭Number of clients')
    parser.add_argument('--krum_k', type=int, default=16, help='Number of models selected by Krum')

    # FLTrust defense parameters
    parser.add_argument('--fltrust_threshold', type=float, default=0.0, help='FLTrust trust threshold')
    parser.add_argument('--fltrust_norm_clipping', action='store_true', default=True, help='Use norm clipping in FLTrust')
    parser.add_argument('--fltrust_root_size', type=int, default=100, help='FLTrust root dataset size')
    parser.add_argument('--fltrust_root_batch_size', type=int, default=32, help='FLTrust根数据集Batch size')

    # Non-IID data partitioning parameters
    parser.add_argument(
        '--dirichlet_alpha', type=float, default=0.1,
        help='Dirichlet concentration parameter; smaller values produce more imbalanced data'
    )

    return parser.parse_args()


def create_config_from_args(args) -> Dict[str, Any]:
    """Create the configuration from command-line arguments."""
    config = defaults.copy()

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Maintain runner compatibility
    config['n_byzantine_clients'] = config['n_byzantine']

    if config['device'] is None:
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if config['dataset'] in ['mnist', 'fashionmnist']:
        config['input_channels'] = 1
    else:
        config['input_channels'] = 3

    # Use FedAvg as the fixed strategy
    config['strategy'] = 'fedavg'

    return config


def main():
    """Main function."""
    args = parse_args()
    config = create_config_from_args(args)

    logger.info(f"Using configuration: {config}")

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    large_dir = os.path.join(config['output_dir'], 'large')
    os.makedirs(large_dir, exist_ok=True)

    defence_dir = os.path.join(large_dir, config['defence'])
    os.makedirs(defence_dir, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config['attack']}_{date_str}"
    experiment_dir = os.path.join(defence_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    config_path = os.path.join(experiment_dir, 'config.json')
    Utils.save_results_to_json(config, config_path)

    with tempdir() as tmp_dir:
        logger.info(f"Using temporary directory: {tmp_dir}")
        config['experiment_dir'] = experiment_dir

        runner = FedAVGRunner(
            config=config,
            tmp_dir=tmp_dir,
            debug=args.debug
        )

        try:
            runner.run()
            logger.info('Training completed')
        except Exception as e:
            logger.error(f'An error occurred during training: {e}')
            raise

        final_results = {
            'config': config,
            'final_metrics': {
                'server': runner.server.get_metrics(),
                'clients': {
                    f'client_{i+1}': client.get_metrics()
                    for i, client in enumerate(runner.clients)
                }
            },
            'total_epochs_completed': runner.total_epochs_completed,
            'total_bytes_communicated': runner.total_bytes_communicated
        }

        results_path = os.path.join(experiment_dir, 'final_results.json')
        Utils.save_results_to_json(final_results, results_path)

        model_path = os.path.join(experiment_dir, 'final_model.pth')
        Utils.save_model(
            runner.server.model,
            model_path,
            optimizer=runner.server.optimizer,
            epoch=runner.total_epochs_completed
        )

        logger.info(f"Results saved to: {experiment_dir}")


if __name__ == '__main__':
    main()
