# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         main.py
# Description:  主运行文件
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 默认配置
# 仅保留：
#   1. 本项目攻击方法：Q-Learning / ICSA
#   2. Q-Learning 动作库所必需的 LIE / Min-Max / Min-Sum / ICSA 参数
#   3. 防御方法及其参数
# 删除了 Gaussian、PoisonedFL、Fang、FedSA、RLFL 等对比攻击及对应参数。
defaults = {
    # 系统
    'run_id': 1,
    'seed': None,

    # 数据与模型
    'dataset': 'cifar10',
    'arch': 'resnet18',
    'batch_size': 128,
    'input_channels': 3,

    # 效率
    'use_amp': True,

    # 优化器
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr': 0.01,
    'client_lr': 0.01,
    'server_lr': 0.01,
    'scheduler': 'linear',

    # 联邦学习设置
    'n_clients': 20,
    'n_total_local_epochs': 10,
    'n_communications': 5,
    'n_server_epochs_per_round': 1,
    'restart_client_lr': False,
    'reinit_server': False,

    # 设备
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    # ------------------------------------------------------------------
    # 攻击与防御基础配置
    # ------------------------------------------------------------------
    # 仅暴露本项目需要的攻击入口：
    #   none       : 无攻击
    #   qlearning  : Q-Learning 自适应攻击，动作库为
    #                LIE / Min-Max / Min-Sum / ICSA
    #   second     : 独立运行 ICSA
    'attack': 'none',
    'defence': 'none',
    'n_byzantine_clients': 2,

    # ------------------------------------------------------------------
    # Q-Learning 参数
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
    # Q-Learning 动作库参数
    # 这些不是独立对比攻击的入口，而是四动作攻击库内部所需参数。
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
    # 共谋 / 高级 Q-Learning 参数
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
    # 防御方法参数
    # ------------------------------------------------------------------
    # 常规鲁棒聚合
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
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='ADRL - 对抗性深度强化学习')

    # 基本设置
    parser.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['mnist', 'cifar10', 'cifar100', 'fashionmnist'],
        help='数据集名称'
    )
    parser.add_argument(
        '--arch', type=str, default='resnet18',
        choices=['simple_cnn', 'mnist_cnn', 'simple_resnet', 'resnet18'],
        help='模型架构'
    )

    # 联邦学习设置
    parser.add_argument('--n_clients', type=int, default=20, help='客户端数量')
    parser.add_argument('--n_communications', type=int, default=5, help='通信轮数')
    parser.add_argument('--n_total_local_epochs', type=int, default=10, help='总本地训练轮数')
    parser.add_argument('--n_server_epochs_per_round', type=int, default=1, help='服务器每轮训练轮数')

    # 训练设置
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--client_lr', type=float, default=0.01, help='客户端学习率')
    parser.add_argument('--server_lr', type=float, default=0.01, help='服务器学习率')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='优化器')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument(
        '--scheduler', type=str, default='linear',
        choices=['linear', 'cosine', 'step'], help='学习率调度器'
    )

    # 系统设置
    parser.add_argument('--device', type=str, default='cuda:0', help='设备 (cuda:0, cpu等)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用自动混合精度')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='不使用自动混合精度')

    # 其他设置
    parser.add_argument('--restart_client_lr', action='store_true', help='重启客户端学习率')
    parser.add_argument('--reinit_server', action='store_true', help='重新初始化服务器')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--output_dir', type=str, default='./result', help='输出目录')

    # ------------------------------------------------------------------
    # 攻击入口：只保留本项目攻击
    # ------------------------------------------------------------------
    parser.add_argument(
        '--attack', type=str, default='none',
        choices=['none', 'qlearning', 'second'],
        help='攻击方法: none / qlearning / second(ICSA)'
    )
    parser.add_argument(
        '--n_byzantine', type=int, default=2,
        help='拜占庭客户端数量'
    )

    # ------------------------------------------------------------------
    # ICSA 参数
    # ------------------------------------------------------------------
    parser.add_argument('--tau', type=float, default=0.1, help='ICSA 的目标 Top-1/Top-2 logit margin')
    parser.add_argument(
        '--max_attack_samples', type=int, default=None,
        help='ICSA 最多处理的本地样本数；不设置则使用全部本地样本'
    )
    parser.add_argument('--lambda_eps', type=float, default=1e-12, help='ICSA lambda 计算的数值稳定项')

    # ------------------------------------------------------------------
    # Q-Learning 参数
    # ------------------------------------------------------------------
    parser.add_argument('--ql_learning_rate', type=float, default=0.05, help='Q-Learning学习率')
    parser.add_argument('--ql_discount_factor', type=float, default=0.9, help='Q-Learning折扣因子')
    parser.add_argument('--ql_epsilon', type=float, default=0.1, help='Q-Learning探索率')
    parser.add_argument('--ql_epsilon_decay', type=float, default=0.995, help='Q-Learning探索率衰减')
    parser.add_argument('--ql_epsilon_min', type=float, default=0.01, help='Q-Learning最小探索率')
    parser.add_argument('--ql_use_nn', action='store_true', default=True, help='Q-Learning使用神经网络')
    parser.add_argument('--ql_memory_size', type=int, default=10000, help='Q-Learning经验回放缓冲区大小')
    parser.add_argument('--ql_batch_size', type=int, default=64, help='Q-Learning批次大小')
    parser.add_argument('--ql_update_target_freq', type=int, default=100, help='Q-Learning目标网络更新频率')
    parser.add_argument('--ql_training_mode', action='store_true', default=True, help='Q-Learning训练模式')
    parser.add_argument('--ql_save_path', type=str, default='./qlearning_models', help='Q-Learning模型保存路径')

    # Q-Learning 四动作内部参数
    parser.add_argument('--lie_z_max', type=float, default=3.0, help='Q-Learning动作库中LIE的最大z-score')
    parser.add_argument(
        '--lie_direction', type=str, default='negative',
        choices=['negative', 'positive'], help='Q-Learning动作库中LIE攻击方向'
    )
    parser.add_argument('--minmax_safety_margin', type=float, default=0.95, help='Q-Learning动作库中Min-Max安全边界')
    parser.add_argument('--minsum_safety_margin', type=float, default=0.95, help='Q-Learning动作库中Min-Sum安全边界')

    # 共谋 / 高级 Q-Learning
    parser.add_argument('--enable_collusion', action='store_true', help='启用拜占庭客户端共谋')
    parser.add_argument(
        '--collusion_strategy', type=str, default='coordinated',
        choices=['coordinated', 'distributed', 'adaptive'],
        help='共谋策略'
    )
    parser.add_argument('--collusion_strength', type=float, default=1.0, help='共谋强度')
    parser.add_argument(
        '--ql_coordination_mode', type=str, default='diverse',
        choices=['diverse', 'focused', 'adaptive'],
        help='Q-Learning协调模式'
    )
    parser.add_argument('--ql_diversity_bonus', type=float, default=0.1, help='Q-Learning多样化奖励')
    parser.add_argument('--ql_synergy_bonus', type=float, default=0.2, help='Q-Learning协同奖励')
    parser.add_argument('--use_advanced_collusion', action='store_true', default=True, help='使用高级共谋攻击')
    parser.add_argument('--reward_accuracy_weight', type=float, default=10.0, help='准确率奖励权重')
    parser.add_argument('--reward_collusion_weight', type=float, default=2.0, help='共谋奖励权重')
    parser.add_argument('--reward_diversity_weight', type=float, default=0.5, help='多样性奖励权重')
    parser.add_argument('--reward_stealth_weight', type=float, default=1.0, help='隐蔽性奖励权重')
    parser.add_argument('--reward_consistency_weight', type=float, default=1.0, help='一致性奖励权重')
    parser.add_argument('--reward_penalty_weight', type=float, default=10.0, help='奖励惩罚项权重')

    # ------------------------------------------------------------------
    # 防御方法
    # ------------------------------------------------------------------
    parser.add_argument(
        '--defence', type=str, default='none',
        choices=['none', 'median', 'trimmed_mean', 'krum', 'bulyan', 'fltrust', 'fool', 'ada', 'momentum'],
        help='拜占庭防御方法'
    )
    parser.add_argument('--trim_ratio', type=float, default=0.2, help='Trimmed Mean 修剪比例')
    parser.add_argument('--f', type=int, default=2, help='最多容忍的拜占庭客户端数量')
    parser.add_argument('--krum_k', type=int, default=16, help='Krum选择的模型数量')

    # FLTrust 防御参数
    parser.add_argument('--fltrust_threshold', type=float, default=0.0, help='FLTrust信任阈值')
    parser.add_argument('--fltrust_norm_clipping', action='store_true', default=True, help='FLTrust使用范数裁剪')
    parser.add_argument('--fltrust_root_size', type=int, default=100, help='FLTrust根数据集大小')
    parser.add_argument('--fltrust_root_batch_size', type=int, default=32, help='FLTrust根数据集批次大小')

    # Non-IID 数据划分参数
    parser.add_argument(
        '--dirichlet_alpha', type=float, default=0.1,
        help='狄利克雷分布浓度参数，越小数据越不均衡'
    )

    return parser.parse_args()


def create_config_from_args(args) -> Dict[str, Any]:
    """从命令行参数创建配置。"""
    config = defaults.copy()

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # 保持 runner 兼容
    config['n_byzantine_clients'] = config['n_byzantine']

    if config['device'] is None:
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if config['dataset'] in ['mnist', 'fashionmnist']:
        config['input_channels'] = 1
    else:
        config['input_channels'] = 3

    # 固定使用 FedAvg
    config['strategy'] = 'fedavg'

    return config


def main():
    """主函数。"""
    args = parse_args()
    config = create_config_from_args(args)

    logger.info(f"使用配置: {config}")

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
        logger.info(f"使用临时目录: {tmp_dir}")
        config['experiment_dir'] = experiment_dir

        runner = FedAVGRunner(
            config=config,
            tmp_dir=tmp_dir,
            debug=args.debug
        )

        try:
            runner.run()
            logger.info('训练完成')
        except Exception as e:
            logger.error(f'训练过程中出现错误: {e}')
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

        logger.info(f"结果已保存到: {experiment_dir}")


if __name__ == '__main__':
    main()
