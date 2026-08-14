# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         main.py
# Description:  主运行文件
# ===========================================================================

import argparse
import os
import sys
import tempfile
import shutil
import logging
from typing import Dict, Any
from tempfile import TemporaryDirectory as tempdir
import torch
from datetime import datetime
from runners.FedAVGRunner import FedAVGRunner
from utilities import Utilities as Utils

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
defaults = {
    # 系统
    'run_id': 1,
    'seed': None,

    # 设置
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

    # FL设置
    'n_total_local_epochs': 10,  # 每个客户端的总本地训练轮数
    'n_communications': 5,  # 服务器和客户端之间的通信轮数
    'n_server_epochs_per_round': 1,  # 服务器每轮应该训练的轮数
    'restart_client_lr': False,  # 如果为True，在每次通信后重启学习率
    'reinit_server': False,  # 在每次通信后重新初始化服务器模型、优化器、调度器

    # 设备
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    
    # 拜占庭攻击和防御
    'attack': 'no_attack',
    'defence': 'no_defence',
    'n_byzantine_clients': 2,
    'attack_strength': 1.0,
    'flip_ratio': 0.5,
    'noise_std': 0.1,
    'trim_ratio': 0.2,
    'f': 2,
    'krum_k': 16,
    
    # FLTrust防御参数
    'fltrust_threshold': 0.0,
    'fltrust_norm_clipping': True,
    'fltrust_root_size': 100,
    'fltrust_root_batch_size': 32,
    
    # Q-Learning参数
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
    
    # RLFL参数
    'rlfl_lr': 3e-4,
    'rlfl_gamma': 0.99,
    'rlfl_gae_lambda': 0.95,
    'rlfl_ppo_epochs': 10,
    'rlfl_ppo_clip': 0.2,
    'rlfl_value_loss_coef': 0.5,
    'rlfl_entropy_coef': 0.01,
    'rlfl_max_grad_norm': 0.5,
    'rlfl_batch_size': 64,
    'rlfl_buffer_size': 10000,
    'rlfl_action_dim': 1,
    'rlfl_action_scale': 10.0,
    'simulator_max_rounds': 50,
    'rlfl_n_episodes': 100,
    'rlfl_max_steps_per_episode': 50,
    'rlfl_pretrain': True,
    'rlfl_save_path': './rlfl_models',
    
    # 共谋攻击参数
    'enable_collusion': False,
    'collusion_strategy': 'coordinated',
    'collusion_strength': 1.0,
    
    # Q-Learning共谋参数
    'ql_coordination_mode': 'diverse',
    'ql_diversity_bonus': 0.1,
    'ql_synergy_bonus': 0.2,
    
    # 高级共谋参数
    'use_advanced_collusion': True,
    'reward_accuracy_weight': 10.0,
    'reward_collusion_weight': 2.0,
    'reward_diversity_weight': 0.5,
    'reward_stealth_weight': 1.0,
    
    # Non-IID数据划分参数
    'dirichlet_alpha': 0.5,  # 狄利克雷分布的浓度参数，越小数据越不均衡
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ADRL - 对抗性深度强化学习')
    
    # 基本设置
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['mnist', 'cifar10', 'cifar100', 'fashionmnist'],
                       help='数据集名称')
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['simple_cnn', 'mnist_cnn', 'simple_resnet', 'resnet18'],
                       help='模型架构')
    
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
    parser.add_argument('--scheduler', type=str, default='linear', 
                       choices=['linear', 'cosine', 'step'], help='学习率调度器')
    
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
    
    # 拜占庭攻击和防御参数
    parser.add_argument('--attack', type=str, default='none',
                       choices=['none', 'gaussian', 'lie', 'poisonedfl', 'fang', 'qlearning', 'minmax', 'minsum', 'fedsa', 'second', 'rlfl'],
                       help='拜占庭攻击方法')
    parser.add_argument('--defence', type=str, default='none',
                       choices=['none', 'median', 'trimmed_mean', 'krum', 'bulyan', 'fltrust', 'fool',  'ada','momentum'],
                       help='拜占庭防御方法')
    parser.add_argument('--n_byzantine', type=int, default=2, help='拜占庭客户端数量')
    parser.add_argument('--attack_strength', type=float, default=1.0, help='攻击强度')
    parser.add_argument('--flip_ratio', type=float, default=0.5, help='翻转比例')
    parser.add_argument('--noise_std', type=float, default=0.1, help='噪声标准差')
    parser.add_argument('--lie_z_max', type=float, default=3.0, help='LIE攻击的最大z-score')
    parser.add_argument('--lie_direction', type=str, default='negative', choices=['negative', 'positive'], 
                       help='LIE攻击方向')
    parser.add_argument('--krum_attack_lambda', type=float, default=1.0, help='Krum攻击强度')
    parser.add_argument('--trim_attack_scale', type=float, default=2.0, help='Trimmed Mean攻击缩放因子')
    parser.add_argument('--median_attack_deviation', type=float, default=3.0, help='Median攻击偏离程度')
    parser.add_argument('--trim_ratio', type=float, default=0.2, help='修剪比例')
    parser.add_argument('--f', type=int, default=2, help='最多容忍的拜占庭客户端数量')
    parser.add_argument('--krum_k', type=int, default=16, help='Krum选择的模型数量')
    
    # FLTrust防御参数
    parser.add_argument('--fltrust_threshold', type=float, default=0.0, help='FLTrust信任阈值')
    parser.add_argument('--fltrust_norm_clipping', action='store_true', default=True, help='FLTrust使用范数裁剪')
    parser.add_argument('--fltrust_root_size', type=int, default=100, help='FLTrust根数据集大小')
    parser.add_argument('--fltrust_root_batch_size', type=int, default=32, help='FLTrust根数据集批次大小')
    
    # Q-Learning参数
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
    
    # RLFL参数
    parser.add_argument('--rlfl_lr', type=float, default=3e-4, help='RLFL学习率')
    parser.add_argument('--rlfl_gamma', type=float, default=0.99, help='RLFL折扣因子')
    parser.add_argument('--rlfl_gae_lambda', type=float, default=0.95, help='RLFL GAE参数')
    parser.add_argument('--rlfl_ppo_epochs', type=int, default=10, help='RLFL PPO更新轮数')
    parser.add_argument('--rlfl_ppo_clip', type=float, default=0.2, help='RLFL PPO裁剪参数')
    parser.add_argument('--rlfl_value_loss_coef', type=float, default=0.5, help='RLFL价值损失系数')
    parser.add_argument('--rlfl_entropy_coef', type=float, default=0.01, help='RLFL熵系数')
    parser.add_argument('--rlfl_max_grad_norm', type=float, default=0.5, help='RLFL梯度裁剪')
    parser.add_argument('--rlfl_batch_size', type=int, default=64, help='RLFL批次大小')
    parser.add_argument('--rlfl_buffer_size', type=int, default=10000, help='RLFL缓冲区大小')
    parser.add_argument('--rlfl_action_dim', type=int, default=1, help='RLFL动作维度')
    parser.add_argument('--rlfl_action_scale', type=float, default=10.0, help='RLFL动作缩放')
    parser.add_argument('--simulator_max_rounds', type=int, default=50, help='RLFL模拟器最大轮数')
    parser.add_argument('--rlfl_n_episodes', type=int, default=100, help='RLFL训练回合数')
    parser.add_argument('--rlfl_max_steps_per_episode', type=int, default=50, help='RLFL每回合最大步数')
    parser.add_argument('--rlfl_pretrain', action='store_true', default=True, help='RLFL是否预训练策略')
    parser.add_argument('--rlfl_save_path', type=str, default='./rlfl_models', help='RLFL模型保存路径')
    
    # 共谋攻击参数
    parser.add_argument('--enable_collusion', action='store_true', help='启用拜占庭客户端共谋')
    parser.add_argument('--collusion_strategy', type=str, default='coordinated',
                       choices=['coordinated', 'distributed', 'adaptive'],
                       help='共谋策略: coordinated(协调), distributed(分布式), adaptive(自适应)')
    parser.add_argument('--collusion_strength', type=float, default=1.0, help='共谋强度')
    
    # Q-Learning共谋参数
    parser.add_argument('--ql_coordination_mode', type=str, default='diverse',
                       choices=['diverse', 'focused', 'adaptive'],
                       help='Q-Learning协调模式: diverse(多样化), focused(集中), adaptive(自适应)')
    parser.add_argument('--ql_diversity_bonus', type=float, default=0.1, help='Q-Learning多样化奖励')
    parser.add_argument('--ql_synergy_bonus', type=float, default=0.2, help='Q-Learning协同奖励')

    
    # AGR-agnostic攻击参数
    parser.add_argument('--minmax_safety_margin', type=float, default=0.95, help='MinMax攻击安全边界')
    parser.add_argument('--minsum_safety_margin', type=float, default=0.95, help='MinSum攻击安全边界')
    
    # 高级共谋参数
    parser.add_argument('--use_advanced_collusion', action='store_true', default=True, 
                       help='使用高级共谋攻击（包含拜占庭通信）')
    parser.add_argument('--reward_accuracy_weight', type=float, default=10.0, help='准确率奖励权重')
    parser.add_argument('--reward_collusion_weight', type=float, default=2.0, help='共谋奖励权重')
    parser.add_argument('--reward_diversity_weight', type=float, default=0.5, help='多样性奖励权重')
    parser.add_argument('--reward_stealth_weight', type=float, default=1.0, help='隐蔽性奖励权重')
    
    # Non-IID数据划分参数
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                       help='狄利克雷分布的浓度参数(仅用于MNIST)，越小数据越不均衡，推荐范围: 0.1-1.0')
    
    return parser.parse_args()


def create_config_from_args(args) -> Dict[str, Any]:
    """从命令行参数创建配置"""
    config = defaults.copy()
    
    # 更新配置
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # 映射n_byzantine到n_byzantine_clients以保持一致性
    if 'n_byzantine' in config:
        config['n_byzantine_clients'] = config['n_byzantine']
    
    # 设置设备
    if config['device'] is None:
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 根据数据集设置输入通道数
    if config['dataset'] in ['mnist', 'fashionmnist']:
        config['input_channels'] = 1
    else:
        config['input_channels'] = 3
    
    # 固定使用FedAVG策略
    config['strategy'] = 'fedavg'
    

    return config



def main():
    """主函数"""
    args = parse_args()
    config = create_config_from_args(args)
    
    logger.info(f"使用配置: {config}")
    # 创建输出目录
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    # result/large
    large_dir = os.path.join(
        config['output_dir'],
        "large"
    )

    os.makedirs(
        large_dir,
        exist_ok=True
    )

    # 第一层：防御方法
    defence_dir = os.path.join(
        large_dir,
        config['defence']
    )

    os.makedirs(
        defence_dir,
        exist_ok=True
    )

    # 第二层：攻击方法+日期
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        f"{config['attack']}_{date_str}"
    )

    experiment_dir = os.path.join(
        defence_dir,
        experiment_name
    )

    # 创建实验目录
    os.makedirs(
        experiment_dir,
        exist_ok=True
    )
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    Utils.save_results_to_json(config, config_path)

    with tempdir() as tmp_dir:
        logger.info(f"使用临时目录: {tmp_dir}")

        # 创建运行器
        config['experiment_dir'] = experiment_dir

        runner = FedAVGRunner(
            config=config,
            tmp_dir=tmp_dir,
            debug=args.debug
        )
        # 运行训练
        try:
            runner.run()
            logger.info("训练完成")
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise

        # 保存最终结果
        final_results = {
            'config': config,
            'final_metrics': {
                'server': runner.server.get_metrics(),
                'clients': {f'client_{i+1}': client.get_metrics()
                           for i, client in enumerate(runner.clients)}
            },
            'total_epochs_completed': runner.total_epochs_completed,
            'total_bytes_communicated': runner.total_bytes_communicated
        }

        results_path = os.path.join(experiment_dir, 'final_results.json')
        Utils.save_results_to_json(final_results, results_path)

        # 保存模型
        model_path = os.path.join(experiment_dir, 'final_model.pth')
        Utils.save_model(runner.server.model, model_path,
                        optimizer=runner.server.optimizer,
                        epoch=runner.total_epochs_completed)

        logger.info(f"结果已保存到: {experiment_dir}")


if __name__ == '__main__':
    from contextlib import contextmanager
    main()
