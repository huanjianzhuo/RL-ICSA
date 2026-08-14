# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         strategies.py
# Description:  FedAVG联邦学习策略
# ===========================================================================

import logging

logger = logging.getLogger(__name__)


class FedAVG:
    """联邦平均：服务器模型通过平均客户端模型更新，然后广播给客户端"""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.runner = kwargs['runner_instance']

    def do_clients_train_on_public_data(self):
        """客户端只在私有数据上训练，不在公共数据上训练"""
        return False

    def verify_input(self):
        """验证策略输入"""
        assert self.config['n_total_local_epochs'] is not None, '需要指定总本地训练轮数'
        assert self.config['n_total_local_epochs'] >= 0, '总本地训练轮数应该为正数'

        assert self.config['n_communications'] is not None, '需要指定通信轮数'
        assert self.config['n_communications'] >= 0, '通信轮数应该为正数'
        assert self.config['n_communications'] <= self.config[
            'n_total_local_epochs'], '通信轮数应该小于等于总本地训练轮数'

    def get_phase_length(self, current_round: int) -> int:
        """返回给定轮次的本地训练轮数"""
        n_epochs_total = self.config['n_total_local_epochs']
        n_communications = self.config['n_communications']

        # 将总本地训练轮数均匀分配到各通信轮次
        epochs_per_round, remainder = divmod(n_epochs_total, n_communications)
        epochs_per_round_schedule = [epochs_per_round if idx >= remainder else epochs_per_round + 1
                                     for idx in range(n_communications)]

        phase_length = epochs_per_round_schedule[current_round - 1]  # 索引从0开始
        return phase_length

    def before_local_training(self):
        """本地训练前，将服务器模型广播给所有客户端"""
        logger.info("广播服务器模型到所有客户端")
        self.runner.broadcast_server_model_to_clients()

    def after_local_training(self):
        """本地训练后，聚合客户端模型并更新服务器"""
        logger.info("聚合客户端模型并更新服务器")
        self.runner.aggregate_client_models_to_server()

    def at_round_end(self):
        """轮次结束时的操作"""
        logger.info(f"FedAVG轮次结束，当前轮次: {self.runner.current_round}")


def get_strategy(strategy_name: str, config: dict, runner_instance):
    """获取FedAVG策略（唯一支持的策略）"""
    if strategy_name.lower() != 'fedavg':
        logger.warning(f"只支持FedAVG策略，忽略策略名称: {strategy_name}")

    return FedAVG(config=config, runner_instance=runner_instance)
