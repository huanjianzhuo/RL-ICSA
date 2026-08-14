# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         runners/FedAVGRunner.py
# Description:  FedAVG运行器类，用于从零开始运行
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
    """处理联邦训练，通过并发训练客户端和服务器。我们不再获取预训练模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_round = None
        self.attack = None
        self.defence = None

    @torch.no_grad()
    def broadcast_server_model_to_clients(self):
        """将服务器模型参数广播给所有客户端"""
        logger.info("将服务器模型广播给客户端")
        server_state_dict = self.server.model.state_dict()
        for client in self.clients:
            client.model.load_state_dict(server_state_dict)

        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    @torch.no_grad()
    def aggregate_client_models_to_server(self):
        """聚合客户端模型并更新服务器"""
        logger.info("聚合客户端模型并更新服务器")
        
        # 获取所有客户端模型
        client_models = Utils.get_client_models(self.clients)
        
        # 应用攻击（如果有）
        if self.attack is not None:
            logger.info(f"应用攻击: {self.config['attack']}")
            perturbed_models = self.attack.get_perturbed_client_models()
            
            # ✅ 修复：为PoisonedFL攻击更新历史信息
            if hasattr(self.attack, 'update_history'):
                # 提取所有扰动模型的参数向量
                param_list = []
                for model in perturbed_models:
                    if model is not None:
                        params_vector = torch.cat([p.data.view(-1) for p in model.parameters()])
                        param_list.append(params_vector)
                
                if param_list:
                    self.attack.update_history(param_list)
                    logger.info(f"[PoisonedFL] 历史信息已更新，当前轮次: {self.attack.current_round - 1}")
        else:
            perturbed_models = client_models
        
        # 应用防御（如果有）
        if self.defence is not None:
            logger.info(f"应用防御: {self.config['defence']}")
            aggregated_model = self.defence.get_aggregated_model(perturbed_models)
        else:
            # 简单平均聚合
            aggregated_model = Utils.average_client_models(perturbed_models)
        
        # 更新服务器模型
        self.server.model.load_state_dict(aggregated_model)

        del client_models
        del perturbed_models
        del aggregated_model
        # 计算通信成本
        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    def set_client_models(self):
        """为每个客户端：初始化模型"""
        for client in self.clients:
            client.set_model(reinit=True, fileName=None)

    def set_client_optimizers(self, reinit_optimizer: bool = True, lr_duration: Optional[int] = None):
        """设置客户端的优化器/调度器"""
        for client in self.clients:
            n_batches_per_epoch = len(client.dataloader)
            n_epochs = lr_duration or self.config["n_total_local_epochs"]
            client.set_optimizer_and_scheduler(
                n_epochs=n_epochs, 
                n_batches_per_epoch=n_batches_per_epoch,
                reinit_optimizer=reinit_optimizer
            )

    def set_server_optimizer(self, reinit_server: bool, first_init: bool):
        """设置服务器的优化器/调度器（服务器不进行本地训练，不需要优化器）"""
        # 服务器不进行本地训练，不需要优化器
        logger.info("服务器不进行本地训练，跳过优化器设置")
    
    def set_byzantine_attack_defence(self):
        """设置拜占庭攻击和防御"""
        print(f"[DEBUG] set_byzantine_attack_defence 被调用")
        print(f"[DEBUG] config中的defence值: {self.config.get('defence', 'NOT_FOUND')}")
        # 为AdaAggRL防御准备验证数据
        defence_name = self.config.get('defence', 'no_defence')
        print(f"[DEBUG] defence_name = '{defence_name}'")
        if defence_name.lower() == 'ada':
            try:
                logger.info("为AdaAggRL防御准备验证数据")
                
                # 检查验证数据加载器是否存在
                if not hasattr(self, 'dataloaders_public') or 'val' not in self.dataloaders_public:
                    logger.warning("验证数据加载器不存在，跳过验证数据准备")
                else:
                    # 从验证数据加载器中提取所有数据
                    x_val_list = []
                    y_val_list = []
                    
                    for x_batch, y_batch, _ in self.dataloaders_public['val']:
                        x_val_list.append(x_batch)
                        y_val_list.append(y_batch)
                    
                    x_val = torch.cat(x_val_list, dim=0)
                    y_val = torch.cat(y_val_list, dim=0)
                    
                    # 确保y_val是长整型（类别索引，不是one-hot编码）
                    if y_val.dtype != torch.long:
                        y_val = y_val.long()
                    
                    # ✅ 修复：AdaAggRL 需要类别索引，不是 one-hot 编码
                    # 添加到配置中
                    self.config['x_val'] = x_val
                    self.config['y_val'] = y_val  # 直接使用类别索引
                    self.config['num_labels'] = self.n_classes  # 添加类别数量
                    
                    logger.info(f"AdaAggRL验证数据已准备: x_val shape={x_val.shape}, y_val shape={y_val.shape}, num_labels={self.n_classes}")
            except Exception as e:
                logger.warning(f"准备AdaAggRL验证数据失败: {e}")
        
        # 设置拜占庭客户端
        n_byzantine = self.config.get('n_byzantine_clients', 0)
        if n_byzantine > 0:
            logger.info(f"设置 {n_byzantine} 个拜占庭客户端")
            byzantine_indices = torch.randperm(len(self.clients))[:n_byzantine]
            for idx in byzantine_indices:
                self.clients[idx].is_byzantine = True
                logger.info(f"客户端 {self.clients[idx].client_id} 被设置为拜占庭客户端")
        
        # 设置攻击
        attack_name = self.config.get('attack', 'no_attack')
        print(f"[DEBUG] 攻击配置: attack_name='{attack_name}', type={type(attack_name)}")
        logger.info(f"检查攻击配置: attack_name='{attack_name}'")
        print(f"[DEBUG] 检查条件: attack_name != 'no_attack' = {attack_name != 'no_attack'}")
        print(f"[DEBUG] 检查条件: attack_name != 'none' = {attack_name != 'none'}")
        print(f"[DEBUG] 检查条件: attack_name.lower() not in ['no_attack', 'none'] = {attack_name.lower() not in ['no_attack', 'none']}")
        
        if attack_name.lower() not in ['no_attack', 'none']:
            try:
                logger.info(f"尝试初始化攻击: {attack_name}")
                self.attack = get_attack(attack_name, self.clients, self.config, self)
                logger.info(f"✓ 设置攻击成功: {attack_name}")
                print(f"[DEBUG] self.attack = {self.attack}")
                print(f"[DEBUG] self.attack is not None = {self.attack is not None}")

                # 如果是Q-Learning攻击，设置性能跟踪
                if attack_name == 'qlearning':
                    self.previous_performance = None
                    self.current_performance = None
            except Exception as e:
                logger.error(f"✗ 设置攻击失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.attack = None
        else:
            logger.info(f"未启用攻击 (attack_name='{attack_name}')")
            self.attack = None
        
        # 设置防御
        print(f"[DEBUG] 检查防御配置: defence_name='{defence_name}', type={type(defence_name)}")
        logger.info(f"检查防御配置: defence_name='{defence_name}'")
        print(f"[DEBUG] 检查条件: defence_name != 'no_defence' = {defence_name != 'no_defence'}")
        print(f"[DEBUG] 检查条件: defence_name != 'none' = {defence_name != 'none'}")
        if defence_name != 'no_defence' and defence_name != 'none':
            try:
                logger.info(f"尝试初始化防御: {defence_name}")
                self.defence = get_defence(defence_name, self.clients, self.config, self)
                logger.info(f"✓ 设置防御成功: {defence_name}")
            except Exception as e:
                logger.error(f"✗ 设置防御失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.defence = None
        else:
            logger.info(f"未启用防御 (defence_name='{defence_name}')")
            self.defence = None

    @torch.no_grad()
    def compute_accuracy(self, loader, prediction):
        """计算加载器上的准确率，其中prediction是包含集成所有预测的张量"""
        logger.info("评估集成准确率")
        accuracy_meter = Accuracy(num_classes=self.n_classes).to(device=self.device)
        
        with tqdm(loader, leave=True) as pbar:
            for _, y_target, indices in pbar:
                y_target = y_target.to(device=self.device)
                accuracy_meter(prediction[indices], y_target)

        return accuracy_meter.compute()

    @torch.no_grad()
    def evaluate_global_model(self):
        """评估全局模型（服务器模型）在所有客户端私有数据上的准确度"""
        logger.info("评估全局模型准确度")
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        self.server.model.eval()
        
        # 在所有客户端的私有数据上评估全局模型
        for client in self.clients:
            with tqdm(client.dataloader, desc=f"评估客户端-{client.client_id}", leave=False) as pbar:
                for x_input, y_target, _ in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    
                    with autocast(enabled=(self.use_amp is True)):
                        output = self.server.model(x_input)
                        loss = self.server.loss_criterion(output, y_target)
                    
                    # 计算准确率
                    pred = output.argmax(dim=1)
                    correct = (pred == y_target).sum().item()
                    
                    total_correct += correct
                    total_samples += y_target.size(0)
                    total_loss += loss.item() * y_target.size(0)
        
        # 计算平均准确率和损失
        global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        global_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # 记录到history
        self.history['global_model_accuracy'].append(global_accuracy)
        self.history['global_model_loss'].append(global_loss)
        
        logger.info(f"全局模型准确率: {global_accuracy:.4f}, 损失: {global_loss:.4f}")
        
        return global_accuracy, global_loss

    def train_client_local(self, n_epochs: int, current_round: int):
        """在本地数据集上训练每个客户端n_epochs轮"""
        for epoch in range(1, n_epochs + 1, 1):
            for client in self.clients:
                client.reset_averaged_metrics()  # 重置客户端指标
                
                if client.is_byzantine:
                    attack_name = self.config.get('attack', 'no_attack').lower()
                    # Q-Learning 攻击：跳过主模型训练，只训练 Q 网络
                    if attack_name == 'qlearning':
                        logger.info(f"\n轮次 {current_round}/{self.config['n_communications']} - 本地Epoch {epoch}/{n_epochs}: 客户端-{client.client_id} (Q-Learning拜占庭，跳过主模型训练)")
                        if client.q_network is not None and self.attack is not None:
                            if hasattr(self.attack, '_learn_from_memory'):
                                if len(client.q_memory) >= self.attack.batch_size:
                                    q_loss = self.attack._learn_from_memory(client)
                                    if q_loss is not None:
                                        logger.info(f"客户端-{client.client_id} Q网络训练损失: {q_loss:.4f}")
                    else:
                        # 其他攻击（如 SecondHighestConfidenceAttack）：先正常训练主模型，
                        # 攻击扰动在 get_perturbed_client_models() 聚合阶段统一施加
                        logger.info(f"\n轮次 {current_round}/{self.config['n_communications']} - 本地Epoch {epoch}/{n_epochs}: 本地训练客户端-{client.client_id} (拜占庭客户端，正常训练后聚合时施加攻击)")
                        self.train_epoch(actor=client, data='train', epoch=epoch)
                else:
                    # 正常客户端训练主模型
                    logger.info(f"\n轮次 {current_round}/{self.config['n_communications']} - 本地Epoch {epoch}/{n_epochs}: 本地训练客户端-{client.client_id}")
                    self.train_epoch(actor=client, data='train', epoch=epoch)
                
                # 客户端不评估
                # self.evaluate_model(actor=client, data='val')
                # if epoch == n_epochs:
                #     self.evaluate_model(actor=client, data='test')

            self.log_clients_at_epoch_end(epoch=self.client_epochs_done + epoch, commit=True)
        self.client_epochs_done += n_epochs

    def train_federated(self):
        """以联邦方式训练服务器和客户端"""
        for current_round in range(0, self.config['n_communications'] + 1, 1):
            self.current_round = current_round
            is_training = current_round > 0
            
            logger.info(f"\nFL - 轮次 {current_round}/{self.config['n_communications']}") if is_training else logger.info(f"\nFL - 评估轮次")
            t_start = time.time()

            # 重置所有actor的指标
            for client in self.clients:
                client.reset_averaged_metrics()
            self.server.reset_averaged_metrics()

            if is_training:
                # 本地训练前，服务器可能向客户端发送聚合信息
                self.strategy.before_local_training()

                # 确定此轮次的epoch数
                round_n_epochs = self.strategy.get_phase_length(current_round=current_round)

                if self.config.get('restart_client_lr'):
                    self.set_client_optimizers(reinit_optimizer=False, lr_duration=round_n_epochs)
                
                if self.config.get('reinit_server'):
                    self.server.set_model(reinit=True)
                    self.set_server_optimizer(reinit_server=self.config['reinit_server'], first_init=False)

                self.train_client_local(n_epochs=round_n_epochs, current_round=current_round)
                
                # 本地训练后，客户端可能向服务器发送聚合信息
                self.strategy.after_local_training()
            else:
                round_n_epochs = 0

            # 评估全局模型（服务器模型）的准确度
            # 使用所有客户端的私有数据来评估全局模型
            if is_training:  # 只在训练轮次后评估
                self.evaluate_global_model()
            
            # 更新Q-Learning攻击的性能信息（使用clients的平均性能）
            if self.attack is not None and hasattr(self.attack, 'update_performance'):
                # 计算所有clients在训练集上的平均准确率（不使用公共验证集）
                total_train_acc = 0.0
                num_clients = 0
                for client in self.clients:
                    client_metrics = client.get_metrics()
                    train_acc = client_metrics.get('train', {}).get('accuracy', 0.0)
                    if train_acc > 0:
                        total_train_acc += train_acc
                        num_clients += 1
                
                current_performance = total_train_acc / num_clients if num_clients > 0 else 0.0
                self.attack.update_performance(current_performance)
                
                # 更新奖励（如果有性能变化）
                if hasattr(self, 'previous_performance') and self.previous_performance is not None:
                    self.attack.update_reward(self.previous_performance, current_performance)
                
                self.previous_performance = current_performance
            
            # ✅ 为 RLFL 攻击更新分布学习和奖励
            if self.attack is not None and hasattr(self.attack, 'update_distribution'):
                logger.info("更新 RLFL 攻击的分布学习器")
                
                # 更新分布学习器（从全局模型学习数据分布）
                self.attack.update_distribution(self.server.model, gradient=None)
                
                # 更新奖励（基于全局模型性能）
                if hasattr(self.attack, 'update_reward') and len(self.history['global_model_loss']) >= 2:
                    previous_loss = self.history['global_model_loss'][-2]
                    current_loss = self.history['global_model_loss'][-1]
                    self.attack.update_reward(previous_loss, current_loss)
                    logger.info(f"RLFL 奖励更新: 上一轮损失={previous_loss:.4f}, 当前损失={current_loss:.4f}")
            
            self.strategy.at_round_end()  # 轮次结束时的策略特定操作
            self.total_epochs_completed += round_n_epochs
            self.log_at_round_end(round=current_round, round_n_epochs=round_n_epochs,
                                 round_runtime=time.time() - t_start)

    def save_global_accuracy_curve(self):
        """
        保存全局模型accuracy变化曲线
        """

        import matplotlib.pyplot as plt
        import os

        accuracy_list = self.history['global_model_accuracy']

        if len(accuracy_list) == 0:
            logger.warning("没有全局模型accuracy数据，跳过绘图")
            return

        # 获取实验目录
        save_dir = self.config.get('experiment_dir', './result')

        os.makedirs(save_dir, exist_ok=True)

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
            f"全局模型精度曲线保存至: {save_path}"
        )
    def run(self):
        """控制工作流程的函数"""
        # 在设置种子之前初始化模型！
        self.set_client_models()  # 每个客户端初始化自己的模型
        self.server.set_model(reinit=True)  # 服务器初始化自己的模型
        self.set_seed()  # 设置种子

        # 设置种子后初始化数据加载器！
        self.assign_dataloaders()  # 分配数据加载器

        self.set_client_optimizers()
        self.set_server_optimizer(reinit_server=self.config.get('reinit_server', False), first_init=True)

        # 设置拜占庭攻击和防御
        print("[DEBUG] 即将调用 set_byzantine_attack_defence()")
        self.set_byzantine_attack_defence()
        print("[DEBUG] set_byzantine_attack_defence() 调用完成")

        self.train_federated()
        self.save_global_accuracy_curve()
        self.final_log()  # 记录所有客户端和服务器
