# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         runners/BaseRunner.py
# Description:  基础运行器类，所有其他运行器都继承自此类
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
    """所有运行器的基类，定义通用函数"""

    def __init__(self, config: Dict[str, Any], tmp_dir: str, debug: bool):
        """初始化变量、服务器和客户端

        Args:
            config: 配置字典
            debug: 如果为True，使用本地数据集而不是集群上的特定数据集
        """
        self.history = {
            'round': [],
            'server_val_accuracy': [],
            'server_val_loss': [],
            'server_train_accuracy': [],
            'server_train_loss': [],
            'client_val_accuracies': [],  # 存储所有客户端的验证精度
            'client_train_accuracies': [],  # 存储所有客户端的训练精度
            'global_model_accuracy': [],  # 全局模型（服务器模型）的准确度
            'global_model_loss': []  # 全局模型（服务器模型）的损失
        }

        # 创建结果目录
        self.result_dir = Path('result')
        self.result_dir.mkdir(exist_ok=True)


        self.config = config
        self.debug = debug

        # 有用变量
        self.tmp_dir = tmp_dir
        logger.info(f"使用临时目录: {self.tmp_dir}")
        self.num_workers = num_workersDict[self.config['dataset']]
        self.n_classes = n_classesDict[self.config['dataset']]
        self.use_amp = torch.cuda.is_available() and self.config.get('use_amp', True)
        logger.info(f"使用AMP: {self.use_amp}")

        # 待设置的变量
        self.device = 'cuda:0'
        self.seed = None
        self.strategy = FedAVG
        self.dataloaders_public = {}
        self.total_epochs_completed = 0
        self.total_bytes_communicated = 0
        self.client_epochs_done, self.server_epochs_done = 0, 0
        self.current_round = 0

        # 配置工作设备
        self.configure_comp_device()

        # 定义策略
        self.define_strategy()

        # 验证输入
        self.verify_input()

        # 定义客户端
        self.clients = [Client(
            use_amp=self.use_amp, 
            client_id=client_id, 
            n_classes=self.n_classes, 
            tmp_dir=self.tmp_dir,
            num_workers=self.num_workers, 
            config=self.config, 
            callbacks=None, 
            device=self.device
        ) for client_id in range(1, self.config['n_clients'] + 1, 1)]

        # 定义服务器模型
        self.server = Server(
            use_amp=self.use_amp, 
            n_classes=self.n_classes, 
            tmp_dir=self.tmp_dir,
            num_workers=self.num_workers, 
            config=self.config, 
            callbacks=None, 
            device=self.device
        )

        # 在客户端间分割数据集
        self.dataset_rootPath = './datasets/' + self.config['dataset']

    def configure_comp_device(self):
        """配置工作设备（GPU/CPU，cudnn.benchmark）"""
        if 'device' in self.config:
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if 'cuda' in str(self.device):
            torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True  # 效率基准测试

    def verify_input(self):
        """验证BaseRunner的输入。此函数只检查非策略特定的事物"""
        assert self.config['n_clients'] > 0, "必须至少有一个客户端"
        
        # 验证策略输入
        self.strategy.verify_input()

    def set_seed(self, seed: Optional[int] = None):
        """设置种子"""
        if seed is None:
            # 生成随机种子
            self.seed = int((os.getpid() + 1) * time.time()) % 2 ** 32
        else:
            self.seed = seed

        # 设置唯一随机种子
        Utils.set_seed(self.seed)
        logger.info(f"设置随机种子: {self.seed}")

    def assign_dataloaders(self):
        """加载数据集并分割到客户端，创建数据加载器"""
        # 加载训练和测试数据集
        raw_train = datasetDict[self.config['dataset']](
            root=self.dataset_rootPath,
            train=True,
            download=True,
            transform=trainTransformDict[self.config['dataset']]
        )
        trainData = Utils.get_index_dataset(raw_train)

        raw_test = datasetDict[self.config['dataset']](
            root=self.dataset_rootPath,
            train=False,
            transform=testTransformDict[self.config['dataset']]
        )
        testData = Utils.get_index_dataset(raw_test)

        # 分割验证数据和根数据集（从训练数据中分割）
        # 为FLTrust防御预留一小部分干净数据作为根数据集
        defence_name = self.config.get('defence', 'none').lower()
        
        # AdaAggRL需要更大的验证集用于评估
        if defence_name == 'ada':
            n_val_samples = int(0.2 * len(trainData))  # 20%作为验证集
        else:
            n_val_samples = int(0.1 * len(trainData))  # 10%作为验证集
        
        # 检查是否使用FLTrust防御
        use_fltrust = defence_name == 'fltrust'
        
        if use_fltrust:
            # 优先使用比例，如果没有则使用绝对数量
            clean_data_ratio = self.config.get('clean_data_ratio', 0.05)  # 默认5%
            fltrust_root_size = self.config.get('fltrust_root_size', None)
            
            if fltrust_root_size is not None:
                # 使用绝对数量
                n_root_samples = min(fltrust_root_size, len(trainData) - n_val_samples)
            else:
                # 使用比例
                n_root_samples = int(clean_data_ratio * len(trainData))
            
            # 确保根数据集大小合理
            n_root_samples = max(1, min(n_root_samples, len(trainData) - n_val_samples - 100))
            n_train_samples = len(trainData) - n_val_samples - n_root_samples
            
            # 如果使用FLTrust，分割出根数据集
            trainData_private, valData, rootData = torch.utils.data.random_split(
                trainData,
                [n_train_samples, n_val_samples, n_root_samples],
                generator=torch.Generator().manual_seed(self.seed)
            )
            logger.info(f"为FLTrust防御分配根数据集: {n_root_samples} 个样本 (占总训练数据的 {n_root_samples/len(trainData)*100:.2f}%)")
        else:
            # 否则只分割训练集和验证集
            n_train_samples = len(trainData) - n_val_samples
            trainData_private, valData = torch.utils.data.random_split(
                trainData,
                [n_train_samples, n_val_samples],
                generator=torch.Generator().manual_seed(self.seed)
            )
            rootData = None

        # 定义数据加载器（只用于评估，客户端不使用）
        self.dataloaders_public = {
            'val': torch.utils.data.DataLoader(
                valData, 
                batch_size=self.config['batch_size'],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=self.num_workers
            ),
            'test': torch.utils.data.DataLoader(
                testData, 
                batch_size=self.config['batch_size'],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=self.num_workers
            )
        }

        # 为FLTrust防御创建根数据集加载器
        if rootData is not None:
            # 使用FLTrust配置的批次大小，如果没有则使用默认批次大小
            root_batch_size = self.config.get('fltrust_batch_size', 
                                            self.config.get('fltrust_root_batch_size', 
                                                           self.config.get('batch_size', 32)))
            self.root_dataloader = torch.utils.data.DataLoader(
                rootData,
                batch_size=root_batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=self.num_workers
            )
            logger.info(f"FLTrust根数据集加载器已创建，批次大小: {root_batch_size}")
        else:
            self.root_dataloader = None
        
        # 保存训练数据集引用（供FLTrust防御使用，如果需要从train_dataset创建干净数据集）
        self.train_dataset = trainData_private

        # 服务器不分配数据集（服务器不进行本地训练）
        # self.server.assign_dataset(trainData=trainData_private)

        # 将训练数据在客户端间分割
        # 对于MNIST数据集，使用狄利克雷分布进行non-IID划分
        if self.config['dataset'].lower() == 'mnist':
            logger.info("使用狄利克雷分布进行non-IID数据划分")
            trainData_private_split = self._dirichlet_split_noniid(
                trainData_private, 
                n_clients=self.config['n_clients'],
                alpha=self.config.get('dirichlet_alpha', 0.5)
            )
        else:
            # 其他数据集使用均匀分割
        splitFractions = self.config['n_clients'] * [len(trainData_private) // self.config['n_clients']]
        splitFractions[0] += len(trainData_private) % self.config['n_clients']  # 余数给第一个客户端

        # 均匀分割
        trainData_private_split = torch.utils.data.random_split(
            trainData_private, 
            splitFractions,
            generator=torch.Generator().manual_seed(self.seed)
        )

        for client in self.clients:
            client_id = client.client_id
            client_data_split = trainData_private_split[client_id - 1]
            client.assign_dataset(trainData=client_data_split)
            logger.info(f"客户端 {client_id} 有 {len(client_data_split)} 个样本")

    def _dirichlet_split_noniid(self, dataset, n_clients, alpha=0.5):
        """使用狄利克雷分布进行non-IID数据划分
        
        Args:
            dataset: 要划分的数据集
            n_clients: 客户端数量
            alpha: 狄利克雷分布的浓度参数，越小数据越不均衡
        
        Returns:
            list: 每个客户端的数据子集列表
        """
        # 获取数据集的标签
        if hasattr(dataset, 'dataset'):
            # 如果是Subset，需要获取原始数据集
            labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        else:
            labels = np.array(dataset.targets)
        
        n_classes = self.n_classes
        n_samples = len(labels)
        
        # 为每个客户端初始化索引列表
        client_indices = [[] for _ in range(n_clients)]
        
        # 对每个类别进行处理
        for k in range(n_classes):
            # 获取该类别的所有样本索引
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # 使用狄利克雷分布生成每个客户端获得该类别样本的比例
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            
            # 根据比例分配样本
            proportions = np.array([p * (len(idx_j) < n_samples / n_clients) 
                                   for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 将样本分配给各个客户端
            client_idx_k = np.split(idx_k, proportions)
            for i in range(n_clients):
                client_indices[i].extend(client_idx_k[i].tolist())
        
        # 为每个客户端创建Subset
        client_datasets = []
        for i in range(n_clients):
            # 如果原始dataset是Subset，需要映射回原始索引
            if hasattr(dataset, 'dataset'):
                # 将client_indices[i]映射回原始数据集的索引
                original_indices = [dataset.indices[idx] for idx in client_indices[i]]
                client_subset = torch.utils.data.Subset(dataset.dataset, original_indices)
            else:
                client_subset = torch.utils.data.Subset(dataset, client_indices[i])
            
            client_datasets.append(client_subset)
            
            # 统计每个客户端的类别分布
            if hasattr(dataset, 'dataset'):
                client_labels = [dataset.dataset.targets[dataset.indices[j]] for j in client_indices[i]]
            else:
                client_labels = [dataset.targets[idx] for idx in client_indices[i]]
            
            class_counts = np.bincount(client_labels, minlength=n_classes)
            logger.info(f"客户端 {i+1} 类别分布: {class_counts.tolist()}")
        
        return client_datasets

    def define_strategy(self):
        """定义训练策略"""
        from strategies import get_strategy
        self.strategy = get_strategy(
            strategy_name=self.config['strategy'],
            config=self.config,
            runner_instance=self
        )

    def log_at_round_end(self, round: int, round_n_epochs: int, round_runtime: float):
        """在轮次结束时记录日志"""
        # 记录历史数据
        # 注释掉server的评估记录，改为记录clients的平均值
        self.history['round'].append(round)
        
        # 计算clients的平均验证和训练准确率/损失
        client_val_accs = []
        client_val_losses = []
        client_train_accs = []
        client_train_losses = []
        for client in self.clients:
            client_metrics = client.get_metrics()
            client_val_accs.append(client_metrics.get('val', {}).get('accuracy', 0))
            client_val_losses.append(client_metrics.get('val', {}).get('loss', 0))
            client_train_accs.append(client_metrics.get('train', {}).get('accuracy', 0))
            client_train_losses.append(client_metrics.get('train', {}).get('loss', 0))
        
        # 使用clients的平均值作为整体性能指标
        avg_val_acc = sum(client_val_accs) / len(client_val_accs) if client_val_accs else 0
        avg_val_loss = sum(client_val_losses) / len(client_val_losses) if client_val_losses else 0
        avg_train_acc = sum(client_train_accs) / len(client_train_accs) if client_train_accs else 0
        avg_train_loss = sum(client_train_losses) / len(client_train_losses) if client_train_losses else 0
        
        self.history['server_val_accuracy'].append(avg_val_acc)
        self.history['server_val_loss'].append(avg_val_loss)
        self.history['server_train_accuracy'].append(avg_train_acc)
        self.history['server_train_loss'].append(avg_train_loss)
        
        # 记录每个客户端的精度（复用已计算的值）
        self.history['client_val_accuracies'].append(client_val_accs)
        self.history['client_train_accuracies'].append(client_train_accs)

        # 获取涉及所有客户端的指标
        loggingDict = {}

        # 添加轮次指标
        loggingDict.update({
            "round": round,
            "round_runtime": round_runtime,
            "round_n_epochs": round_n_epochs,
            "epoch": self.total_epochs_completed
        })

        # 添加服务器指标
        server_metrics = self.server.get_metrics()
        loggingDict.update({"server": server_metrics})

        # 记录到控制台
        epoch = self.total_epochs_completed
        loss = server_metrics.get('val', {}).get('loss', None)
        acc = server_metrics.get('val', {}).get('accuracy', None)
        
        logger.info(f"轮次 {round} 完成 - Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def plot_accuracy_curves(self, save_path: str = None):
        """绘制精度折线图并保存

        Args:
            save_path: 保存路径，如果为None则使用默认路径
        """
        if not self.history['round']:
            logger.warning("没有历史数据可绘制")
            return

        if save_path is None:
            # 获取数据集、攻击和防御名称
            dataset_name = self.config.get('dataset', 'unknown')
            attack_name = self.config.get('attack', 'no_attack')
            defence_name = self.config.get('defence', 'no_defence')
            
            # 处理名称映射
            attack_display = attack_name if attack_name not in ['none', 'no_attack'] else 'NoAttack'
            defence_display = defence_name if defence_name not in ['none', 'no_defence'] else 'NoDefence'
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = self.result_dir / f"{dataset_name}_{attack_display}_{defence_display}_{timestamp}.png"

        # 只绘制全局模型准确率
        plt.figure(figsize=(10, 6))
        
        if self.history['global_model_accuracy']:
            # 获取攻击和防御名称用于标题
            attack_name = self.config.get('attack', 'no_attack')
            defence_name = self.config.get('defence', 'no_defence')
            
            attack_display = attack_name if attack_name not in ['none', 'no_attack'] else 'No Attack'
            defence_display = defence_name if defence_name not in ['none', 'no_defence'] else 'No Defence'
            
            plt.plot(self.history['round'][1:], self.history['global_model_accuracy'],
                     'r-', label='Global Model Accuracy', linewidth=2, marker='o')
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Global Model Accuracy - Attack: {attack_display}, Defence: {defence_display}', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
            # 添加数值标注（可选）
            for i, (x, y) in enumerate(zip(self.history['round'][1:], self.history['global_model_accuracy'])):
                if i % 2 == 0:  # 每隔一个点标注，避免过于密集
                    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                               xytext=(0,5), ha='center', fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"精度曲线已保存到: {save_path}")
        plt.close()

    def save_training_history(self, save_path: str = None):
        """保存训练历史到CSV文件

        Args:
            save_path: 保存路径，如果为None则使用默认路径
        """
        if save_path is None:
            # 获取数据集、攻击和防御名称
            dataset_name = self.config.get('dataset', 'unknown')
            attack_name = self.config.get('attack', 'no_attack')
            defence_name = self.config.get('defence', 'no_defence')
            
            # 处理名称映射
            attack_display = attack_name if attack_name not in ['none', 'no_attack'] else 'NoAttack'
            defence_display = defence_name if defence_name not in ['none', 'no_defence'] else 'NoDefence'
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = self.result_dir / f"{dataset_name}_{attack_display}_{defence_display}_{timestamp}.csv"

        # 创建DataFrame
        history_data = {
            'round': self.history['round'],
            'server_train_accuracy': self.history['server_train_accuracy'],
            'server_train_loss': self.history['server_train_loss'],
        }
        
        # 添加全局模型数据（从第1轮开始）
        if self.history['global_model_accuracy']:
            # 为第0轮添加NaN
            global_acc = [float('nan')] + self.history['global_model_accuracy']
            global_loss = [float('nan')] + self.history['global_model_loss']
            history_data['global_model_accuracy'] = global_acc[:len(self.history['round'])]
            history_data['global_model_loss'] = global_loss[:len(self.history['round'])]
        
        df = pd.DataFrame(history_data)
        df.to_csv(save_path, index=False)
        logger.info(f"训练历史已保存到: {save_path}")

    def log_clients_at_epoch_end(self, epoch: int, commit: bool = True):
        """记录所有客户端信息"""
        loggingDict = {f"client{client.client_id}": client.get_metrics() for client in self.clients}
        loggingDict.update({"client_epoch": epoch, "epoch": epoch})
        
        # 记录到控制台
        for client in self.clients:
            metrics = client.get_metrics()
            train_acc = metrics.get('train', {}).get('accuracy', 0)
            val_acc = metrics.get('val', {}).get('accuracy', 0)
            logger.info(f"客户端 {client.client_id} - 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")

    def log_server(self, epoch: int, commit: bool = True):
        """记录服务器到日志"""
        loggingDict = {
            "server": self.server.get_metrics(),
            "server_epoch": epoch,
            "epoch": epoch
        }
        
        # 记录到控制台
        metrics = self.server.get_metrics()
        train_acc = metrics.get('train', {}).get('accuracy', 0)
        val_acc = metrics.get('val', {}).get('accuracy', 0)
        logger.info(f"服务器 - 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")

    def final_log(self, client: Optional[Client] = None):
        """执行最终评估和记录"""
        logger.info("最终记录")
        # 客户端不评估
        # actors = [client] if client else [client for client in self.clients]
        # 
        # for actor in actors:
        #     actor.reset_averaged_metrics()
        #     self.evaluate_model(actor=actor, data='val')
        #     self.evaluate_model(actor=actor, data='test')
        
        # 生成统一的时间戳和文件名前缀
        dataset_name = self.config.get('dataset', 'unknown')
        attack_name = self.config.get('attack', 'no_attack')
        defence_name = self.config.get('defence', 'no_defence')
        attack_display = attack_name if attack_name not in ['none', 'no_attack'] else 'NoAttack'
        defence_display = defence_name if defence_name not in ['none', 'no_defence'] else 'NoDefence'
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 使用统一的文件名保存图表和历史数据
        base_filename = f"{dataset_name}_{attack_display}_{defence_display}_{timestamp}"
        png_path = self.result_dir / f"{base_filename}.png"
        csv_path = self.result_dir / f"{base_filename}.csv"
        
        # 绘制并保存精度曲线
        self.plot_accuracy_curves(save_path=png_path)
        self.save_training_history(save_path=csv_path)

    def train_epoch(self, actor: Actor, data: str = 'train', is_training: bool = True, epoch: Optional[int] = None):
        """训练actor一个epoch。也用于评估"""
        assert data in ['train', 'val', 'test']
        assert not (data in ['test', 'val'] and is_training), "不能在测试/验证集上训练"
        
        if data == 'train':
            loader = actor.dataloader  # 使用客户端的私有训练数据
            assert actor.actor_type == 'client' or not is_training, "不能在标记的公共训练数据集上训练服务器"
        else:
            loader = self.dataloaders_public[data]

        epochStr = f"\nEpoch {epoch} - " if epoch is not None else ""
        logger.info(f"{epochStr}{'训练' if is_training else '评估'} {actor.actor_name} 在{'私有' if data == 'train' else '公共'}数据上")
        
        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, y_target, _ in pbar:
                    # 移动到CUDA如果可能
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    actor.optimizer.zero_grad()  # 清零梯度缓冲区

                    if is_training:
                        with autocast(enabled=(self.use_amp is True)):
                            output = actor.model.train()(x_input)
                            loss = actor.loss_criterion(output, y_target)

                        if self.use_amp:
                            actor.gradScaler.scale(loss).backward()  # AMP梯度缩放 + 反向传播
                            actor.gradScaler.step(actor.optimizer)  # 优化步骤
                            actor.gradScaler.update()  # 更新AMP gradScaler
                        else:
                            loss.backward()
                            actor.optimizer.step()

                        actor.scheduler.step()
                    else:
                        with autocast(enabled=(self.use_amp is True)):
                            # 我们使用train(mode=True)用于训练数据集，这样不会因为BN的运行平均而损失下降
                            output = actor.model.train(mode=(data == 'train'))(x_input)
                            loss = actor.loss_criterion(output, y_target)

                    actor.update_batch_metrics(mode=data, loss=loss, output=output, y_target=y_target)

    def evaluate_model(self, actor: Actor, data: str = 'train'):
        """评估客户端在给定数据上的模型"""
        self.train_epoch(actor=actor, data=data, is_training=False)
