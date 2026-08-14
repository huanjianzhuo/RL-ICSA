# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         utilities.py
# Description:  工具函数
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlteredDatasetWrapper(torch.utils.data.Dataset):
    """包装数据集以返回索引"""
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
        """暴露原始数据集的targets属性"""
        if hasattr(self.dataset, 'targets'):
            return self.dataset.targets
        elif hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        else:
            raise AttributeError(f"原始数据集 {type(self.dataset)} 没有 'targets' 或 'labels' 属性")


class Utilities:
    """工具函数类"""

    @staticmethod
    def get_index_dataset(original_dataset):
        """为数据集添加索引"""
        return AlteredDatasetWrapper(original_dataset)

    @staticmethod
    def reset_dataset_subset_indices(dataset: torch.utils.data.dataset.Subset):
        """重置数据集子集索引"""
        dataset.indices = list(range(len(dataset.indices)))

    @staticmethod
    def get_model_communication_cost(model: torch.nn.Module) -> int:
        """计算模型通信成本（字节）"""
        total_bytes = 0
        for param in model.parameters():
            bytes_per_number = param.element_size()
            total_bytes += param.numel() * bytes_per_number
        return int(total_bytes)

    @staticmethod
    def calculate_communication_cost(tensorList: List[torch.Tensor]) -> int:
        """计算张量列表的通信成本（字节）"""
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
        """平均聚合客户端模型"""
        if not client_model_list:
            raise ValueError("客户端模型列表不能为空")
            
        average_state_dict = OrderedDict()
        factor = 1.0 / len(client_model_list)
        
        # 确定目标设备（使用第一个模型的设备）
        first_model = client_model_list[0]
        if isinstance(first_model, dict):
            # 如果是state_dict，从第一个参数获取设备
            first_param = next(iter(first_model.values()))
            target_device = first_param.device
        else:
            # 如果是模型对象，从模型参数获取设备
            first_param = next(first_model.parameters())
            target_device = first_param.device
        
        for client_model in client_model_list:
            # 如果是模型对象，获取其state_dict；如果已经是state_dict，直接使用
            if isinstance(client_model, dict):
                client_state_dict = client_model
            else:
                client_state_dict = client_model.state_dict()
            
            for key in client_state_dict:
                # 确保所有张量在同一设备上
                param = client_state_dict[key].clone().detach().to(target_device)
                if key not in average_state_dict:
                    average_state_dict[key] = factor * param
                else:
                    average_state_dict[key] += factor * param

        return average_state_dict

    @staticmethod
    def save_results_to_json(results: Dict[str, Any], filepath: str):
        """保存结果到JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(Utilities._to_serializable(results), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_serializable(obj):
        """将对象递归转换为可JSON序列化的结构"""
        # 基本类型直接返回
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        # PyTorch Tensor
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
        # Numpy 标量
        if isinstance(obj, (np.generic,)):
            return obj.item()
        # Numpy 数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 字典
        if isinstance(obj, dict):
            return {k: Utilities._to_serializable(v) for k, v in obj.items()}
        # 可迭代（列表/元组/集合）
        if isinstance(obj, (list, tuple, set)):
            return [Utilities._to_serializable(v) for v in obj]
        # 其他常见对象转字符串
        if isinstance(obj, (torch.device,)):
            return str(obj)
        return str(obj)

    @staticmethod
    def load_results_from_json(filepath: str) -> Dict[str, Any]:
        """从JSON文件加载结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
        """创建实验目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    @staticmethod
    def log_metrics(metrics: Dict[str, Any], epoch: int, prefix: str = ""):
        """记录指标"""
        log_str = f"Epoch {epoch}"
        if prefix:
            log_str += f" [{prefix}]"
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_str += f", {key}: {value:.4f}"
            else:
                log_str += f", {key}: {value}"
        
        logger.info(log_str)

    @staticmethod
    def compute_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """计算模型在数据加载器上的准确率"""
        model.eval()
        accuracy_meter = Accuracy(num_classes=10).to(device=device)
        
        with torch.no_grad():
            for x_input, y_target, _ in dataloader:
                x_input = x_input.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                
                output = model(x_input)
                accuracy_meter(output, y_target)
        
        return float(accuracy_meter.compute())

    @staticmethod
    def compute_loss(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                    criterion: nn.Module, device: torch.device) -> float:
        """计算模型在数据加载器上的损失"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_input, y_target, _ in dataloader:
                x_input = x_input.to(device, non_blocking=True)
                y_target = y_target.to(device, non_blocking=True)
                
                output = model(x_input)
                loss = criterion(output, y_target)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    @staticmethod
    def set_seed(seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def save_model(model: nn.Module, filepath: str, optimizer=None, epoch=None, loss=None):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        logger.info(f"模型已保存到: {filepath}")

    @staticmethod
    def load_model(model: nn.Module, filepath: str, optimizer=None, device=None):
        """加载模型"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        
        logger.info(f"模型已从 {filepath} 加载，epoch: {epoch}, loss: {loss}")
        return epoch, loss

class WarmupLRWrapper:
    """Takes an existing optimizer with corresponding scheduler and warms-up the learning rate."""

    def __init__(self, optimizer, scheduler, warmup_steps):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Simulate the scheduler for warmup_steps many steps
        for _ in range(self.warmup_steps):
            self.scheduler.step()

        # Get the end lr
        self.end_lr = [pg['lr'] for pg in self.optimizer.param_groups]

        # Set initial lr to a small starting value (i.e. 1/2 of the first actual warmup value)
        for pg_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.end_lr[pg_idx] * 0.5 / self.warmup_steps

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Set the lr to the warmup value (from 0 to self.end_lr)
            for pg_idx, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.end_lr[pg_idx] * self.current_step / self.warmup_steps
        else:
            self.scheduler.step()


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()
