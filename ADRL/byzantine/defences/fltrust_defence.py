# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/fltrust_defence.py
# Description:  FLTrust防御
# ===========================================================================

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
import copy
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class FLTrustDefence(ByzantineDefence):
    """
    FLTrust防御：基于信任引导的拜占庭鲁棒联邦学习
    
    论文: "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"
    
    实现基于ourFLTrust项目：
    https://github.com/yourusername/ourFLTrust-main
    
    核心算法：
    1. 服务器将全局模型发送给所有客户端
    2. 客户端在本地数据上训练并返回模型更新 delta_i = w_new - w_old
    3. 服务器在根数据集（干净数据）上训练并得到根更新 delta_0
    4. 计算信任分数和归一化：
       - TS_i = ReLU(cosine_similarity(delta_i, delta_0))
       - norm_i = ||delta_0|| / ||delta_i||
       - TSnorm_i = TS_i * norm_i
    5. 加权聚合：delta = (Σ TSnorm_i * delta_i) / (Σ TS_i)
    6. 更新全局模型：w = w + alpha * delta
    
    关键特性：
    1. 使用根数据集（干净数据）作为信任锚点
    2. ReLU裁剪确保只信任与根更新方向相似的客户端
    3. 范数归一化防止恶意客户端通过放大更新来主导聚合
    4. 完全按照原始FLTrust论文和ourFLTrust实现
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.trust_threshold = config.get('fltrust_threshold', 0.0)  # 信任阈值（可选）
        self.clean_data_ratio = config.get('clean_data_ratio', 0.05)  # 干净数据集比例
        self.clean_dataset = None  # 服务器的干净数据集
        self.clean_dataloader = None  # 干净数据集的DataLoader
        self.baseline_accuracy = 0.0  # 基准准确率
        self._clean_dataset_initialized = False  # 标记是否已初始化干净数据集
        
        # FLTrust参数（与ourFLTrust保持一致）
        self.learning_rate = config.get('fltrust_lr', config.get('lr', 0.01))  # 学习率
        self.epochs = config.get('fltrust_epochs', 1)  # 本地训练轮数
        self.batch_size = config.get('fltrust_batch_size', config.get('batch_size', 32))  # 批次大小
        self.alpha = config.get('fltrust_alpha', 1.0)  # 全局聚合学习率（通常为 lr/lr = 1）
        
        logger.info(f"FLTrust防御已初始化（基于ourFLTrust实现）")
        logger.info(f"参数: 学习率={self.learning_rate}, 训练轮数={self.epochs}, 批次大小={self.batch_size}, alpha={self.alpha}")
    
    def _train_on_dataset(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """
        在给定数据集上训练模型并返回模型更新
        
        对应ourFLTrust中Client的train方法：
        - 使用服务器模型权重初始化
        - 在本地数据上训练指定轮数
        - 返回 delta_weights = new_weights - old_weights
        
        Args:
            model: 当前全局模型
            dataloader: 训练数据集
        
        Returns:
            delta_weights: 模型更新量（字典形式）
        """
        device = self.runner.device
        
        # 保存原始模型权重
        original_weights = OrderedDict()
        for name, param in model.state_dict().items():
            original_weights[name] = param.clone().detach()
        
        # 创建模型副本用于训练
        model_copy = copy.deepcopy(model).to(device)
        model_copy.train()
        
        # 确保所有参数都启用梯度追踪
        for param in model_copy.parameters():
            param.requires_grad_(True)
        
        # 创建优化器
        trainable_params = [p for p in model_copy.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            logger.warning("模型没有可训练参数，返回零更新")
            return {name: torch.zeros_like(param) for name, param in original_weights.items()}
        
        optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 训练指定轮数
        try:
            for epoch in range(self.epochs):
                for batch_idx, (data, target, _) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)
                    
                    # 确保在梯度追踪上下文中
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        output = model_copy(data)
                        loss = loss_fn(output, target)
                        
                        # 检查是否有梯度
                        if not loss.requires_grad:
                            logger.error("损失函数不需要梯度！检查模型参数设置。")
                            raise RuntimeError("损失函数没有梯度")
                        
                        loss.backward()
                        optimizer.step()
                    
                    # ourFLTrust中每个epoch只训练一个batch（steps_per_epoch=1）
                    break
        except Exception as e:
            logger.warning(f"训练失败: {e}，返回零更新")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return {name: torch.zeros_like(param) for name, param in original_weights.items()}
        
        # 计算模型更新: delta = new_weights - old_weights
        delta_weights = OrderedDict()
        new_weights = model_copy.state_dict()
        for name in original_weights.keys():
            if name in new_weights:
                # 确保设备匹配
                original_tensor = original_weights[name].to(new_weights[name].device)
                delta_weights[name] = (new_weights[name] - original_tensor).detach()
            else:
                delta_weights[name] = torch.zeros_like(original_weights[name])
        
        return delta_weights
    
    def _init_clean_dataset(self):
        """从训练数据中划分出小的干净数据集"""
        # 首先检查是否已经有root_dataloader（在BaseRunner中预先分配）
        if hasattr(self.runner, 'root_dataloader') and self.runner.root_dataloader is not None:
            self.clean_dataloader = self.runner.root_dataloader
            logger.info(f"使用BaseRunner中预先分配的根数据集")
            return
            
        # 如果没有root_dataloader，尝试从train_dataset创建
        if hasattr(self.runner, 'train_dataset'):
            train_dataset = self.runner.train_dataset
            
            # 计算干净数据集大小
            total_size = len(train_dataset)
            clean_size = int(total_size * self.clean_data_ratio)
            
            # 随机划分
            indices = torch.randperm(total_size).tolist()
            clean_indices = indices[:clean_size]
            
            # 创建子集
            self.clean_dataset = torch.utils.data.Subset(train_dataset, clean_indices)
            self.clean_dataloader = torch.utils.data.DataLoader(
                self.clean_dataset,
                batch_size=self.batch_size,  # 使用配置的批次大小
                shuffle=True
            )
            
            logger.info(f"已从训练数据中划分出 {len(self.clean_dataset)} 个样本作为干净数据集")
        else:
            logger.warning("无法初始化干净数据集：未找到root_dataloader或train_dataset")
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> torch.nn.Module:
        """
        FLTrust聚合算法（基于ourFLTrust实现）
        
        对应server.py的train方法中的聚合逻辑：
        1. 获取所有客户端模型更新 deltas
        2. 在根数据集上训练得到根更新 root_delta
        3. 计算信任分数: TS_i = ReLU(cos_sim(delta_i, root_delta))
        4. 计算归一化因子: norm_i = ||root_delta|| / ||delta_i||
        5. 加权聚合: delta = Σ(TS_i * norm_i * delta_i) / Σ(TS_i)
        6. 更新模型: w = w + alpha * delta
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        # 延迟初始化干净数据集
        if not self._clean_dataset_initialized:
            self._init_clean_dataset()
            self._clean_dataset_initialized = True
        
        if self.clean_dataloader is None:
            logger.warning("FLTrust: 没有根数据集，回退到简单平均")
            from utilities import Utilities as Utils
            return Utils.average_client_models(client_models)
        
        # 获取服务器当前模型
        server_model = self.runner.server.model
        device = self.runner.device
        
        # 确保服务器模型处于正确状态
        server_model.train()
        for param in server_model.parameters():
            param.requires_grad_(True)
        
        # 评估当前模型在根数据集上的性能
        self.baseline_accuracy = self._evaluate_on_clean_dataset(server_model)
        logger.info(f"FLTrust: 根数据集上的准确率: {self.baseline_accuracy:.2f}%")
        
        # Step 1: 计算所有客户端的更新（对应ourFLTrust中收集deltas）
        logger.info(f"FLTrust: 计算 {len(client_models)} 个客户端的模型更新")
        client_deltas = []
        for i, client_model in enumerate(client_models):
            # 计算 delta = client_model - server_model
            delta = self._compute_model_delta(server_model, client_model)
            client_deltas.append(delta)
        
        # Step 2: 在根数据集上训练得到根更新（对应ourFLTrust中的root_client.train）
        logger.info("FLTrust: 在根数据集上训练以获取根更新")
        root_delta = self._train_on_dataset(server_model, self.clean_dataloader)
        
        # Step 3: FLTrust聚合（完全按照ourFLTrust的实现）
        logger.info("FLTrust: 使用信任分数和范数归一化进行聚合")
        aggregated_delta = self._fltrust_aggregate(client_deltas, root_delta)
        
        # Step 4: 应用聚合更新到服务器模型
        new_weights = OrderedDict()
        server_state = server_model.state_dict()
        for name in server_state.keys():
            if name in aggregated_delta:
                # w = w + alpha * delta（确保设备匹配）
                delta_tensor = aggregated_delta[name].to(server_state[name].device)
                new_weights[name] = server_state[name] + self.alpha * delta_tensor
            else:
                new_weights[name] = server_state[name]
        
        return new_weights
    
    def _compute_model_delta(self, server_model: torch.nn.Module, client_model) -> Dict[str, torch.Tensor]:
        """
        计算模型更新 delta = client_model - server_model
        
        Args:
            server_model: 服务器模型
            client_model: 客户端模型（可能是模型对象或state_dict）
        
        Returns:
            delta: 模型更新量
        """
        delta = OrderedDict()
        server_state = server_model.state_dict()
        
        # 检查client_model是否已经是state_dict
        if isinstance(client_model, dict):
            client_state = client_model
        else:
            client_state = client_model.state_dict()
        
        # 计算delta（确保设备匹配）
        for name in server_state.keys():
            if name in client_state:
                # 将client_state的张量移动到与server_state相同的设备
                client_tensor = client_state[name].to(server_state[name].device)
                delta[name] = (client_tensor - server_state[name]).detach()
            else:
                delta[name] = torch.zeros_like(server_state[name])
        
        return delta
    
    def _fltrust_aggregate(self, client_deltas: List[Dict[str, torch.Tensor]], 
                          root_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        FLTrust聚合算法（完全按照ourFLTrust实现）
        
        对应server.py第58-82行的逻辑：
        1. 展平所有更新向量
        2. 计算每个客户端与根更新的余弦相似度
        3. 应用ReLU: TS_i = max(0, cos_sim_i)
        4. 计算归一化因子: norm_i = ||root_delta|| / ||client_delta_i||
        5. 计算 TSnorm_i = TS_i * norm_i
        6. 加权聚合: delta = Σ(TSnorm_i * client_delta_i) / Σ(TS_i)
        
        Args:
            client_deltas: 客户端更新列表
            root_delta: 根更新
        
        Returns:
            aggregated_delta: 聚合后的更新
        """
        if not client_deltas:
            return root_delta
        
        # Step 1: 展平根更新（对应第58-59行）
        root_flat = torch.cat([param.flatten() for param in root_delta.values()])
        root_norm = torch.norm(root_flat).item()
        
        # 获取根更新的设备
        root_device = root_flat.device
        
        if root_norm < 1e-10:
            logger.warning("根更新范数接近零，返回零更新")
            return {name: torch.zeros_like(param) for name, param in root_delta.items()}
        
        # Step 2: 计算信任分数和归一化因子（对应第62-72行）
        total_TS = 0.0
        TSnorm_list = []
        
        for i, client_delta in enumerate(client_deltas):
            # 展平客户端更新（确保与根更新在同一设备）
            client_flat = torch.cat([param.flatten().to(root_device) for param in client_delta.values()])
            client_norm = torch.norm(client_flat).item()
            
            # 计算余弦相似度: cos_sim = dot(client, root) / (||client|| * ||root||)
            if client_norm > 1e-10:
                cos_sim = torch.dot(client_flat, root_flat).item() / (client_norm * root_norm)
            else:
                cos_sim = 0.0
            
            # 应用ReLU: TS = max(0, cos_sim)
            TS = max(0.0, cos_sim)
            
            # 可选：应用信任阈值
            if TS < self.trust_threshold:
                logger.debug(f"客户端{i}更新被过滤，信任分数{TS:.4f}低于阈值{self.trust_threshold}")
                TS = 0.0
            
            total_TS += TS
            
            # 计算归一化因子: norm = ||root|| / ||client||
            if client_norm > 1e-10:
                norm_factor = root_norm / client_norm
            else:
                norm_factor = 0.0
            
            # TSnorm = TS * norm_factor
            TSnorm = TS * norm_factor
            TSnorm_list.append(TSnorm)
        
        # 如果所有信任分数都为0，返回根更新
        if total_TS < 1e-10:
            logger.warning("所有客户端信任分数都为零，使用根更新")
            return root_delta
        
        # Step 3: 加权聚合（对应第76-82行）
        # delta_weight = Σ(TSnorm_i * delta_i) / total_TS
        aggregated_delta = OrderedDict()
        
        for name in root_delta.keys():
            # 获取目标设备（使用根更新的设备）
            target_device = root_delta[name].device
            
            # 初始化为第一个客户端的加权更新（确保设备匹配）
            weighted_sum = TSnorm_list[0] * client_deltas[0][name].to(target_device)
            
            # 累加其他客户端的加权更新
            for i in range(1, len(client_deltas)):
                weighted_sum += TSnorm_list[i] * client_deltas[i][name].to(target_device)
            
            # 除以总信任分数进行归一化
            aggregated_delta[name] = weighted_sum / total_TS
        
        logger.info(f"FLTrust聚合完成，有效客户端: {sum(1 for ts in TSnorm_list if ts > 0)}/{len(client_deltas)}")
        
        return aggregated_delta
    
    def _evaluate_on_clean_dataset(self, model: torch.nn.Module) -> float:
        """在干净数据集上评估模型性能"""
        if self.clean_dataloader is None:
            return 0.0
        
        # 保存原始模式
        original_training_mode = model.training
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target, _ in self.clean_dataloader:
                data, target = data.to(self.runner.device), target.to(self.runner.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 恢复原始模式
        if original_training_mode:
            model.train()
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy
    
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用FLTrust聚合预测
        注意：预测聚合不需要根数据集，使用简单的加权平均
        """
        # 计算预测之间的相似度作为信任分数
        n_predictions = len(client_predictions)
        trust_scores = []
        
        # 使用所有预测的平均作为"根预测"
        mean_prediction = torch.mean(torch.stack(client_predictions), dim=0)
        mean_norm = torch.norm(mean_prediction)
        
        for pred in client_predictions:
            pred_norm = torch.norm(pred)
            if pred_norm > 0 and mean_norm > 0:
                cos_sim = torch.dot(pred.flatten(), mean_prediction.flatten()) / (pred_norm * mean_norm)
                ts = max(0.0, cos_sim.item())
            else:
                ts = 0.0
            trust_scores.append(ts)
        
        # 归一化
        total_ts = sum(trust_scores)
        if total_ts > 0:
            trust_scores = [ts / total_ts for ts in trust_scores]
        else:
            trust_scores = [1.0 / n_predictions] * n_predictions
        
        # 加权聚合
        aggregated_pred = torch.zeros_like(client_predictions[0])
        for pred, ts in zip(client_predictions, trust_scores):
            aggregated_pred += ts * pred
        
        # 计算异常分数
        distances = torch.norm(torch.stack(client_predictions) - aggregated_pred.unsqueeze(0), dim=-1)
        outlier_scores = distances / (torch.max(distances) + 1e-10)
        
        return aggregated_pred, outlier_scores

