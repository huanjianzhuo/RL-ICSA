# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/ada_defence.py
# Description:  AdaAggRL 自适应聚合防御
# ===========================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from collections import OrderedDict
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


def create_adaagg_eval_fn(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> Callable:
    """
    创建 AdaAggRL 评估函数
    
    Args:
        model: 用于评估的模型（仅用于获取结构）
        x_val: 验证数据
        y_val: 验证标签
    
    Returns:
        评估函数 fn(weights_list) -> (loss, acc_dict, label_acc, label_loss)
    """
    device = next(model.parameters()).device
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    # 获取标签数量
    num_labels = len(torch.unique(y_val))
    
    def eval_fn(weights_list: List[np.ndarray]) -> Tuple[float, Dict, List[float], List[float]]:
        """
        评估给定权重的模型
        
        Args:
            weights_list: 模型权重列表（numpy数组）
        
        Returns:
            (loss, acc_dict, label_acc, label_loss)
            - loss: 整体损失
            - acc_dict: {'accuracy': overall_accuracy}
            - label_acc: 每个标签的准确率列表
            - label_loss: 每个标签的损失列表
        """
        # 创建临时模型并加载权重
        temp_model = type(model)().to(device)
        state_dict = temp_model.state_dict()
        
        # 将numpy权重转换为torch张量
        for i, key in enumerate(state_dict.keys()):
            if i < len(weights_list):
                weight_np = weights_list[i]
                target_shape = state_dict[key].shape
                target_dtype = state_dict[key].dtype
                
                # 跳过非浮点参数
                if not target_dtype.is_floating_point:
                    continue
                
                # 检查形状是否匹配
                if weight_np.shape != tuple(target_shape):
                    # 如果元素数量相同，尝试reshape
                    if weight_np.size == np.prod(target_shape):
                        logger.debug(f"AdaAggRL eval_fn: reshape参数{key}: {weight_np.shape} -> {target_shape}")
                        try:
                            weight_np = weight_np.reshape(target_shape)
                        except Exception as e:
                            logger.warning(f"AdaAggRL eval_fn: 无法reshape参数{key}: {e}，跳过")
                            continue
                    else:
                        logger.warning(f"AdaAggRL eval_fn: 参数{key}形状不匹配，跳过 (期望{target_shape}, 实际{weight_np.shape})")
                        continue
                
                # 安全地转换为torch张量
                try:
                    weight_tensor = torch.from_numpy(weight_np).to(device=device, dtype=target_dtype)
                    state_dict[key] = weight_tensor
                except Exception as e:
                    logger.warning(f"AdaAggRL eval_fn: 无法转换参数{key}: {e}")
                    continue
        
        temp_model.load_state_dict(state_dict)
        temp_model.eval()
        
        # 计算整体损失和准确率
        with torch.no_grad():
            outputs = temp_model(x_val)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_val).item()
            
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_val).sum().item()
            total = y_val.size(0)
            overall_accuracy = correct / total if total > 0 else 0.0
        
        # 计算每个标签的准确率和损失
        label_acc = []
        label_loss = []
        
        for label_idx in range(num_labels):
            mask = (y_val == label_idx)
            if mask.sum() > 0:
                label_outputs = outputs[mask]
                label_targets = y_val[mask]
                
                # 标签损失
                label_loss_val = criterion(label_outputs, label_targets).item()
                label_loss.append(label_loss_val)
                
                # 标签准确率
                _, label_predicted = torch.max(label_outputs, 1)
                label_correct = (label_predicted == label_targets).sum().item()
                label_total = label_targets.size(0)
                label_accuracy = label_correct / label_total if label_total > 0 else 0.0
                label_acc.append(label_accuracy)
            else:
                label_acc.append(0.0)
                label_loss.append(0.0)
        
        acc_dict = {'accuracy': overall_accuracy}
        
        return loss, acc_dict, label_acc, label_loss
    
    return eval_fn


class AdaAggRLDefence(ByzantineDefence):
    """
    AdaAggRL 自适应聚合防御
    
    基于 AdaAggRL-main/utilities.py -> Poison_detect 机制的完整torch版实现
    
    核心算法：
    1. 对每个客户端模型进行评估，获得整体损失和逐标签损失
    2. 计算整体得分：基于损失与平均值的偏差（MAD）
    3. 计算逐标签得分：基于逐标签损失与平均值的偏差，加权因子为 (mean/overall_mean)^s2
    4. 自适应调整s2参数：尝试5个不同的s2值，选择验证集损失最小的
    5. 根据得分计算客户端权重，加权聚合模型更新
    
    关键参数：
    - s1_overall: 整体得分斜率（默认2）
    - s1_label: 逐标签得分斜率（默认3）
    - s2: 标签重要性指数（默认3，自适应调整）
    - eval_fn: 评估函数，输入权重列表，返回 (loss, acc_dict, label_acc, label_loss)
    """
    
    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        self.eval_fn = config.get('agg_evaluate_fn')  # fn(weights) -> (loss, acc_dict, label_acc, label_loss)
        self.s1_overall = config.get('adaagg_s1_overall', 2)
        self.s1_label = config.get('adaagg_s1_label', 3)
        self.s2 = config.get('adaagg_s2', 3)
        self.pre_reset_s2 = self.s2
        self.no_labels = config.get('num_labels', 10)
        
        # 如果未提供评估函数，尝试从配置中获取验证数据创建
        if self.eval_fn is None:
            x_val = config.get('x_val')
            y_val = config.get('y_val')
            
            if x_val is not None and y_val is not None:
                try:
                    logger.info("AdaAggRL: 从配置中的验证数据创建评估函数")
                    # 使用runner的全局模型或创建临时模型
                    if hasattr(runner_instance, 'global_model'):
                        model = runner_instance.global_model
                    elif hasattr(runner_instance, 'server') and hasattr(runner_instance.server, 'model'):
                        model = runner_instance.server.model
                    else:
                        logger.warning("AdaAggRL: 无法获取模型，将使用简单平均")
                        model = None
                    
                    if model is not None:
                        self.eval_fn = create_adaagg_eval_fn(model, x_val, y_val)
                        logger.info("AdaAggRL: 评估函数已成功创建")
                except Exception as e:
                    logger.warning(f"AdaAggRL: 创建评估函数失败: {e}，将使用简单平均")
            else:
                logger.warning("AdaAggRL: 未提供评估函数和验证数据，将使用简单平均")
    
    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> OrderedDict:
        """
        使用AdaAggRL自适应聚合算法聚合模型
        
        Args:
            client_models: 客户端模型列表
        
        Returns:
            聚合后的模型参数（OrderedDict）
        """
        if not client_models:
            raise ValueError("客户端模型列表为空")
        
        if self.eval_fn is None:
            logger.warning("AdaAggRL: 评估函数未定义，回退到简单平均")
            from utilities import Utilities as Utils
            return Utils.average_client_models(client_models)
        
        # Step 1: 将模型转换为结果格式 (client_id, params_list)
        # 为了保证评估/聚合时的权重顺序一致，严格按照服务器模型state_dict的key顺序对齐
        results = []
        if hasattr(self.runner, 'server') and hasattr(self.runner.server, 'model'):
            ref_keys = list(self.runner.server.model.state_dict().keys())
        else:
            ref_keys = list(client_models[0].state_dict().keys())
        
        for i, model in enumerate(client_models):
            sd = model.state_dict()
            params = []
            for k in ref_keys:
                arr = sd[k].data.cpu().numpy()
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)
                params.append(arr)
            results.append((i, params))
        
        # Step 2: 获取上一轮的全局参数作为基准（用于计算模型更新）
        # 这里使用第一个客户端的参数作为参考，实际应该使用服务器的全局模型
        if hasattr(self.runner, 'server') and hasattr(self.runner.server, 'model'):
            global_params = []
            sd = self.runner.server.model.state_dict()
            for k in ref_keys:
                arr = sd[k].data.cpu().numpy()
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)
                global_params.append(arr)
        else:
            global_params = []
            sd0 = client_models[0].state_dict()
            for k in ref_keys:
                arr = sd0[k].data.cpu().numpy()
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)
                global_params.append(arr)
        
        # Step 3: 执行自适应聚合算法
        agg_weights = self._calculate_new_aggregated(results, global_params)
        
        # Step 4: 将聚合后的权重转换回模型state_dict格式
        base_state = client_models[0].state_dict()
        agg_state = OrderedDict()
        for i, k in enumerate(base_state.keys()):
            shape = base_state[k].shape
            param_np = agg_weights[i].reshape(shape)
            agg_state[k] = torch.from_numpy(param_np).to(base_state[k].device).type(base_state[k].dtype)
        
        return agg_state
    
    def _calculate_new_aggregated(self, results: List[Tuple], last_agg_w: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算新的聚合权重
        
        对应AdaAggRL中的calculate_new_aggregated方法：
        1. 评估所有客户端模型
        2. 尝试5个不同的s2值
        3. 为每个s2值计算客户端得分和聚合权重
        4. 选择验证集损失最小的s2值
        
        Args:
            results: 客户端结果列表 [(client_id, params), ...]
            last_agg_w: 上一轮的全局权重
        
        Returns:
            聚合后的权重列表
        """
        # 评估所有客户端
        # 与原AdaAggRL-main中的 Poison_detect.calculate_accs 对齐：
        # 返回 label_acc_dict（逐标签精度）、nodes_acc（整体精度）、loss_dict、label_loss_dict
        label_acc_dict, nodes_acc, loss_dict, label_loss_dict = self._calculate_accs(results)
        
        adaptives2Loss = []
        adaptives2Parts = []
        weights = []
        
        # 尝试5个不同的s2值（对应AdaAggRL第1155行）
        adaptives2Tests = [
            self.s2,                           # 当前s2值
            max(1, self.s2 - 0.5),             # s2 - 0.5
            self.s2 + 0.5,                     # s2 + 0.5
            3,                                 # 固定值3
            self.pre_reset_s2                  # 上一个最优s2值
        ]
        
        action_names = [
            f"动作0: 当前s2={adaptives2Tests[0]:.2f}",
            f"动作1: s2-0.5={adaptives2Tests[1]:.2f}",
            f"动作2: s2+0.5={adaptives2Tests[2]:.2f}",
            f"动作3: 固定值={adaptives2Tests[3]:.2f}",
            f"动作4: 历史最优={adaptives2Tests[4]:.2f}"
        ]
        
        logger.info(f"AdaAggRL RL策略: 尝试5个动作 {adaptives2Tests}")
        print(f"\n{'='*70}")
        print(f"AdaAggRL 强化学习 - 动作选择")
        print(f"{'='*70}")
        print(f"可选动作空间: {len(adaptives2Tests)} 个动作")
        for i, name in enumerate(action_names):
            print(f"  {name}")
        
        print(f"\n评估各动作的奖励:")
        print(f"{'-'*70}")
        
        for idx, s2_test in enumerate(adaptives2Tests):
            self.s2 = s2_test
            
            # ✅ 修复：使用损失而非准确率计算得分（与原始AdaAggRL实现一致）
            # 计算客户端得分（完全对齐 Poison_detect.get_points_overall / get_points_label）
            # 原始实现使用 loss_dict 和 label_loss_dict
            points = {}
            points, overall_mean = self._get_points_overall(loss_dict, results, points=points)
            points = self._get_points_label(label_loss_dict, results, overall_mean, points)
            
            # 将得分转换为权重
            part_agg = self._points_to_parts(points)
            
            # 根据权重聚合模型
            agg_copy_weights = self._agg_copy_weights(results, part_agg, last_agg_w)
            weights.append(agg_copy_weights)
            
            # 评估聚合后的模型
            try:
                loss, acc, _, _ = self.eval_fn(agg_copy_weights)
                adaptives2Parts.append(part_agg)
                adaptives2Loss.append(loss)
                acc_val = acc.get('accuracy', 0)
                print(f"  {action_names[idx]:<35} → 损失: {loss:.6f}, 准确率: {acc_val:.4f}")
                logger.info(f"AdaAggRL: s2={s2_test:.2f}, loss={loss:.6f}, acc={acc_val:.4f}")
            except Exception as e:
                logger.warning(f"AdaAggRL: 评估s2={s2_test}失败: {e}，使用最大损失")
                adaptives2Loss.append(float('inf'))
                adaptives2Parts.append(part_agg)
                print(f"  {action_names[idx]:<35} → 评估失败")
        
        # 选择损失最小的s2值
        idx_max = np.argmin(adaptives2Loss)
        
        # 如果最优s2是固定值3，则更新pre_reset_s2
        if idx_max == 3:
            self.pre_reset_s2 = self.s2
        
        # 更新当前s2值
        self.s2 = adaptives2Tests[idx_max]
        
        # 显示最终选择的动作
        print(f"\n{'='*70}")
        print(f"✓ 强化学习决策: 选择 {action_names[idx_max]}")
        print(f"  → 最小损失: {adaptives2Loss[idx_max]:.6f}")
        print(f"  → 策略更新: s2 从 {adaptives2Tests[0]:.2f} 调整为 {self.s2:.2f}")
        print(f"{'='*70}\n")
        
        logger.info(f"AdaAggRL RL决策: 选择动作{idx_max} (s2={self.s2:.2f}), 最小损失={adaptives2Loss[idx_max]:.6f}")
        
        return weights[idx_max]
    
    def _agg_copy_weights(self, results: List[Tuple], part_agg: Dict[int, float], 
                         last_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        根据客户端权重聚合模型
        
        对应AdaAggRL中的agg_copy_weights方法：
        1. 计算每个客户端的模型更新（delta = current - last）
        2. 按权重加权求和
        3. 加上基准权重
        
        Args:
            results: 客户端结果列表
            part_agg: 客户端权重字典 {client_id: weight}
            last_weights: 基准权重（上一轮全局权重）
        
        Returns:
            聚合后的权重列表
        """
        # 计算每个客户端的模型更新（仅对浮点权重聚合；整数/布尔等权重直接保留）
        norms_dict = {}
        float_mask = [np.issubdtype(w.dtype, np.floating) for w in last_weights]
        for client_id, params in results:
            delta = []
            for p, lp, is_float in zip(params, last_weights, float_mask):
                if is_float:
                    # 检查形状是否匹配
                    if p.shape != lp.shape:
                        logger.warning(f"AdaAggRL: 客户端{client_id}权重形状不匹配: {p.shape} vs {lp.shape}，尝试reshape")
                        try:
                            # 尝试reshape到目标形状
                            if p.size == lp.size:
                                p_reshaped = p.reshape(lp.shape)
                                delta.append(p_reshaped.astype(lp.dtype, copy=False) - lp)
                            else:
                                logger.error(f"AdaAggRL: 无法reshape，元素数量不匹配: {p.size} vs {lp.size}，使用零更新")
                                delta.append(np.zeros_like(lp))
                        except Exception as e:
                            logger.error(f"AdaAggRL: reshape失败: {e}，使用零更新")
                            delta.append(np.zeros_like(lp))
                    else:
                        # 统一用原dtype进行计算，避免不必要的类型提升
                        delta.append(p.astype(lp.dtype, copy=False) - lp)
                else:
                    # 对非浮点权重不做更新
                    delta.append(np.zeros_like(lp))
            norms_dict[client_id] = delta
        
        # 初始化聚合权重：浮点权重初始化为0，其他权重直接拷贝上一轮权重
        ret_weights = []
        for w, is_float in zip(last_weights, float_mask):
            if is_float:
                ret_weights.append(np.zeros_like(w))
            else:
                ret_weights.append(w.copy())
        
        # 加权求和（仅对浮点权重）
        for client_id, deltas in norms_dict.items():
            weight = np.float32(part_agg.get(client_id, 0.0))
            if weight == 0.0:
                continue
            for i, (d, is_float) in enumerate(zip(deltas, float_mask)):
                if is_float:
                    # 保证结果dtype与目标一致，避免float64 -> float32/ int64的冲突
                    ret_weights[i] += (d * weight).astype(ret_weights[i].dtype, copy=False)
        
        # 加上基准权重（仅对浮点权重）；非浮点权重已在初始化时拷贝
        for i, is_float in enumerate(float_mask):
            if is_float:
                ret_weights[i] += last_weights[i]
        
        return ret_weights
    
    def _get_points_overall(self, nodes_metric: Dict[int, float], results: List[Tuple], 
                           points: Dict[int, float] = None) -> Tuple[Dict[int, float], float]:
        """整体得分（与 AdaAggRL-main 中 Poison_detect.get_points_overall 完全对齐）

        使用每个客户端的整体指标（损失或准确率）计算得分：
        - 先计算 mean_metric
        - 对每个客户端计算 all_for_score = mean_metric - metric_i
        - 用平均绝对偏差 MAD = mean(|all_for_score|) 计算斜率
        - points[client_id] += slope * all_for_score[i] + 10
        
        注意：原始AdaAggRL使用损失（loss_dict），损失越低得分越高
        """
        if points is None:
            points = {}

        # 计算所有客户端的指标均值
        mean_calc = []
        for cid in nodes_metric:
            mean_calc.append(nodes_metric[cid])
        mean = np.mean(mean_calc) if mean_calc else 0.0

        # 计算每个客户端的偏差
        # 对于损失：mean_loss - client_loss，损失越低得分越高
        # 对于准确率：mean_acc - client_acc，准确率越高得分越高
        all_for_score = []
        for elem in mean_calc:
            all_for_score.append(mean - elem)

        # 计算平均绝对偏差 MAD（与原代码保持一致，用 mean 而非 median）
        mad_calc = all_for_score.copy()
        for i in range(len(mad_calc)):
            mad_calc[i] = abs(mad_calc[i])
        no_elems = round(len(mad_calc)) if mad_calc else 0
        mad_calc.sort()
        mad_calc = mad_calc[:no_elems]
        mad = np.mean(mad_calc) if mad_calc else 0.0

        # 斜率（原实现未做数值保护，这里保持一致）
        slope = self.s1_overall / mad if mad != 0 else 0.0

        # 为每个客户端累加整体得分
        for i in range(len(all_for_score)):
            client_id = results[i][0]
            points[client_id] = points.get(client_id, 0) + slope * all_for_score[i] + 10

        return points, mean
    
    def _get_points_label(self, label_metric_dict: Dict[int, List[float]], results: List[Tuple],
                         overall_mean: float, points: Dict[int, float]) -> Dict[int, float]:
        """逐标签得分（与 AdaAggRL-main 中 Poison_detect.get_points_label 完全对齐）

        对每个标签类别 i：
        - 收集所有客户端在该标签上的指标（损失或准确率）mean_calc
        - 计算 all_for_score = mean_i - metric_i
        - MAD = mean(|all_for_score|)，斜率 slope = s1_label / MAD
        - 因子 factor = ((overall_mean + (mean_i - overall_mean)) / overall_mean) ** s2
                      = (mean_i / overall_mean) ** s2
        - points[client_id] += max(1, factor) * slope * all_for_score[k] + 10
        
        注意：原始AdaAggRL使用逐标签损失（label_loss_dict）
        """
        for label_idx in range(self.no_labels):
            # 该标签在所有客户端上的指标列表
            mean_calc = []
            for cid in label_metric_dict:
                mean_calc.append(label_metric_dict.get(cid)[label_idx])

            mean = np.mean(mean_calc) if mean_calc else 0.0

            # 偏差：mean - acc_label
            all_for_score = []
            for elem in mean_calc:
                all_for_score.append(mean - elem)

            # 平均绝对偏差 MAD
            mad_calc = all_for_score.copy()
            for j in range(len(mad_calc)):
                mad_calc[j] = abs(mad_calc[j])
            no_elems = round(len(mad_calc)) if mad_calc else 0
            mad_calc.sort()
            mad_calc = mad_calc[:no_elems]
            mad = np.mean(mad_calc) if mad_calc else 0.0

            slope = self.s1_label / mad if mad != 0 else 0.0

            # 因子：((overall_mean + dif) / overall_mean) ** s2，其中 dif = mean - overall_mean
            if overall_mean != 0:
                dif = (mean - overall_mean)
                x = ((overall_mean + dif) / overall_mean)
                factor = x ** self.s2
            else:
                factor = 1.0

            for k in range(len(all_for_score)):
                client_id = results[k][0]
                points[client_id] = points.get(client_id, 0) + max(1, factor) * slope * all_for_score[k] + 10

        return points
    
    def _points_to_parts(self, points: Dict[int, float]) -> Dict[int, float]:
        """
        将得分转换为权重
        
        对应AdaAggRL中的points_to_parts方法：
        1. 确保所有得分非负
        2. 归一化为权重（和为1）
        
        Args:
            points: 得分字典
        
        Returns:
            权重字典
        """
        part_agg = {}
        
        # 确保所有得分非负
        for client_id in points:
            points[client_id] = max(0.0, points[client_id])
        
        # 计算总得分
        sum_points = max(0.1, sum(points.values()))
        
        # 归一化为权重
        for client_id in points:
            part_agg[client_id] = points[client_id] / sum_points
        
        return part_agg
    
    def _calculate_accs(self, results: List[Tuple]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        评估所有客户端模型
        
        对应AdaAggRL中的calculate_accs方法：
        对每个客户端的权重进行评估，获得整体损失和逐标签损失
        
        Args:
            results: 客户端结果列表 [(client_id, params), ...]
        
        Returns:
            (label_acc_dict, nodes_acc, loss_dict, label_loss_dict)
            - label_acc_dict: {client_id: [per_label_accuracy]}
            - nodes_acc: {client_id: overall_accuracy}
            - loss_dict: {client_id: overall_loss}
            - label_loss_dict: {client_id: [per_label_loss]}
        """
        label_acc_dict = {}
        nodes_acc = {}
        loss_dict = {}
        label_loss_dict = {}
        
        # 评估每个客户端
        for result in results:
            client_id, loss, acc_dict, label_acc, label_loss = self._par_results_ev(result)
            
            label_acc_dict[client_id] = label_acc
            nodes_acc[client_id] = acc_dict.get('accuracy', 0.0)
            loss_dict[client_id] = loss
            label_loss_dict[client_id] = label_loss
        
        return label_acc_dict, nodes_acc, loss_dict, label_loss_dict
    
    def _par_results_ev(self, result: Tuple) -> Tuple[int, float, Dict, List[float], List[float]]:
        """
        评估单个客户端模型
        
        对应AdaAggRL中的par_results_ev方法：
        使用eval_fn评估客户端权重
        
        Args:
            result: (client_id, params_list)
        
        Returns:
            (client_id, loss, acc_dict, label_acc, label_loss)
        """
        client_id, params = result
        
        try:
            loss, acc_dict, label_acc, label_loss = self.eval_fn(params)
            return client_id, loss, acc_dict, label_acc, label_loss
        except Exception as e:
            logger.warning(f"AdaAggRL: 评估客户端{client_id}失败: {e}")
            # 返回默认值
            return client_id, float('inf'), {'accuracy': 0.0}, [0.0] * self.no_labels, [float('inf')] * self.no_labels
    
    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用AdaAggRL聚合预测
        
        对于预测聚合，使用简化的方法：
        1. 计算所有预测的平均值作为基准
        2. 基于预测与平均值的距离计算异常分数
        
        Args:
            client_predictions: 客户端预测列表
        
        Returns:
            (聚合预测, 异常分数)
        """
        if not client_predictions:
            raise ValueError("客户端预测列表为空")
        
        # 计算平均预测
        stacked_predictions = torch.stack(client_predictions)
        aggregated_pred = torch.mean(stacked_predictions, dim=0)
        
        # 计算异常分数：基于与平均预测的L2距离
        distances = torch.norm(stacked_predictions - aggregated_pred.unsqueeze(0), dim=-1)
        outlier_scores = distances / (torch.max(distances) + 1e-10)
        
        return aggregated_pred, outlier_scores

