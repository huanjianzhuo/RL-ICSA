# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/qlearning_attack.py
# Description:  Q-Learning智能攻击
# ===========================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
import copy
import random
from typing import List, Dict, Any, Tuple
from collections import deque
from .base import NoAttack
from .lie_attack import LIEAttack
from .secondhighestconfidence_attack_ICSA_paper_version import SecondHighestConfidenceAttack

logger = logging.getLogger(__name__)


class QLearningAttack(NoAttack):
    """基于Q-Learning的智能拜占庭攻击（四动作：LIE/MinMax/MinSum/ICSA）"""

    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)
        # 区分拜占庭与良性客户端
        self.byzantine_clients = [client for client in self.clients if getattr(client, 'is_byzantine', False)]
        self.benign_clients = [client for client in self.clients if not getattr(client, 'is_byzantine', False)]

        if not self.byzantine_clients:
            logger.warning("QLearningAttack: 未检测到拜占庭客户端，攻击将不会生效")
        
        # Q-Learning超参
        self.learning_rate = config.get('ql_learning_rate', 0.001)
        self.discount_factor = config.get('ql_discount_factor', 0.9)
        self.epsilon = config.get('ql_epsilon', 0.1)
        self.epsilon_decay = config.get('ql_epsilon_decay', 0.995)
        self.epsilon_min = config.get('ql_epsilon_min', 0.01)
        
        logger.info("Q-Learning高级共谋攻击已启用（简化状态空间）")
        self._init_advanced_collusion(config, runner_instance)
    
    def _init_advanced_collusion(self, config: Dict[str, Any], runner_instance):
        """初始化高级共谋模式"""
        # 创建通信器（内部实现）
        self.communicator = self._create_communicator(config)
        
        # 创建协调器（内部实现）
        self.coordinator = self._create_coordinator(config)
        
        # 创建状态构建器（内部实现，简化状态）
        # 状态维度：1（全局模型精度）+ N（其他拜占庭客户端是否参与共谋）
        n_other_byzantine = max(len(self.byzantine_clients) - 1, 0)
        self.state_dim = 1 + n_other_byzantine
        self.state_builder = self._create_state_builder(config)
        
        # 创建奖励计算器（内部实现）
        self.reward_calculator = self._create_reward_calculator(config)
        
        # Q-Learning动作空间（仅保留论文所需的四种攻击）：
        # 0 = LIE
        # 1 = Min-Max
        # 2 = Min-Sum
        # 3 = ICSA (Inter-class Similarity Attack)
        # 注意：Q网络输出维度必须与这里保持一致。
        self.action_dim = 4
        self.attack_methods = {
            0: self._lie_attack,
            1: self._minmax_attack,
            2: self._minsum_attack,
            3: self._icsa_attack,
        }
        
        # 为每个拜占庭客户端创建Q网络和经验回放
        self._init_q_networks(config, runner_instance)
        
        # 当前轮次和准确率
        self.current_round = 0
        self.previous_accuracy = None
        
        # 持久化 ICSA 攻击器。ICSA 本身按照论文实现为：
        # 逐样本生成 Top-1/Top-2 边界扰动，并进行 element-wise median。
        self.icsa = SecondHighestConfidenceAttack(
            self.clients,
            config,
            runner_instance,
        )

        logger.info(
            f"四动作Q-Learning: 状态维度={self.state_dim} "
            f"(1个全局精度 + {self.state_dim-1}个其他拜占庭参与状态), "
            f"动作维度={self.action_dim} "
            f"(LIE, MinMax, MinSum, ICSA)"
        )
    
    def _init_q_networks(self, config: Dict[str, Any], runner_instance):
        """为每个拜占庭客户端创建Q网络"""
        self.batch_size = config.get('ql_batch_size', 32)
        self.update_target_frequency = config.get('ql_update_target_freq', 100)
        
        for client in self.byzantine_clients:
            # 创建Q网络
            client.q_network = nn.Sequential(
                nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, self.action_dim)
            ).to(runner_instance.device)
            
            client.q_target_network = nn.Sequential(
                nn.Linear(self.state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, self.action_dim)
            ).to(runner_instance.device)
            
            client.q_target_network.load_state_dict(client.q_network.state_dict())
            client.q_optimizer = torch.optim.Adam(client.q_network.parameters(), lr=self.learning_rate)
            
            # 经验回放
            client.q_memory = deque(maxlen=config.get('ql_memory_size', 10000))
            client.q_episode_count = 0
            client.action_history = deque(maxlen=10)

    def _select_action(self, state: np.ndarray, client) -> int:
        """使用客户端自己的Q网络选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(client.q_network[0].weight.device)
            q = client.q_network(s)
            return int(q.argmax().item())

    def _learn_from_memory(self, client):
        """使用客户端自己的Q网络和经验回放进行学习"""
        if len(client.q_memory) < self.batch_size:
            return
        
        batch = random.sample(client.q_memory, self.batch_size)
        device = client.q_network[0].weight.device
        
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)

        current_q = client.q_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = client.q_target_network(next_states).max(1)[0]
            target_q = rewards + self.discount_factor * next_q
        loss = nn.MSELoss()(current_q, target_q)
        
        client.q_optimizer.zero_grad()
        loss.backward()
        client.q_optimizer.step()

        if client.q_episode_count % self.update_target_frequency == 0:
            client.q_target_network.load_state_dict(client.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    # 攻击方法实现
    def _lie_attack(self, client) -> torch.nn.Module:
        """LIE攻击"""
        lie_attacker = LIEAttack(self.clients, self.config, self.runner)
        attacked_models = lie_attacker.get_perturbed_client_models()
        client_idx = self.clients.index(client)
        return attacked_models[client_idx]
    
    def _get_params_vector(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    def _set_params_vector(self, model: torch.nn.Module, vec: torch.Tensor) -> None:
        pointer = 0
        for p in model.parameters():
            n = p.numel()
            p.data = vec[pointer:pointer+n].view_as(p.data)
            pointer += n
    
    def _get_global_vector(self) -> torch.Tensor:
        if hasattr(self.runner, 'server') and self.runner.server is not None:
            return torch.cat([p.data.view(-1) for p in self.runner.server.model.parameters()])
        return self._get_params_vector(self.clients[0].model)
    
    # ========== 内部通信器实现 ==========
    def _create_communicator(self, config: Dict[str, Any]):
        """创建通信器（内部实现）"""
        class Communicator:
            def __init__(self, byzantine_clients, config):
                self.byzantine_clients = byzantine_clients
                self.n_byzantine = len(byzantine_clients)
                self.config = config
                
                # 客户端状态
                self.client_states = {}
                for idx in range(self.n_byzantine):
                    self.client_states[idx] = {
                        'client_id': idx,
                        'current_round': 0,
                        'is_participating': False,
                        'selected_attack': None,
                        'attack_strength': 0.0,
                        'update_norm': 0.0,
                        'participation_history': deque(maxlen=10),
                        'attack_history': deque(maxlen=10),
                    }
                
                # 全局信息
                self.current_round = 0
                self.global_accuracy_history = deque(maxlen=50)
                self.benign_updates_stats = None
                
                logger.info(f"共谋通信器已初始化: {self.n_byzantine}个拜占庭客户端")
            
            def start_round(self, round_num):
                self.current_round = round_num
                for state in self.client_states.values():
                    state['current_round'] = round_num
            
            def broadcast_state(self, client_idx, is_participating, selected_attack, attack_strength, model_update=None):
                if client_idx in self.client_states:
                    state = self.client_states[client_idx]
                    state['is_participating'] = is_participating
                    state['selected_attack'] = selected_attack
                    state['attack_strength'] = attack_strength
                    state['participation_history'].append(is_participating)
                    state['attack_history'].append(selected_attack)
            
            def get_other_clients_states(self, client_idx):
                states = []
                for idx, state in self.client_states.items():
                    if idx != client_idx:
                        states.append({
                            'client_id': state['client_id'],
                            'is_participating': state['is_participating'],
                            'selected_attack': state['selected_attack'],
                            'attack_strength': state['attack_strength'],
                            'participation_rate': np.mean(state['participation_history']) if state['participation_history'] else 0.0,
                            'recent_attacks': list(state['attack_history'])[-3:],
                        })
                return states
            
            def get_participation_info(self):
                participating_clients = []
                attack_distribution = {}
                
                for idx, state in self.client_states.items():
                    if state['is_participating']:
                        participating_clients.append(idx)
                        attack = state['selected_attack']
                        if attack is not None:
                            attack_distribution[attack] = attack_distribution.get(attack, 0) + 1
                
                return {
                    'participating_clients': participating_clients,
                    'participation_rate': len(participating_clients) / self.n_byzantine if self.n_byzantine > 0 else 0.0,
                    'attack_distribution': attack_distribution,
                    'n_participating': len(participating_clients),
                }
            
            def update_global_accuracy(self, accuracy):
                self.global_accuracy_history.append(accuracy)
                logger.info(f"轮次 {self.current_round}: 全局准确率 = {accuracy:.4f}")
            
            def get_accuracy_trend(self):
                if len(self.global_accuracy_history) < 2:
                    return {
                        'current': self.global_accuracy_history[-1] if self.global_accuracy_history else 0.0,
                        'previous': 0.0,
                        'change': 0.0,
                        'trend': 0.0,
                    }
                
                current = self.global_accuracy_history[-1]
                previous = self.global_accuracy_history[-2]
                change = current - previous
                
                # 计算趋势
                if len(self.global_accuracy_history) >= 5:
                    recent = list(self.global_accuracy_history)[-5:]
                    trend = (recent[-1] - recent[0]) / 4
                else:
                    trend = change
                
                return {
                    'current': current,
                    'previous': previous,
                    'change': change,
                    'trend': trend,
                }
            
            def analyze_benign_updates(self, benign_clients):
                if not benign_clients:
                    return
                
                updates = []
                for client in benign_clients:
                    if hasattr(client, 'model') and client.model is not None:
                        update = []
                        for param in client.model.parameters():
                            update.append(param.data.flatten())
                        updates.append(torch.cat(update))
                
                if updates:
                    updates_tensor = torch.stack(updates)
                    self.benign_updates_stats = {
                        'norm_mean': torch.norm(updates_tensor, dim=1).mean().item(),
                        'norm_std': torch.norm(updates_tensor, dim=1).std().item(),
                    }
            
            def get_benign_stats(self):
                return self.benign_updates_stats
            
            def get_statistics(self):
                return {
                    'current_round': self.current_round,
                    'n_byzantine': self.n_byzantine,
                    'participation_info': self.get_participation_info(),
                    'accuracy_trend': self.get_accuracy_trend(),
                }
        
        return Communicator(self.byzantine_clients, config)
    
    def _create_coordinator(self, config: Dict[str, Any]):
        """创建协调器（内部实现）"""
        class Coordinator:
            def __init__(self, byzantine_clients, benign_clients, config, communicator):
                self.byzantine_clients = byzantine_clients
                self.benign_clients = benign_clients
                self.config = config
                self.communicator = communicator
                self.n_byzantine = len(byzantine_clients)
                
                # 只使用协调攻击策略
                self.coordination_strategy = 'coordinated'
                self.attack_strength = config.get('collusion_strength', 1.0)
                
                logger.info(f"高级共谋协调器已初始化: 策略=协调攻击")
            
            def coordinate_attack(self, round_num):
                self.communicator.start_round(round_num)
                self.communicator.analyze_benign_updates(self.benign_clients)
                
                # 生成协调攻击计划
                attack_plan = self._coordinated_attack()
                
                # 广播攻击计划
                for client_idx, plan in attack_plan.items():
                    self.communicator.broadcast_state(
                        client_idx,
                        plan['is_participating'],
                        plan['selected_attack'],
                        plan['attack_strength']
                    )
                
                return attack_plan
            
            def _coordinated_attack(self):
                """
                协调攻击：共谋体现在统一的攻击目标和信息互通
                - 攻击方法：由每个拜占庭客户端的Q网络独立决定
                - 攻击目标：统一（通过通信器共享良性客户端的统计信息）
                - 攻击方向：统一（所有客户端朝同一方向）
                - 攻击强度：统一
                - 信息互通：通过通信器共享状态和决策
                """
                attack_plan = {}
                for idx in range(self.n_byzantine):
                    # 不预先分配攻击方法，由Q网络决定
                    # 这里只设置共谋的基本信息
                    attack_plan[idx] = {
                        'is_participating': True,
                        'selected_attack': None,  # 由Q网络决定
                        'attack_strength': self.attack_strength,
                    }
                return attack_plan
        
        return Coordinator(self.byzantine_clients, self.benign_clients, config, self.communicator)
    
    # ========== 内部状态构建器实现 ==========
    def _create_state_builder(self, config: Dict[str, Any]):
        """创建状态构建器（内部实现，简化状态）"""
        class StateBuilder:
            def __init__(self, config, n_other_byzantine):
                self.config = config
                # 状态维度：1（全局模型精度）+ N（其他拜占庭客户端是否参与共谋）
                self.state_dim = 1 + n_other_byzantine
                self.n_other_byzantine = n_other_byzantine
            
            def get_state_dim(self):
                return self.state_dim
            
            def build_state(self, client, agent_id, attack_info, current_round):
                """
                构建简化状态向量
                
                状态组成：
                - 0: 全局模型精度（当前准确率）
                - 1~N: 其他拜占庭客户端是否参与共谋（1表示参与，0表示不参与）
                """
                state = np.zeros(self.state_dim, dtype=np.float32)
                
                # === 状态0: 全局模型精度 ===
                accuracy_trend = attack_info.get('accuracy_trend', {})
                state[0] = accuracy_trend.get('current', 0.0)
                
                # === 状态1~N: 其他拜占庭客户端是否参与共谋 ===
                other_clients = attack_info.get('other_clients', [])
                participation_info = attack_info.get('participation_info', {})
                
                # 获取其他拜占庭客户端的参与状态
                for idx, other_client_info in enumerate(other_clients):
                    if idx < self.n_other_byzantine:
                        # 检查该客户端是否参与共谋（is_participating）
                        is_participating = other_client_info.get('is_participating', False)
                        state[1 + idx] = 1.0 if is_participating else 0.0
                
                # 如果其他客户端数量少于预期，剩余位置填充0
                # （这种情况不应该发生，但为了安全起见）
                for idx in range(len(other_clients), self.n_other_byzantine):
                    state[1 + idx] = 0.0
                
                return state
        
        # 计算其他拜占庭客户端数量（不包括当前客户端）
        n_other_byzantine = max(len(self.byzantine_clients) - 1, 0)
        return StateBuilder(config, n_other_byzantine)
    
    def _create_reward_calculator(self, config: Dict[str, Any]):
        """创建奖励计算器（内部实现）"""
        class RewardCalculator:
            def __init__(self, config):
                self.config = config
                # 奖励权重
                self.accuracy_weight = config.get('reward_accuracy_weight', 10.0)
                self.collusion_weight = config.get('reward_collusion_weight', 2.0)
                self.diversity_weight = config.get('reward_diversity_weight', 0.5)
                self.consistency_weight = config.get('reward_consistency_weight', 1.0)
            
            def calculate_reward(self, agent_id, action, attack_info, accuracy_change):
                """
                计算多维奖励
                
                奖励组成：
                1. 准确率下降奖励：准确率下降越多，奖励越高
                2. 共谋协调奖励：与其他拜占庭客户端协调一致（参与率）
                """
                reward = 0.0
                
                # 1. 准确率下降奖励（主要奖励）
                if accuracy_change is not None:
                    # 准确率下降（负变化）给予正奖励
                    accuracy_reward = -accuracy_change * self.accuracy_weight
                    reward += accuracy_reward

                    # 如果准确率上升超过阈值，给予额外惩罚
                    if accuracy_change > 0.13:
                        penalty_weight = self.config.get('reward_penalty_weight', self.accuracy_weight)
                        penalty = (accuracy_change - 0.13) * penalty_weight
                        reward -= penalty
                
                # 2. 共谋协调奖励
                participation_info = attack_info.get('participation_info', {})
                participation_rate = participation_info.get('participation_rate', 0.0)
                
                # 参与率越高，协调奖励越高
                collusion_reward = participation_rate * self.collusion_weight
                reward += collusion_reward
                
                return reward
        
        return RewardCalculator(config)


