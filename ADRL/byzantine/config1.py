# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/config.py
# Description:  拜占庭攻击和防御配置
# ===========================================================================

# 攻击方法配置
ATTACK_CONFIGS = {
    'no_attack': {
        'description': '无攻击（基线）',
        'parameters': {}
    },
    'random': {
        'description': '随机攻击：将模型参数替换为随机值',
        'parameters': {
            'attack_strength': 1.0  # 攻击强度
        }
    },
    'sign_flip': {
        'description': '符号翻转攻击：翻转模型参数的符号',
        'parameters': {
            'flip_ratio': 1.0  # 翻转比例
        }
    },
    'gaussian': {
        'description': '高斯噪声攻击：向模型参数添加高斯噪声',
        'parameters': {
            'noise_std': 0.1  # 噪声标准差
        }
    },
    'label_flip': {
        'description': '标签翻转攻击：翻转训练标签',
        'parameters': {
            'flip_ratio': 0.5  # 翻转比例
        }
    },
    'model_replacement': {
        'description': '模型替换攻击：用恶意模型替换客户端模型',
        'parameters': {
            'malicious_model_path': None,  # 恶意模型路径
            'scale_factor': 2.0  # 缩放因子
        }
    },
    'qlearning': {
        'description': 'Q-Learning智能攻击：使用强化学习选择最优攻击策略',
        'parameters': {
            'ql_learning_rate': 0.01,  # Q-Learning学习率（降低以提高稳定性，但更新更频繁）
            'ql_discount_factor': 0.95,  # 折扣因子（提高以更重视长期回报）
            'ql_epsilon': 0.1,  # 探索率（提高初始探索率）
            'ql_epsilon_decay': 0.995,  # 探索率衰减（加快衰减使动作选择更快收敛）
            'ql_epsilon_min': 0.01,  # 最小探索率
            'ql_use_nn': True,  # 是否使用神经网络
            'ql_memory_size': 10000,  # 经验回放缓冲区大小
            'ql_batch_size': 64,  # 批次大小（增大以提高学习效率）
            'ql_update_target_freq': 50,  # 目标网络更新频率（降低以更频繁更新，从150改为50）
            'ql_training_mode': True,  # 训练模式
            'ql_save_path': './qlearning_models',  # 模型保存路径
            'weight_reward_scale': 1.5,  # 权重奖励缩放因子（增加权重奖励的影响）
        }
    },
    'rlfl': {
        'description': 'RLFL攻击：基于强化学习的联邦学习攻击（完整Policy Learning）',
        'parameters': {
            # Policy Learning 配置
            'rlfl_lr': 3e-4,                    # 学习率
            'rlfl_gamma': 0.99,                 # 折扣因子
            'rlfl_gae_lambda': 0.95,            # GAE参数
            
            # PPO 特定参数
            'rlfl_ppo_epochs': 10,              # PPO更新轮数
            'rlfl_ppo_clip': 0.2,               # PPO裁剪参数
            'rlfl_value_loss_coef': 0.5,        # 价值损失系数
            'rlfl_entropy_coef': 0.01,          # 熵系数
            'rlfl_max_grad_norm': 0.5,          # 梯度裁剪
            
            # 经验回放
            'rlfl_batch_size': 64,              # 批次大小
            'rlfl_buffer_size': 10000,          # 缓冲区大小
            
            # MDP 定义
            'rlfl_action_dim': 1,               # 动作维度
            'rlfl_action_scale': 10.0,          # 动作缩放
            
            # 模拟器配置
            'simulator_max_rounds': 50,         # 模拟器最大轮数
            
            # 训练配置
            'rlfl_n_episodes': 100,             # 训练回合数
            'rlfl_max_steps_per_episode': 50,   # 每回合最大步数
            'rlfl_pretrain': True,              # 是否预训练策略
            'rlfl_save_path': './rlfl_models',  # 模型保存路径
            
            # 共谋配置
            'enable_collusion': True,           # 启用共谋模式
            'collusion_strength': 1.0,          # 共谋强度
        }
    },
    'gradient_inversion_collusion': {
        'description': '梯度反演共谋攻击：结合梯度反演、分布对齐、拜占庭共谋和相似性损失的高级攻击',
        'parameters': {
            # 梯度反演参数
            'gi_inversion_lr': 0.1,  # 梯度反演学习率
            'gi_inversion_steps': 100,  # 梯度反演优化步数
            'gi_inversion_batch_size': 32,  # 虚拟数据批次大小
            
            # 分布一致性损失权重（对应图2中的λ1和λ2）
            'gi_lambda_1': 1.0,  # 当前梯度与历史梯度的MMD权重
            'gi_lambda_2': 1.0,  # 当前梯度与良性分布的MMD权重
            
            # 相似性损失权重（对应图1中的β）
            'gi_beta': 0.5,  # 避免过度相似的权重
            
            # 攻击强度
            'gi_attack_scale': 1.5,  # 攻击缩放因子
            
            # 历史长度
            'gi_history_length': 10,  # 保留的历史梯度数量
            
            # 共谋配置
            'enable_collusion': True,  # 启用共谋模式
            'collusion_strength': 1.0,  # 共谋强度
        }
    }
}

# 防御方法配置
DEFENCE_CONFIGS = {
    'no_defence': {
        'description': '无防御（基线）',
        'parameters': {}
    },
    'median': {
        'description': '中位数防御：使用中位数聚合',
        'parameters': {}
    },
    'trimmed_mean': {
        'description': '修剪均值防御：移除异常值后计算均值',
        'parameters': {
            'trim_ratio': 0.2  # 修剪比例
        }
    },
    'krum': {
        'description': 'Krum防御：选择最相似的模型',
        'parameters': {
            'f': 1  # 最多容忍的拜占庭客户端数量
        }
    },
    'bulyan': {
        'description': 'Bulyan防御：结合Krum和修剪均值',
        'parameters': {
            'f': 1,  # 最多容忍的拜占庭客户端数量
            'krum_k': 4  # Krum选择的模型数量
        }
    },
    'adaptive': {
        'description': '自适应防御：根据攻击类型调整防御策略',
        'parameters': {
            'defence_strategy': 'auto',  # 防御策略
            'adaptation_threshold': 0.1  # 适应阈值
        }
    }
}

# 默认配置
DEFAULT_ATTACK_CONFIG = {
    'attack': 'no_attack',
    'n_byzantine_clients': 0,
    'attack_strength': 1.0,
    'flip_ratio': 0.5,
    'noise_std': 0.1,
    'malicious_model_path': None,
    'scale_factor': 2.0,
    'adaptive_attack_type': 'gaussian',
    'adaptation_rate': 0.1
}

DEFAULT_DEFENCE_CONFIG = {
    'defence': 'no_defence',
    'trim_ratio': 0.2,
    'f': 1,
    'krum_k': 4,
    'defence_strategy': 'auto',
    'adaptation_threshold': 0.1
}

# 攻击-防御组合建议
ATTACK_DEFENCE_RECOMMENDATIONS = {
    'random': ['median', 'trimmed_mean', 'krum'],
    'sign_flip': ['median', 'krum', 'bulyan'],
    'gaussian': ['trimmed_mean', 'krum', 'bulyan'],
    'label_flip': ['median', 'trimmed_mean'],
    'model_replacement': ['krum', 'bulyan'],
    'adaptive': ['adaptive', 'bulyan'],
    'gradient_inversion_collusion': ['bulyan', 'adaptive', 'krum'],  # 需要强鲁棒性防御
    'qlearning': ['bulyan', 'krum', 'fltrust'],  # Q-Learning智能攻击需要强防御
    'rlfl': ['bulyan', 'fltrust', 'krum'],  # RLFL攻击需要最强防御
}

def get_attack_config(attack_name: str) -> dict:
    """获取攻击配置"""
    if attack_name not in ATTACK_CONFIGS:
        raise ValueError(f"未知的攻击方法: {attack_name}")
    
    config = DEFAULT_ATTACK_CONFIG.copy()
    config.update(ATTACK_CONFIGS[attack_name]['parameters'])
    config['attack'] = attack_name
    return config

def get_defence_config(defence_name: str) -> dict:
    """获取防御配置"""
    if defence_name not in DEFENCE_CONFIGS:
        raise ValueError(f"未知的防御方法: {defence_name}")
    
    config = DEFAULT_DEFENCE_CONFIG.copy()
    config.update(DEFENCE_CONFIGS[defence_name]['parameters'])
    config['defence'] = defence_name
    return config

def get_recommended_defences(attack_name: str) -> list:
    """获取推荐的防御方法"""
    return ATTACK_DEFENCE_RECOMMENDATIONS.get(attack_name, ['no_defence'])

def validate_config(config: dict) -> bool:
    """验证配置的有效性"""
    required_keys = ['attack', 'defence', 'n_byzantine_clients']
    
    for key in required_keys:
        if key not in config:
            return False
    
    # 验证攻击方法
    if config['attack'] not in ATTACK_CONFIGS:
        return False
    
    # 验证防御方法
    if config['defence'] not in DEFENCE_CONFIGS:
        return False
    
    # 验证拜占庭客户端数量
    if config['n_byzantine_clients'] < 0:
        return False
    
    return True

# 在配置文件中添加 RLFL 攻击配置
config = {
    # RLFL 攻击配置
    'rlfl_lr': 3e-4,                    # 学习率
    'rlfl_gamma': 0.99,                 # 折扣因子
    'rlfl_gae_lambda': 0.95,            # GAE参数
    'rlfl_ppo_epochs': 10,              # PPO更新轮数
    'rlfl_ppo_clip': 0.2,               # PPO裁剪参数
    'rlfl_value_loss_coef': 0.5,        # 价值损失系数
    'rlfl_entropy_coef': 0.01,          # 熵系数
    'rlfl_max_grad_norm': 0.5,          # 梯度裁剪
    'rlfl_batch_size': 64,              # 批次大小
    'rlfl_buffer_size': 10000,          # 缓冲区大小
    'rlfl_action_dim': 1,               # 动作维度
    'rlfl_action_scale': 10.0,          # 动作缩放
    'simulator_max_rounds': 50,         # 模拟器最大轮数
}

# 创建 RLFL 攻击实例
rlfl_attack = RLFLAttack(clients, config, runner_instance)

# 训练策略（Policy Learning 阶段）
rlfl_attack.train_policy(n_episodes=100, max_steps_per_episode=50)

# 在联邦学习过程中使用训练好的策略
perturbed_models = rlfl_attack.get_perturbed_client_models()

# 每轮更新性能
rlfl_attack.update_performance(current_accuracy)

# 保存训练好的策略
save_policy(rlfl_attack.actor, rlfl_attack.critic, 'rlfl_policy.pth')