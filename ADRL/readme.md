# ADRL - Adversarial Deep Reinforcement Learning

基于FedDistill项目创建的联邦学习框架，使用FedAvg方法进行联邦学习，不使用wandb。

## 项目结构

```
ADRL/
├── actors.py              # 客户端和服务器Actor类
├── config.py              # 数据集配置和变换
├── main.py                # 主运行文件
├── strategies.py          # 联邦学习策略
├── utilities.py           # 工具函数
├── requirements.txt       # 依赖包
├── README.md             # 项目说明
├── models/               # 模型定义
│   ├── __init__.py
│   └── simple_cnn.py     # 简单CNN模型
├── byzantine/            # 拜占庭攻击和防御
│   ├── __init__.py
│   ├── attacks.py        # 攻击方法
│   ├── defences.py       # 防御方法
│   └── config.py         # 配置
├── runners/              # 运行器
│   ├── __init__.py
│   ├── BaseRunner.py     # 基础运行器
│   └── FedAVGRunner.py   # FedAVG运行器
└── byzantine_example.py  # 拜占庭攻击和防御示例
```

## 功能特性

- **FedAVG策略**：使用联邦平均算法进行联邦学习
- **多种模型**：支持简单CNN、ResNet等模型架构
- **多数据集**：支持MNIST、CIFAR-10、CIFAR-100、Fashion-MNIST
- **私有数据训练**：客户端只在私有数据上训练，不涉及公共数据集
- **拜占庭攻击**：支持多种拜占庭攻击方法（随机、符号翻转、高斯噪声等）
- **拜占庭防御**：支持多种防御方法（中位数、修剪均值、Krum、Bulyan等）
- **自动混合精度**：支持AMP加速训练
- **灵活配置**：支持命令行参数和配置文件
- **无wandb依赖**：使用本地日志记录，不依赖wandb

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python main.py --dataset cifar10 --n_clients 5 --n_communications 10
```

### 参数说明

- `--dataset`: 数据集名称 (mnist, cifar10, cifar100, fashionmnist)
- `--arch`: 模型架构 (simple_cnn, mnist_cnn, simple_resnet)
- `--n_clients`: 客户端数量
- `--n_communications`: 通信轮数
- `--n_total_local_epochs`: 总本地训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备 (cuda:0, cpu等)

### 示例命令

```bash
# CIFAR-10上的FedAVG训练
python main.py --dataset cifar10 --n_clients 10 --n_communications 20 --n_total_local_epochs 50

# MNIST上的FedAVG训练
python main.py --dataset mnist --n_clients 5 --n_communications 100 --n_total_local_epochs 20

# 使用ResNet在CIFAR-100上训练
python main.py --dataset cifar100 --arch simple_resnet --n_clients 8 --n_communications 30

# 带拜占庭攻击和防御的训练
python main.py --dataset cifar10 --arch resnet18 --n_clients 50 --n_communications 30 --n_total_local_epochs 150 --attack poisonedfl --defence momentum  --n_byzantine 10
# 拜占庭攻击和防御示例```
1234
## 配置说明

项目支持通过命令行参数或修改`main.py`中的`defaults`字典来配置训练参数。

### 主要配置项

- **数据集设置**: `dataset`, `batch_size`
- **模型设置**: `arch`, `input_channels`
- **联邦学习设置**: `n_clients`, `n_communications` (固定使用FedAVG策略)
- **拜占庭设置**: `attack`, `defence`, `n_byzantine_clients`
- **训练设置**: `lr`, `optimizer`, `scheduler`
- **系统设置**: `device`, `seed`, `use_amp`

## 输出结果

训练结果将保存在`./results/`目录下，包括：

- `config.json`: 训练配置
- `final_results.json`: 最终结果和指标
- `final_model.pth`: 训练好的模型

## 扩展功能

### 添加新模型

在`models/`目录下创建新的模型文件，并在`models/simple_cnn.py`的`get_model`函数中添加新模型。

### 修改策略

项目固定使用FedAVG策略，如需修改可在`strategies.py`中调整FedAVG类的实现。

### 添加新数据集

在`config.py`中添加新数据集的配置，包括均值、标准差、变换等。

### 拜占庭攻击和防御

项目支持多种拜占庭攻击和防御方法：

#### 攻击方法
- `no_attack`: 无攻击（基线）
- `random`: 随机攻击
- `sign_flip`: 符号翻转攻击
- `gaussian`: 高斯噪声攻击
- `label_flip`: 标签翻转攻击
- `model_replacement`: 模型替换攻击
- `adaptive`: 自适应攻击
- `qlearning`: Q-Learning智能攻击

#### 防御方法
- `no_defence`: 无防御（基线）
- `median`: 中位数防御
- `trimmed_mean`: 修剪均值防御
- `krum`: Krum防御
- `bulyan`: Bulyan防御
- `adaptive`: 自适应防御

#### 使用示例
```bash
# 查看可用攻击和防御方法
python byzantine_example.py --list_attacks
python byzantine_example.py --list_defences

# 交互式选择攻击和防御
python byzantine_example.py --interactive

# 直接运行特定攻击和防御
python byzantine_example.py --attack gaussian --defence median --n_byzantine 2

# Q-Learning智能攻击示例
python qlearning_example.py --example basic
python qlearning_example.py --example interactive
```

### Q-Learning智能攻击

Q-Learning攻击是一种基于强化学习的智能拜占庭攻击方法，具有以下特点：

#### 状态空间
- **本地信息**：模型参数特征、数据分布特征、当前动作、历史动作序列
- **全局信息**：全局模型性能、模型更新幅度
- **协同信息**：其他恶意客户端的动作

#### 动作空间
- **动作0**：不攻击
- **动作1**：攻击（随机选择攻击方法）

#### 奖励函数
- **主奖励**：全局模型性能下降程度
- **探索奖励**：鼓励尝试不同动作

#### 使用方法
```bash
# 基本使用
python main.py --attack qlearning --defence median --n_byzantine 2

# 自定义Q-Learning参数
python main.py --attack qlearning --ql_learning_rate 0.05 --ql_epsilon 0.2 --n_byzantine 3

# 运行Q-Learning示例
python qlearning_example.py --example basic
```

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 根据数据集大小调整批次大小
3. 联邦学习的效果取决于客户端数量和通信轮数
4. 建议在开始大规模实验前先进行小规模测试

## 许可证

本项目基于MIT许可证开源。
