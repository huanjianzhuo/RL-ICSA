# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         config.py
# Description:  数据集、标准化和变换配置
# ===========================================================================

import torch
import torchvision
from torchvision import transforms

# 数据集均值和标准差
means = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'mnist': (0.1307,),
    'fashionmnist': (0.2860,),
}

stds = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.3081,),
    'fashionmnist': (0.3530,),
}

# 类别数量
n_classesDict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'fashionmnist': 10,
}

# 数据加载器工作线程数
num_workersDict = {
    'mnist': 2 if torch.cuda.is_available() else 0,
    'cifar10': 2 if torch.cuda.is_available() else 0,
    'cifar100': 4 if torch.cuda.is_available() else 0,
    'fashionmnist': 2 if torch.cuda.is_available() else 0,
}

# 数据集字典
datasetDict = {
    'mnist': getattr(torchvision.datasets, 'MNIST'),
    'cifar10': getattr(torchvision.datasets, 'CIFAR10'),
    'cifar100': getattr(torchvision.datasets, 'CIFAR100'),
    'fashionmnist': getattr(torchvision.datasets, 'FashionMNIST'),
}

# 训练数据变换
trainTransformDict = {
    'mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['mnist'], std=stds['mnist'])
    ]),
    'cifar10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']),
    ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']),
    ]),
    'fashionmnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['fashionmnist'], std=stds['fashionmnist'])
    ]),
}

# 测试数据变换
testTransformDict = {
    'mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['mnist'], std=stds['mnist'])
    ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']),
    ]),
    'fashionmnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['fashionmnist'], std=stds['fashionmnist'])
    ]),
}


