# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         models/cifar10.py
# Description:  model definitions
# ===========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN model suitable for CIFAR-10/100 and MNIST."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate linear layer input size assuming 32x32 input dimension
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.25)
