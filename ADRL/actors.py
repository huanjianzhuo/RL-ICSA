# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         actors.py
# Description:  Client and Server Actor classes
# ===========================================================================

import importlib
import os
from collections import OrderedDict
from typing import Optional
import time
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models import cifar10
from utilities import WarmupLRWrapper, SequentialSchedulers

logger = logging.getLogger(__name__)


class Actor(ABC):
    """Base Actor class defining common interfaces for clients and servers."""

    def __init__(self, use_amp, **kwargs):
        self.use_amp = use_amp
        self.n_classes = kwargs['n_classes']
        self.tmp_dir = kwargs['tmp_dir']
        self.num_workers = kwargs['num_workers']
        self.config = kwargs['config']
        self.callbacks = kwargs['callbacks']
        self.device = kwargs['device']
        
        # Model related
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_criterion = nn.CrossEntropyLoss()
        self.gradScaler = GradScaler() if use_amp else None
        
        # Data related
        self.dataloader = None
        self.trainData = None
        
        # Metrics related
        self.metrics = {}
        self.best_checkpoint_val_accuracy = 0.0
        
        # Checkpoint related
        self.checkpoint_path = None
        self.best_model_state = None
        self.best_optimizer_state = None

    def set_model(self, reinit: bool = False, fileName: Optional[str] = None):
        """Sets the model."""

        if reinit:
            model_cls = getattr(importlib.import_module('models.' + self.config['dataset']), self.config['arch'])
            model = model_cls(num_classes=self.n_classes, input_channels=self.config.get('input_channels', 3))
        else:
            # The model has been initialized already
            model = self.model

        if fileName is not None:
            fPath = os.path.join(self.tmp_dir, fileName)

            state_dict = torch.load(fPath, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                v_new = v  # Remains unchanged if not in _orig format
                if k.endswith("_orig"):
                    # We loaded the _orig tensor and corresponding mask
                    name = name[:-5]  # Truncate the "_orig"
                    if f"{k[:-5]}_mask" in state_dict.keys():
                        v_new = v * state_dict[f"{k[:-5]}_mask"]

                new_state_dict[name] = v_new

            maskKeys = [k for k in new_state_dict.keys() if k.endswith("_mask")]
            for k in maskKeys:
                del new_state_dict[k]

            # Load the state_dict
            model.load_state_dict(new_state_dict)
        self.model = model.to(device=self.device)

    def reset_averaged_metrics(self):
        """Resets averaged metrics."""
        self.metrics = {
            'train': {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0},
            'val': {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0},
            'test': {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}
        }

    @torch.no_grad()
    def update_batch_metrics(self, mode: str, loss: torch.Tensor, output: torch.Tensor, y_target: torch.Tensor):
        """Updates batch metrics."""
        if mode not in self.metrics:
            self.metrics[mode] = {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}
        
        batch_size = output.size(0)
        self.metrics[mode]['loss'] += loss.item() * batch_size
        self.metrics[mode]['num_samples'] += batch_size
        
        # Calculate accuracy
        if y_target is not None:
            pred = output.argmax(dim=1)
            correct = pred.eq(y_target).sum().item()
            self.metrics[mode]['accuracy'] += correct

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Gets averaged metrics."""
        averaged_metrics = {}
        for mode, metrics in self.metrics.items():
            if metrics['num_samples'] > 0:
                averaged_metrics[mode] = {
                    'loss': metrics['loss'] / metrics['num_samples'],
                    'accuracy': metrics['accuracy'] / metrics['num_samples']
                }
            else:
                averaged_metrics[mode] = {'loss': 0.0, 'accuracy': 0.0}
        return averaged_metrics

    def update_checkpoint(self):
        """Updates checkpoint."""
        val_metrics = self.metrics.get('val', {})
        val_accuracy = val_metrics.get('accuracy', 0.0)
        
        if val_accuracy > self.best_checkpoint_val_accuracy:
            self.best_checkpoint_val_accuracy = val_accuracy
            self.best_model_state = self.model.state_dict().copy()
            if self.optimizer is not None:
                self.best_optimizer_state = self.optimizer.state_dict().copy()

    def reset_val_and_test_metrics(self):
        """Resets validation and test metrics."""
        self.metrics['val'] = {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}
        self.metrics['test'] = {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}


class Client(Actor):
    """Client class."""
    actor_type = 'client'

    def __init__(self, use_amp, client_id, **kwargs):
        super().__init__(use_amp=use_amp, **kwargs)
        self.client_id = client_id
        self.actor_name = f'client-{self.client_id}'

        # Define private variables which are to be set
        self.trainData = None
        self.dataloader = None
        self.model = None
        self.original_loss = None
        self.is_byzantine = False

        # Q-Learning attack related (used for Byzantine clients only)
        self.q_network = None
        self.q_target_network = None
        self.q_optimizer = None
        self.q_memory = None
        self.q_state = None
        self.q_action = None
        self.q_episode_count = 0

        # Checkpoint/Early Stopping variables
        self.best_checkpoint_model = None
        self.best_checkpoint_val_accuracy = 0

    def assign_dataset(self, trainData: torch.utils.data.Subset):
        """Assigns dataset, creates dataloader
        Args:
            trainData (torch.utils.data.Subset): Private Training dataset of client
        """
        self.trainData = trainData
        self.dataloader = torch.utils.data.DataLoader(trainData, batch_size=self.config['batch_size'], shuffle=True,
                                                      pin_memory=torch.cuda.is_available(),
                                                      num_workers=self.num_workers)

    def load_checkpoint(self):
        """Loads the checkpoint of the client."""
        # Take the self.best_checkpoint_model and load it
        if self.best_checkpoint_model is not None:
            # Move all tensors to GPU
            self.best_checkpoint_model = {key: val.to(device=self.device) for key, val in
                                          self.best_checkpoint_model.items()}
            # Load the state dict directly from self.best_checkpoint_model
            self.model.load_state_dict(self.best_checkpoint_model)
            self.model = self.model.to(device=self.device)

            del self.best_checkpoint_model
            self.best_checkpoint_model = None
            self.best_checkpoint_val_accuracy = 0

    def update_checkpoint(self):
        """Updates the checkpoint of the client."""
        # Get the current validation accuracy
        val_accuracy = self.metrics['val']['accuracy']
        if val_accuracy >= self.best_checkpoint_val_accuracy:
            self.best_checkpoint_val_accuracy = val_accuracy

            # Delete the old checkpoint model if existing
            if self.best_checkpoint_model is not None:
                del self.best_checkpoint_model

            # Save the state dict directly to self.best_checkpoint_model, copying the tensors and moving to CPU
            self.best_checkpoint_model = {key: val.detach().clone().cpu() for key, val in
                                          self.model.state_dict().items()}

    def detach_model(self):
        """Detach the model to avoid OOM, i.e., we save the state dict and reload it when needed.
        """
        pass

    def attach_model(self):
        """Re-attach the model, i.e., we reload the state dict.
        """
        pass

    def save_model(self, modelType: str) -> str:
        """Saves current model to os.path.join(self.tmp_dir, f"{modelType}_model.pt"), returns the complete file path.

        Args:
            modelType (str): Name of model type such as 'initial'.

        Returns:
            str: Absolute path to saved model state dict.
        """

        fName = f"{modelType}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict
        return fPath

    def set_optimizer_and_scheduler(self, n_epochs: int, n_batches_per_epoch: int, reinit_optimizer: bool = True):
        """
        Sets the optimizer and scheduler.
        Args:
            n_epochs (int): Use the specified amount of epochs for the learning rate.
            n_batches_per_epoch (int): Number of batches per epoch.
            reinit_optimizer (bool): If True, reinit the optimizer, otherwise keep.
        """
        if self.actor_type == 'client':
            learning_rate = self.config['client_lr']
        elif self.actor_type == 'server':
            learning_rate = self.config['server_lr']
        else:
            raise NotImplementedError(f"Actor type {self.actor_type} not implemented.")

        self.define_optimizer_scheduler(learning_rate=learning_rate, n_epochs=n_epochs,
                                        n_batches_per_epoch=n_batches_per_epoch, reinit_optimizer=reinit_optimizer,
                                        do_warmup=False)

    def define_optimizer_scheduler(self, learning_rate, n_epochs: int, n_batches_per_epoch: int,
                                   do_warmup: bool = False, reinit_optimizer: bool = True):
        """
        Defines optimizer and learning rate scheduler, sets self.optimizer and self.scheduler.
        Args:
            learning_rate (str or float): Learning rate schedule in the form of (type, kwargs) or a float value
            n_epochs (int): Number of epochs to run scheduler for
            n_batches_per_epoch (int): Number of batches per epoch
            do_warmup (bool): If True, warmup for 5% of iterations
            reinit_optimizer (bool): If True, reinit the optimizer, otherwise keep.
        """
        # If learning_rate is numerical, convert to Constant scheduler format
        if isinstance(learning_rate, (int, float)):
            learning_rate = f"(Constant, {learning_rate})"
        
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
            scheduler = (name, kwargs)
            initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        wd = self.config['weight_decay'] or 0.
        if reinit_optimizer:
            if self.config['optimizer'] == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                                 momentum=self.config['momentum'],
                                                 weight_decay=wd, nesterov=wd > 0.)
            elif self.config['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr, weight_decay=wd)
            else:
                raise NotImplementedError("Only SGD and AdamW implemented at the moment.")

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = n_batches_per_epoch
        n_total_iterations = iterations_per_epoch * n_epochs
        n_warmup_iterations = 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if do_warmup and int(0.05 * n_total_iterations) > 0:
            n_warmup_iterations = int(0.05 * n_total_iterations)
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        name, kwargs = scheduler
        scheduler = None
        if name == 'Constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif name == 'StepLR':
            # Tuple of form ('StepLR', initial_lr, step_size, gamma)
            # Reduces initial_lr by gamma every step_size epochs
            step_size, gamma = int(kwargs[1]), float(kwargs[2])

            # Convert to iterations
            step_size = iterations_per_epoch * step_size

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size,
                                                        gamma=gamma)
        elif name == 'MultiStepLR':
            # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
            # Reduces initial_lr by gamma every epoch that is in the list milestones
            milestones, gamma = kwargs[1].strip(), float(kwargs[2])
            # Remove square bracket
            if milestones[0] == '[':
                milestones = milestones[1:]
            if milestones[-1] == ']':
                milestones = milestones[:-1]
            # Convert to iterations directly
            milestones = [int(ms) * iterations_per_epoch for ms in milestones.split('|')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                             gamma=gamma)
        elif name == 'ExponentialLR':
            # Tuple of form ('ExponentialLR', initial_lr, gamma)
            gamma = float(kwargs[1])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif name in ['Linear']:
            if len(kwargs) == 2:
                # The final learning rate has also been passed
                end_factor = float(kwargs[1]) / float(initial_lr)
            else:
                end_factor = 0.
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=end_factor,
                                                          total_iters=n_remaining_iterations)

        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        elif name in ['StepLR', 'MultiStepLR']:
            # We need parallel schedulers, since the steps should be counted during warmup
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def warmup_scheduler(self, warmup_steps: int):
        """Adds a short warmup of the learning rate to the current scheduler."""
        if warmup_steps > 0:
            self.scheduler = WarmupLRWrapper(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                warmup_steps=warmup_steps)


class Server(Client):
    """Server class."""
    actor_type = 'server'

    def __init__(self, use_amp, **kwargs):
        super().__init__(use_amp=use_amp, client_id=None, **kwargs)
        self.actor_name = 'server'
