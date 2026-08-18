# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/foolsgold_defence.py
# Description:  FoolsGold Defense
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
import copy
import logging

from .base import ByzantineDefence

logger = logging.getLogger(__name__)


class FoolsGoldDefence(ByzantineDefence):
    """
    FoolsGold Defense: Downweights suspicious (Sybil/colluding) clients based on update similarity.
    Reference: Fung et al., RAID 2020 / arXiv:1808.04866
    """

    def _flatten_update(self, client_model: torch.nn.Module, base_state: OrderedDict) -> torch.Tensor:
        vec = []
        sd = client_model.state_dict()
        for k, base in base_state.items():
            param = sd[k]
            if not param.dtype.is_floating_point:
                continue
            # Ensure param and base are on the same device
            param_tensor = param.detach().to(base.device)
            vec.append((param_tensor - base).flatten().to(dtype=torch.float32, device='cpu'))
        if len(vec) == 0:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(vec)

    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> OrderedDict:
        base_state = copy.deepcopy(self.runner.server.model.state_dict())
        updates = [self._flatten_update(cm, base_state) for cm in client_models]
        updates = torch.stack(updates, dim=0)
        w = self._compute_fg_weights(updates)
        agg_state = OrderedDict()
        n = len(client_models)
        for k, base in base_state.items():
            if not base.dtype.is_floating_point:
                agg_state[k] = base
                continue
            dev = base.device
            dtype = base.dtype
            delta = torch.zeros_like(base, device=dev, dtype=dtype)
            for i in range(n):
                ci = client_models[i].state_dict()[k].to(device=dev, dtype=dtype)
                delta = delta + float(w[i].item()) * (ci - base)
            agg_state[k] = base + delta
        return agg_state

    def get_aggregated_predictions(self, client_predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not client_predictions:
            raise ValueError("Client predictions list is empty")
        preds = [p.detach().flatten().to(dtype=torch.float32, device='cpu') for p in client_predictions]
        max_len = max(p.numel() for p in preds)
        padded = [F.pad(p, (0, max_len - p.numel())) for p in preds]
        U = torch.stack(padded, dim=0)
        w = self._compute_fg_weights(U)
        stacked = torch.stack(client_predictions)
        
        aggregated_pred = torch.zeros_like(client_predictions[0])
        for i in range(len(client_predictions)):
            aggregated_pred += w[i] * client_predictions[i]
            
        return aggregated_pred, w
