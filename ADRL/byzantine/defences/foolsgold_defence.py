# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/defences/foolsgold_defence.py
# Description:  FoolsGold防御
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
    FoolsGold 防御：基于更新相似度对可疑（女巫/共谋）客户端降权。
    参考：Fung et al., RAID 2020 / arXiv:1808.04866
    """

    def _flatten_update(self, client_model: torch.nn.Module, base_state: OrderedDict) -> torch.Tensor:
        vec = []
        sd = client_model.state_dict()
        for k, base in base_state.items():
            param = sd[k]
            if not param.dtype.is_floating_point:
                continue
            # 确保 param 和 base 在同一设备上
            param_tensor = param.detach().to(base.device)
            vec.append((param_tensor - base).flatten().to(dtype=torch.float32, device='cpu'))
        if len(vec) == 0:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(vec)

    def _compute_fg_weights(self, updates: torch.Tensor) -> torch.Tensor:
        n = updates.size(0)
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        eps = 1e-12
        norms = updates.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        U = updates / norms
        S = U @ U.t()
        S = torch.clamp(S, min=0.0)
        S.fill_diagonal_(0.0)
        smax = S.max(dim=1).values
        w = 1.0 - smax
        # Pardoning
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if w[i] < w[j] and S[i, j] > 0:
                    S[i, j] = S[i, j] * (w[i] / (w[j] + eps))
        smax = S.max(dim=1).values
        w = 1.0 - smax
        w = torch.clamp(w, min=0.0)
        m = float(w.max().item())
        if m > 0:
            w = w / m
        return w

    def get_aggregated_model(self, client_models: List[torch.nn.Module]) -> OrderedDict:
        if not client_models:
            raise ValueError("客户端模型列表为空")
        if hasattr(self.runner, 'server') and hasattr(self.runner.server, 'model') and self.runner.server.model is not None:
            base_state = copy.deepcopy(self.runner.server.model.state_dict())
        else:
            base_state = copy.deepcopy(client_models[0].state_dict())
        updates = [self._flatten_update(m, base_state) for m in client_models]
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
            raise ValueError("客户端预测列表为空")
        preds = [p.detach().flatten().to(dtype=torch.float32, device='cpu') for p in client_predictions]
        max_len = max(p.numel() for p in preds)
        padded = [F.pad(p, (0, max_len - p.numel())) for p in preds]
        U = torch.stack(padded, dim=0)
        w = self._compute_fg_weights(U)
        stacked = torch.stack(client_predictions)
        wn = w / (w.sum() + 1e-12)
        agg = torch.tensordot(wn.to(stacked.device, dtype=stacked.dtype), stacked, dims=([0], [0]))
        outlier = 1.0 - w
        if outlier.max() > 0:
            outlier = outlier / outlier.max()
        return agg, outlier

