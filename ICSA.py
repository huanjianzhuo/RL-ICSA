# ===========================================================================
# Project:      ADRL - Adversarial Deep Reinforcement Learning
# File:         byzantine/attacks/secondhighestconfidence_attack_ICSA.py
# Description:  ICSA - Inter-class Similarity Attack
# ===========================================================================

import copy
import logging
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from .base import CollusionAttackBase

logger = logging.getLogger(__name__)


class SecondHighestConfidenceAttack(CollusionAttackBase):
    """
    Inter-class Similarity Attack (ICSA)

    Paper-aligned implementation:
        1. Process local samples in a sample-wise manner.
        2. For each sample, extract the feature vector h entering the final
           classification layer and obtain the logit vector z.
        3. Identify Top-1 class i and Top-2 class j.
        4. Compute the sample-specific scaling factor

               lambda = (z_(1) - z_(2) - tau) / (2 * ||h||^2)

           according to the constraint
               (w_i')^T h - (w_j')^T h = tau.
        5. Build a sample-specific perturbation matrix delta_sim^(k):
               row i = -lambda * h
               row j = +lambda * h

           Note that PyTorch stores the final Linear weight as
           [C, d], i.e., each class prototype is a row. This is the
           transpose of the paper's conceptual W_last in R^{d x C}.
        6. Aggregate all sample-specific perturbation matrices using
           element-wise median.
        7. Apply the resulting median perturbation only to the final
           classification-layer weights. Bias is not modified.

    This implementation intentionally does NOT use:
        - softmax probability matching;
        - binary search;
        - a globally selected second-highest class;
        - bias-only perturbation;
        - client-wise averaging of scalar increments;
        - persistent accumulation across rounds;
        - an adjustable attack-strength multiplier.

    These mechanisms belonged to the previous implementation and are not
    part of the current paper formulation.
    """

    def __init__(self, clients: List, config: Dict[str, Any], runner_instance):
        super().__init__(clients, config, runner_instance)

        self.config = config

        # tau in the paper: desired Top-1/Top-2 logit margin after attack.
        # Example: tau = 0.1
        self.tau = float(config.get("tau", 0.1))

        # Optional engineering limit. None means: process the entire local
        # dataloader, which is the default behavior required by the paper.
        self.max_attack_samples = config.get("max_attack_samples", None)

        # Numerical stability for lambda computation.
        self.eps = float(config.get("lambda_eps", 1e-12))

        logger.info("ICSA initialized:")
        logger.info("  - tau: %.6f", self.tau)
        logger.info("  - max_attack_samples: %s", self.max_attack_samples)
        logger.info("  - element-wise median aggregation: enabled")
        logger.info("  - final-layer bias perturbation: disabled")

    # ------------------------------------------------------------------
    # Final classification layer utilities
    # ------------------------------------------------------------------
    def _get_last_linear(
        self, model: torch.nn.Module
    ) -> Optional[nn.Linear]:
        """
        Find the last nn.Linear module in the model.

        Searching from the end is more robust than assuming the model has an
        attribute named 'fc', 'classifier', etc. It also handles structures
        such as nn.Sequential(..., nn.Linear(...)).
        """
        if isinstance(model, nn.Linear):
            return model

        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            logger.warning("Unable to find a final nn.Linear classification layer.")
        return last_linear

    def _get_last_layer_params(
        self, model: torch.nn.Module
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (weight, bias) of the final classification layer.
        """
        last_layer = self._get_last_linear(model)
        if last_layer is None:
            return None, None

        weight = last_layer.weight.data
        bias = last_layer.bias.data if last_layer.bias is not None else None
        return weight, bias

    def _set_last_layer_params(
        self,
        model: torch.nn.Module,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> None:
        """
        Set final classification-layer parameters in-place.
        """
        last_layer = self._get_last_linear(model)
        if last_layer is None:
            return

        if weight is not None:
            if tuple(last_layer.weight.shape) != tuple(weight.shape):
                raise ValueError(
                    "Final-layer weight shape mismatch: "
                    f"module={tuple(last_layer.weight.shape)}, "
                    f"new={tuple(weight.shape)}"
                )
            last_layer.weight.data.copy_(weight)

        # ICSA does not intentionally modify bias, but keep this setter
        # generic so it remains compatible with the existing framework.
        if bias is not None and last_layer.bias is not None:
            if tuple(last_layer.bias.shape) != tuple(bias.shape):
                raise ValueError(
                    "Final-layer bias shape mismatch: "
                    f"module={tuple(last_layer.bias.shape)}, "
                    f"new={tuple(bias.shape)}"
                )
            last_layer.bias.data.copy_(bias)

    # ------------------------------------------------------------------
    # Model output / feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_logits(model_output: Any) -> torch.Tensor:
        """
        Normalize common model output formats to a logits tensor [B, C].
        """
        if torch.is_tensor(model_output):
            return model_output

        if isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError("Model returned an empty tuple/list.")
            if torch.is_tensor(model_output[0]):
                return model_output[0]

        if isinstance(model_output, dict):
            for key in ("logits", "output", "outputs"):
                if key in model_output and torch.is_tensor(model_output[key]):
                    return model_output[key]

        raise TypeError(
            "Unsupported model output type. Expected Tensor, tuple/list, "
            "or dict containing logits."
        )

    def _collect_sample_features_and_logits(
        self, client, model: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the local dataset through the model and capture, for every sample:

            h_k: input to the final Linear layer, shape [d]
            z_k: final logits, shape [C]

        A forward_pre_hook is used because the paper's h is exactly the input
        to the final classification layer (the penultimate representation).
        """
        if not hasattr(client, "dataloader") or client.dataloader is None:
            raise RuntimeError("Client does not provide a valid dataloader.")

        last_layer = self._get_last_linear(model)
        if last_layer is None:
            raise RuntimeError("Cannot identify the final classification layer.")

        captured_features = []
        collected_logits = []
        sample_count = 0

        def capture_feature(module, inputs):
            if not inputs:
                raise RuntimeError(
                    "Final Linear layer received no input; unable to extract h."
                )

            feature = inputs[0]
            if not torch.is_tensor(feature):
                raise TypeError(
                    "Input to final Linear layer is not a tensor; "
                    "cannot construct ICSA perturbation."
                )

            # Expected shape is [B, d]. Flattening is intentionally NOT used,
            # because h must correspond to the d-dimensional input of W_last.
            if feature.ndim != 2:
                feature = feature.reshape(feature.shape[0], -1)

            captured_features.append(feature.detach().clone())

        was_training = model.training
        hook = last_layer.register_forward_pre_hook(capture_feature)

        model.eval()
        try:
            with torch.no_grad():
                for batch_data in client.dataloader:
                    if isinstance(batch_data, (tuple, list)):
                        x = batch_data[0]
                    elif isinstance(batch_data, dict):
                        if "x" in batch_data:
                            x = batch_data["x"]
                        elif "data" in batch_data:
                            x = batch_data["data"]
                        else:
                            raise KeyError(
                                "Unable to find input tensor in batch dictionary."
                            )
                    else:
                        raise TypeError(
                            "Unsupported dataloader batch type. "
                            "Expected tuple/list/dict."
                        )

                    device = next(model.parameters()).device
                    x = x.to(device)

                    model_output = model(x)
                    logits = self._extract_logits(model_output)

                    if logits.ndim != 2:
                        logits = logits.reshape(logits.shape[0], -1)

                    collected_logits.append(logits.detach().clone())

                    sample_count += logits.shape[0]
                    if (
                        self.max_attack_samples is not None
                        and sample_count >= int(self.max_attack_samples)
                    ):
                        break
        finally:
            hook.remove()
            if was_training:
                model.train()

        if not captured_features or not collected_logits:
            raise RuntimeError("No local samples were collected for ICSA.")

        features = torch.cat(captured_features, dim=0)
        logits = torch.cat(collected_logits, dim=0)

        # Ensure feature/logit correspondence remains sample-wise.
        if features.shape[0] != logits.shape[0]:
            n = min(features.shape[0], logits.shape[0])
            logger.warning(
                "Feature/logit sample count mismatch (%d vs %d); truncating to %d.",
                features.shape[0],
                logits.shape[0],
                n,
            )
            features = features[:n]
            logits = logits[:n]

        if (
            self.max_attack_samples is not None
            and features.shape[0] > int(self.max_attack_samples)
        ):
            n = int(self.max_attack_samples)
            features = features[:n]
            logits = logits[:n]

        return features, logits

    # ------------------------------------------------------------------
    # ICSA mathematical core
    # ------------------------------------------------------------------
    def _compute_sample_perturbation(
        self,
        h: torch.Tensor,
        logit: torch.Tensor,
        weight_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Construct delta_sim^(k) for one sample.

        PyTorch final-layer convention:
            W_torch in R^{C x d}

        Paper notation:
            W_last in R^{d x C}

        Therefore:
            w_c in the paper corresponds to W_torch[c, :].

        The perturbation is:
            delta[i, :] = -lambda * h
            delta[j, :] = +lambda * h

        where:
            i = Top-1
            j = Top-2
            lambda = (z_(1) - z_(2) - tau) / (2 ||h||^2)
        """
        if len(weight_shape) != 2:
            raise ValueError(
                f"Expected final classification weight to be 2-D, got {weight_shape}."
            )

        num_classes, feature_dim = int(weight_shape[0]), int(weight_shape[1])

        h = h.reshape(-1)
        if h.numel() != feature_dim:
            raise ValueError(
                "Feature dimensionality mismatch: "
                f"h has {h.numel()} elements, final layer expects {feature_dim}."
            )

        logit = logit.reshape(-1)
        if logit.numel() != num_classes:
            raise ValueError(
                "Logit/class mismatch: "
                f"logit has {logit.numel()} classes, final layer has {num_classes}."
            )

        # Top-1 / Top-2 are determined independently for each sample.
        top2_values, top2_indices = torch.topk(logit, k=2, dim=0)
        z1, z2 = top2_values[0], top2_values[1]
        i, j = top2_indices[0].item(), top2_indices[1].item()

        # Eq. in the paper:
        # lambda = (z_(1) - z_(2) - tau) / (2 ||h||^2)
        h_norm_sq = torch.sum(h * h)

        if h_norm_sq <= self.eps:
            # A zero/near-zero feature vector cannot define a meaningful
            # parameter-direction perturbation.
            return torch.zeros(
                (num_classes, feature_dim),
                dtype=h.dtype,
                device=h.device,
            )

        raw_lambda = (z1 - z2 - self.tau) / (2.0 * h_norm_sq)

        # Top-1 >= Top-2 by definition. In the intended operating regime,
        # margin >= tau and lambda >= 0. If a sample already has margin < tau,
        # we do not increase the margin or reverse the attack direction.
        lam = torch.clamp(raw_lambda, min=0.0)

        delta = torch.zeros(
            (num_classes, feature_dim),
            dtype=h.dtype,
            device=h.device,
        )

        delta[i, :] = -lam * h
        delta[j, :] = +lam * h

        return delta

    def _build_median_perturbation(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate one perturbation matrix per sample and aggregate them with
        element-wise median.
        """
        sample_deltas = []

        for sample_idx in range(features.shape[0]):
            delta_k = self._compute_sample_perturbation(
                h=features[sample_idx],
                logit=logits[sample_idx],
                weight_shape=weight.shape,
            )
            sample_deltas.append(delta_k)

        if not sample_deltas:
            return torch.zeros_like(weight)

        deltas = torch.stack(sample_deltas, dim=0)

        # Element-wise median over the sample dimension, exactly matching
        # the paper's aggregation rule.
        median_delta = torch.median(deltas, dim=0).values

        return median_delta

    # ------------------------------------------------------------------
    # Attack generation
    # ------------------------------------------------------------------
    def _generate_attack_params(
        self, client, agent_id: int
    ) -> Optional[torch.Tensor]:
        """
        Generate the poisoned model parameters using the paper-version ICSA.
        """
        if client.model is None:
            logger.warning("Client model is None; ICSA attack skipped.")
            return None

        model = client.model

        # 1. Locate the final classification layer.
        weight, bias = self._get_last_layer_params(model)
        if weight is None:
            logger.warning("Final-layer weight not found; ICSA attack skipped.")
            return None

        # 2. Sample-wise extraction of h and z.
        try:
            features, logits = self._collect_sample_features_and_logits(
                client, model
            )
        except Exception as exc:
            client_id = getattr(client, "agent_id", agent_id)
            logger.warning(
                "Client %s: failed to collect features/logits for ICSA: %s",
                client_id,
                exc,
            )
            return self._flatten_model_params(model)

        # 3. Compute one delta_sim^(k) per sample and aggregate with
        #    element-wise median.
        median_delta = self._build_median_perturbation(
            features=features,
            logits=logits,
            weight=weight,
        )

        # 4. Apply the median perturbation only to the final-layer weights.
        attacked_model = copy.deepcopy(model)
        attacked_weight, attacked_bias = self._get_last_layer_params(attacked_model)

        if attacked_weight is None:
            logger.warning("Cannot access final-layer weight in model copy.")
            return self._flatten_model_params(model)

        if tuple(attacked_weight.shape) != tuple(median_delta.shape):
            raise ValueError(
                "Median perturbation shape does not match final-layer weight: "
                f"delta={tuple(median_delta.shape)}, "
                f"weight={tuple(attacked_weight.shape)}"
            )

        new_weight = attacked_weight + median_delta

        # Bias is intentionally kept unchanged in ICSA.
        self._set_last_layer_params(
            attacked_model,
            weight=new_weight,
            bias=attacked_bias,
        )

        client_id = getattr(client, "agent_id", agent_id)
        delta_norm = torch.norm(median_delta).item()
        logger.debug(
            "Client %s - ICSA applied: samples=%d, tau=%.6f, "
            "median_delta_norm=%.6f",
            client_id,
            features.shape[0],
            self.tau,
            delta_norm,
        )

        return self._flatten_model_params(attacked_model)

    @staticmethod
    def _flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
        """
        Flatten all model parameters into the representation expected by the
        existing CollusionAttackBase interface.
        """
        params = [param.data.flatten() for param in model.parameters()]
        if not params:
            raise RuntimeError("Model contains no parameters.")
        return torch.cat(params)


def get_attack(
    attack_name: str,
    clients: List,
    config: Dict[str, Any],
    runner_instance,
):
    """获取指定的攻击方法"""
    attacks = {
        'none': NoAttack,
        'random': RandomAttack,
        'gaussian': GaussianAttack,
        'lie': LIEAttack,
        'qlearning': QLearningAttack,
        'poisonedfl': PoisonedFLAttack,
        'fang': FangAttack,
        'minmax': MinMaxAttack,
        'minsum': MinSumAttack,
        'fedsa': FedSAAttack,
        'second': SecondHighestConfidenceAttack,
    }

    attack_name = attack_name.lower()
    if attack_name not in attacks:
        raise ValueError(f"未知的攻击方法: {attack_name}")

    return attacks[attack_name](clients, config, runner_instance)
