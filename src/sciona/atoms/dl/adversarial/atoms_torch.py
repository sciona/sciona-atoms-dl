"""GPU port of adversarial attack atoms using PyTorch.

Provides differentiable torch.Tensor implementations of the numpy adversarial
atoms for use in autograd-enabled attack pipelines. These are NOT registered
atoms; they are port implementations referenced by the artifact_ports system.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def auxiliary_logit_loss_fusion_torch(
    main_logits: "torch.Tensor",
    labels_onehot: "torch.Tensor",
    aux_logits: "torch.Tensor | None" = None,
    aux_weight: float = 0.4,
) -> "torch.Tensor":
    """Differentiable auxiliary logit loss for adversarial attacks.

    Same as auxiliary_logit_loss_fusion but operates on torch tensors with
    autograd support. Computes softmax cross-entropy on main logits and
    optionally adds a weighted auxiliary branch (e.g. Inception AuxLogits).
    """
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")
    log_probs = F.log_softmax(main_logits, dim=-1)
    loss = -(labels_onehot * log_probs).sum(dim=-1).mean()
    if aux_logits is not None:
        aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        loss = loss + aux_weight * (-(labels_onehot * aux_log_probs).sum(dim=-1).mean())
    return loss
