"""GPU port of loss function atoms using PyTorch.

Provides differentiable torch.Tensor implementations of the numpy loss atoms
for use in autograd-enabled training pipelines. These are NOT registered atoms;
they are port implementations referenced by the artifact_ports system.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def miss_penalty_loss_torch(
    predictions: "torch.Tensor",
    labels: "torch.Tensor",
    threshold: float = 0.03,
) -> "torch.Tensor":
    """Differentiable miss penalty loss for autograd.

    Same as miss_penalty_loss but operates on torch tensors and is
    differentiable -- gradients flow back through predictions.

    Penalizes cases where the model predicts below threshold on positive
    examples (labels > 0.5). The loss is the negative log-likelihood of
    the predicted probability for these missed positives.
    """
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")
    mask = (labels > 0.5) & (predictions < threshold)
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    return -torch.log(predictions[mask] + 1e-8).sum()
