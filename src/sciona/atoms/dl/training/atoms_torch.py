"""GPU port of training strategy atoms using PyTorch.

Provides torch.Tensor implementations of the numpy training atoms for use in
GPU-accelerated training pipelines. These are NOT registered atoms; they are
port implementations referenced by the artifact_ports system.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def online_hard_negative_mining_torch(
    neg_scores: "torch.Tensor",
    num_hard: int,
) -> "torch.Tensor":
    """GPU-accelerated hard negative mining via torch.topk (O(n log k)).

    Same as online_hard_negative_mining but operates on torch tensors,
    using torch.topk for efficient partial sorting on the GPU instead of
    full argsort.
    """
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")
    k = min(num_hard, len(neg_scores))
    _, indices = torch.topk(neg_scores, k)
    return indices


def softmax_temperature_proposal_sampling_torch(
    scores: "torch.Tensor",
    k: int,
    temperature: float = 1.0,
) -> "torch.Tensor":
    """GPU-accelerated temperature-scaled proposal sampling.

    Same as softmax_temperature_proposal_sampling but operates on torch
    tensors, using torch.multinomial for GPU-resident sampling without
    CPU transfer.
    """
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")
    probs = F.softmax(scores / temperature, dim=0)
    return torch.multinomial(probs, k, replacement=False)
