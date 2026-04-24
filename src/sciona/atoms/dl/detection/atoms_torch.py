"""GPU port of 3D detection atoms using PyTorch.

Provides torch.Tensor implementations of the numpy detection atoms for use in
GPU-resident inference and training pipelines. These are NOT registered atoms;
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


def center_feature_extraction_3d_torch(
    feature_map: "torch.Tensor",
) -> "torch.Tensor":
    """GPU-resident center feature extraction avoiding CPU transfer.

    Same as center_feature_extraction_3d but operates on torch tensors,
    using F.max_pool3d for GPU-accelerated spatial reduction of the
    center 2x2x2 cube from a 5D feature map to (N, C).
    """
    if not HAS_TORCH:
        raise ImportError("torch required for GPU port")
    N, C, D, H, W = feature_map.shape
    center = feature_map[:, :, D // 2 - 1 : D // 2 + 1, H // 2 - 1 : H // 2 + 1, W // 2 - 1 : W // 2 + 1]
    return F.max_pool3d(center, kernel_size=center.shape[2:]).squeeze(-1).squeeze(-1).squeeze(-1)
