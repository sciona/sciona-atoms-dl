"""Ghost witnesses for detection atoms.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray


def witness_lung_mask_with_bone_removal(
    ct_volume: AbstractArray,
    spacing: AbstractArray,
    intensity_th: float = -600.0,
    pad_value: float = 170.0,
) -> AbstractArray:
    """Ghost witness for lung mask with bone removal.

    Takes a 3D CT volume and spacing array, returns a float64 mask of the
    same shape as the input volume with 1.0 for lung voxels and 0.0 elsewhere.
    """
    return ct_volume


def witness_anchor_label_mapping_with_iou_dilation(
    input_size: tuple[int, int, int],
    target: AbstractArray,
    anchors: AbstractArray,
    stride: int,
    pos_th: float = 0.5,
    neg_th: float = 0.02,
    dilation_iterations: int = 1,
) -> AbstractArray:
    """Ghost witness for anchor label mapping with IoU dilation.

    Takes detection grid parameters and targets, returns a 5D label array
    of shape (D//stride, H//stride, W//stride, num_anchors, 5) where
    channel 0 is the class label and channels 1-4 are regression targets.
    """
    return AbstractArray()


def witness_center_feature_extraction_3d(
    feature_map: AbstractArray,
) -> AbstractArray:
    """Ghost witness for center feature extraction 3D.

    Takes a 5D feature map (N, C, D, H, W) and returns a 2D array (N, C)
    by max-pooling the center 2x2x2 spatial region.
    """
    return AbstractArray()
