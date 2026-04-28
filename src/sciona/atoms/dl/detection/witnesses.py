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


def witness_iou_matrix(
    boxes_a: AbstractArray,
    boxes_b: AbstractArray,
) -> AbstractArray:
    """Mirror pairwise IoU matrix construction for two box sets."""
    return AbstractArray()


def witness_giou_matrix(
    boxes_a: AbstractArray,
    boxes_b: AbstractArray,
) -> AbstractArray:
    """Mirror pairwise generalized IoU matrix construction."""
    return AbstractArray()


def witness_nms(
    boxes: AbstractArray,
    scores: AbstractArray,
    iou_threshold: float,
) -> AbstractArray:
    """Mirror greedy NMS as retained integer indices."""
    return AbstractArray()


def witness_soft_nms(
    boxes: AbstractArray,
    scores: AbstractArray,
    iou_threshold: float,
    sigma: float = 0.5,
    method: str = "linear",
    score_threshold: float = 0.001,
) -> tuple[AbstractArray, AbstractArray]:
    """Mirror Soft-NMS as filtered boxes and decayed scores."""
    return AbstractArray(), AbstractArray()


def witness_wbf(
    boxes_list: list[AbstractArray],
    scores_list: list[AbstractArray],
    labels_list: list[AbstractArray],
    weights: list[float],
    iou_threshold: float = 0.55,
    skip_box_thr: float = 0.0,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Mirror weighted box fusion as boxes, scores, and labels."""
    return AbstractArray(), AbstractArray(), AbstractArray()


def witness_wbf_1d(
    spans_list: list[AbstractArray],
    scores_list: list[AbstractArray],
    labels_list: list[AbstractArray],
    weights: list[float],
    iou_threshold: float = 0.55,
    skip_span_thr: float = 0.0,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Mirror one-dimensional weighted span fusion."""
    return AbstractArray(), AbstractArray(), AbstractArray()


def witness_generate_anchors(
    feature_map_size: tuple[int, int],
    stride: int,
    sizes: tuple[float, ...],
    aspect_ratios: tuple[float, ...],
) -> AbstractArray:
    """Mirror grid anchor generation as an anchor matrix."""
    return AbstractArray()


def witness_encode_boxes(
    anchors: AbstractArray,
    gt_boxes: AbstractArray,
    scales: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> AbstractArray:
    """Mirror box delta encoding relative to anchors."""
    return AbstractArray()


def witness_decode_boxes(
    anchors: AbstractArray,
    deltas: AbstractArray,
    scales: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> AbstractArray:
    """Mirror box delta decoding back to boxes."""
    return AbstractArray()


def witness_nms_1d(
    signal: AbstractArray,
    min_distance: int,
    threshold: float,
) -> AbstractArray:
    """Mirror one-dimensional peak suppression as retained indices."""
    return AbstractArray()


def witness_masks_to_boxes(binary_masks: AbstractArray) -> AbstractArray:
    """Mirror binary mask projection into bounding boxes."""
    return AbstractArray()


def witness_associate_boxes(
    boxes_a: AbstractArray,
    boxes_b: AbstractArray,
    iou_threshold: float,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Mirror IoU-based bipartite association index outputs."""
    return AbstractArray(), AbstractArray(), AbstractArray(), AbstractArray()


def witness_threshold_detections(
    boxes: AbstractArray,
    scores: AbstractArray,
    threshold: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Mirror confidence filtering as retained boxes and scores."""
    return AbstractArray(), AbstractArray()
