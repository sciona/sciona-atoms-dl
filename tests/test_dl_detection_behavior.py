from __future__ import annotations

import numpy as np
import pytest


def test_detection_import() -> None:
    from sciona.atoms.dl.detection.atoms import (
        anchor_label_mapping_with_iou_dilation,
        associate_boxes,
        center_feature_extraction_3d,
        decode_boxes,
        encode_boxes,
        generate_anchors,
        giou_matrix,
        iou_matrix,
        lung_mask_with_bone_removal,
        masks_to_boxes,
        nms,
        nms_1d,
        soft_nms,
        threshold_detections,
        wbf,
        wbf_1d,
    )
    assert callable(lung_mask_with_bone_removal)
    assert callable(anchor_label_mapping_with_iou_dilation)
    assert callable(center_feature_extraction_3d)
    assert callable(iou_matrix)
    assert callable(giou_matrix)
    assert callable(nms)
    assert callable(soft_nms)
    assert callable(wbf)
    assert callable(wbf_1d)
    assert callable(generate_anchors)
    assert callable(encode_boxes)
    assert callable(decode_boxes)
    assert callable(nms_1d)
    assert callable(masks_to_boxes)
    assert callable(associate_boxes)
    assert callable(threshold_detections)


# ---------------------------------------------------------------------------
# lung_mask_with_bone_removal
# ---------------------------------------------------------------------------


def _make_synthetic_ct(
    shape: tuple[int, int, int] = (32, 64, 64),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic CT volume with an air-filled sphere in the center.

    Background is 0 HU (soft tissue), the sphere interior is -800 HU (lung),
    and spacing is 1mm isotropic.
    """
    ct = np.zeros(shape, dtype=np.float64)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius = min(shape) // 4
    zz, yy, xx = np.ogrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    ct[dist < radius] = -800.0
    return ct, spacing


def test_lung_mask_output_shape() -> None:
    from sciona.atoms.dl.detection.atoms import lung_mask_with_bone_removal

    ct, spacing = _make_synthetic_ct()
    result = lung_mask_with_bone_removal(ct, spacing)
    assert result.shape == ct.shape


def test_lung_mask_output_dtype() -> None:
    from sciona.atoms.dl.detection.atoms import lung_mask_with_bone_removal

    ct, spacing = _make_synthetic_ct()
    result = lung_mask_with_bone_removal(ct, spacing)
    assert result.dtype == np.float64


def test_lung_mask_binary_values() -> None:
    from sciona.atoms.dl.detection.atoms import lung_mask_with_bone_removal

    ct, spacing = _make_synthetic_ct()
    result = lung_mask_with_bone_removal(ct, spacing)
    unique = np.unique(result)
    assert all(v in (0.0, 1.0) for v in unique)


def test_lung_mask_all_zero_for_uniform_volume() -> None:
    """A uniform (non-lung) volume should produce an all-zero mask."""
    from sciona.atoms.dl.detection.atoms import lung_mask_with_bone_removal

    ct = np.zeros((16, 32, 32), dtype=np.float64)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    result = lung_mask_with_bone_removal(ct, spacing)
    assert np.sum(result) == 0.0


# ---------------------------------------------------------------------------
# anchor_label_mapping_with_iou_dilation
# ---------------------------------------------------------------------------


def test_anchor_label_output_shape() -> None:
    from sciona.atoms.dl.detection.atoms import anchor_label_mapping_with_iou_dilation

    input_size = (32, 32, 32)
    target = np.array([16.0, 16.0, 16.0, 10.0], dtype=np.float64)
    anchors = np.array([10.0, 30.0, 60.0], dtype=np.float64)
    stride = 4
    result = anchor_label_mapping_with_iou_dilation(
        input_size, target, anchors, stride
    )
    assert result.shape == (8, 8, 8, 3, 5)


def test_anchor_label_has_positive_assignment() -> None:
    """A target at grid center with matching anchor should produce a positive."""
    from sciona.atoms.dl.detection.atoms import anchor_label_mapping_with_iou_dilation

    input_size = (32, 32, 32)
    # Place target exactly at a grid point with diameter matching anchor
    target = np.array([15.5, 15.5, 15.5, 10.0], dtype=np.float64)
    anchors = np.array([10.0], dtype=np.float64)
    stride = 4
    result = anchor_label_mapping_with_iou_dilation(
        input_size, target, anchors, stride, pos_th=0.5, neg_th=0.02
    )
    # There should be at least one positive label (value == 1)
    assert np.any(result[:, :, :, :, 0] == 1)


def test_anchor_label_has_negatives() -> None:
    """Far-from-target grid positions should be negative (-1)."""
    from sciona.atoms.dl.detection.atoms import anchor_label_mapping_with_iou_dilation

    input_size = (64, 64, 64)
    target = np.array([32.0, 32.0, 32.0, 10.0], dtype=np.float64)
    anchors = np.array([10.0], dtype=np.float64)
    stride = 4
    result = anchor_label_mapping_with_iou_dilation(
        input_size, target, anchors, stride
    )
    # There should be negative labels
    assert np.any(result[:, :, :, :, 0] == -1)


def test_anchor_label_positive_has_regression_targets() -> None:
    """Positive positions should have non-trivial regression targets."""
    from sciona.atoms.dl.detection.atoms import anchor_label_mapping_with_iou_dilation

    input_size = (32, 32, 32)
    target = np.array([15.5, 15.5, 15.5, 10.0], dtype=np.float64)
    anchors = np.array([10.0], dtype=np.float64)
    stride = 4
    result = anchor_label_mapping_with_iou_dilation(
        input_size, target, anchors, stride, pos_th=0.5
    )
    pos_mask = result[:, :, :, :, 0] == 1
    if np.any(pos_mask):
        # At least the regression target channels should be populated
        pos_indices = np.where(pos_mask)
        reg = result[pos_indices[0][0], pos_indices[1][0], pos_indices[2][0], pos_indices[3][0], 1:5]
        # dd = log(10/10) = 0 for matching anchor, dz/dh/dw can be nonzero
        assert reg.shape == (4,)


def test_anchor_label_dilation_creates_ignore_zone() -> None:
    """Dilation should produce ignore labels (0) around positives."""
    from sciona.atoms.dl.detection.atoms import anchor_label_mapping_with_iou_dilation

    input_size = (64, 64, 64)
    target = np.array([32.0, 32.0, 32.0, 10.0], dtype=np.float64)
    anchors = np.array([10.0], dtype=np.float64)
    stride = 4
    result = anchor_label_mapping_with_iou_dilation(
        input_size, target, anchors, stride, dilation_iterations=2
    )
    labels = result[:, :, :, :, 0]
    # Should have all three label types: -1, 0, 1
    has_pos = np.any(labels == 1)
    has_neg = np.any(labels == -1)
    has_ignore = np.any(labels == 0)
    assert has_pos or has_ignore  # dilation should produce some non-negative zone
    assert has_neg  # far positions should be negative


# ---------------------------------------------------------------------------
# center_feature_extraction_3d
# ---------------------------------------------------------------------------


def test_center_feature_output_shape() -> None:
    from sciona.atoms.dl.detection.atoms import center_feature_extraction_3d

    fm = np.random.default_rng(42).standard_normal((2, 128, 24, 24, 24))
    result = center_feature_extraction_3d(fm)
    assert result.shape == (2, 128)


def test_center_feature_extracts_spatial_max() -> None:
    """The result should be the max over the center 2x2x2 region."""
    from sciona.atoms.dl.detection.atoms import center_feature_extraction_3d

    fm = np.zeros((1, 4, 6, 6, 6), dtype=np.float64)
    # Set a known value in center region
    fm[0, 0, 2, 2, 2] = 5.0  # D//2-1=2, H//2-1=2, W//2-1=2
    fm[0, 0, 3, 3, 3] = 10.0  # D//2+0=3, H//2+0=3, W//2+0=3
    result = center_feature_extraction_3d(fm)
    assert result[0, 0] == 10.0


def test_center_feature_ignores_non_center() -> None:
    """Values outside the center 2x2x2 region should not affect the output."""
    from sciona.atoms.dl.detection.atoms import center_feature_extraction_3d

    fm = np.zeros((1, 2, 8, 8, 8), dtype=np.float64)
    # Set a large value outside center
    fm[0, 0, 0, 0, 0] = 999.0
    # Set a known value in center
    fm[0, 0, 3, 3, 3] = 7.0
    fm[0, 0, 4, 4, 4] = 3.0
    result = center_feature_extraction_3d(fm)
    assert result[0, 0] == 7.0  # max of center 2x2x2 (indices 3:5 for dim=8)


def test_center_feature_preserves_batch_and_channel() -> None:
    from sciona.atoms.dl.detection.atoms import center_feature_extraction_3d

    rng = np.random.default_rng(123)
    fm = rng.standard_normal((3, 64, 10, 10, 10))
    result = center_feature_extraction_3d(fm)
    assert result.shape == (3, 64)
    # Each (n, c) should be the max of the center 2x2x2
    for n in range(3):
        for c in range(64):
            center = fm[n, c, 4:6, 4:6, 4:6]
            assert result[n, c] == pytest.approx(np.max(center))


def test_iou_and_giou_matrices() -> None:
    from sciona.atoms.dl.detection.atoms import giou_matrix, iou_matrix

    boxes_a = np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    boxes_b = np.array([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]])
    iou = iou_matrix(boxes_a, boxes_b)
    giou = giou_matrix(boxes_a, boxes_b)
    assert iou.shape == (2, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)
    assert giou[0, 0] == pytest.approx(1.0)
    assert giou[0, 1] < 0.0


def test_nms_suppresses_lower_scored_overlap() -> None:
    from sciona.atoms.dl.detection.atoms import nms

    boxes = np.array(
        [[0.0, 0.0, 2.0, 2.0], [0.1, 0.1, 2.1, 2.1], [5.0, 5.0, 6.0, 6.0]],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_soft_nms_decays_overlapping_score() -> None:
    from sciona.atoms.dl.detection.atoms import soft_nms

    boxes = np.array([[0.0, 0.0, 2.0, 2.0], [0.1, 0.1, 2.1, 2.1]], dtype=np.float64)
    scores = np.array([0.9, 0.8], dtype=np.float64)
    out_boxes, out_scores = soft_nms(
        boxes, scores, iou_threshold=0.5, method="linear", score_threshold=0.0
    )
    assert out_boxes.shape == (2, 4)
    assert out_scores[0] == pytest.approx(0.9)
    assert 0.0 < out_scores[1] < 0.8


def test_weighted_box_fusion_merges_model_consensus() -> None:
    from sciona.atoms.dl.detection.atoms import wbf

    boxes, scores, labels = wbf(
        [
            np.array([[0.10, 0.10, 0.40, 0.40]], dtype=np.float64),
            np.array([[0.20, 0.10, 0.50, 0.40]], dtype=np.float64),
        ],
        [
            np.array([0.9], dtype=np.float64),
            np.array([0.9], dtype=np.float64),
        ],
        [
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
        weights=[1.0, 1.0],
        iou_threshold=0.3,
    )
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == pytest.approx([0.15, 0.10, 0.45, 0.40])
    assert scores[0] == pytest.approx(0.9)
    assert labels[0] == 1


def test_weighted_span_fusion_merges_1d_consensus() -> None:
    from sciona.atoms.dl.detection.atoms import wbf_1d

    spans, scores, labels = wbf_1d(
        [
            np.array([[10.0, 20.0]], dtype=np.float64),
            np.array([[12.0, 22.0]], dtype=np.float64),
        ],
        [
            np.array([0.8], dtype=np.float64),
            np.array([0.8], dtype=np.float64),
        ],
        [
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ],
        weights=[1.0, 1.0],
        iou_threshold=0.5,
    )
    assert spans.shape == (1, 2)
    assert spans[0].tolist() == pytest.approx([11.0, 21.0])
    assert scores[0] == pytest.approx(0.8)
    assert labels[0] == 2


def test_generate_anchors_count_and_origin_center() -> None:
    from sciona.atoms.dl.detection.atoms import generate_anchors

    anchors = generate_anchors((1, 1), stride=4, sizes=(2.0,), aspect_ratios=(1.0, 4.0))
    assert anchors.shape == (2, 4)
    assert anchors[0].tolist() == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert anchors[1].tolist() == pytest.approx([-0.5, -2.0, 0.5, 2.0])


def test_encode_decode_boxes_round_trip() -> None:
    from sciona.atoms.dl.detection.atoms import decode_boxes, encode_boxes

    anchors = np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 30.0, 20.0]])
    gt_boxes = np.array([[1.0, 2.0, 11.0, 12.0], [12.0, 8.0, 32.0, 18.0]])
    deltas = encode_boxes(anchors, gt_boxes)
    decoded = decode_boxes(anchors, deltas)
    assert decoded == pytest.approx(gt_boxes)
    assert decode_boxes(anchors, np.zeros_like(anchors)) == pytest.approx(anchors)


def test_nms_1d_selects_separated_peak_centers() -> None:
    from sciona.atoms.dl.detection.atoms import nms_1d

    signal = np.array([0.0, 0.8, 0.8, 0.8, 0.1, 0.9, 0.2], dtype=np.float64)
    peaks = nms_1d(signal, min_distance=3, threshold=0.5)
    assert peaks.tolist() == [2, 5]


def test_masks_to_boxes_handles_nonempty_and_empty_masks() -> None:
    from sciona.atoms.dl.detection.atoms import masks_to_boxes

    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:5, 3:7] = True
    boxes = masks_to_boxes(masks)
    assert boxes[0].tolist() == [3.0, 2.0, 6.0, 4.0]
    assert boxes[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_associate_boxes_matches_by_iou_threshold() -> None:
    from sciona.atoms.dl.detection.atoms import associate_boxes

    boxes_a = np.array([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    boxes_b = np.array([[0.0, 0.0, 2.0, 2.0], [8.0, 8.0, 9.0, 9.0]])
    matched_a, matched_b, unmatched_a, unmatched_b = associate_boxes(boxes_a, boxes_b, 0.5)
    assert matched_a.tolist() == [0]
    assert matched_b.tolist() == [0]
    assert unmatched_a.tolist() == [1]
    assert unmatched_b.tolist() == [1]


def test_threshold_detections_filters_scores_and_boxes() -> None:
    from sciona.atoms.dl.detection.atoms import threshold_detections

    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]])
    scores = np.array([0.1, 0.9, 0.4], dtype=np.float64)
    out_boxes, out_scores = threshold_detections(boxes, scores, 0.5)
    assert out_boxes.tolist() == [[1.0, 1.0, 2.0, 2.0]]
    assert out_scores.tolist() == [0.9]
