"""3D detection atoms from the DSB2017 1st-place solution.

Implements core building blocks for pulmonary nodule detection: CT lung
segmentation with bone removal, anchor-based label mapping with IoU-guided
dilation, and center feature extraction for classification.

All computation is pure numpy/scipy -- no PyTorch dependency. The U-Net
detector is represented as an opaque conceptual node in the CDG only.

Source: DSB2017 1st place (MIT license)
        preprocessing/step1.py, data_detector.py, net_classifier.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    map_coordinates,
)
from scipy.optimize import linear_sum_assignment
from skimage import measure

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_anchor_label_mapping_with_iou_dilation,
    witness_associate_boxes,
    witness_center_feature_extraction_3d,
    witness_margin_expanded_face_crop,
    witness_decode_boxes,
    witness_encode_boxes,
    witness_face_similarity_align,
    witness_generate_anchors,
    witness_giou_matrix,
    witness_iou_matrix,
    witness_lung_mask_with_bone_removal,
    witness_masks_to_boxes,
    witness_nms,
    witness_nms_1d,
    witness_soft_nms,
    witness_threshold_detections,
    witness_wbf,
    witness_wbf_1d,
)

_BOX_DELTA_CLIP = float(np.log(1000.0 / 16.0))


def _valid_xyxy_boxes(boxes: NDArray[np.float64]) -> bool:
    return bool(
        boxes.ndim == 2
        and boxes.shape[1] == 4
        and np.all(np.isfinite(boxes))
        and np.all(boxes[:, 0] <= boxes[:, 2])
        and np.all(boxes[:, 1] <= boxes[:, 3])
    )


def _positive_area_xyxy_boxes(boxes: NDArray[np.float64]) -> bool:
    return bool(_valid_xyxy_boxes(boxes) and np.all(boxes[:, 0] < boxes[:, 2]) and np.all(boxes[:, 1] < boxes[:, 3]))


def _valid_normalized_xyxy_boxes(boxes: NDArray[np.float64]) -> bool:
    return bool(_valid_xyxy_boxes(boxes) and np.all((boxes >= 0.0) & (boxes <= 1.0)))


def _valid_spans(spans: NDArray[np.float64]) -> bool:
    return bool(
        spans.ndim == 2
        and spans.shape[1] == 2
        and np.all(np.isfinite(spans))
        and np.all(spans[:, 0] <= spans[:, 1])
    )


def _valid_image_array(image: NDArray[np.float64]) -> bool:
    return bool(
        image.ndim in {2, 3}
        and image.shape[0] > 0
        and image.shape[1] > 0
        and np.issubdtype(image.dtype, np.number)
        and np.all(np.isfinite(image))
    )


def _valid_landmarks(landmarks: NDArray[np.float64]) -> bool:
    return bool(landmarks.shape == (5, 2) and np.all(np.isfinite(landmarks)))


def _nondegenerate_landmarks(landmarks: NDArray[np.float64]) -> bool:
    if not _valid_landmarks(landmarks):
        return False
    centered = landmarks - np.mean(landmarks, axis=0)
    return bool(np.linalg.matrix_rank(centered) == 2 and np.sum(centered**2) > 0.0)


def _positive_output_size(output_size: tuple[int, int]) -> bool:
    return bool(len(output_size) == 2 and output_size[0] > 0 and output_size[1] > 0)


def _crop_bounds(
    image_shape: tuple[int, ...],
    bbox: NDArray[np.float64],
    margin: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = float(margin) * box_width
    pad_y = float(margin) * box_height
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return left, top, right, bottom


def _crop_has_area(image: NDArray[np.float64], bbox: NDArray[np.float64], margin: float) -> bool:
    if not (_valid_xyxy_boxes(bbox.reshape(1, 4)) and np.isfinite(margin) and margin >= 0.0):
        return False
    left, top, right, bottom = _crop_bounds(image.shape, bbox, margin)
    return bool(right > left and bottom > top)


def _estimate_similarity_transform(
    src_landmarks: NDArray[np.float64],
    dst_landmarks: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    src = np.asarray(src_landmarks, dtype=np.float64)
    dst = np.asarray(dst_landmarks, dtype=np.float64)
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = u_matrix @ vt_matrix
    scale = float(np.sum(singular_values) / np.sum(src_centered**2))
    translation = dst_mean - scale * (src_mean @ rotation)
    return scale, rotation, translation


def _sample_similarity_aligned(
    image: NDArray[np.float64],
    scale: float,
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    output_size: tuple[int, int],
    order: int,
) -> NDArray[np.float64]:
    out_height, out_width = output_size
    rows, cols = np.indices((out_height, out_width), dtype=np.float64)
    dst_points = np.stack([cols.ravel(), rows.ravel()], axis=1)
    src_points = ((dst_points - translation) @ rotation.T) / scale
    coords = [src_points[:, 1].reshape(output_size), src_points[:, 0].reshape(output_size)]
    if image.ndim == 2:
        return map_coordinates(image, coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
    channels = [
        map_coordinates(image[:, :, channel], coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=2)


def _box_areas(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return widths * heights


def _pairwise_iou(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = _box_areas(boxes_a)[:, None] + _box_areas(boxes_b)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def _span_iou(span: NDArray[np.float64], spans: NDArray[np.float64]) -> NDArray[np.float64]:
    starts = np.maximum(span[0], spans[:, 0])
    ends = np.minimum(span[1], spans[:, 1])
    intersection = np.maximum(ends - starts, 0.0)
    union = (span[1] - span[0]) + (spans[:, 1] - spans[:, 0]) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def _lists_aligned(
    boxes_list: list[NDArray[np.float64]],
    scores_list: list[NDArray[np.float64]],
    labels_list: list[NDArray[np.int64]],
) -> bool:
    if not (len(boxes_list) == len(scores_list) == len(labels_list)):
        return False
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        if not (
            _valid_normalized_xyxy_boxes(boxes)
            and scores.shape == (boxes.shape[0],)
            and labels.shape == (boxes.shape[0],)
            and np.all(np.isfinite(scores))
            and np.all((scores >= 0.0) & (scores <= 1.0))
        ):
            return False
    return True


def _span_lists_aligned(
    spans_list: list[NDArray[np.float64]],
    scores_list: list[NDArray[np.float64]],
    labels_list: list[NDArray[np.int64]],
) -> bool:
    if not (len(spans_list) == len(scores_list) == len(labels_list)):
        return False
    for spans, scores, labels in zip(spans_list, scores_list, labels_list):
        if not (
            _valid_spans(spans)
            and scores.shape == (spans.shape[0],)
            and labels.shape == (spans.shape[0],)
            and np.all(np.isfinite(scores))
            and np.all((scores >= 0.0) & (scores <= 1.0))
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Private helpers for lung_mask_with_bone_removal
# ---------------------------------------------------------------------------


def _binarize_per_slice(
    image: NDArray[np.float64],
    spacing: NDArray[np.float64],
    intensity_th: float = -600.0,
    sigma: float = 1.0,
    area_th: float = 30.0,
    eccen_th: float = 0.99,
    bg_patch_size: int = 10,
) -> NDArray[np.bool_]:
    """Gaussian filter each slice, threshold, and keep valid components.

    For slices with uniform corner regions (padded scans), a circular NaN mask
    is applied before filtering to avoid edge artifacts. Components are kept
    only if their physical area exceeds area_th mm^2 and eccentricity is below
    eccen_th (to reject elongated bone/table artifacts).

    Derived from step1.py binarize_per_slice (lines 52-79).
    """
    bw = np.zeros(image.shape, dtype=bool)
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2 + y**2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan

    for i in range(image.shape[0]):
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = (
                gaussian_filter(
                    np.multiply(image[i].astype("float32"), nan_mask),
                    sigma,
                    truncate=2.0,
                )
                < intensity_th
            )
        else:
            current_bw = (
                gaussian_filter(image[i].astype("float32"), sigma, truncate=2.0)
                < intensity_th
            )

        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if (
                prop.area * spacing[1] * spacing[2] > area_th
                and prop.eccentricity < eccen_th
            ):
                valid_label.add(prop.label)
        current_bw = np.isin(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def _all_slice_analysis(
    bw: NDArray[np.bool_],
    spacing: NDArray[np.float64],
    cut_num: int = 0,
    vol_limit: tuple[float, float] = (0.68, 8.2),
    area_th: float = 6e3,
    dist_th: float = 62.0,
) -> tuple[NDArray[np.bool_], int]:
    """3D connected component analysis: keep lung-like components.

    Removes background-touching components, filters by volume (in liters),
    and validates remaining components by their average distance to the
    center axis across slices. Optionally removes top slices (cut_num)
    to handle trachea/artifact leaks, then restores them via dilation overlap.

    Derived from step1.py all_slice_analysis (lines 81-141).
    """
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)

    mid = int(label.shape[2] / 2)
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1 - cut_num, 0, 0],
            label[-1 - cut_num, 0, -1],
            label[-1 - cut_num, -1, 0],
            label[-1 - cut_num, -1, -1],
            label[0, 0, mid],
            label[0, -1, mid],
            label[-1 - cut_num, 0, mid],
            label[-1 - cut_num, -1, mid],
        ]
    )
    for lbl in bg_label:
        label[label == lbl] = 0

    properties = measure.regionprops(label)
    for prop in properties:
        vol = prop.area * spacing.prod()
        if vol < vol_limit[0] * 1e6 or vol > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    x_axis = (
        np.linspace(
            -label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]
        )
        * spacing[1]
    )
    y_axis = (
        np.linspace(
            -label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]
        )
        * spacing[2]
    )
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2 + y**2) ** 0.5

    vols = measure.regionprops(label)
    valid_label = set()
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(
                single_vol[i] * d + (1 - single_vol[i]) * np.max(d)
            )

        large_slices = [
            i for i in range(label.shape[0]) if slice_area[i] > area_th
        ]
        if len(large_slices) > 0:
            avg_dist = np.average([min_distance[i] for i in large_slices])
            if avg_dist < dist_th:
                valid_label.add(vol.label)

    bw = np.isin(label, list(valid_label)).reshape(label.shape)

    if cut_num > 0:
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for lbl in l_list:
            indices = np.nonzero(label == lbl)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.isin(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def _fill_hole(bw: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fill 3D holes by inverting, labeling, and removing corner components.

    Derived from step1.py fill_hole (lines 143-151).
    """
    label = measure.label(~bw)
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1, 0, 0],
            label[-1, 0, -1],
            label[-1, -1, 0],
            label[-1, -1, -1],
        ]
    )
    bw = ~np.isin(label, list(bg_label)).reshape(label.shape)
    return bw


def _two_lung_only(
    bw: NDArray[np.bool_],
    spacing: NDArray[np.float64],
    max_iter: int = 22,
    max_ratio: float = 4.8,
) -> NDArray[np.bool_]:
    """Separate left/right lungs via iterative erosion + distance transform.

    If two comparably-sized components are found (ratio < max_ratio), assigns
    each voxel in the original mask to the nearest component using distance
    transforms, then extracts the main connected component per lung and fills
    2D holes. If separation fails after max_iter erosions, the entire mask is
    treated as a single lung.

    Derived from step1.py two_lung_only (lines 156-226).
    """

    def _extract_main(bw_in: NDArray[np.bool_], cover: float = 0.95) -> NDArray[np.bool_]:
        bw_work = np.copy(bw_in)
        for i in range(bw_work.shape[0]):
            current_slice = bw_work[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            if len(properties) == 0:
                continue
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            area_sum = 0
            total = np.sum(area)
            while area_sum < total * cover and count < len(area):
                area_sum += area[count]
                count += 1
            filt = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filt[bb[0] : bb[2], bb[1] : bb[3]] = (
                    filt[bb[0] : bb[2], bb[1] : bb[3]] | properties[j].convex_image
                )
            bw_work[i] = bw_work[i] & filt

        label = measure.label(bw_work)
        properties = measure.regionprops(label)
        if len(properties) == 0:
            return bw_work
        properties.sort(key=lambda x: x.area, reverse=True)
        bw_work = label == properties[0].label
        return bw_work

    def _fill_2d_hole(bw_in: NDArray[np.bool_]) -> NDArray[np.bool_]:
        bw_work = np.copy(bw_in)
        for i in range(bw_work.shape[0]):
            current_slice = bw_work[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0] : bb[2], bb[1] : bb[3]] = (
                    current_slice[bb[0] : bb[2], bb[1] : bb[3]] | prop.filled_image
                )
            bw_work[i] = current_slice
        return bw_work

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if (
            len(properties) > 1
            and properties[0].area / properties[1].area < max_ratio
        ):
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = binary_erosion(bw)
            iter_count += 1

    if found_flag:
        d1 = distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
        bw1 = _extract_main(bw1)
        bw2 = _extract_main(bw2)
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape, dtype=bool)

    bw1 = _fill_2d_hole(bw1)
    bw2 = _fill_2d_hole(bw2)
    bw = bw1 | bw2
    return bw


# ---------------------------------------------------------------------------
# Atom 1: Lung mask with bone removal
# ---------------------------------------------------------------------------


@register_atom(witness_lung_mask_with_bone_removal)
@icontract.require(lambda ct_volume: ct_volume.ndim == 3, "ct_volume must be 3D")
@icontract.require(
    lambda spacing: spacing.shape == (3,), "spacing must have shape (3,)"
)
@icontract.ensure(
    lambda ct_volume, result: result.shape == ct_volume.shape,
    "output mask must match input volume shape",
)
def lung_mask_with_bone_removal(
    ct_volume: NDArray[np.float64],
    spacing: NDArray[np.float64],
    intensity_th: float = -600.0,
    pad_value: float = 170.0,
) -> NDArray[np.float64]:
    """Segment lungs from a CT volume, removing bone and table artifacts.

    Pipeline:
    1. Binarize each 2D slice via Gaussian smoothing + intensity thresholding,
       filtering components by area and eccentricity.
    2. 3D connected component analysis to keep lung-sized volumes near the
       center axis, with iterative top-slice removal for trachea leaks.
    3. Fill 3D holes (e.g., vessels enclosed by lung parenchyma).
    4. Separate left/right lungs via iterative erosion and distance transform
       assignment, handling merged-lung and single-lung edge cases.

    The output is a float64 mask (0.0 or 1.0) with the same shape as the
    input volume.

    Derived from step1.py step1_python (lines 228-243), composing
    binarize_per_slice, all_slice_analysis, fill_hole, two_lung_only.
    """
    bw = _binarize_per_slice(ct_volume, spacing, intensity_th=intensity_th)

    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = _all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=(0.68, 7.5))
        cut_num += cut_step

    bw = _fill_hole(bw)
    bw = _two_lung_only(bw, spacing)

    return bw.astype(np.float64)


# ---------------------------------------------------------------------------
# Private helper for anchor_label_mapping_with_iou_dilation
# ---------------------------------------------------------------------------


def _select_samples(
    bbox: NDArray[np.float64],
    anchor: float,
    th: float,
    oz: NDArray[np.float64],
    oh: NDArray[np.float64],
    ow: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Select grid positions where 3D IoU with bbox exceeds threshold.

    For each anchor at each grid position, computes the 3D IoU between the
    anchor cube and the target sphere (approximated as a cube). Returns
    indices into oz/oh/ow where IoU >= th.

    Derived from data_detector.py select_samples (lines 342-405).
    """
    z, h, w, d = bbox[0], bbox[1], bbox[2], bbox[3]
    max_overlap = min(d, anchor)
    min_overlap = max(d, anchor) ** 3 * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return (
            np.zeros((0,), np.int64),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.int64),
        )

    s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
    e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
    mz = np.logical_and(oz >= s, oz <= e)
    iz = np.where(mz)[0]

    s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
    e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
    mh = np.logical_and(oh >= s, oh <= e)
    ih = np.where(mh)[0]

    s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
    e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
    mw = np.logical_and(ow >= s, ow <= e)
    iw = np.where(mw)[0]

    if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
        return (
            np.zeros((0,), np.int64),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.int64),
        )

    lz, lh, lw = len(iz), len(ih), len(iw)
    iz = iz.reshape((-1, 1, 1))
    ih = ih.reshape((1, -1, 1))
    iw = iw.reshape((1, 1, -1))
    iz = np.tile(iz, (1, lh, lw)).reshape((-1))
    ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
    iw = np.tile(iw, (lz, lh, 1)).reshape((-1))

    centers = np.concatenate(
        [oz[iz].reshape((-1, 1)), oh[ih].reshape((-1, 1)), ow[iw].reshape((-1, 1))],
        axis=1,
    )

    r0 = anchor / 2
    s0 = centers - r0
    e0 = centers + r0

    r1 = d / 2
    s1 = (bbox[:3] - r1).reshape((1, -1))
    e1 = (bbox[:3] + r1).reshape((1, -1))

    overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
    intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
    union = anchor**3 + d**3 - intersection
    iou = intersection / union

    mask = iou >= th
    iz = iz[mask]
    ih = ih[mask]
    iw = iw[mask]
    return iz, ih, iw


# ---------------------------------------------------------------------------
# Atom 2: Anchor label mapping with IoU dilation
# ---------------------------------------------------------------------------


@register_atom(witness_anchor_label_mapping_with_iou_dilation)
@icontract.require(lambda stride: stride > 0, "stride must be positive")
@icontract.require(
    lambda anchors: len(anchors) > 0 and np.all(anchors > 0),
    "all anchors must be positive",
)
@icontract.require(
    lambda input_size: len(input_size) == 3, "input_size must be a 3-tuple (D, H, W)"
)
@icontract.ensure(
    lambda result: result.ndim == 5 and result.shape[-1] == 5,
    "output must be 5D with last dimension 5",
)
def anchor_label_mapping_with_iou_dilation(
    input_size: tuple[int, int, int],
    target: NDArray[np.float64],
    anchors: NDArray[np.float64],
    stride: int,
    pos_th: float = 0.5,
    neg_th: float = 0.02,
    dilation_iterations: int = 1,
) -> NDArray[np.float64]:
    """Map ground-truth targets to anchor-grid labels via 3D IoU.

    Creates a label volume at stride resolution where each position and anchor
    is assigned: 1 (positive, IoU >= pos_th), -1 (negative, IoU < neg_th and
    not in dilation zone), or 0 (ignore, in the dilated boundary region).

    For positive anchors, regression targets encode the offset of the target
    center from the anchor center (normalized by anchor size) and the log-ratio
    of target diameter to anchor size.

    The dilation step expands positive regions by dilation_iterations using a
    3-connectivity structure, creating an ignore buffer that prevents hard
    negatives near true positives from destabilizing training.

    Derived from data_detector.py LabelMapping.__call__ (lines 276-340)
    and select_samples (lines 342-405).
    """
    struct = generate_binary_structure(3, 1)

    output_size = []
    for i in range(3):
        output_size.append(int(input_size[i] // stride))

    label = np.zeros(output_size + [len(anchors), 5], np.float32)
    offset = (float(stride) - 1) / 2
    oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

    # Phase 1: mark positive candidates per bounding box and dilate
    if target.ndim == 1:
        bboxes = target.reshape(1, -1)
    else:
        bboxes = target

    for bbox in bboxes:
        if np.isnan(bbox[0]):
            continue
        for i, anchor in enumerate(anchors):
            iz, ih, iw = _select_samples(bbox, anchor, neg_th, oz, oh, ow)
            label[iz, ih, iw, i, 0] = 1
            label[:, :, :, i, 0] = binary_dilation(
                label[:, :, :, i, 0].astype("bool"),
                structure=struct,
                iterations=dilation_iterations,
            ).astype("float32")

    # Convert: 1 -> 0 (ignore zone from dilation), original 0 -> -1 (negative)
    label[:, :, :, :, 0] = label[:, :, :, :, 0] - 1

    # Phase 2: assign exact positive labels and regression targets for first target
    first_target = bboxes[0] if len(bboxes) > 0 else target
    if not np.isnan(first_target[0]):
        iz_list, ih_list, iw_list, ia_list = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = _select_samples(first_target, anchor, pos_th, oz, oh, ow)
            iz_list.append(iiz)
            ih_list.append(iih)
            iw_list.append(iiw)
            ia_list.append(i * np.ones((len(iiz),), np.int64))
        iz_all = np.concatenate(iz_list, 0)
        ih_all = np.concatenate(ih_list, 0)
        iw_all = np.concatenate(iw_list, 0)
        ia_all = np.concatenate(ia_list, 0)

        if len(iz_all) == 0:
            # Fallback: assign nearest grid position
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((first_target[i] - offset) / stride))))
            idx = int(np.argmin(np.abs(np.log(first_target[3] / anchors))))
            pos.append(idx)
        else:
            idx = 0
            pos = [int(iz_all[idx]), int(ih_all[idx]), int(iw_all[idx]), int(ia_all[idx])]

        dz = (first_target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (first_target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (first_target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(first_target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]

    return label.astype(np.float64)


# ---------------------------------------------------------------------------
# Atom 3: Center feature extraction 3D
# ---------------------------------------------------------------------------


@register_atom(witness_center_feature_extraction_3d)
@icontract.require(
    lambda feature_map: feature_map.ndim == 5,
    "feature_map must be 5D (N, C, D, H, W)",
)
@icontract.require(
    lambda feature_map: all(feature_map.shape[i] >= 2 for i in range(2, 5)),
    "each spatial dimension must be >= 2",
)
@icontract.ensure(
    lambda feature_map, result: result.shape == (feature_map.shape[0], feature_map.shape[1]),
    "output must be (N, C)",
)
def center_feature_extraction_3d(
    feature_map: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract center 2x2x2 cube from a 5D feature map and max-pool.

    Takes the spatial center region of the feature map --
    fm[:, :, D//2-1:D//2+1, H//2-1:H//2+1, W//2-1:W//2+1] -- and reduces
    it to (N, C) by taking the maximum over the 2x2x2 spatial dimensions.
    This provides a spatially-focused feature vector for downstream
    classification (e.g., nodule malignancy scoring).

    Derived from net_classifier.py CaseNet.forward (lines 163-166):
        centerFeat = self.pool(noduleFeat[:,:,D/2-1:D/2+1,H/2-1:H/2+1,W/2-1:W/2+1])
        centerFeat = centerFeat[:,:,0,0,0]
    """
    _, _, d, h, w = feature_map.shape
    center_cube = feature_map[
        :,
        :,
        d // 2 - 1 : d // 2 + 1,
        h // 2 - 1 : h // 2 + 1,
        w // 2 - 1 : w // 2 + 1,
    ]
    result: NDArray[np.float64] = np.max(center_cube, axis=(2, 3, 4))
    return result


@register_atom(witness_iou_matrix)
@icontract.require(lambda boxes_a: _valid_xyxy_boxes(boxes_a), "boxes_a must be xyxy boxes")
@icontract.require(lambda boxes_b: _valid_xyxy_boxes(boxes_b), "boxes_b must be xyxy boxes")
@icontract.ensure(
    lambda boxes_a, boxes_b, result: result.shape == (boxes_a.shape[0], boxes_b.shape[0]),
    "IoU matrix shape must match input box counts",
)
@icontract.ensure(
    lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))),
    "IoU values must be in [0, 1]",
)
def iou_matrix(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise intersection-over-union for xyxy boxes."""
    return _pairwise_iou(boxes_a, boxes_b)


@register_atom(witness_giou_matrix)
@icontract.require(lambda boxes_a: _valid_xyxy_boxes(boxes_a), "boxes_a must be xyxy boxes")
@icontract.require(lambda boxes_b: _valid_xyxy_boxes(boxes_b), "boxes_b must be xyxy boxes")
@icontract.ensure(
    lambda boxes_a, boxes_b, result: result.shape == (boxes_a.shape[0], boxes_b.shape[0]),
    "GIoU matrix shape must match input box counts",
)
@icontract.ensure(
    lambda result: bool(np.all((result >= -1.0 - 1e-12) & (result <= 1.0 + 1e-12))),
    "GIoU values must stay in [-1, 1]",
)
def giou_matrix(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise generalized IoU for xyxy boxes."""
    iou = _pairwise_iou(boxes_a, boxes_b)
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return iou
    top_left = np.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    enclosure = wh[:, :, 0] * wh[:, :, 1]
    intersection_top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    intersection_bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = np.maximum(intersection_bottom_right - intersection_top_left, 0.0)
    intersection = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    union = _box_areas(boxes_a)[:, None] + _box_areas(boxes_b)[None, :] - intersection
    penalty = np.divide(
        enclosure - union,
        enclosure,
        out=np.zeros_like(enclosure),
        where=enclosure > 0.0,
    )
    return iou - penalty


@register_atom(witness_nms)
@icontract.require(lambda boxes: _valid_xyxy_boxes(boxes), "boxes must be xyxy boxes")
@icontract.require(
    lambda boxes, scores: scores.shape == (boxes.shape[0],) and np.all(np.isfinite(scores)),
    "scores must be finite and match boxes",
)
@icontract.require(lambda iou_threshold: 0.0 <= iou_threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.ensure(lambda boxes, result: bool(np.all((result >= 0) & (result < boxes.shape[0]))), "indices must refer to input boxes")
def nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    iou_threshold: float,
) -> NDArray[np.int64]:
    """Select boxes by greedy non-maximum suppression."""
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while order.size > 0:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        overlaps = _pairwise_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


@register_atom(witness_soft_nms)
@icontract.require(lambda boxes: _valid_xyxy_boxes(boxes), "boxes must be xyxy boxes")
@icontract.require(
    lambda boxes, scores: scores.shape == (boxes.shape[0],)
    and np.all(np.isfinite(scores))
    and np.all((scores >= 0.0) & (scores <= 1.0)),
    "scores must be probabilities matching boxes",
)
@icontract.require(lambda iou_threshold: 0.0 <= iou_threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.require(lambda sigma: sigma > 0.0, "sigma must be positive")
@icontract.require(lambda method: method in {"linear", "gaussian"}, "method must be linear or gaussian")
@icontract.require(lambda score_threshold: 0.0 <= score_threshold <= 1.0, "score threshold must be in [0, 1]")
@icontract.ensure(lambda result: result[0].shape[0] == result[1].shape[0], "boxes and scores must align")
def soft_nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    iou_threshold: float,
    sigma: float = 0.5,
    method: str = "linear",
    score_threshold: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Soft-NMS by decaying scores of overlapping boxes."""
    remaining_boxes = boxes.astype(np.float64, copy=True)
    remaining_scores = scores.astype(np.float64, copy=True)
    selected_boxes: list[NDArray[np.float64]] = []
    selected_scores: list[float] = []

    while remaining_scores.size > 0:
        best = int(np.argmax(remaining_scores))
        selected_boxes.append(remaining_boxes[best].copy())
        selected_scores.append(float(remaining_scores[best]))
        if remaining_scores.size == 1:
            break

        current_box = remaining_boxes[best : best + 1]
        keep_mask = np.ones(remaining_scores.shape[0], dtype=bool)
        keep_mask[best] = False
        candidate_boxes = remaining_boxes[keep_mask]
        candidate_scores = remaining_scores[keep_mask]
        overlaps = _pairwise_iou(current_box, candidate_boxes)[0]
        if method == "linear":
            decay = np.where(overlaps > iou_threshold, 1.0 - overlaps, 1.0)
        else:
            decay = np.exp(-((overlaps * overlaps) / sigma))
        candidate_scores = candidate_scores * decay
        alive = candidate_scores >= score_threshold
        remaining_boxes = candidate_boxes[alive]
        remaining_scores = candidate_scores[alive]

    if not selected_boxes:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.vstack(selected_boxes).astype(np.float64), np.asarray(selected_scores, dtype=np.float64)


@register_atom(witness_wbf)
@icontract.require(lambda boxes_list, scores_list, labels_list: _lists_aligned(boxes_list, scores_list, labels_list), "box, score, and label lists must align")
@icontract.require(lambda weights, boxes_list: len(weights) == len(boxes_list) and all(weight > 0.0 for weight in weights), "weights must be positive and match models")
@icontract.require(lambda iou_threshold: 0.0 <= iou_threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.require(lambda skip_box_thr: 0.0 <= skip_box_thr <= 1.0, "skip threshold must be in [0, 1]")
@icontract.ensure(lambda result: result[0].shape[0] == result[1].shape[0] == result[2].shape[0], "outputs must align")
def wbf(
    boxes_list: list[NDArray[np.float64]],
    scores_list: list[NDArray[np.float64]],
    labels_list: list[NDArray[np.int64]],
    weights: list[float],
    iou_threshold: float = 0.55,
    skip_box_thr: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Fuse normalized boxes from multiple models by confidence-weighted averaging."""
    candidates: list[tuple[int, int, NDArray[np.float64], float, int, float]] = []
    for model_id, (boxes, scores, labels, weight) in enumerate(
        zip(boxes_list, scores_list, labels_list, weights)
    ):
        for row in range(boxes.shape[0]):
            if float(scores[row]) >= skip_box_thr:
                candidates.append(
                    (
                        int(labels[row]),
                        model_id,
                        boxes[row].astype(np.float64, copy=True),
                        float(scores[row]),
                        row,
                        float(scores[row]) * float(weight),
                    )
                )
    candidates.sort(key=lambda item: item[5], reverse=True)
    clusters: list[dict[str, object]] = []
    for label, model_id, box, score, _, weighted_score in candidates:
        best_cluster = -1
        best_iou = 0.0
        for cluster_index, cluster in enumerate(clusters):
            if int(cluster["label"]) != label:
                continue
            overlap = float(_pairwise_iou(box.reshape(1, 4), np.asarray(cluster["box"]).reshape(1, 4))[0, 0])
            if overlap > best_iou:
                best_iou = overlap
                best_cluster = cluster_index
        if best_cluster >= 0 and best_iou >= iou_threshold:
            cluster = clusters[best_cluster]
            cluster["boxes"].append(box)
            cluster["scores"].append(score)
            cluster["model_ids"].add(model_id)
            cluster["weighted_scores"].append(weighted_score)
        else:
            cluster = {
                "label": label,
                "boxes": [box],
                "scores": [score],
                "model_ids": {model_id},
                "weighted_scores": [weighted_score],
            }
            clusters.append(cluster)
        target = clusters[best_cluster] if best_cluster >= 0 and best_iou >= iou_threshold else clusters[-1]
        box_stack = np.vstack(target["boxes"])
        score_weights = np.asarray(target["weighted_scores"], dtype=np.float64)
        target["box"] = np.average(box_stack, axis=0, weights=score_weights)

    fused: list[tuple[NDArray[np.float64], float, int]] = []
    model_count = max(len(weights), 1)
    for cluster in clusters:
        participation = min(len(cluster["model_ids"]), model_count) / float(model_count)
        fused_score = float(np.mean(np.asarray(cluster["scores"], dtype=np.float64)) * participation)
        fused.append((np.clip(np.asarray(cluster["box"], dtype=np.float64), 0.0, 1.0), fused_score, int(cluster["label"])))
    fused.sort(key=lambda item: item[1], reverse=True)
    if not fused:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.vstack([item[0] for item in fused]).astype(np.float64),
        np.asarray([item[1] for item in fused], dtype=np.float64),
        np.asarray([item[2] for item in fused], dtype=np.int64),
    )


@register_atom(witness_wbf_1d)
@icontract.require(lambda spans_list, scores_list, labels_list: _span_lists_aligned(spans_list, scores_list, labels_list), "span, score, and label lists must align")
@icontract.require(lambda weights, spans_list: len(weights) == len(spans_list) and all(weight > 0.0 for weight in weights), "weights must be positive and match models")
@icontract.require(lambda iou_threshold: 0.0 <= iou_threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.require(lambda skip_span_thr: 0.0 <= skip_span_thr <= 1.0, "skip threshold must be in [0, 1]")
@icontract.ensure(lambda result: result[0].shape[0] == result[1].shape[0] == result[2].shape[0], "outputs must align")
def wbf_1d(
    spans_list: list[NDArray[np.float64]],
    scores_list: list[NDArray[np.float64]],
    labels_list: list[NDArray[np.int64]],
    weights: list[float],
    iou_threshold: float = 0.55,
    skip_span_thr: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Fuse 1D spans from multiple models by confidence-weighted averaging."""
    candidates: list[tuple[int, int, NDArray[np.float64], float, float]] = []
    for model_id, (spans, scores, labels, weight) in enumerate(
        zip(spans_list, scores_list, labels_list, weights)
    ):
        for row in range(spans.shape[0]):
            if float(scores[row]) >= skip_span_thr:
                candidates.append(
                    (
                        int(labels[row]),
                        model_id,
                        spans[row].astype(np.float64, copy=True),
                        float(scores[row]),
                        float(scores[row]) * float(weight),
                    )
                )
    candidates.sort(key=lambda item: item[4], reverse=True)
    clusters: list[dict[str, object]] = []
    for label, model_id, span, score, weighted_score in candidates:
        best_cluster = -1
        best_iou = 0.0
        for cluster_index, cluster in enumerate(clusters):
            if int(cluster["label"]) != label:
                continue
            overlap = float(_span_iou(span, np.asarray(cluster["span"]).reshape(1, 2))[0])
            if overlap > best_iou:
                best_iou = overlap
                best_cluster = cluster_index
        if best_cluster >= 0 and best_iou >= iou_threshold:
            cluster = clusters[best_cluster]
            cluster["spans"].append(span)
            cluster["scores"].append(score)
            cluster["model_ids"].add(model_id)
            cluster["weighted_scores"].append(weighted_score)
        else:
            cluster = {
                "label": label,
                "spans": [span],
                "scores": [score],
                "model_ids": {model_id},
                "weighted_scores": [weighted_score],
            }
            clusters.append(cluster)
        target = clusters[best_cluster] if best_cluster >= 0 and best_iou >= iou_threshold else clusters[-1]
        span_stack = np.vstack(target["spans"])
        score_weights = np.asarray(target["weighted_scores"], dtype=np.float64)
        target["span"] = np.average(span_stack, axis=0, weights=score_weights)

    fused: list[tuple[NDArray[np.float64], float, int]] = []
    model_count = max(len(weights), 1)
    for cluster in clusters:
        participation = min(len(cluster["model_ids"]), model_count) / float(model_count)
        fused_score = float(np.mean(np.asarray(cluster["scores"], dtype=np.float64)) * participation)
        fused.append((np.asarray(cluster["span"], dtype=np.float64), fused_score, int(cluster["label"])))
    fused.sort(key=lambda item: item[1], reverse=True)
    if not fused:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.vstack([item[0] for item in fused]).astype(np.float64),
        np.asarray([item[1] for item in fused], dtype=np.float64),
        np.asarray([item[2] for item in fused], dtype=np.int64),
    )


@register_atom(witness_generate_anchors)
@icontract.require(lambda feature_map_size: len(feature_map_size) == 2 and all(size > 0 for size in feature_map_size), "feature_map_size must be positive H,W")
@icontract.require(lambda stride: stride > 0, "stride must be positive")
@icontract.require(lambda sizes: len(sizes) > 0 and all(size > 0.0 for size in sizes), "sizes must be positive")
@icontract.require(lambda aspect_ratios: len(aspect_ratios) > 0 and all(ratio > 0.0 for ratio in aspect_ratios), "ratios must be positive")
@icontract.ensure(
    lambda feature_map_size, sizes, aspect_ratios, result: result.shape == (
        feature_map_size[0] * feature_map_size[1] * len(sizes) * len(aspect_ratios),
        4,
    ),
    "anchor count must match grid and base-anchor count",
)
def generate_anchors(
    feature_map_size: tuple[int, int],
    stride: int,
    sizes: tuple[float, ...],
    aspect_ratios: tuple[float, ...],
) -> NDArray[np.float64]:
    """Generate grid anchors in xyxy format for a feature map."""
    base_anchors = []
    for size in sizes:
        area = float(size) * float(size)
        for ratio in aspect_ratios:
            width = np.sqrt(area / float(ratio))
            height = width * float(ratio)
            base_anchors.append([-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height])
    base = np.asarray(base_anchors, dtype=np.float64)
    height, width = feature_map_size
    shift_x, shift_y = np.meshgrid(
        np.arange(width, dtype=np.float64) * float(stride),
        np.arange(height, dtype=np.float64) * float(stride),
    )
    shifts = np.stack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()], axis=1)
    return (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4).astype(np.float64)


@register_atom(witness_encode_boxes)
@icontract.require(lambda anchors: _positive_area_xyxy_boxes(anchors), "anchors must be positive-area xyxy boxes")
@icontract.require(lambda gt_boxes: _positive_area_xyxy_boxes(gt_boxes), "gt boxes must be positive-area xyxy boxes")
@icontract.require(lambda anchors, gt_boxes: anchors.shape == gt_boxes.shape, "anchors and gt boxes must align")
@icontract.require(lambda scales: len(scales) == 4 and all(scale > 0.0 for scale in scales), "scales must contain four positive values")
@icontract.ensure(lambda anchors, result: result.shape == anchors.shape, "deltas must match anchor shape")
def encode_boxes(
    anchors: NDArray[np.float64],
    gt_boxes: NDArray[np.float64],
    scales: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> NDArray[np.float64]:
    """Encode ground-truth boxes as Faster R-CNN deltas relative to anchors."""
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = scales
    deltas = np.zeros_like(anchors, dtype=np.float64)
    deltas[:, 0] = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
    deltas[:, 1] = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
    deltas[:, 2] = ww * np.log(gt_widths / anchor_widths)
    deltas[:, 3] = wh * np.log(gt_heights / anchor_heights)
    return deltas


@register_atom(witness_decode_boxes)
@icontract.require(lambda anchors: _positive_area_xyxy_boxes(anchors), "anchors must be positive-area xyxy boxes")
@icontract.require(lambda anchors, deltas: deltas.shape == anchors.shape and np.all(np.isfinite(deltas)), "deltas must be finite and match anchors")
@icontract.require(lambda scales: len(scales) == 4 and all(scale > 0.0 for scale in scales), "scales must contain four positive values")
@icontract.ensure(lambda anchors, result: result.shape == anchors.shape, "decoded boxes must match anchor shape")
@icontract.ensure(lambda result: _positive_area_xyxy_boxes(result), "decoded boxes must have positive area")
def decode_boxes(
    anchors: NDArray[np.float64],
    deltas: NDArray[np.float64],
    scales: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> NDArray[np.float64]:
    """Decode Faster R-CNN deltas back into xyxy boxes."""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    wx, wy, ww, wh = scales
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = np.clip(deltas[:, 2] / ww, -_BOX_DELTA_CLIP, _BOX_DELTA_CLIP)
    dh = np.clip(deltas[:, 3] / wh, -_BOX_DELTA_CLIP, _BOX_DELTA_CLIP)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    decoded = np.zeros_like(deltas, dtype=np.float64)
    decoded[:, 0] = pred_ctr_x - 0.5 * pred_w
    decoded[:, 1] = pred_ctr_y - 0.5 * pred_h
    decoded[:, 2] = pred_ctr_x + 0.5 * pred_w
    decoded[:, 3] = pred_ctr_y + 0.5 * pred_h
    return decoded


@register_atom(witness_nms_1d)
@icontract.require(lambda signal: signal.ndim == 1 and np.all(np.isfinite(signal)), "signal must be a finite vector")
@icontract.require(lambda min_distance: min_distance > 0, "min_distance must be positive")
@icontract.require(lambda threshold: 0.0 <= threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.ensure(lambda signal, result: bool(np.all((result >= 0) & (result < signal.shape[0]))), "peaks must be valid indices")
def nms_1d(
    signal: NDArray[np.float64],
    min_distance: int,
    threshold: float,
) -> NDArray[np.int64]:
    """Find local 1D peaks and suppress lower peaks within a fixed distance."""
    if signal.size == 0:
        return np.zeros((0,), dtype=np.int64)
    left = np.r_[signal[0] - 1.0, signal[:-1]]
    right = np.r_[signal[1:], signal[-1] - 1.0]
    candidate_mask = (signal >= threshold) & (signal >= left) & (signal >= right)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    plateau_peaks: list[int] = []
    splits = np.split(candidate_indices, np.where(np.diff(candidate_indices) > 1)[0] + 1)
    for group in splits:
        values = signal[group]
        max_value = np.max(values)
        best = group[np.flatnonzero(values == max_value)]
        plateau_peaks.append(int(best[len(best) // 2]))

    peaks = np.asarray(plateau_peaks, dtype=np.int64)
    order = peaks[np.argsort(signal[peaks])[::-1]]
    kept: list[int] = []
    for index in order:
        if all(abs(int(index) - kept_index) >= min_distance for kept_index in kept):
            kept.append(int(index))
    return np.asarray(sorted(kept), dtype=np.int64)


@register_atom(witness_masks_to_boxes)
@icontract.require(
    lambda binary_masks: binary_masks.ndim == 3 and binary_masks.dtype == np.bool_,
    "binary_masks must have shape (N, H, W) and bool dtype",
)
@icontract.ensure(lambda binary_masks, result: result.shape == (binary_masks.shape[0], 4), "one box per mask is required")
def masks_to_boxes(binary_masks: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Convert binary instance masks into xyxy bounding boxes."""
    n_masks, height, width = binary_masks.shape
    boxes = np.zeros((n_masks, 4), dtype=np.float64)
    if n_masks == 0:
        return boxes
    x_projection = np.any(binary_masks, axis=1)
    y_projection = np.any(binary_masks, axis=2)
    nonempty = np.any(x_projection, axis=1) & np.any(y_projection, axis=1)
    boxes[nonempty, 0] = np.argmax(x_projection[nonempty], axis=1)
    boxes[nonempty, 1] = np.argmax(y_projection[nonempty], axis=1)
    boxes[nonempty, 2] = width - 1 - np.argmax(x_projection[nonempty, ::-1], axis=1)
    boxes[nonempty, 3] = height - 1 - np.argmax(y_projection[nonempty, ::-1], axis=1)
    return boxes


@register_atom(witness_associate_boxes)
@icontract.require(lambda boxes_a: _valid_xyxy_boxes(boxes_a), "boxes_a must be xyxy boxes")
@icontract.require(lambda boxes_b: _valid_xyxy_boxes(boxes_b), "boxes_b must be xyxy boxes")
@icontract.require(lambda iou_threshold: 0.0 <= iou_threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.ensure(lambda result: len(result) == 4, "association must return four index arrays")
def associate_boxes(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
    iou_threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Associate two box sets by Hungarian matching on IoU distance."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.arange(boxes_a.shape[0], dtype=np.int64),
            np.arange(boxes_b.shape[0], dtype=np.int64),
        )
    overlaps = _pairwise_iou(boxes_a, boxes_b)
    row_ind, col_ind = linear_sum_assignment(1.0 - overlaps)
    valid = overlaps[row_ind, col_ind] >= iou_threshold
    matched_a = row_ind[valid].astype(np.int64)
    matched_b = col_ind[valid].astype(np.int64)
    unmatched_a = np.setdiff1d(np.arange(boxes_a.shape[0], dtype=np.int64), matched_a)
    unmatched_b = np.setdiff1d(np.arange(boxes_b.shape[0], dtype=np.int64), matched_b)
    return matched_a, matched_b, unmatched_a, unmatched_b


@register_atom(witness_threshold_detections)
@icontract.require(lambda boxes: _valid_xyxy_boxes(boxes), "boxes must be xyxy boxes")
@icontract.require(
    lambda boxes, scores: scores.shape == (boxes.shape[0],)
    and np.all(np.isfinite(scores))
    and np.all((scores >= 0.0) & (scores <= 1.0)),
    "scores must be probabilities matching boxes",
)
@icontract.require(lambda threshold: 0.0 <= threshold <= 1.0, "threshold must be in [0, 1]")
@icontract.ensure(lambda result: result[0].shape[0] == result[1].shape[0], "boxes and scores must align")
@icontract.ensure(lambda threshold, result: bool(np.all(result[1] >= threshold)), "scores must satisfy threshold")
def threshold_detections(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter detections by a minimum confidence score."""
    keep = scores >= threshold
    return boxes[keep].astype(np.float64, copy=True), scores[keep].astype(np.float64, copy=True)


@register_atom(witness_margin_expanded_face_crop)
@icontract.require(lambda image: _valid_image_array(image), "image must be a finite 2D or HWC numeric array")
@icontract.require(lambda bbox: isinstance(bbox, np.ndarray) and bbox.shape == (4,), "bbox must be a length-4 xyxy array")
@icontract.require(lambda bbox: _valid_xyxy_boxes(bbox.reshape(1, 4)), "bbox must be a finite xyxy box")
@icontract.require(lambda margin: np.isfinite(margin) and margin >= 0.0, "margin must be non-negative")
@icontract.require(lambda image, bbox, margin: _crop_has_area(image, bbox, margin), "expanded crop must overlap image")
@icontract.ensure(lambda result: result.shape[0] > 0 and result.shape[1] > 0, "crop must have positive height and width")
@icontract.ensure(lambda image, result: result.ndim == image.ndim, "crop preserves image rank")
def margin_expanded_face_crop(
    image: NDArray[np.float64],
    bbox: NDArray[np.float64],
    margin: float,
) -> NDArray[np.float64]:
    """Expand an xyxy face box by a fractional margin, clip to image bounds, and crop."""
    left, top, right, bottom = _crop_bounds(image.shape, bbox, float(margin))
    return np.asarray(image[top:bottom, left:right], dtype=image.dtype).copy()


@register_atom(witness_face_similarity_align)
@icontract.require(lambda image: _valid_image_array(image), "image must be a finite 2D or HWC numeric array")
@icontract.require(lambda src_landmarks: _nondegenerate_landmarks(src_landmarks), "source landmarks must be 5 non-collinear xy points")
@icontract.require(lambda dst_landmarks: _nondegenerate_landmarks(dst_landmarks), "target landmarks must be 5 non-collinear xy points")
@icontract.require(lambda output_size: _positive_output_size(output_size), "output_size must be positive H,W")
@icontract.require(lambda order: order in {0, 1, 2, 3, 4, 5}, "order must be a scipy spline order")
@icontract.ensure(lambda output_size, result: result.shape[:2] == output_size, "aligned image must match requested size")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "aligned image must be finite")
def face_similarity_align(
    image: NDArray[np.float64],
    src_landmarks: NDArray[np.float64],
    dst_landmarks: NDArray[np.float64],
    output_size: tuple[int, int],
    order: int = 1,
) -> NDArray[np.float64]:
    """Align a face image to target landmarks using a 2D similarity transform."""
    scale, rotation, translation = _estimate_similarity_transform(src_landmarks, dst_landmarks)
    return _sample_similarity_aligned(
        np.asarray(image, dtype=np.float64),
        scale,
        rotation,
        translation,
        output_size,
        int(order),
    )
