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
)
from skimage import measure

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_anchor_label_mapping_with_iou_dilation,
    witness_center_feature_extraction_3d,
    witness_lung_mask_with_bone_removal,
)


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
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
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

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

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
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

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
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
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
