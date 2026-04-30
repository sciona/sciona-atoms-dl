from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.dl.segmentation import (
    dense_crf_2d,
    dilate_mask,
    erode_mask,
    false_color_composite,
    fill_holes,
    filter_components_by_area,
    mask_to_rle,
    morphological_close,
    morphological_open,
    rle_to_mask,
    smooth_contour,
    watershed_instances,
    wkt_to_mask,
)


def test_binary_morphology_atoms_preserve_expected_mask_semantics() -> None:
    isolated = np.zeros((5, 5), dtype=np.uint8)
    isolated[2, 2] = 1
    assert int(morphological_open(isolated, 3).sum()) == 0
    assert int(dilate_mask(isolated, 1).sum()) == 5

    block_with_hole = np.ones((7, 7), dtype=np.uint8)
    block_with_hole[0, :] = 0
    block_with_hole[-1, :] = 0
    block_with_hole[:, 0] = 0
    block_with_hole[:, -1] = 0
    block_with_hole[3, 3] = 0
    closed = morphological_close(block_with_hole, 3)
    filled = fill_holes(block_with_hole)
    assert closed[3, 3] == 1
    assert filled[3, 3] == 1

    square = np.zeros((5, 5), dtype=np.uint8)
    square[1:4, 1:4] = 1
    assert int(erode_mask(square, 1).sum()) == 1


def test_filter_components_by_area_removes_small_components() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2:4, 2:4] = 1

    filtered = filter_components_by_area(mask, min_area=2)

    assert filtered[0, 0] == 0
    assert int(filtered.sum()) == 4


def test_dense_crf_2d_returns_hard_label_map() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    unary = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.1, 0.9], [0.8, 0.2]],
        ],
        dtype=np.float64,
    )

    labels = dense_crf_2d(image, unary, sxy=1.0, srgb=1.0, compat=0.0, iterations=1)

    assert labels.shape == (2, 2)
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels)).issubset({0, 1})


def test_watershed_instances_keeps_labels_inside_mask() -> None:
    distance_map = np.zeros((5, 5), dtype=np.float64)
    distance_map[1, 1] = 2.0
    distance_map[3, 3] = 2.0
    markers = np.zeros((5, 5), dtype=np.int64)
    markers[1, 1] = 1
    markers[3, 3] = 2
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    labels = watershed_instances(distance_map, markers, mask)

    assert labels.shape == mask.shape
    assert np.all(labels[mask == 0] == 0)
    assert {1, 2}.issubset(set(np.unique(labels)))


def test_mask_rle_round_trip_uses_one_based_column_major_runs() -> None:
    mask = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)

    rle = mask_to_rle(mask)

    assert rle == [1, 2, 4, 1]
    assert np.array_equal(rle_to_mask(rle, mask.shape), mask)


def test_smooth_contour_simplifies_ordered_points() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.02], [2.0, -0.01], [3.0, 0.0]])

    simplified = smooth_contour(points, epsilon=0.1)

    assert np.array_equal(simplified, np.array([[0.0, 0.0], [3.0, 0.0]]))


def test_wkt_to_mask_rasterizes_polygon_with_affine_transform() -> None:
    mask = wkt_to_mask(
        "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
        image_shape=(5, 5),
        transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )

    assert mask.shape == (5, 5)
    assert mask[2, 2] == 1
    assert int(mask.sum()) >= 4


def test_false_color_composite_stretches_bands_to_uint8_rgb() -> None:
    bands = {
        "red": np.array([[0.0, 0.5], [1.0, 1.5]]),
        "green": np.array([[10.0, 15.0], [20.0, 25.0]]),
        "blue": np.array([[-1.0, 0.0], [1.0, 2.0]]),
    }
    bounds = {"red": (0.0, 1.0), "green": (10.0, 20.0), "blue": (-1.0, 1.0)}

    image = false_color_composite(bands, bounds)

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[1, 0].tolist() == [255, 255, 255]


def test_contracts_reject_non_binary_masks() -> None:
    with pytest.raises(ViolationError):
        fill_holes(np.array([[0, 2]], dtype=np.uint8))

