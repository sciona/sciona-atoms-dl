from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.dl.skeletonization import (
    medial_axis_2d,
    skeleton_to_graph,
    skeletonize_2d,
)


def test_skeletonize_2d_preserves_shape_and_stays_inside_mask() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 2:10] = 1

    skeleton = skeletonize_2d(mask)

    assert skeleton.shape == mask.shape
    assert skeleton.dtype == np.bool_
    assert np.all(~skeleton | mask.astype(bool))
    assert 0 < int(skeleton.sum()) < int(mask.sum())


def test_skeletonize_2d_rejects_nonbinary_input() -> None:
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)

    with pytest.raises(Exception, match="binary 2D"):
        skeletonize_2d(mask)


def test_medial_axis_2d_can_return_distance_map_deterministically() -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1

    first_skeleton, first_distance = medial_axis_2d(mask, return_distance=True, rng_seed=7)
    second_skeleton, second_distance = medial_axis_2d(mask, return_distance=True, rng_seed=7)

    assert np.array_equal(first_skeleton, second_skeleton)
    assert np.array_equal(first_distance, second_distance)
    assert first_skeleton.shape == mask.shape
    assert first_distance.shape == mask.shape
    assert np.all(first_distance >= 0.0)
    assert np.all(~first_skeleton | mask.astype(bool))


def test_skeleton_to_graph_extracts_cross_endpoints_and_junction() -> None:
    skeleton = np.zeros((11, 11), dtype=np.uint8)
    skeleton[5, 1:10] = 1
    skeleton[1:10, 5] = 1

    graph = skeleton_to_graph(skeleton)

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 4
    degrees = sorted(degree for _, degree in graph.degree())
    assert degrees == [1, 1, 1, 1, 4]
    for _, data in graph.nodes(data=True):
        assert data["pts"].ndim == 2
        assert data["pts"].shape[1] == 2
        assert data["o"].shape == (2,)
    for _, _, data in graph.edges(data=True):
        assert data["pts"].ndim == 2
        assert data["pts"].shape[1] == 2
        assert data["weight"] > 0.0

