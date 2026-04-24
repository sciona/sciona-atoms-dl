from __future__ import annotations

import numpy as np


def test_graph_atoms_import() -> None:
    from sciona.atoms.dl.graph.atoms import (
        adjacency_smoothing,
        feature_clip_standardize,
        node_degree_bucketing,
        time_budget_estimator,
    )

    assert callable(node_degree_bucketing)
    assert callable(feature_clip_standardize)
    assert callable(time_budget_estimator)
    assert callable(adjacency_smoothing)


def test_node_degree_bucketing_uses_log2_buckets() -> None:
    from sciona.atoms.dl.graph.atoms import node_degree_bucketing

    degrees = np.array([0, 1, 2, 3, 7, 8, 15], dtype=np.int64)
    result = node_degree_bucketing(degrees)
    assert np.array_equal(result, np.array([0, 1, 1, 2, 3, 3, 4], dtype=np.int64))


def test_node_degree_bucketing_clips_to_num_buckets() -> None:
    from sciona.atoms.dl.graph.atoms import node_degree_bucketing

    degrees = np.array([0, 1, 1024], dtype=np.int64)
    result = node_degree_bucketing(degrees, num_buckets=3)
    assert np.array_equal(result, np.array([0, 1, 2], dtype=np.int64))


def test_feature_clip_standardize_matches_zscore_with_clip() -> None:
    from sciona.atoms.dl.graph.atoms import feature_clip_standardize

    features = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]], dtype=np.float64)
    result = feature_clip_standardize(features, clip_range=1.0)
    expected = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    assert np.allclose(result, expected, atol=1e-4)


def test_time_budget_estimator_picks_largest_feasible_complexity() -> None:
    from sciona.atoms.dl.graph.atoms import time_budget_estimator

    complexities = np.array([10.0, 30.0, 60.0], dtype=np.float64)
    result = time_budget_estimator(20.0, 70.0, complexities)
    assert result == 1


def test_time_budget_estimator_falls_back_to_cheapest_when_none_fit() -> None:
    from sciona.atoms.dl.graph.atoms import time_budget_estimator

    complexities = np.array([10.0, 30.0, 60.0], dtype=np.float64)
    result = time_budget_estimator(68.0, 70.0, complexities)
    assert result == 0


def test_adjacency_smoothing_returns_symmetric_normalization() -> None:
    from sciona.atoms.dl.graph.atoms import adjacency_smoothing

    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    result = adjacency_smoothing(adjacency)
    assert np.allclose(result, adjacency)


def test_adjacency_smoothing_handles_zero_degree_nodes() -> None:
    from sciona.atoms.dl.graph.atoms import adjacency_smoothing

    adjacency = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = adjacency_smoothing(adjacency)
    assert np.allclose(result[2], np.zeros(3))
    assert np.allclose(result[:, 2], np.zeros(3))
