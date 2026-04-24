"""Ghost witnesses for graph atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_node_degree_bucketing(
    degrees: AbstractArray,
    num_buckets: int = 10,
) -> AbstractArray:
    """Ghost witness for node degree bucketing."""
    return degrees


def witness_feature_clip_standardize(
    features: AbstractArray,
    clip_range: float = 3.0,
) -> AbstractArray:
    """Ghost witness for feature clip standardization."""
    return features


def witness_time_budget_estimator(
    elapsed_seconds: float,
    total_budget: float,
    model_complexities: AbstractArray,
) -> int:
    """Ghost witness for time budget estimation."""
    return 0


def witness_adjacency_smoothing(
    adjacency: AbstractArray,
) -> AbstractArray:
    """Ghost witness for symmetric adjacency smoothing."""
    return adjacency
