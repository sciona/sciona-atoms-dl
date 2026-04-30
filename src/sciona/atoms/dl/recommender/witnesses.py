"""Ghost witnesses for recommender-system atoms."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


def witness_co_occurrence_matrix(
    sessions: list[list[int]],
    n_items: int,
    time_weights: AbstractArray | None = None,
) -> AbstractArray:
    """Witness for sparse co-occurrence matrix construction."""
    return AbstractArray(shape=(n_items, n_items), dtype="float64")


def witness_cooccurrence_candidates(
    user_items: AbstractArray,
    cooccurrence: AbstractArray,
    k: int,
    filter_seen: bool = True,
) -> AbstractArray:
    """Witness for co-occurrence candidate retrieval."""
    return AbstractArray(shape=(k,), dtype="int64")


def witness_als_user_update(
    user_interactions: AbstractArray,
    item_factors: AbstractArray,
    yt_y: AbstractArray,
    regularization: float,
    alpha: float = 40.0,
) -> AbstractArray:
    """Witness for one ALS user-vector update."""
    return item_factors


def witness_als_item_update(
    item_interactions: AbstractArray,
    user_factors: AbstractArray,
    xt_x: AbstractArray,
    regularization: float,
    alpha: float = 40.0,
) -> AbstractArray:
    """Witness for one ALS item-vector update."""
    return user_factors


def witness_item_popularity_decay(
    item_ids: AbstractArray,
    timestamps: AbstractArray,
    reference_time: float,
    half_life: float,
) -> AbstractArray:
    """Witness for time-decayed item popularity."""
    return item_ids


def witness_session_features(
    item_sequence: list[int],
    timestamps: AbstractArray,
) -> dict[str, float]:
    """Witness for session feature extraction."""
    return {"session_length": 0.0, "unique_items": 0.0, "avg_gap": 0.0, "duration": 0.0}


def witness_user_item_affinity(
    user_item_timestamps: AbstractArray,
    reference_time: float,
) -> dict[str, float]:
    """Witness for user-item affinity features."""
    return {"interaction_count": 0.0, "recency": -1.0}


def witness_reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int = 100,
) -> list[int]:
    """Witness for reciprocal rank fusion."""
    return []


def witness_sampled_softmax_loss(
    positive_scores: AbstractArray,
    negative_scores: AbstractArray,
) -> float:
    """Witness for sampled softmax loss."""
    return 0.0


def witness_bpr_max_loss(
    positive_scores: AbstractArray,
    negative_scores: AbstractArray,
    reg_lambda: float,
) -> float:
    """Witness for BPR-Max loss."""
    return 0.0


def witness_uniform_negative_sampling(
    num_items: int,
    num_samples: int,
    exclude: AbstractArray,
    rng: np.random.Generator,
) -> AbstractArray:
    """Witness for uniform negative sampling."""
    return exclude


def witness_in_batch_negative_sampling(
    batch_items: AbstractArray,
    num_negatives: int,
) -> AbstractArray:
    """Witness for in-batch negative sampling."""
    return batch_items


def witness_ranking_moments_extractor(
    rank_matrix: AbstractArray,
) -> AbstractArray:
    """Witness for ranking moment extraction."""
    return rank_matrix
