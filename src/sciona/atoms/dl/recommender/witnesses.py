"""Ghost witnesses for recommender-system atoms."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


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
