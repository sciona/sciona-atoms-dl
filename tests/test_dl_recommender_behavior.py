from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.sparse import csr_matrix
import pytest


def test_recommender_import() -> None:
    from sciona.atoms.dl.recommender.atoms import (
        als_item_update,
        als_user_update,
        bpr_max_loss,
        co_occurrence_matrix,
        cooccurrence_candidates,
        in_batch_negative_sampling,
        item_popularity_decay,
        ranking_moments_extractor,
        reciprocal_rank_fusion,
        sampled_softmax_loss,
        session_features,
        uniform_negative_sampling,
        user_item_affinity,
    )

    assert callable(co_occurrence_matrix)
    assert callable(cooccurrence_candidates)
    assert callable(als_user_update)
    assert callable(als_item_update)
    assert callable(item_popularity_decay)
    assert callable(session_features)
    assert callable(user_item_affinity)
    assert callable(reciprocal_rank_fusion)
    assert callable(sampled_softmax_loss)
    assert callable(bpr_max_loss)
    assert callable(uniform_negative_sampling)
    assert callable(in_batch_negative_sampling)
    assert callable(ranking_moments_extractor)


def test_co_occurrence_matrix_builds_weighted_sparse_counts() -> None:
    from sciona.atoms.dl.recommender import co_occurrence_matrix

    matrix = co_occurrence_matrix(
        sessions=[[0, 1, 2], [2, 1, 0]],
        n_items=4,
        time_weights=np.array([0.0, 1.0, 0.5], dtype=np.float64),
    )

    dense = matrix.toarray()
    assert dense.shape == (4, 4)
    assert np.allclose(dense.diagonal(), 0.0)
    assert dense[0, 1] == pytest.approx(2.0)
    assert dense[0, 2] == pytest.approx(1.0)
    assert np.allclose(dense, dense.T)


def test_cooccurrence_candidates_filters_seen_and_ranks_scores() -> None:
    from sciona.atoms.dl.recommender import cooccurrence_candidates

    matrix = csr_matrix(
        np.array(
            [
                [0.0, 0.0, 5.0, 2.0],
                [0.0, 0.0, 1.0, 8.0],
                [5.0, 1.0, 0.0, 0.0],
                [2.0, 8.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    result = cooccurrence_candidates(np.array([0, 1], dtype=np.int64), matrix, k=2)
    assert np.array_equal(result, np.array([3, 2], dtype=np.int64))


def test_als_user_update_matches_closed_form_reference() -> None:
    from sciona.atoms.dl.recommender import als_user_update

    item_factors = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float64)
    interactions = csr_matrix(([2.0, 1.0], ([0, 0], [0, 2])), shape=(1, 3))
    yt_y = item_factors.T @ item_factors

    result = als_user_update(interactions, item_factors, yt_y, regularization=0.5, alpha=2.0)

    rows = item_factors[[0, 2]]
    confidence = np.array([5.0, 3.0])
    lhs = yt_y + rows.T @ (rows * (confidence - 1.0)[:, None]) + 0.5 * np.eye(2)
    rhs = rows.T @ confidence
    assert np.allclose(result, np.linalg.solve(lhs, rhs))


def test_als_item_update_matches_user_update_formula() -> None:
    from sciona.atoms.dl.recommender import als_item_update

    user_factors = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    interactions = csr_matrix(([1.0], ([0], [1])), shape=(1, 3))
    xt_x = user_factors.T @ user_factors

    result = als_item_update(interactions, user_factors, xt_x, regularization=1.0, alpha=1.0)
    row = user_factors[[1]]
    lhs = xt_x + row.T @ row + np.eye(2)
    rhs = row.T @ np.array([2.0])
    assert np.allclose(result, np.linalg.solve(lhs, rhs))


def test_item_popularity_decay_uses_half_life_weights() -> None:
    from sciona.atoms.dl.recommender import item_popularity_decay

    scores = item_popularity_decay(
        np.array([2, 2, 1], dtype=np.int64),
        np.array([100.0, 90.0, 100.0], dtype=np.float64),
        reference_time=100.0,
        half_life=10.0,
    )

    assert scores[2] == pytest.approx(1.5)
    assert scores[1] == pytest.approx(1.0)


def test_session_features_extracts_duration_and_gap_stats() -> None:
    from sciona.atoms.dl.recommender import session_features

    result = session_features([5, 5, 6], np.array([10.0, 20.0, 40.0], dtype=np.float64))

    assert result == {
        "session_length": 3.0,
        "unique_items": 2.0,
        "avg_gap": 15.0,
        "duration": 30.0,
    }


def test_user_item_affinity_handles_seen_and_unseen_items() -> None:
    from sciona.atoms.dl.recommender import user_item_affinity

    seen = user_item_affinity(np.array([5.0, 9.0, 12.0], dtype=np.float64), reference_time=15.0)
    unseen = user_item_affinity(np.array([], dtype=np.float64), reference_time=15.0)

    assert seen == {"interaction_count": 3.0, "recency": 3.0}
    assert unseen == {"interaction_count": 0.0, "recency": -1.0}


def test_reciprocal_rank_fusion_rewards_consistent_candidates() -> None:
    from sciona.atoms.dl.recommender import reciprocal_rank_fusion

    result = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 4]], k=60, top_n=3)

    assert result[:2] == [3, 2]
    assert len(result) == len(set(result))
    assert len(result) == 3


def test_sampled_softmax_loss_matches_cross_entropy() -> None:
    from sciona.atoms.dl.recommender.atoms import sampled_softmax_loss

    positive = np.array([2.0], dtype=np.float64)
    negatives = np.array([[1.0, 0.0]], dtype=np.float64)
    result = sampled_softmax_loss(positive, negatives)
    expected = -2.0 + np.log(np.exp(2.0) + np.exp(1.0) + np.exp(0.0))
    assert result == pytest.approx(expected)


def test_sampled_softmax_loss_rewards_stronger_positive_scores() -> None:
    from sciona.atoms.dl.recommender.atoms import sampled_softmax_loss

    negatives = np.array([[0.0, -1.0], [0.5, 0.0]], dtype=np.float64)
    weak = sampled_softmax_loss(np.array([0.1, 0.1]), negatives)
    strong = sampled_softmax_loss(np.array([3.0, 3.0]), negatives)
    assert strong < weak


def test_bpr_max_loss_matches_reference_formula() -> None:
    from sciona.atoms.dl.recommender.atoms import bpr_max_loss

    positive = np.array([2.0], dtype=np.float64)
    negatives = np.array([[0.0, -1.0]], dtype=np.float64)
    result = bpr_max_loss(positive, negatives, reg_lambda=0.1)

    diff = 1.0 / (1.0 + np.exp(-(positive.reshape(-1, 1) - negatives)))
    shifted = negatives - negatives.max(axis=1, keepdims=True)
    softmax_shifted = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    unregularized = -np.log(np.sum(softmax_shifted * diff, axis=1))
    softmax_neg = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    regularized = 0.1 * np.sum(softmax_neg * negatives * negatives, axis=1)
    expected = float(np.mean(unregularized + regularized))
    assert result == pytest.approx(expected)


def test_bpr_max_loss_regularization_increases_loss() -> None:
    from sciona.atoms.dl.recommender.atoms import bpr_max_loss

    positive = np.array([1.0, 1.0], dtype=np.float64)
    negatives = np.array([[0.5, 0.0], [0.2, -0.2]], dtype=np.float64)
    unregularized = bpr_max_loss(positive, negatives, reg_lambda=0.0)
    regularized = bpr_max_loss(positive, negatives, reg_lambda=0.5)
    assert regularized > unregularized


def test_uniform_negative_sampling_excludes_positive_items() -> None:
    from sciona.atoms.dl.recommender.atoms import uniform_negative_sampling

    rng = np.random.default_rng(42)
    result = uniform_negative_sampling(
        num_items=5,
        num_samples=100,
        exclude=np.array([2, 4], dtype=np.int64),
        rng=rng,
    )
    assert result.shape == (100,)
    assert set(np.unique(result).tolist()) <= {1, 3, 5}
    assert 2 not in result
    assert 4 not in result


def test_uniform_negative_sampling_zero_samples() -> None:
    from sciona.atoms.dl.recommender.atoms import uniform_negative_sampling

    rng = np.random.default_rng(7)
    result = uniform_negative_sampling(
        num_items=3,
        num_samples=0,
        exclude=np.array([1], dtype=np.int64),
        rng=rng,
    )
    assert result.shape == (0,)


def test_in_batch_negative_sampling_uses_other_items() -> None:
    from sciona.atoms.dl.recommender.atoms import in_batch_negative_sampling

    batch_items = np.array([10, 20, 30, 40], dtype=np.int64)
    result = in_batch_negative_sampling(batch_items, num_negatives=2)
    assert result.shape == (4, 2)
    for row_idx, row in enumerate(result):
        assert batch_items[row_idx] not in row
        assert set(row.tolist()) <= set(batch_items.tolist())


def test_in_batch_negative_sampling_zero_negatives() -> None:
    from sciona.atoms.dl.recommender.atoms import in_batch_negative_sampling

    batch_items = np.array([10, 20, 30], dtype=np.int64)
    result = in_batch_negative_sampling(batch_items, num_negatives=0)
    assert result.shape == (3, 0)


def test_ranking_moments_extractor_matches_scipy_moments() -> None:
    from sciona.atoms.dl.recommender.atoms import ranking_moments_extractor

    rank_matrix = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float64)
    result = ranking_moments_extractor(rank_matrix)

    expected = np.column_stack(
        [
            rank_matrix.mean(axis=1),
            rank_matrix.std(axis=1, ddof=1),
            skew(rank_matrix, axis=1),
            kurtosis(rank_matrix, axis=1),
        ]
    )
    assert np.allclose(result, expected)
