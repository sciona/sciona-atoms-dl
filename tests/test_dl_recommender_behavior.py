from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
import pytest


def test_recommender_import() -> None:
    from sciona.atoms.dl.recommender.atoms import (
        bpr_max_loss,
        in_batch_negative_sampling,
        ranking_moments_extractor,
        sampled_softmax_loss,
        uniform_negative_sampling,
    )

    assert callable(sampled_softmax_loss)
    assert callable(bpr_max_loss)
    assert callable(uniform_negative_sampling)
    assert callable(in_batch_negative_sampling)
    assert callable(ranking_moments_extractor)


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
