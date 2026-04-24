"""Recommender-system training primitives in pure numpy.

Implements optimized negative-sampling losses and samplers from the
MIT-licensed TRON RecSys 2023 codebase, plus rank-distribution feature
extraction from the RecSys 2024 winning reranking pipeline.

Source: recsys-2023-tron/src/shared/loss.py (MIT)
        recsys-2023-tron/src/shared/sample.py (MIT)
        recsys-2024-1st/Utils/xgboost_functions.py (AGPL-3.0)
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bpr_max_loss,
    witness_in_batch_negative_sampling,
    witness_ranking_moments_extractor,
    witness_sampled_softmax_loss,
    witness_uniform_negative_sampling,
)


def _positive_column(
    positive_scores: NDArray[np.float64],
    negative_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if neg.ndim != 2:
        raise ValueError("negative_scores must be a 2-D array")
    if pos.shape[0] != neg.shape[0]:
        raise ValueError("positive_scores and negative_scores must align on axis 0")
    return pos


def _stable_softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


@register_atom(witness_sampled_softmax_loss)
@icontract.require(
    lambda negative_scores: np.asarray(negative_scores).ndim == 2,
    "negative_scores must be a 2-D array",
)
@icontract.require(
    lambda positive_scores, negative_scores: np.asarray(positive_scores).size
    == np.asarray(negative_scores).shape[0],
    "positive_scores must contain one value per row of negative_scores",
)
@icontract.require(
    lambda negative_scores: np.asarray(negative_scores).shape[1] >= 1,
    "negative_scores must contain at least one negative per row",
)
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def sampled_softmax_loss(
    positive_scores: NDArray[np.float64],
    negative_scores: NDArray[np.float64],
) -> float:
    """Compute sampled softmax cross-entropy with the positive score as class 0.

    TRON concatenates the positive logit with sampled negatives and applies a
    cross-entropy target of zero. This atom returns the mean loss over rows.
    """
    pos = _positive_column(positive_scores, negative_scores)
    neg = np.asarray(negative_scores, dtype=np.float64)
    logits = np.concatenate((pos, neg), axis=1)
    logsumexp = np.log(np.sum(np.exp(logits), axis=1))
    losses = -pos[:, 0] + logsumexp
    return float(np.mean(losses))


@register_atom(witness_bpr_max_loss)
@icontract.require(
    lambda negative_scores: np.asarray(negative_scores).ndim == 2,
    "negative_scores must be a 2-D array",
)
@icontract.require(
    lambda positive_scores, negative_scores: np.asarray(positive_scores).size
    == np.asarray(negative_scores).shape[0],
    "positive_scores must contain one value per row of negative_scores",
)
@icontract.require(
    lambda negative_scores: np.asarray(negative_scores).shape[1] >= 1,
    "negative_scores must contain at least one negative per row",
)
@icontract.require(lambda reg_lambda: reg_lambda >= 0.0, "reg_lambda must be non-negative")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def bpr_max_loss(
    positive_scores: NDArray[np.float64],
    negative_scores: NDArray[np.float64],
    reg_lambda: float,
) -> float:
    """Compute TRON's BPR-Max ranking loss with softmax-weighted regularization."""
    pos = _positive_column(positive_scores, negative_scores)
    neg = np.asarray(negative_scores, dtype=np.float64)

    logits_diff = 1.0 / (1.0 + np.exp(-(pos - neg)))
    softmax_neg_shifted = _stable_softmax(neg)
    weighted_pairwise = softmax_neg_shifted * logits_diff
    unregularized = -np.log(np.sum(weighted_pairwise, axis=1))

    softmax_neg = _stable_softmax(neg)
    regularization = reg_lambda * np.sum(softmax_neg * neg * neg, axis=1)
    return float(np.mean(unregularized + regularization))


@register_atom(witness_uniform_negative_sampling)
@icontract.require(lambda num_items: num_items >= 1, "num_items must be positive")
@icontract.require(lambda num_samples: num_samples >= 0, "num_samples must be non-negative")
@icontract.require(
    lambda exclude, num_items: np.all((np.asarray(exclude, dtype=np.int64) >= 1))
    and np.all((np.asarray(exclude, dtype=np.int64) <= num_items)),
    "exclude items must lie in [1, num_items]",
)
@icontract.require(
    lambda num_items, exclude, num_samples: num_samples == 0
    or len(np.setdiff1d(np.arange(1, num_items + 1), np.asarray(exclude, dtype=np.int64)))
    >= 1,
    "at least one candidate item must remain after exclusion",
)
@icontract.ensure(
    lambda num_samples, result: result.shape == (num_samples,),
    "result must have shape (num_samples,)",
)
def uniform_negative_sampling(
    num_items: int,
    num_samples: int,
    exclude: NDArray[np.int64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample uniform negatives with replacement while avoiding excluded items.

    TRON samples item ids in the inclusive range ``[1, num_items]``. Rejection
    sampling in the original code is implemented here by drawing uniformly from
    the remaining candidate set, which preserves the same distribution.
    """
    if num_samples == 0:
        return np.array([], dtype=np.int64)

    excluded = np.asarray(exclude, dtype=np.int64)
    candidates = np.setdiff1d(
        np.arange(1, num_items + 1, dtype=np.int64),
        excluded,
        assume_unique=False,
    )
    return rng.choice(candidates, size=num_samples, replace=True).astype(np.int64)


@register_atom(witness_in_batch_negative_sampling)
@icontract.require(
    lambda batch_items: np.asarray(batch_items).ndim == 1,
    "batch_items must be a 1-D array",
)
@icontract.require(lambda num_negatives: num_negatives >= 0, "num_negatives must be non-negative")
@icontract.require(
    lambda batch_items, num_negatives: num_negatives <= max(0, np.asarray(batch_items).size - 1),
    "num_negatives cannot exceed the number of other batch items",
)
@icontract.ensure(
    lambda batch_items, num_negatives, result: result.shape
    == (np.asarray(batch_items).size, num_negatives),
    "result must have shape (len(batch_items), num_negatives)",
)
def in_batch_negative_sampling(
    batch_items: NDArray[np.int64],
    num_negatives: int,
) -> NDArray[np.int64]:
    """Sample negatives for each item from other items in the same batch."""
    items = np.asarray(batch_items, dtype=np.int64).reshape(-1)
    if num_negatives == 0:
        return np.empty((len(items), 0), dtype=np.int64)

    rng = np.random.default_rng()
    negatives = np.empty((len(items), num_negatives), dtype=np.int64)
    for idx in range(len(items)):
        candidates = np.delete(items, idx)
        negatives[idx] = rng.choice(candidates, size=num_negatives, replace=False)
    return negatives


@register_atom(witness_ranking_moments_extractor)
@icontract.require(
    lambda rank_matrix: np.asarray(rank_matrix).ndim == 2,
    "rank_matrix must be a 2-D array",
)
@icontract.require(
    lambda rank_matrix: np.asarray(rank_matrix).shape[1] >= 2,
    "rank_matrix must contain at least two rank columns",
)
@icontract.ensure(
    lambda rank_matrix, result: result.shape == (np.asarray(rank_matrix).shape[0], 4),
    "output must have shape (n_items, 4)",
)
def ranking_moments_extractor(
    rank_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute rank-distribution moments across recommender models.

    Derived from the RecSys 2024 XGBoost feature builder, which adds position
    mean, sample standard deviation, skewness, and kurtosis as reranking
    features.
    """
    ranks = np.asarray(rank_matrix, dtype=np.float64)
    mean_rank = np.mean(ranks, axis=1)
    std_rank = np.std(ranks, axis=1, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skew_rank = skew(ranks, axis=1)
        kurtosis_rank = kurtosis(ranks, axis=1)
    return np.column_stack([mean_rank, std_rank, skew_rank, kurtosis_rank])
