"""Recommender-system primitives in pure numpy and scipy.

Implements optimized negative-sampling losses and samplers from the
MIT-licensed TRON RecSys 2023 codebase, plus rank-distribution feature
extraction from the RecSys 2024 winning reranking pipeline. The retrieval
and feature atoms follow the local recommender-systems research report.

Source: recsys-2023-tron/src/shared/loss.py (MIT)
        recsys-2023-tron/src/shared/sample.py (MIT)
        recsys-2024-1st/Utils/xgboost_functions.py (AGPL-3.0)
        sciona-atoms/research/13_research.pdf
"""

from __future__ import annotations

from collections import defaultdict
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import kurtosis, skew

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_als_item_update,
    witness_als_user_update,
    witness_bpr_max_loss,
    witness_co_occurrence_matrix,
    witness_cooccurrence_candidates,
    witness_in_batch_negative_sampling,
    witness_item_popularity_decay,
    witness_reciprocal_rank_fusion,
    witness_ranking_moments_extractor,
    witness_sampled_softmax_loss,
    witness_session_features,
    witness_uniform_negative_sampling,
    witness_user_item_affinity,
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


def _sessions_valid(sessions: list[list[int]], n_items: int) -> bool:
    if int(n_items) <= 0:
        return False
    for session in sessions:
        for item in session:
            if int(item) < 0 or int(item) >= int(n_items):
                return False
    return True


def _time_weights_valid(sessions: list[list[int]], time_weights: NDArray[np.float64] | None) -> bool:
    if time_weights is None:
        return True
    weights = np.asarray(time_weights, dtype=np.float64).reshape(-1)
    max_session_len = max((len(session) for session in sessions), default=0)
    return bool(weights.shape[0] >= max_session_len and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _sparse_square(matrix: csr_matrix) -> bool:
    sparse = csr_matrix(matrix)
    return bool(sparse.ndim == 2 and sparse.shape[0] == sparse.shape[1] and sparse.shape[0] > 0)


def _valid_item_ids(items: NDArray[np.int64], n_items: int) -> bool:
    values = np.asarray(items, dtype=np.int64).reshape(-1)
    return bool(values.size > 0 and np.all(values >= 0) and np.all(values < int(n_items)))


def _candidate_result_valid(result: NDArray[np.int64], k: int, n_items: int) -> bool:
    values = np.asarray(result, dtype=np.int64).reshape(-1)
    return bool(
        values.shape[0] <= int(k)
        and np.unique(values).shape[0] == values.shape[0]
        and np.all(values >= 0)
        and np.all(values < int(n_items))
    )


def _als_inputs_valid(
    interactions: csr_matrix,
    factors: NDArray[np.float64],
    gram: NDArray[np.float64],
) -> bool:
    sparse = csr_matrix(interactions)
    factor_matrix = np.asarray(factors, dtype=np.float64)
    gram_matrix = np.asarray(gram, dtype=np.float64)
    return bool(
        sparse.shape == (1, factor_matrix.shape[0])
        and sparse.nnz >= 1
        and factor_matrix.ndim == 2
        and factor_matrix.shape[1] >= 1
        and gram_matrix.shape == (factor_matrix.shape[1], factor_matrix.shape[1])
        and np.all(np.isfinite(sparse.data))
        and np.all(sparse.data > 0.0)
        and np.all(np.isfinite(factor_matrix))
        and np.all(np.isfinite(gram_matrix))
    )


def _als_vector_update(
    interactions: csr_matrix,
    factors: NDArray[np.float64],
    gram: NDArray[np.float64],
    regularization: float,
    alpha: float,
) -> NDArray[np.float64]:
    sparse = csr_matrix(interactions)
    factor_matrix = np.asarray(factors, dtype=np.float64)
    gram_matrix = np.asarray(gram, dtype=np.float64)
    rank = factor_matrix.shape[1]
    item_rows = factor_matrix[sparse.indices]
    confidence = 1.0 + float(alpha) * sparse.data.astype(np.float64)
    weighted_rows = item_rows * (confidence - 1.0)[:, None]
    lhs = gram_matrix + item_rows.T @ weighted_rows + float(regularization) * np.eye(rank)
    rhs = item_rows.T @ confidence
    return np.linalg.solve(lhs, rhs).astype(np.float64)


def _nonnegative_item_events(item_ids: NDArray[np.int64], timestamps: NDArray[np.float64]) -> bool:
    items = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    return bool(
        items.shape == times.shape
        and items.size > 0
        and np.all(items >= 0)
        and np.all(np.isfinite(times))
    )


def _session_inputs_valid(item_sequence: list[int], timestamps: NDArray[np.float64]) -> bool:
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    gaps = np.diff(times)
    return bool(
        len(item_sequence) == times.shape[0]
        and len(item_sequence) >= 1
        and all(int(item) >= 0 for item in item_sequence)
        and np.all(np.isfinite(times))
        and np.all(gaps > 0.0)
    )


def _affinity_timestamps_valid(user_item_timestamps: NDArray[np.float64], reference_time: float) -> bool:
    times = np.asarray(user_item_timestamps, dtype=np.float64).reshape(-1)
    return bool(
        np.isfinite(float(reference_time))
        and np.all(np.isfinite(times))
        and (times.size == 0 or float(reference_time) >= float(np.max(times)))
    )


def _rrf_lists_valid(ranked_lists: list[list[int]]) -> bool:
    return all(all(int(item) >= 0 for item in ranked_list) for ranked_list in ranked_lists)


def _feature_dict_has(result: dict[str, float], keys: set[str]) -> bool:
    return bool(set(result) == keys and all(np.isfinite(float(value)) for value in result.values()))


@register_atom(witness_co_occurrence_matrix)
@icontract.require(lambda sessions, n_items: _sessions_valid(sessions, n_items), "sessions must contain valid item ids")
@icontract.require(lambda sessions, time_weights: _time_weights_valid(sessions, time_weights), "time weights must cover session distances")
@icontract.ensure(lambda result, n_items: result.shape == (int(n_items), int(n_items)), "co-occurrence matrix must be square")
@icontract.ensure(lambda result: np.allclose(result.diagonal(), 0.0), "self co-occurrence must be zero")
def co_occurrence_matrix(
    sessions: list[list[int]],
    n_items: int,
    time_weights: NDArray[np.float64] | None = None,
) -> csr_matrix:
    """Build a symmetric sparse item-item co-occurrence matrix from sessions."""
    weights = None if time_weights is None else np.asarray(time_weights, dtype=np.float64).reshape(-1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for session in sessions:
        for left_pos, left_item in enumerate(session):
            for right_pos in range(left_pos + 1, len(session)):
                right_item = int(session[right_pos])
                left_id = int(left_item)
                if left_id == right_item:
                    continue
                distance = right_pos - left_pos
                weight = 1.0 if weights is None else float(weights[distance])
                if weight == 0.0:
                    continue
                rows.extend([left_id, right_item])
                cols.extend([right_item, left_id])
                data.extend([weight, weight])
    matrix = coo_matrix((data, (rows, cols)), shape=(int(n_items), int(n_items)), dtype=np.float64).tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


@register_atom(witness_cooccurrence_candidates)
@icontract.require(lambda cooccurrence: _sparse_square(cooccurrence), "cooccurrence must be a non-empty square sparse matrix")
@icontract.require(lambda user_items, cooccurrence: _valid_item_ids(user_items, csr_matrix(cooccurrence).shape[0]), "user_items must be valid indices")
@icontract.require(lambda k: int(k) > 0, "k must be positive")
@icontract.ensure(lambda result, k, cooccurrence: _candidate_result_valid(result, k, csr_matrix(cooccurrence).shape[0]), "candidate ids must be unique valid top-k indices")
@icontract.ensure(lambda result, user_items, filter_seen: (not filter_seen) or np.intersect1d(result, np.asarray(user_items, dtype=np.int64)).size == 0, "filtered candidates must exclude seen items")
def cooccurrence_candidates(
    user_items: NDArray[np.int64],
    cooccurrence: csr_matrix,
    k: int,
    filter_seen: bool = True,
) -> NDArray[np.int64]:
    """Return top scoring item candidates from rows linked to a user's history."""
    matrix = csr_matrix(cooccurrence)
    history = np.unique(np.asarray(user_items, dtype=np.int64).reshape(-1))
    scores = np.asarray(matrix[history].sum(axis=0), dtype=np.float64).reshape(-1)
    if filter_seen:
        scores[history] = -np.inf
    positive = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
    if positive.size == 0:
        return np.array([], dtype=np.int64)
    limit = min(int(k), positive.size)
    selected = positive[np.argpartition(scores[positive], -limit)[-limit:]]
    ordered = selected[np.argsort(-scores[selected], kind="mergesort")]
    return ordered.astype(np.int64)


@register_atom(witness_als_user_update)
@icontract.require(lambda user_interactions, item_factors, yt_y: _als_inputs_valid(user_interactions, item_factors, yt_y), "ALS user inputs must align")
@icontract.require(lambda regularization: np.isfinite(float(regularization)) and float(regularization) > 0.0, "regularization must be positive")
@icontract.require(lambda alpha: np.isfinite(float(alpha)) and float(alpha) >= 0.0, "alpha must be finite and non-negative")
@icontract.ensure(lambda result, item_factors: result.shape == (np.asarray(item_factors).shape[1],), "user factor must match latent rank")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "user factor must be finite")
def als_user_update(
    user_interactions: csr_matrix,
    item_factors: NDArray[np.float64],
    yt_y: NDArray[np.float64],
    regularization: float,
    alpha: float = 40.0,
) -> NDArray[np.float64]:
    """Solve one implicit-feedback ALS update for a single user vector."""
    return _als_vector_update(user_interactions, item_factors, yt_y, regularization, alpha)


@register_atom(witness_als_item_update)
@icontract.require(lambda item_interactions, user_factors, xt_x: _als_inputs_valid(item_interactions, user_factors, xt_x), "ALS item inputs must align")
@icontract.require(lambda regularization: np.isfinite(float(regularization)) and float(regularization) > 0.0, "regularization must be positive")
@icontract.require(lambda alpha: np.isfinite(float(alpha)) and float(alpha) >= 0.0, "alpha must be finite and non-negative")
@icontract.ensure(lambda result, user_factors: result.shape == (np.asarray(user_factors).shape[1],), "item factor must match latent rank")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "item factor must be finite")
def als_item_update(
    item_interactions: csr_matrix,
    user_factors: NDArray[np.float64],
    xt_x: NDArray[np.float64],
    regularization: float,
    alpha: float = 40.0,
) -> NDArray[np.float64]:
    """Solve one implicit-feedback ALS update for a single item vector."""
    return _als_vector_update(item_interactions, user_factors, xt_x, regularization, alpha)


@register_atom(witness_item_popularity_decay)
@icontract.require(lambda item_ids, timestamps: _nonnegative_item_events(item_ids, timestamps), "item ids and timestamps must be finite aligned arrays")
@icontract.require(lambda timestamps, reference_time: float(reference_time) >= float(np.max(np.asarray(timestamps, dtype=np.float64))), "reference_time must not precede events")
@icontract.require(lambda half_life: np.isfinite(float(half_life)) and float(half_life) > 0.0, "half_life must be positive")
@icontract.ensure(lambda result: result.ndim == 1 and np.all(np.isfinite(result)) and np.all(result >= 0.0), "popularity scores must be finite and non-negative")
def item_popularity_decay(
    item_ids: NDArray[np.int64],
    timestamps: NDArray[np.float64],
    reference_time: float,
    half_life: float,
) -> NDArray[np.float64]:
    """Aggregate item popularity with exponential time decay by item id."""
    items = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ages = float(reference_time) - times
    weights = np.power(0.5, ages / float(half_life))
    return np.bincount(items, weights=weights, minlength=int(np.max(items)) + 1).astype(np.float64)


@register_atom(witness_session_features)
@icontract.require(lambda item_sequence, timestamps: _session_inputs_valid(item_sequence, timestamps), "session items and timestamps must align chronologically")
@icontract.ensure(lambda result: _feature_dict_has(result, {"session_length", "unique_items", "avg_gap", "duration"}), "session feature dictionary has fixed finite keys")
def session_features(
    item_sequence: list[int],
    timestamps: NDArray[np.float64],
) -> dict[str, float]:
    """Extract scalar length, diversity, duration, and gap features for one session."""
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    gaps = np.diff(times)
    return {
        "session_length": float(len(item_sequence)),
        "unique_items": float(len(set(int(item) for item in item_sequence))),
        "avg_gap": float(np.mean(gaps)) if gaps.size else 0.0,
        "duration": float(times[-1] - times[0]) if times.size > 1 else 0.0,
    }


@register_atom(witness_user_item_affinity)
@icontract.require(lambda user_item_timestamps, reference_time: _affinity_timestamps_valid(user_item_timestamps, reference_time), "affinity timestamps must be finite and not after reference_time")
@icontract.ensure(lambda result: _feature_dict_has(result, {"interaction_count", "recency"}), "affinity feature dictionary has fixed finite keys")
@icontract.ensure(lambda result: result["interaction_count"] >= 0.0, "interaction count must be non-negative")
def user_item_affinity(
    user_item_timestamps: NDArray[np.float64],
    reference_time: float,
) -> dict[str, float]:
    """Summarize a pre-filtered user-item timestamp history for ranking models."""
    times = np.asarray(user_item_timestamps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return {"interaction_count": 0.0, "recency": -1.0}
    return {
        "interaction_count": float(times.size),
        "recency": float(reference_time) - float(np.max(times)),
    }


@register_atom(witness_reciprocal_rank_fusion)
@icontract.require(lambda ranked_lists: _rrf_lists_valid(ranked_lists), "ranked lists must contain non-negative ids")
@icontract.require(lambda k: int(k) > 0, "k must be positive")
@icontract.require(lambda top_n: int(top_n) > 0, "top_n must be positive")
@icontract.ensure(lambda result, top_n: len(result) <= int(top_n), "fused ranking must respect top_n")
@icontract.ensure(lambda result: len(result) == len(set(result)), "fused ranking must contain unique ids")
def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int = 100,
) -> list[int]:
    """Fuse ranked candidate lists using reciprocal rank fusion scores."""
    scores: defaultdict[int, float] = defaultdict(float)
    first_seen: dict[int, int] = {}
    order = 0
    for ranked_list in ranked_lists:
        seen_in_list: set[int] = set()
        for rank, raw_item in enumerate(ranked_list, start=1):
            item = int(raw_item)
            if item in seen_in_list:
                continue
            seen_in_list.add(item)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
            scores[item] += 1.0 / (float(k) + float(rank))
    ranked = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return ranked[: int(top_n)]


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
