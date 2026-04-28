"""Training strategy primitives in pure numpy.

Implements core training utilities from the 1st-place solution to the
Data Science Bowl 2017 lung cancer detection competition. The atoms decompose
reusable training building blocks: hard negative mining, size-aware
oversampling, and temperature-based proposal sampling.

All computation is pure numpy -- no PyTorch dependency. Tensors are assumed
to be converted to numpy arrays by the caller.

Source: dsb2017-1st/layers.py (MIT)
        dsb2017-1st/data_detector.py (MIT)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_online_hard_negative_mining,
    witness_size_aware_nodule_oversampling,
    witness_softmax_temperature_proposal_sampling,
    witness_ternary_search_threshold,
)


# ---------------------------------------------------------------------------
# Atom 1: Online hard negative mining
# ---------------------------------------------------------------------------


@register_atom(witness_online_hard_negative_mining)
@icontract.require(lambda num_hard: num_hard >= 1, "num_hard must be at least 1")
@icontract.require(
    lambda neg_scores: len(neg_scores) >= 1,
    "neg_scores must contain at least one element",
)
@icontract.ensure(
    lambda neg_scores, num_hard, result: len(result) == min(num_hard, len(neg_scores)),
    "result length must equal min(num_hard, len(neg_scores))",
)
def online_hard_negative_mining(
    neg_scores: NDArray[np.float64],
    num_hard: int,
) -> NDArray[np.int64]:
    """Select indices of the highest-scoring negatives for hard mining.

    Returns the indices that sort neg_scores in descending order, truncated
    to at most num_hard entries. These correspond to the negative examples
    the model is most confident about (i.e., the hardest negatives), which
    are the most informative for training.

    Derived from layers.py lines 149-153:
        _, idcs = torch.topk(neg_output, min(num_hard, ...))
    Reimplemented in numpy without PyTorch dependency.
    """
    indices: NDArray[np.int64] = np.argsort(neg_scores)[::-1][
        : min(num_hard, len(neg_scores))
    ].astype(np.int64)
    return indices


# ---------------------------------------------------------------------------
# Atom 2: Size-aware nodule oversampling
# ---------------------------------------------------------------------------


@register_atom(witness_size_aware_nodule_oversampling)
@icontract.require(
    lambda size_thresholds, repeat_counts: (
        size_thresholds is None
        and repeat_counts is None
    )
    or (
        size_thresholds is not None
        and repeat_counts is not None
        and len(size_thresholds) == len(repeat_counts)
    ),
    "size_thresholds and repeat_counts must have the same length",
)
@icontract.require(
    lambda size_thresholds: size_thresholds is None
    or all(
        size_thresholds[i] <= size_thresholds[i + 1]
        for i in range(len(size_thresholds) - 1)
    ),
    "size_thresholds must be sorted in ascending order",
)
@icontract.require(
    lambda bboxes, diameter_column: 0 <= diameter_column < bboxes.shape[1],
    "diameter_column must be a valid column index",
)
@icontract.ensure(lambda bboxes, result: result.ndim == 2 and result.shape[1] == bboxes.shape[1] and len(result) >= len(bboxes), "result must preserve columns and include originals")
def size_aware_nodule_oversampling(
    bboxes: NDArray[np.float64],
    diameter_column: int,
    size_thresholds: NDArray[np.float64] | None = None,
    repeat_counts: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Oversample bounding boxes based on nodule diameter thresholds.

    For each threshold, selects bboxes whose diameter exceeds that threshold
    and appends repeat_counts[i] copies. Larger nodules are rarer and more
    clinically significant, so they receive more copies to rebalance the
    training distribution.

    The result is the concatenation of the original bboxes with all
    oversampled copies.

    Derived from data_detector.py lines 60-70:
        for threshold, repeat in zip(sizes, repeats):
            sel = bboxes[bboxes[:, diameter_col] > threshold]
            bboxes = np.concatenate([bboxes] + [sel] * repeat)
    """
    if size_thresholds is None:
        size_thresholds = np.array([6.0, 30.0, 40.0], dtype=np.float64)
    if repeat_counts is None:
        repeat_counts = np.array([1, 2, 4], dtype=np.int64)

    parts: list[NDArray[np.float64]] = [bboxes]
    for threshold, count in zip(size_thresholds, repeat_counts):
        mask = bboxes[:, diameter_column] > threshold
        selected = bboxes[mask]
        if len(selected) > 0:
            for _ in range(int(count)):
                parts.append(selected)
    result: NDArray[np.float64] = np.concatenate(parts, axis=0)
    return result


# ---------------------------------------------------------------------------
# Atom 3: Softmax temperature proposal sampling
# ---------------------------------------------------------------------------


@register_atom(witness_softmax_temperature_proposal_sampling)
@icontract.require(lambda temperature: temperature > 0.0, "temperature must be positive")
@icontract.require(lambda k: k >= 1, "k must be at least 1")
@icontract.require(
    lambda scores, k: k <= len(scores),
    "k must not exceed the number of scores",
)
@icontract.ensure(
    lambda k, result: len(result) == k,
    "result length must equal k",
)
@icontract.ensure(
    lambda result: len(set(result.tolist())) == len(result),
    "result indices must be unique",
)
def softmax_temperature_proposal_sampling(
    scores: NDArray[np.float64],
    k: int,
    temperature: float = 1.0,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Sample k indices without replacement using temperature-scaled softmax.

    Applies temperature scaling to the raw scores before converting to
    probabilities via softmax. Higher temperature flattens the distribution
    (more exploration), lower temperature sharpens it (more exploitation).
    Indices are sampled without replacement according to the resulting
    probability distribution.

    This is a general-purpose proposal mechanism useful in training
    pipelines where deterministic top-k selection would reduce diversity.
    """
    rng = np.random.default_rng(random_state)
    scaled = scores / temperature
    shifted = scaled - np.max(scaled)
    exp_scores = np.exp(shifted)
    probs: NDArray[np.float64] = exp_scores / np.sum(exp_scores)
    indices: NDArray[np.int64] = rng.choice(
        len(scores), size=k, replace=False, p=probs
    ).astype(np.int64)
    return indices


# ---------------------------------------------------------------------------
# Atom 4: Ternary search threshold
# ---------------------------------------------------------------------------


@register_atom(witness_ternary_search_threshold)
@icontract.require(
    lambda scores, labels: scores.shape == labels.shape,
    "scores and labels must have the same shape",
)
@icontract.require(
    lambda n_iterations: n_iterations >= 1,
    "n_iterations must be at least 1",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "threshold must be finite")
def ternary_search_threshold(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    metric_fn: Callable[[NDArray[np.int64], NDArray[np.int64]], float],
    n_iterations: int = 50,
) -> float:
    """Find a near-optimal score threshold using ternary search.

    Generalizes BirdCLEF's baseline threshold tuning, which repeatedly
    compares two interior thresholds and keeps the more promising interval.
    `metric_fn` is expected to consume `(labels, predictions)`.
    """
    lower = float(np.min(scores))
    upper = float(np.max(scores))

    if lower == upper:
        return lower

    def objective(threshold: float) -> float:
        predictions = (scores > threshold).astype(np.int64)
        return float(metric_fn(labels, predictions))

    for _ in range(n_iterations):
        th1 = (2.0 * lower + upper) / 3.0
        th2 = (lower + 2.0 * upper) / 3.0
        if objective(th1) < objective(th2):
            lower = th1
        else:
            upper = th2

    return (lower + upper) / 2.0
