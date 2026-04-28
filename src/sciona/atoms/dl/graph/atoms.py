"""Graph preprocessing and scheduling atoms in pure numpy.

Implements reusable graph utilities extracted from KDD 2020 graph-learning
solutions: degree bucketing for categorical graph features, feature
standardization with clipping, time-budget-aware complexity selection, and
symmetrically normalized adjacency smoothing.

All computation is pure numpy/scipy.

Source: kdd-2020-autograph-1st (Apache 2.0)
        kdd-2020-graph-adversarial-6th (MIT)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_adjacency_smoothing,
    witness_feature_clip_standardize,
    witness_node_degree_bucketing,
    witness_time_budget_estimator,
)


@register_atom(witness_node_degree_bucketing)
@icontract.require(lambda num_buckets: num_buckets >= 1, "num_buckets must be at least 1")
@icontract.require(
    lambda degrees: np.all(degrees >= 0),
    "degrees must be non-negative",
)
@icontract.ensure(
    lambda degrees, result: result.shape == degrees.shape,
    "output must preserve the input shape",
)
def node_degree_bucketing(
    degrees: NDArray[np.int64],
    num_buckets: int = 10,
) -> NDArray[np.int64]:
    """Map node degrees to log-spaced integer buckets.

    AutoGraph engineers categorical degree features to reduce hub overfitting.
    This atom captures the same intent with logarithmic buckets:
    `floor(log2(degree + 1))`, clipped to `num_buckets - 1`.
    """
    buckets = np.floor(np.log2(degrees.astype(np.float64) + 1.0)).astype(np.int64)
    return np.minimum(buckets, num_buckets - 1)


@register_atom(witness_feature_clip_standardize)
@icontract.require(lambda clip_range: clip_range > 0.0, "clip_range must be positive")
@icontract.require(
    lambda features: features.ndim == 2,
    "features must be a 2-D array",
)
@icontract.ensure(
    lambda features, result: result.shape == features.shape,
    "output must preserve the input shape",
)
def feature_clip_standardize(
    features: NDArray[np.float64],
    clip_range: float = 3.0,
) -> NDArray[np.float64]:
    """Standardize features column-wise, then clip to a bounded range.

    Derived from AutoGraph `process_gnn_data`, which computes z-scores and
    clips them to `[-3, 3]` before feeding the graph model.
    """
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    standardized = (features - means) / (stds + 1e-5)
    return np.clip(standardized, -clip_range, clip_range)


@register_atom(witness_time_budget_estimator)
@icontract.require(
    lambda elapsed_seconds: elapsed_seconds >= 0.0,
    "elapsed_seconds must be non-negative",
)
@icontract.require(
    lambda total_budget: total_budget > 0.0,
    "total_budget must be positive",
)
@icontract.require(
    lambda model_complexities: len(model_complexities) >= 1,
    "model_complexities must contain at least one element",
)
@icontract.require(
    lambda model_complexities: np.all(model_complexities >= 0.0),
    "model_complexities must be non-negative",
)
@icontract.ensure(lambda model_complexities, result: 0 <= result < len(model_complexities), "result must be a valid model index")
def time_budget_estimator(
    elapsed_seconds: float,
    total_budget: float,
    model_complexities: NDArray[np.float64],
) -> int:
    """Select the most complex model that still fits the remaining budget.

    Mirrors AutoGraph's runtime scheduling heuristic: reserve a small safety
    margin, estimate remaining time, and choose the largest feasible training
    budget. If none fit, fall back to the cheapest option.
    """
    remaining = max(total_budget - elapsed_seconds - 5.0, 0.0)
    feasible = np.flatnonzero(model_complexities <= remaining)
    if len(feasible) == 0:
        return int(np.argmin(model_complexities))
    best_local = feasible[np.argmax(model_complexities[feasible])]
    return int(best_local)


@register_atom(witness_adjacency_smoothing)
@icontract.require(
    lambda adjacency: adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1],
    "adjacency must be a square 2-D matrix",
)
@icontract.ensure(
    lambda adjacency, result: result.shape == adjacency.shape,
    "output must preserve the adjacency shape",
)
def adjacency_smoothing(
    adjacency: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute symmetrically normalized adjacency `D^(-1/2) A D^(-1/2)`.

    The upstream defense stack passes sparse adjacency into graph convolution
    layers that apply degree normalization internally. This atom exposes that
    low-pass smoothing step directly as a reusable numpy primitive.
    """
    degree = np.sum(adjacency, axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0.0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    scale = np.diag(inv_sqrt_degree)
    return scale @ adjacency @ scale
