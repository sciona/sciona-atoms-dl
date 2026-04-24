"""Ghost witnesses for training strategy atoms.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

from collections.abc import Callable

from sciona.ghost.abstract import AbstractArray


def witness_online_hard_negative_mining(
    neg_scores: AbstractArray,
    num_hard: int,
) -> AbstractArray:
    """Ghost witness for online hard negative mining.

    Takes a 1-D score array and a count, returns an integer index array
    of length min(num_hard, len(neg_scores)) with the highest-scoring
    negative indices.
    """
    return neg_scores


def witness_size_aware_nodule_oversampling(
    bboxes: AbstractArray,
    diameter_column: int,
    size_thresholds: AbstractArray | None = None,
    repeat_counts: AbstractArray | None = None,
) -> AbstractArray:
    """Ghost witness for size-aware nodule oversampling.

    Takes a 2-D bounding box array, a column index, thresholds, and
    repeat counts. Returns a 2-D array with the same number of columns
    but more rows (original + oversampled copies).
    """
    return bboxes


def witness_softmax_temperature_proposal_sampling(
    scores: AbstractArray,
    k: int,
    temperature: float = 1.0,
    random_state: int | None = None,
) -> AbstractArray:
    """Ghost witness for softmax temperature proposal sampling.

    Takes a 1-D score array, a count k, temperature, and optional seed.
    Returns an integer index array of length k sampled without replacement
    from the temperature-scaled softmax distribution.
    """
    return scores


def witness_ternary_search_threshold(
    scores: AbstractArray,
    labels: AbstractArray,
    metric_fn: Callable[[AbstractArray, AbstractArray], float],
    n_iterations: int = 50,
) -> float:
    """Ghost witness for ternary threshold search."""
    return 0.0
