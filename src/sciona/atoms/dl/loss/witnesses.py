"""Ghost witnesses for loss function atoms.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_miss_penalty_loss(
    predictions: AbstractArray,
    labels: AbstractArray,
    threshold: float = 0.03,
) -> float:
    """Ghost witness for miss penalty loss.

    Takes prediction and label arrays of the same shape and a threshold,
    returns a non-negative scalar loss penalizing missed positives.
    """
    return 0.0
