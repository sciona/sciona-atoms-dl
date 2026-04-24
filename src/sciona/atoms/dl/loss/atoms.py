"""Loss function primitives in pure numpy.

Implements a miss-penalty loss from the 1st-place solution to the
Data Science Bowl 2017 lung cancer detection competition. The atom captures
the penalty for high-confidence misses where the model predicts below a
threshold on positive examples.

All computation is pure numpy -- no PyTorch dependency.

Source: dsb2017-1st/layers.py (MIT)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_miss_penalty_loss


# ---------------------------------------------------------------------------
# Atom 1: Miss penalty loss
# ---------------------------------------------------------------------------


@register_atom(witness_miss_penalty_loss)
@icontract.require(
    lambda predictions: np.all(predictions >= 0.0) and np.all(predictions <= 1.0),
    "predictions must be in [0, 1]",
)
@icontract.require(
    lambda labels: np.all((labels == 0.0) | (labels == 1.0)),
    "labels must be binary (0 or 1)",
)
@icontract.require(
    lambda threshold: 0.0 < threshold < 1.0,
    "threshold must be in (0, 1)",
)
@icontract.require(
    lambda predictions, labels: predictions.shape == labels.shape,
    "predictions and labels must have the same shape",
)
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def miss_penalty_loss(
    predictions: NDArray[np.float64],
    labels: NDArray[np.float64],
    threshold: float = 0.03,
) -> float:
    """Compute penalty loss for missed positive predictions.

    Penalizes cases where the model predicts below threshold on positive
    examples (labels > 0.5). The loss is the negative log-likelihood of
    the predicted probability for these missed positives, encouraging
    the model to push predictions above the threshold for true positives.

    If no elements satisfy the miss condition, returns 0.0.

    Derived from layers.py lines 155-217:
        Conceptual miss-penalty formula extracted via CDG analysis of
        the focal/hard-example loss structure.
    """
    mask = (labels > 0.5) & (predictions < threshold)
    if not np.any(mask):
        return 0.0
    loss: float = float(-np.sum(np.log(predictions[mask] + 1e-8)))
    return loss
