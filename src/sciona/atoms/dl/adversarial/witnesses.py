"""Ghost witnesses for adversarial attack atoms.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_auxiliary_logit_loss_fusion(
    main_logits: AbstractArray,
    labels_onehot: AbstractArray,
    aux_logits: AbstractArray | None = None,
    aux_weight: float = 0.4,
) -> float:
    """Ghost witness for auxiliary logit loss fusion.

    Takes main logits and one-hot labels of the same shape, an optional
    auxiliary logit array of the same shape, and a non-negative weight.
    Returns a scalar loss value combining softmax cross-entropy on both
    logit branches.
    """
    return 0.0


def witness_std_normalized_momentum_gradient(
    gradient: AbstractArray,
    previous_accumulated: AbstractArray,
    momentum: float = 1.0,
) -> AbstractArray:
    """Ghost witness for std-normalized momentum gradient.

    Takes two same-shape arrays and a scalar momentum, returns an array
    of the same shape. The output is the double-std-normalized momentum
    accumulation of the current gradient and previous accumulation.
    """
    return gradient


def witness_ensemble_prediction_label_inference(
    predictions: list[AbstractArray],
    iteration: int = 0,
    previous_labels: AbstractArray | None = None,
) -> AbstractArray:
    """Ghost witness for ensemble prediction label inference.

    Takes a list of same-shape prediction arrays, an iteration counter,
    and optional previous labels. Returns a 1-D integer array of
    consensus label indices with shape (B,).
    """
    return predictions[0]
