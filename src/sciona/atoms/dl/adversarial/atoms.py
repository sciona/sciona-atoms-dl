"""Framework-agnostic adversarial attack primitives in pure numpy.

Implements targeted-attack variants and label inference building blocks from
the 1st-place solutions to the NIPS 2017 adversarial non-targeted and targeted
attack competitions. These atoms complement the gradient_attacks family in
sciona-atoms-ml with loss computation, std-normalized momentum, and ensemble
label inference.

All computation is pure numpy -- no TensorFlow dependency. The gradient itself
is assumed to be provided by the caller from whatever autodiff framework is
in use.

Source: adversarial-nontarget-1st/attack_iter.py (Apache 2.0)
        adversarial-target-1st/target_attack.py (Apache 2.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_auxiliary_logit_loss_fusion,
    witness_ensemble_prediction_label_inference,
    witness_std_normalized_momentum_gradient,
)


# ---------------------------------------------------------------------------
# Atom 1: Auxiliary logit loss fusion
# ---------------------------------------------------------------------------


@register_atom(witness_auxiliary_logit_loss_fusion)
@icontract.require(
    lambda main_logits, labels_onehot: main_logits.shape == labels_onehot.shape,
    "main_logits and labels_onehot must have the same shape",
)
@icontract.require(
    lambda main_logits, aux_logits: aux_logits is None or main_logits.shape == aux_logits.shape,
    "aux_logits must have the same shape as main_logits if provided",
)
@icontract.require(lambda aux_weight: aux_weight >= 0.0, "aux_weight must be non-negative")
@icontract.require(
    lambda labels_onehot: np.allclose(labels_onehot.sum(axis=1), 1.0, atol=0.01),
    "labels_onehot rows must sum to approximately 1",
)
def auxiliary_logit_loss_fusion(
    main_logits: NDArray[np.float64],
    labels_onehot: NDArray[np.float64],
    aux_logits: NDArray[np.float64] | None = None,
    aux_weight: float = 0.4,
) -> float:
    """Compute softmax cross-entropy loss with optional auxiliary logit branch.

    Fuses the main logit loss with a weighted auxiliary logit loss. The
    auxiliary branch typically comes from Inception-family AuxLogits heads.
    The original code uses TF softmax_cross_entropy with weight=0.4 for
    the auxiliary branch.

    Derived from attack_iter.py lines 176-183:
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits)
        cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, weights=0.4)
    """
    def _softmax_cross_entropy(logits: NDArray[np.float64], labels: NDArray[np.float64]) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_softmax = shifted - log_sum_exp
        return float(-np.mean(np.sum(labels * log_softmax, axis=1)))

    loss = _softmax_cross_entropy(main_logits, labels_onehot)

    if aux_logits is not None:
        loss += aux_weight * _softmax_cross_entropy(aux_logits, labels_onehot)

    return loss


# ---------------------------------------------------------------------------
# Atom 2: Std-normalized momentum gradient
# ---------------------------------------------------------------------------


@register_atom(witness_std_normalized_momentum_gradient)
@icontract.require(
    lambda gradient, previous_accumulated: gradient.shape == previous_accumulated.shape,
    "gradient and previous_accumulated must have the same shape",
)
@icontract.require(lambda momentum: momentum >= 0.0, "momentum must be non-negative")
@icontract.ensure(
    lambda gradient, result: result.shape == gradient.shape,
    "output must preserve the input shape",
)
def std_normalized_momentum_gradient(
    gradient: NDArray[np.float64],
    previous_accumulated: NDArray[np.float64],
    momentum: float = 1.0,
) -> NDArray[np.float64]:
    """Targeted-attack variant of momentum_gradient_accumulation.

    Uses std-normalization (preserves spatial structure) instead of
    L1-mean normalization (flattens to unit L1 ball). The double
    std-normalization -- once on the current gradient, once on the
    accumulated result -- is intentional: it prevents scale drift
    in targeted attacks where gradients must maintain directional
    consistency across many iterations.

    Derived from target_attack.py lines 155-158:
        noise = noise / std(noise)
        noise = momentum * grad + noise
        noise = noise / std(noise)
    """
    noise = gradient / (np.std(gradient) + 1e-12)
    accumulated = momentum * previous_accumulated + noise
    result: NDArray[np.float64] = accumulated / (np.std(accumulated) + 1e-12)
    return result


# ---------------------------------------------------------------------------
# Atom 3: Ensemble prediction label inference
# ---------------------------------------------------------------------------


@register_atom(witness_ensemble_prediction_label_inference)
@icontract.require(
    lambda predictions: len(predictions) >= 1,
    "need at least one prediction array",
)
@icontract.require(
    lambda iteration, previous_labels: iteration == 0 or previous_labels is not None,
    "previous_labels must be provided when iteration > 0",
)
@icontract.ensure(
    lambda predictions, result: result.shape == (predictions[0].shape[0],),
    "output must have shape (B,)",
)
def ensemble_prediction_label_inference(
    predictions: list[NDArray[np.float64]],
    iteration: int = 0,
    previous_labels: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Infer consensus labels from an ensemble of model predictions.

    On the first iteration (iteration == 0), computes argmax over the
    summed predictions from all models. On subsequent iterations, freezes
    the labels to prevent chasing a moving target during iterative
    adversarial attacks. This freeze-after-first-iteration pattern is
    critical: without it, the inferred label shifts as the adversarial
    perturbation changes model outputs, destabilizing the attack.

    Derived from attack_iter.py lines 162-167:
        pred = tf.argmax(sum_of_predictions, 1)
        first_round = tf.cast(tf.equal(i, 0), tf.int64)
        y = first_round * pred + (1 - first_round) * y
    """
    if iteration > 0 and previous_labels is not None:
        return previous_labels

    ensemble = sum(predictions)
    result: NDArray[np.int64] = np.argmax(ensemble, axis=1).astype(np.int64)
    return result
