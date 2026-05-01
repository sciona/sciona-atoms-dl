"""Loss function primitives in pure numpy.

The family contains reusable scalar objectives for competition pipelines:
ordinal agreement, sequence alignment, segmentation overlap, class imbalance,
metric learning, trajectory mixtures, calibration, and weighted aggregation.
All atoms are deterministic NumPy computations with explicit numeric inputs.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_contrastive_loss,
    witness_crps_score,
    witness_ctc_loss,
    witness_dice_loss,
    witness_focal_loss,
    witness_label_smoothing_ce,
    witness_lovasz_softmax_loss,
    witness_miss_penalty_loss,
    witness_multimodal_nll_loss,
    witness_quantile_spread_to_confidence,
    witness_qwk_loss,
    witness_triplet_loss,
    witness_weighted_bce_loss,
    witness_weighted_multitask_loss,
)


def _is_probability_array(values: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all((values >= 0.0) & (values <= 1.0)))


def _one_hot(targets: NDArray[np.int64], num_classes: int) -> NDArray[np.float64]:
    encoded = np.zeros((targets.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(targets.shape[0]), targets.astype(np.int64)] = 1.0
    return encoded


def _logsumexp(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("-inf")
    maximum = float(np.max(finite))
    return float(maximum + np.log(np.sum(np.exp(finite - maximum))))


def _ctc_min_timesteps(labels: NDArray[np.int64]) -> int:
    if labels.size == 0:
        return 0
    repeats = int(np.sum(labels[1:] == labels[:-1]))
    return int(labels.size + repeats)


def _ctc_batch_is_feasible(
    targets: NDArray[np.int64],
    input_lengths: NDArray[np.int64],
    target_lengths: NDArray[np.int64],
) -> bool:
    offset = 0
    for input_length, target_length in zip(input_lengths, target_lengths):
        labels = targets[offset : offset + int(target_length)]
        if int(input_length) < _ctc_min_timesteps(labels):
            return False
        offset += int(target_length)
    return True


def _ctc_single_loss(
    log_probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    blank: int = 0,
) -> float:
    extended = np.empty(labels.size * 2 + 1, dtype=np.int64)
    extended[0::2] = blank
    extended[1::2] = labels
    states = extended.size

    alpha = np.full(states, float("-inf"), dtype=np.float64)
    alpha[0] = log_probs[0, blank]
    if states > 1:
        alpha[1] = log_probs[0, extended[1]]

    for t in range(1, log_probs.shape[0]):
        previous = alpha.copy()
        current = np.full(states, float("-inf"), dtype=np.float64)
        for s in range(states):
            candidates = [previous[s]]
            if s - 1 >= 0:
                candidates.append(previous[s - 1])
            if s - 2 >= 0 and extended[s] != blank and extended[s] != extended[s - 2]:
                candidates.append(previous[s - 2])
            current[s] = log_probs[t, extended[s]] + _logsumexp(
                np.asarray(candidates, dtype=np.float64)
            )
        alpha = current

    terminal = alpha[-1] if states == 1 else _logsumexp(alpha[-2:])
    return float(-terminal)


def _lovasz_grad(labels_sorted: NDArray[np.float64]) -> NDArray[np.float64]:
    total_positive = float(np.sum(labels_sorted))
    intersection = total_positive - np.cumsum(labels_sorted)
    union = total_positive + np.cumsum(1.0 - labels_sorted)
    jaccard = 1.0 - intersection / np.maximum(union, 1e-12)
    if labels_sorted.size > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _stable_log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


@register_atom(witness_miss_penalty_loss)
@icontract.require(
    lambda predictions: _is_probability_array(predictions),
    "predictions must be probabilities in [0, 1]",
)
@icontract.require(
    lambda labels: np.all((labels == 0.0) | (labels == 1.0)),
    "labels must be binary",
)
@icontract.require(lambda threshold: 0.0 < threshold < 1.0, "threshold must be in (0, 1)")
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
    """Penalize positive examples predicted below a probability threshold."""
    mask = (labels > 0.5) & (predictions < threshold)
    if not np.any(mask):
        return 0.0
    return float(-np.sum(np.log(predictions[mask] + 1e-8)))


@register_atom(witness_qwk_loss)
@icontract.require(lambda num_classes: num_classes >= 2, "num_classes must be at least 2")
@icontract.require(
    lambda predictions, num_classes: predictions.ndim == 2 and predictions.shape[1] == num_classes,
    "predictions must have shape (n_examples, num_classes)",
)
@icontract.require(
    lambda predictions: _is_probability_array(predictions)
    and np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-6),
    "prediction rows must be probability distributions",
)
@icontract.require(
    lambda predictions, targets: targets.shape == (predictions.shape[0],),
    "targets must have one class id per prediction row",
)
@icontract.require(
    lambda targets, num_classes: np.all((targets >= 0) & (targets < num_classes)),
    "targets must be class ids in range",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def qwk_loss(
    predictions: NDArray[np.float64],
    targets: NDArray[np.int64],
    num_classes: int,
) -> float:
    """Compute differentiable quadratic weighted kappa loss for ordinal classes."""
    target_probs = _one_hot(targets, num_classes)
    observed = target_probs.T @ predictions
    expected = np.outer(np.sum(target_probs, axis=0), np.sum(predictions, axis=0))
    expected = expected / float(predictions.shape[0])
    class_ids = np.arange(num_classes, dtype=np.float64)
    weights = (class_ids[:, None] - class_ids[None, :]) ** 2
    weights = weights / float((num_classes - 1) ** 2)
    numerator = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * expected)) + 1e-12
    return numerator / denominator


@register_atom(witness_ctc_loss)
@icontract.require(
    lambda log_probs: log_probs.ndim == 3 and np.all(np.isfinite(log_probs)),
    "log_probs must have shape (time, batch, classes) and be finite",
)
@icontract.require(
    lambda log_probs: log_probs.shape[0] >= 1 and log_probs.shape[1] >= 1 and log_probs.shape[2] >= 2,
    "log_probs must include time, batch, and at least blank plus one label",
)
@icontract.require(
    lambda log_probs, input_lengths: input_lengths.shape == (log_probs.shape[1],)
    and np.all((input_lengths >= 1) & (input_lengths <= log_probs.shape[0])),
    "input_lengths must match batch size and stay within time axis",
)
@icontract.require(
    lambda target_lengths, input_lengths: target_lengths.shape == input_lengths.shape
    and np.all((target_lengths >= 0) & (target_lengths <= input_lengths)),
    "target_lengths must match batch size and not exceed input lengths",
)
@icontract.require(
    lambda targets, target_lengths: targets.ndim == 1 and int(np.sum(target_lengths)) == targets.shape[0],
    "targets must be a flat concatenation matching target_lengths",
)
@icontract.require(
    lambda targets, log_probs: np.all((targets >= 1) & (targets < log_probs.shape[2])),
    "targets must use labels in [1, num_classes); blank label 0 is reserved",
)
@icontract.require(
    lambda targets, input_lengths, target_lengths: _ctc_batch_is_feasible(
        targets, input_lengths, target_lengths
    ),
    "input lengths must allow repeated labels to be separated by blanks",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def ctc_loss(
    log_probs: NDArray[np.float64],
    targets: NDArray[np.int64],
    input_lengths: NDArray[np.int64],
    target_lengths: NDArray[np.int64],
) -> float:
    """Compute log-space CTC negative log-likelihood with blank label zero."""
    losses: list[float] = []
    offset = 0
    for batch_index in range(log_probs.shape[1]):
        time_steps = int(input_lengths[batch_index])
        target_count = int(target_lengths[batch_index])
        labels = targets[offset : offset + target_count].astype(np.int64)
        offset += target_count
        losses.append(_ctc_single_loss(log_probs[:time_steps, batch_index, :], labels))
    return float(np.mean(np.asarray(losses, dtype=np.float64)))


@register_atom(witness_focal_loss)
@icontract.require(
    lambda predictions: _is_probability_array(predictions),
    "predictions must be probabilities in [0, 1]",
)
@icontract.require(
    lambda targets: np.all((targets == 0.0) | (targets == 1.0)),
    "targets must be binary",
)
@icontract.require(
    lambda predictions, targets: predictions.shape == targets.shape,
    "predictions and targets must have the same shape",
)
@icontract.require(lambda alpha: 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]")
@icontract.require(lambda gamma: gamma >= 0.0, "gamma must be non-negative")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def focal_loss(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> float:
    """Compute binary focal loss for imbalanced classification probabilities."""
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    target_probs = np.where(targets == 1.0, clipped, 1.0 - clipped)
    alpha_weight = np.where(targets == 1.0, alpha, 1.0 - alpha)
    losses = -alpha_weight * ((1.0 - target_probs) ** gamma) * np.log(target_probs)
    return float(np.mean(losses))


@register_atom(witness_lovasz_softmax_loss)
@icontract.require(
    lambda probabilities: _is_probability_array(probabilities),
    "probabilities must be in [0, 1]",
)
@icontract.require(
    lambda targets: np.all((targets == 0.0) | (targets == 1.0)),
    "targets must be binary or one-hot",
)
@icontract.require(
    lambda probabilities, targets: probabilities.shape == targets.shape and probabilities.size >= 1,
    "probabilities and targets must have the same non-empty shape",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def lovasz_softmax_loss(
    probabilities: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> float:
    """Compute the Lovasz extension surrogate for Jaccard overlap loss."""
    labels = targets.astype(np.float64).reshape(-1)
    if np.sum(labels) == 0.0:
        errors = probabilities.reshape(-1)
    else:
        errors = np.abs(labels - probabilities.reshape(-1))
    order = np.argsort(-errors)
    errors_sorted = errors[order]
    labels_sorted = labels[order]
    return float(np.dot(errors_sorted, _lovasz_grad(labels_sorted)))


@register_atom(witness_dice_loss)
@icontract.require(
    lambda predictions: _is_probability_array(predictions),
    "predictions must be probabilities in [0, 1]",
)
@icontract.require(
    lambda targets: np.all((targets == 0.0) | (targets == 1.0)),
    "targets must be binary or one-hot",
)
@icontract.require(
    lambda predictions, targets: predictions.shape == targets.shape and predictions.size >= 1,
    "predictions and targets must have the same non-empty shape",
)
@icontract.require(lambda smooth: smooth > 0.0, "smooth must be positive")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0 + 1e-9, "loss must stay near [0, 1]")
def dice_loss(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    smooth: float = 1e-6,
) -> float:
    """Compute soft Dice loss from probability and target masks."""
    intersection = float(np.sum(predictions * targets))
    denominator = float(np.sum(predictions**2) + np.sum(targets**2))
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(1.0 - score)


@register_atom(witness_crps_score)
@icontract.require(
    lambda cdf_predictions: cdf_predictions.ndim >= 1 and _is_probability_array(cdf_predictions),
    "cdf_predictions must be probability values",
)
@icontract.require(
    lambda cdf_predictions: np.all(np.diff(cdf_predictions, axis=-1) >= -1e-6),
    "cdf_predictions must be non-decreasing along the bin axis",
)
@icontract.require(
    lambda cdf_predictions, true_values: true_values.shape == cdf_predictions.shape[:-1],
    "true_values must provide one observed bin index per CDF row",
)
@icontract.require(
    lambda cdf_predictions, true_values: np.all((true_values >= 0) & (true_values < cdf_predictions.shape[-1])),
    "true_values must be valid bin indices",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "score must be finite")
@icontract.ensure(lambda result: result >= 0.0, "score must be non-negative")
def crps_score(
    cdf_predictions: NDArray[np.float64],
    true_values: NDArray[np.int64],
) -> float:
    """Compute discrete CRPS against a step CDF at each observed bin index."""
    bins = np.arange(cdf_predictions.shape[-1], dtype=np.int64)
    step = (bins >= np.expand_dims(true_values.astype(np.int64), axis=-1)).astype(np.float64)
    return float(np.mean((cdf_predictions - step) ** 2))


@register_atom(witness_contrastive_loss)
@icontract.require(
    lambda embedding_a, embedding_b: embedding_a.shape == embedding_b.shape
    and embedding_a.ndim == 2
    and embedding_a.shape[0] >= 1,
    "embeddings must be non-empty matrices with the same shape",
)
@icontract.require(
    lambda labels, embedding_a: labels.shape == (embedding_a.shape[0],)
    and np.all((labels == 0.0) | (labels == 1.0)),
    "labels must contain one binary pair label per row",
)
@icontract.require(lambda margin: margin > 0.0, "margin must be positive")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def contrastive_loss(
    embedding_a: NDArray[np.float64],
    embedding_b: NDArray[np.float64],
    labels: NDArray[np.float64],
    margin: float,
) -> float:
    """Compute pairwise contrastive loss for same and different embedding pairs."""
    distances = np.linalg.norm(embedding_a - embedding_b, axis=1)
    positive = labels * (distances**2)
    negative = (1.0 - labels) * (np.maximum(margin - distances, 0.0) ** 2)
    return float(np.mean(positive + negative))


@register_atom(witness_triplet_loss)
@icontract.require(
    lambda anchor, positive, negative: anchor.shape == positive.shape == negative.shape
    and anchor.ndim == 2
    and anchor.shape[0] >= 1,
    "anchor, positive, and negative embeddings must be non-empty matrices",
)
@icontract.require(lambda margin: margin > 0.0, "margin must be positive")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def triplet_loss(
    anchor: NDArray[np.float64],
    positive: NDArray[np.float64],
    negative: NDArray[np.float64],
    margin: float,
) -> float:
    """Compute margin triplet loss over aligned anchor, positive, and negative rows."""
    positive_distance = np.linalg.norm(anchor - positive, axis=1)
    negative_distance = np.linalg.norm(anchor - negative, axis=1)
    return float(np.mean(np.maximum(positive_distance - negative_distance + margin, 0.0)))


@register_atom(witness_label_smoothing_ce)
@icontract.require(
    lambda logits: logits.ndim == 2 and logits.shape[0] >= 1 and logits.shape[1] >= 2,
    "logits must have shape (n_examples, num_classes)",
)
@icontract.require(
    lambda logits, targets: targets.shape == (logits.shape[0],)
    and np.all((targets >= 0) & (targets < logits.shape[1])),
    "targets must contain one class id per row",
)
@icontract.require(lambda epsilon: 0.0 <= epsilon < 1.0, "epsilon must be in [0, 1)")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def label_smoothing_ce(
    logits: NDArray[np.float64],
    targets: NDArray[np.int64],
    epsilon: float,
) -> float:
    """Compute cross-entropy after mixing one-hot labels with uniform mass."""
    num_classes = logits.shape[1]
    smooth_targets = (1.0 - epsilon) * _one_hot(targets, num_classes)
    smooth_targets += epsilon / float(num_classes)
    log_probs = _stable_log_softmax(logits)
    return float(-np.mean(np.sum(smooth_targets * log_probs, axis=1)))


@register_atom(witness_weighted_multitask_loss)
@icontract.require(lambda losses, weights: len(losses) == len(weights), "losses and weights must align")
@icontract.require(lambda losses: len(losses) >= 1, "at least one loss is required")
@icontract.require(lambda losses: all(loss >= 0.0 and math.isfinite(loss) for loss in losses), "losses must be finite and non-negative")
@icontract.require(lambda weights: all(weight >= 0.0 and math.isfinite(weight) for weight in weights), "weights must be finite and non-negative")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative when inputs are non-negative")
def weighted_multitask_loss(losses: list[float], weights: list[float]) -> float:
    """Combine scalar task losses with fixed non-negative task weights."""
    return float(sum(weight * loss for loss, weight in zip(losses, weights)))


@register_atom(witness_multimodal_nll_loss)
@icontract.require(
    lambda trajectories: trajectories.ndim == 4,
    "trajectories must have shape (batch, modes, time, coordinates)",
)
@icontract.require(
    lambda ground_truth, trajectories: ground_truth.shape == (
        trajectories.shape[0],
        trajectories.shape[2],
        trajectories.shape[3],
    ),
    "ground_truth must match the batch, time, and coordinate axes",
)
@icontract.require(
    lambda confidences, trajectories: confidences.shape == trajectories.shape[:2],
    "confidences must have shape (batch, modes)",
)
@icontract.require(
    lambda confidences: _is_probability_array(confidences)
    and np.all(confidences > 0.0)
    and np.allclose(np.sum(confidences, axis=1), 1.0, atol=1e-6),
    "confidences must be positive probability distributions",
)
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def multimodal_nll_loss(
    ground_truth: NDArray[np.float64],
    trajectories: NDArray[np.float64],
    confidences: NDArray[np.float64],
) -> float:
    """Compute mixture trajectory negative log-likelihood with unit variance."""
    residual = trajectories - ground_truth[:, None, :, :]
    mode_energy = 0.5 * np.sum(residual**2, axis=(2, 3))
    log_terms = np.log(confidences) - mode_energy
    sample_losses = -np.asarray([_logsumexp(row) for row in log_terms], dtype=np.float64)
    return float(np.mean(sample_losses))


@register_atom(witness_weighted_bce_loss)
@icontract.require(
    lambda logits, targets, weights: logits.shape == targets.shape == weights.shape and logits.size >= 1,
    "logits, targets, and weights must have the same non-empty shape",
)
@icontract.require(
    lambda targets: np.all((targets == 0.0) | (targets == 1.0)),
    "targets must be binary",
)
@icontract.require(lambda weights: np.all(weights >= 0.0), "weights must be non-negative")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "loss must be finite")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def weighted_bce_loss(
    logits: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Compute stable binary cross-entropy with per-sample weights."""
    per_element = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(weights * per_element))


@register_atom(witness_quantile_spread_to_confidence)
@icontract.require(
    lambda q_low, q_high: q_low.shape == q_high.shape,
    "q_low and q_high must have the same shape",
)
@icontract.require(lambda q_low: q_low.ndim >= 1, "q_low must be at least 1-D")
@icontract.require(lambda min_sigma: min_sigma > 0.0, "min_sigma must be positive")
@icontract.ensure(
    lambda result, q_low: result.shape == q_low.shape,
    "result must preserve shape",
)
@icontract.ensure(lambda result, min_sigma: np.all(result >= min_sigma), "sigma must be at least min_sigma")
def quantile_spread_to_confidence(
    q_low: NDArray[np.float64],
    q_high: NDArray[np.float64],
    min_sigma: float = 70.0,
) -> NDArray[np.float64]:
    """Convert quantile spread to confidence interval width, clipped.

    Takes the difference between upper and lower quantile predictions
    and clips to a minimum width, useful for converting quantile
    regression outputs into confidence-weighted loss terms.
    """
    sigma = np.maximum(q_high - q_low, min_sigma)
    return sigma
