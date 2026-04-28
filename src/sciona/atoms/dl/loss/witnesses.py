"""Ghost witnesses for loss function atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_miss_penalty_loss(
    predictions: AbstractArray,
    labels: AbstractArray,
    threshold: float = 0.03,
) -> float:
    """Mirror a thresholded positive miss penalty as a scalar loss."""
    return 0.0


def witness_qwk_loss(
    predictions: AbstractArray,
    targets: AbstractArray,
    num_classes: int,
) -> float:
    """Mirror ordinal probability agreement as a scalar loss."""
    return 0.0


def witness_ctc_loss(
    log_probs: AbstractArray,
    targets: AbstractArray,
    input_lengths: AbstractArray,
    target_lengths: AbstractArray,
) -> float:
    """Mirror CTC sequence alignment likelihood as a scalar loss."""
    return 0.0


def witness_focal_loss(
    predictions: AbstractArray,
    targets: AbstractArray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> float:
    """Mirror modulated binary cross-entropy as a scalar loss."""
    return 0.0


def witness_lovasz_softmax_loss(
    probabilities: AbstractArray,
    targets: AbstractArray,
) -> float:
    """Mirror Lovasz Jaccard surrogate loss over flattened masks."""
    return 0.0


def witness_dice_loss(
    predictions: AbstractArray,
    targets: AbstractArray,
    smooth: float = 1e-6,
) -> float:
    """Mirror soft overlap loss for probability masks."""
    return 0.0


def witness_crps_score(
    cdf_predictions: AbstractArray,
    true_values: AbstractArray,
) -> float:
    """Mirror squared CDF distance against observed bin indices."""
    return 0.0


def witness_contrastive_loss(
    embedding_a: AbstractArray,
    embedding_b: AbstractArray,
    labels: AbstractArray,
    margin: float,
) -> float:
    """Mirror pairwise metric-learning loss over embedding rows."""
    return 0.0


def witness_triplet_loss(
    anchor: AbstractArray,
    positive: AbstractArray,
    negative: AbstractArray,
    margin: float,
) -> float:
    """Mirror margin ranking loss over aligned embedding triples."""
    return 0.0


def witness_label_smoothing_ce(
    logits: AbstractArray,
    targets: AbstractArray,
    epsilon: float,
) -> float:
    """Mirror smoothed-target categorical cross-entropy as a scalar."""
    return 0.0


def witness_weighted_multitask_loss(losses: list[float], weights: list[float]) -> float:
    """Mirror fixed weighted aggregation of scalar task losses."""
    return 0.0


def witness_multimodal_nll_loss(
    ground_truth: AbstractArray,
    trajectories: AbstractArray,
    confidences: AbstractArray,
) -> float:
    """Mirror mixture trajectory likelihood as a scalar loss."""
    return 0.0


def witness_weighted_bce_loss(
    logits: AbstractArray,
    targets: AbstractArray,
    weights: AbstractArray,
) -> float:
    """Mirror weighted binary cross-entropy over matching arrays."""
    return 0.0
