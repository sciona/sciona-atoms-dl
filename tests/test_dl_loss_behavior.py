from __future__ import annotations

import numpy as np


def test_loss_atoms_import() -> None:
    from sciona.atoms.dl.loss import atoms

    for name in [
        "miss_penalty_loss",
        "qwk_loss",
        "ctc_loss",
        "focal_loss",
        "lovasz_softmax_loss",
        "dice_loss",
        "crps_score",
        "contrastive_loss",
        "triplet_loss",
        "label_smoothing_ce",
        "weighted_multitask_loss",
        "multimodal_nll_loss",
        "weighted_bce_loss",
    ]:
        assert callable(getattr(atoms, name))


def test_miss_penalty_loss_penalizes_missed_positives() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.01, 0.9, 0.02], dtype=np.float64)
    labels = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    result = miss_penalty_loss(predictions, labels, threshold=0.03)
    expected = -np.log(0.01 + 1e-8) + -np.log(0.02 + 1e-8)
    assert np.isclose(result, expected)


def test_qwk_loss_is_zero_for_perfect_ordinal_predictions() -> None:
    from sciona.atoms.dl.loss.atoms import qwk_loss

    predictions = np.eye(3, dtype=np.float64)
    targets = np.array([0, 1, 2], dtype=np.int64)
    assert np.isclose(qwk_loss(predictions, targets, 3), 0.0)


def test_ctc_loss_is_small_for_confident_single_label_alignment() -> None:
    from sciona.atoms.dl.loss.atoms import ctc_loss

    probs = np.array(
        [
            [[0.99, 0.01, 1e-8]],
            [[0.01, 0.98, 0.01]],
            [[0.99, 0.01, 1e-8]],
        ],
        dtype=np.float64,
    )
    log_probs = np.log(probs)
    targets = np.array([1], dtype=np.int64)
    input_lengths = np.array([3], dtype=np.int64)
    target_lengths = np.array([1], dtype=np.int64)
    assert ctc_loss(log_probs, targets, input_lengths, target_lengths) < 0.05


def test_focal_loss_downweights_easy_examples() -> None:
    from sciona.atoms.dl.loss.atoms import focal_loss

    targets = np.array([1.0, 0.0], dtype=np.float64)
    easy = np.array([0.95, 0.05], dtype=np.float64)
    hard = np.array([0.55, 0.45], dtype=np.float64)
    assert focal_loss(easy, targets) < focal_loss(hard, targets)


def test_lovasz_and_dice_losses_are_zero_for_perfect_masks() -> None:
    from sciona.atoms.dl.loss.atoms import dice_loss, lovasz_softmax_loss

    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert np.isclose(lovasz_softmax_loss(mask, mask), 0.0)
    assert np.isclose(dice_loss(mask, mask), 0.0)


def test_crps_score_is_zero_for_matching_step_cdf() -> None:
    from sciona.atoms.dl.loss.atoms import crps_score

    cdf = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64)
    true_values = np.array([2], dtype=np.int64)
    assert np.isclose(crps_score(cdf, true_values), 0.0)


def test_metric_learning_losses_respect_margin() -> None:
    from sciona.atoms.dl.loss.atoms import contrastive_loss, triplet_loss

    a = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    b = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    labels = np.array([1.0, 0.0], dtype=np.float64)
    assert np.isclose(contrastive_loss(a, b, labels, margin=1.0), 0.0)

    positive = np.array([[0.1, 0.0]], dtype=np.float64)
    negative = np.array([[2.0, 0.0]], dtype=np.float64)
    assert np.isclose(triplet_loss(a[:1], positive, negative, margin=0.5), 0.0)


def test_label_smoothing_ce_matches_manual_log_softmax() -> None:
    from sciona.atoms.dl.loss.atoms import label_smoothing_ce

    logits = np.array([[2.0, 0.0]], dtype=np.float64)
    targets = np.array([0], dtype=np.int64)
    epsilon = 0.1
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    smooth = np.array([[0.95, 0.05]], dtype=np.float64)
    expected = -float(np.sum(smooth * log_probs))
    assert np.isclose(label_smoothing_ce(logits, targets, epsilon), expected)


def test_weighted_multitask_loss_uses_fixed_weights() -> None:
    from sciona.atoms.dl.loss.atoms import weighted_multitask_loss

    assert weighted_multitask_loss([1.0, 2.0], [0.5, 0.5]) == 1.5


def test_multimodal_nll_loss_prefers_confident_matching_mode() -> None:
    from sciona.atoms.dl.loss.atoms import multimodal_nll_loss

    ground_truth = np.array([[[0.0, 0.0], [1.0, 1.0]]], dtype=np.float64)
    trajectories = np.array(
        [[[[0.0, 0.0], [1.0, 1.0]], [[5.0, 5.0], [6.0, 6.0]]]],
        dtype=np.float64,
    )
    confidences = np.array([[0.999999, 0.000001]], dtype=np.float64)
    assert multimodal_nll_loss(ground_truth, trajectories, confidences) < 1e-4


def test_weighted_bce_loss_matches_stable_formula() -> None:
    from sciona.atoms.dl.loss.atoms import weighted_bce_loss

    logits = np.array([0.0, 0.0], dtype=np.float64)
    targets = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([1.0, 2.0], dtype=np.float64)
    expected = np.mean(weights * np.log(2.0))
    assert np.isclose(weighted_bce_loss(logits, targets, weights), expected)
