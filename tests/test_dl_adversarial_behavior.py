from __future__ import annotations

import numpy as np
import pytest


def test_adversarial_import() -> None:
    from sciona.atoms.dl.adversarial.atoms import (
        auxiliary_logit_loss_fusion,
        ensemble_prediction_label_inference,
        std_normalized_momentum_gradient,
    )
    assert callable(auxiliary_logit_loss_fusion)
    assert callable(std_normalized_momentum_gradient)
    assert callable(ensemble_prediction_label_inference)


# ---------------------------------------------------------------------------
# auxiliary_logit_loss_fusion
# ---------------------------------------------------------------------------


def test_aux_loss_fusion_main_only() -> None:
    from sciona.atoms.dl.adversarial.atoms import auxiliary_logit_loss_fusion

    # Uniform logits => cross-entropy = log(C)
    logits = np.zeros((2, 4), dtype=np.float64)
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    loss = auxiliary_logit_loss_fusion(logits, labels)
    expected = np.log(4.0)
    assert abs(loss - expected) < 1e-6


def test_aux_loss_fusion_with_aux_logits() -> None:
    from sciona.atoms.dl.adversarial.atoms import auxiliary_logit_loss_fusion

    logits = np.zeros((2, 3), dtype=np.float64)
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    aux = np.zeros((2, 3), dtype=np.float64)
    loss = auxiliary_logit_loss_fusion(logits, labels, aux_logits=aux, aux_weight=0.4)
    expected = np.log(3.0) + 0.4 * np.log(3.0)
    assert abs(loss - expected) < 1e-6


def test_aux_loss_fusion_confident_main_logits() -> None:
    from sciona.atoms.dl.adversarial.atoms import auxiliary_logit_loss_fusion

    # Confident logits for correct class => near-zero loss
    logits = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)
    labels = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    loss = auxiliary_logit_loss_fusion(logits, labels)
    assert loss < 1e-6


def test_aux_loss_fusion_aux_weight_zero_ignores_aux() -> None:
    from sciona.atoms.dl.adversarial.atoms import auxiliary_logit_loss_fusion

    logits = np.zeros((2, 3), dtype=np.float64)
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    aux = np.full((2, 3), 1000.0, dtype=np.float64)  # huge aux logits
    loss_no_aux = auxiliary_logit_loss_fusion(logits, labels)
    loss_zero_w = auxiliary_logit_loss_fusion(logits, labels, aux_logits=aux, aux_weight=0.0)
    assert abs(loss_no_aux - loss_zero_w) < 1e-6


# ---------------------------------------------------------------------------
# std_normalized_momentum_gradient
# ---------------------------------------------------------------------------


def test_std_momentum_shape_preserved() -> None:
    from sciona.atoms.dl.adversarial.atoms import std_normalized_momentum_gradient

    grad = np.random.default_rng(42).standard_normal((3, 4))
    prev = np.zeros_like(grad)
    result = std_normalized_momentum_gradient(grad, prev, momentum=1.0)
    assert result.shape == grad.shape


def test_std_momentum_zero_momentum_ignores_history() -> None:
    from sciona.atoms.dl.adversarial.atoms import std_normalized_momentum_gradient

    rng = np.random.default_rng(42)
    grad = rng.standard_normal((2, 3))
    prev = np.full((2, 3), 100.0, dtype=np.float64)
    result = std_normalized_momentum_gradient(grad, prev, momentum=0.0)
    # With zero momentum: noise = grad / std(grad), then result = noise / std(noise)
    noise = grad / (np.std(grad) + 1e-12)
    expected = noise / (np.std(noise) + 1e-12)
    np.testing.assert_allclose(result, expected)


def test_std_momentum_double_normalization() -> None:
    from sciona.atoms.dl.adversarial.atoms import std_normalized_momentum_gradient

    rng = np.random.default_rng(0)
    grad = rng.standard_normal((4, 8))
    prev = rng.standard_normal((4, 8)) * 5.0
    result = std_normalized_momentum_gradient(grad, prev, momentum=1.0)
    # Output std should be approximately 1 due to final normalization
    assert abs(np.std(result) - 1.0) < 1e-6


def test_std_momentum_differs_from_l1_normalization() -> None:
    """Verify that std-normalization gives a different result from L1-mean."""
    from sciona.atoms.dl.adversarial.atoms import std_normalized_momentum_gradient

    rng = np.random.default_rng(7)
    grad = rng.standard_normal((3, 5))
    prev = rng.standard_normal((3, 5))

    result_std = std_normalized_momentum_gradient(grad, prev, momentum=1.0)

    # L1-norm variant for comparison
    l1_noise = grad / (np.mean(np.abs(grad)) + 1e-12)
    l1_accumulated = 1.0 * prev + l1_noise
    # They should NOT be equal
    assert not np.allclose(result_std, l1_accumulated)


# ---------------------------------------------------------------------------
# ensemble_prediction_label_inference
# ---------------------------------------------------------------------------


def test_ensemble_label_first_iteration() -> None:
    from sciona.atoms.dl.adversarial.atoms import ensemble_prediction_label_inference

    preds = [
        np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64),
        np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64),
    ]
    labels = ensemble_prediction_label_inference(preds, iteration=0)
    np.testing.assert_array_equal(labels, [1, 0])


def test_ensemble_label_freezes_after_first() -> None:
    from sciona.atoms.dl.adversarial.atoms import ensemble_prediction_label_inference

    preds = [
        np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64),
    ]
    prev = np.array([0, 1], dtype=np.int64)
    labels = ensemble_prediction_label_inference(preds, iteration=1, previous_labels=prev)
    # Should return previous_labels, not recompute
    np.testing.assert_array_equal(labels, [0, 1])


def test_ensemble_label_output_shape() -> None:
    from sciona.atoms.dl.adversarial.atoms import ensemble_prediction_label_inference

    rng = np.random.default_rng(42)
    preds = [rng.random((8, 1001)) for _ in range(5)]
    labels = ensemble_prediction_label_inference(preds, iteration=0)
    assert labels.shape == (8,)
    assert labels.dtype == np.int64


def test_ensemble_label_consensus_from_multiple_models() -> None:
    from sciona.atoms.dl.adversarial.atoms import ensemble_prediction_label_inference

    # Model 1 says class 0, model 2 says class 2, model 3 says class 2
    preds = [
        np.array([[10.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 10.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 10.0]], dtype=np.float64),
    ]
    labels = ensemble_prediction_label_inference(preds, iteration=0)
    np.testing.assert_array_equal(labels, [2])  # majority wins
