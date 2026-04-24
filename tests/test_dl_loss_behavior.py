from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_loss_atoms_import() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    assert callable(miss_penalty_loss)


# ---------------------------------------------------------------------------
# miss_penalty_loss
# ---------------------------------------------------------------------------


def test_miss_penalty_loss_zero_when_no_misses() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.9, 0.8, 0.7])
    labels = np.array([1.0, 1.0, 1.0])
    result = miss_penalty_loss(predictions, labels, threshold=0.5)
    assert result == 0.0


def test_miss_penalty_loss_zero_when_no_positives() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.01, 0.02, 0.01])
    labels = np.array([0.0, 0.0, 0.0])
    result = miss_penalty_loss(predictions, labels, threshold=0.5)
    assert result == 0.0


def test_miss_penalty_loss_penalizes_missed_positives() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.01, 0.9, 0.02])
    labels = np.array([1.0, 1.0, 1.0])
    result = miss_penalty_loss(predictions, labels, threshold=0.03)
    # Only predictions[0]=0.01 and predictions[2]=0.02 are below threshold
    expected = -np.log(0.01 + 1e-8) + -np.log(0.02 + 1e-8)
    assert np.isclose(result, expected)


def test_miss_penalty_loss_non_negative() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.001, 0.002, 0.5])
    labels = np.array([1.0, 1.0, 0.0])
    result = miss_penalty_loss(predictions, labels, threshold=0.1)
    assert result >= 0.0


def test_miss_penalty_loss_single_element_miss() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.02])
    labels = np.array([1.0])
    result = miss_penalty_loss(predictions, labels, threshold=0.03)
    expected = -np.log(0.02 + 1e-8)
    assert np.isclose(result, expected)


def test_miss_penalty_loss_default_threshold() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    predictions = np.array([0.01, 0.05])
    labels = np.array([1.0, 1.0])
    # Default threshold is 0.03, so only 0.01 is below
    result = miss_penalty_loss(predictions, labels)
    expected = -np.log(0.01 + 1e-8)
    assert np.isclose(result, expected)


def test_miss_penalty_loss_higher_penalty_for_lower_prediction() -> None:
    from sciona.atoms.dl.loss.atoms import miss_penalty_loss

    preds_low = np.array([0.001])
    preds_high = np.array([0.02])
    labels = np.array([1.0])
    loss_low = miss_penalty_loss(preds_low, labels, threshold=0.03)
    loss_high = miss_penalty_loss(preds_high, labels, threshold=0.03)
    assert loss_low > loss_high
