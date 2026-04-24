from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_training_atoms_import() -> None:
    from sciona.atoms.dl.training.atoms import (
        online_hard_negative_mining,
        size_aware_nodule_oversampling,
        softmax_temperature_proposal_sampling,
        ternary_search_threshold,
    )

    assert callable(online_hard_negative_mining)
    assert callable(size_aware_nodule_oversampling)
    assert callable(softmax_temperature_proposal_sampling)
    assert callable(ternary_search_threshold)


# ---------------------------------------------------------------------------
# online_hard_negative_mining
# ---------------------------------------------------------------------------


def test_hard_negative_mining_returns_top_indices() -> None:
    from sciona.atoms.dl.training.atoms import online_hard_negative_mining

    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    result = online_hard_negative_mining(scores, num_hard=3)
    assert len(result) == 3
    # Highest scores are at indices 1, 3, 2
    assert result[0] == 1
    assert result[1] == 3
    assert result[2] == 2


def test_hard_negative_mining_clamps_to_array_length() -> None:
    from sciona.atoms.dl.training.atoms import online_hard_negative_mining

    scores = np.array([0.3, 0.7])
    result = online_hard_negative_mining(scores, num_hard=10)
    assert len(result) == 2


def test_hard_negative_mining_single_element() -> None:
    from sciona.atoms.dl.training.atoms import online_hard_negative_mining

    scores = np.array([0.5])
    result = online_hard_negative_mining(scores, num_hard=1)
    assert len(result) == 1
    assert result[0] == 0


def test_hard_negative_mining_result_dtype() -> None:
    from sciona.atoms.dl.training.atoms import online_hard_negative_mining

    scores = np.array([0.1, 0.9, 0.5])
    result = online_hard_negative_mining(scores, num_hard=2)
    assert result.dtype == np.int64


# ---------------------------------------------------------------------------
# size_aware_nodule_oversampling
# ---------------------------------------------------------------------------


def test_oversampling_preserves_original() -> None:
    from sciona.atoms.dl.training.atoms import size_aware_nodule_oversampling

    bboxes = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 35.0], [5.0, 6.0, 50.0]])
    result = size_aware_nodule_oversampling(bboxes, diameter_column=2)
    # Original 3 rows are always present
    assert np.array_equal(result[:3], bboxes)


def test_oversampling_increases_rows() -> None:
    from sciona.atoms.dl.training.atoms import size_aware_nodule_oversampling

    bboxes = np.array([[1.0, 2.0, 35.0], [3.0, 4.0, 50.0]])
    result = size_aware_nodule_oversampling(
        bboxes,
        diameter_column=2,
        size_thresholds=np.array([30.0]),
        repeat_counts=np.array([2]),
    )
    # Both rows exceed 30, so 2 copies of both are appended
    assert len(result) == 2 + 2 * 2


def test_oversampling_no_match_returns_original() -> None:
    from sciona.atoms.dl.training.atoms import size_aware_nodule_oversampling

    bboxes = np.array([[1.0, 2.0, 3.0]])
    result = size_aware_nodule_oversampling(
        bboxes,
        diameter_column=2,
        size_thresholds=np.array([100.0]),
        repeat_counts=np.array([5]),
    )
    assert len(result) == 1
    assert np.array_equal(result, bboxes)


def test_oversampling_multiple_thresholds() -> None:
    from sciona.atoms.dl.training.atoms import size_aware_nodule_oversampling

    bboxes = np.array([
        [0.0, 0.0, 5.0],   # below all thresholds
        [0.0, 0.0, 10.0],  # above 6.0 only
        [0.0, 0.0, 45.0],  # above all thresholds
    ])
    result = size_aware_nodule_oversampling(
        bboxes,
        diameter_column=2,
        size_thresholds=np.array([6.0, 30.0, 40.0]),
        repeat_counts=np.array([1, 2, 4]),
    )
    # Original: 3
    # threshold=6: rows 1,2 match -> 1 copy each = +2
    # threshold=30: row 2 matches -> 2 copies = +2
    # threshold=40: row 2 matches -> 4 copies = +4
    assert len(result) == 3 + 2 + 2 + 4


# ---------------------------------------------------------------------------
# softmax_temperature_proposal_sampling
# ---------------------------------------------------------------------------


def test_softmax_sampling_returns_k_unique_indices() -> None:
    from sciona.atoms.dl.training.atoms import softmax_temperature_proposal_sampling

    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = softmax_temperature_proposal_sampling(scores, k=3, random_state=42)
    assert len(result) == 3
    assert len(set(result.tolist())) == 3


def test_softmax_sampling_deterministic_with_seed() -> None:
    from sciona.atoms.dl.training.atoms import softmax_temperature_proposal_sampling

    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r1 = softmax_temperature_proposal_sampling(scores, k=3, random_state=123)
    r2 = softmax_temperature_proposal_sampling(scores, k=3, random_state=123)
    assert np.array_equal(r1, r2)


def test_softmax_sampling_low_temperature_favors_top() -> None:
    from sciona.atoms.dl.training.atoms import softmax_temperature_proposal_sampling

    scores = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
    # Very low temperature should almost always pick index 4
    result = softmax_temperature_proposal_sampling(
        scores, k=1, temperature=0.01, random_state=0
    )
    assert result[0] == 4


def test_softmax_sampling_k_equals_n() -> None:
    from sciona.atoms.dl.training.atoms import softmax_temperature_proposal_sampling

    scores = np.array([1.0, 2.0, 3.0])
    result = softmax_temperature_proposal_sampling(scores, k=3, random_state=42)
    assert len(result) == 3
    assert set(result.tolist()) == {0, 1, 2}


def test_softmax_sampling_result_dtype() -> None:
    from sciona.atoms.dl.training.atoms import softmax_temperature_proposal_sampling

    scores = np.array([1.0, 2.0, 3.0])
    result = softmax_temperature_proposal_sampling(scores, k=2, random_state=0)
    assert result.dtype == np.int64


# ---------------------------------------------------------------------------
# ternary_search_threshold
# ---------------------------------------------------------------------------


def test_ternary_search_threshold_finds_reasonable_cutoff() -> None:
    from sciona.atoms.dl.training.atoms import ternary_search_threshold

    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    threshold = ternary_search_threshold(
        scores,
        labels,
        accuracy_metric,
        n_iterations=40,
    )
    assert 0.2 < threshold < 0.8


def test_ternary_search_threshold_handles_constant_scores() -> None:
    from sciona.atoms.dl.training.atoms import ternary_search_threshold

    scores = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    labels = np.array([1, 0, 1], dtype=np.int64)

    def dummy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    threshold = ternary_search_threshold(scores, labels, dummy_metric)
    assert threshold == 0.5
