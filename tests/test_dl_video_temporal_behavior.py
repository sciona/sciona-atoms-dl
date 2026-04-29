from __future__ import annotations

import numpy as np
import pytest
import icontract


def test_video_temporal_import() -> None:
    from sciona.atoms.dl.video_temporal.atoms import (
        sample_frame_indices,
        sliding_windows,
        stack_adjacent_frames,
        temporal_attention_pool,
        temporal_max_pool,
        temporal_mean_pool,
        temporal_median_filter,
        temporal_unroll,
        uniform_sample_indices,
    )

    assert callable(sample_frame_indices)
    assert callable(uniform_sample_indices)
    assert callable(temporal_mean_pool)
    assert callable(temporal_max_pool)
    assert callable(temporal_attention_pool)
    assert callable(temporal_median_filter)
    assert callable(sliding_windows)
    assert callable(stack_adjacent_frames)
    assert callable(temporal_unroll)


def test_sample_frame_indices_maps_target_fps_to_native_frames() -> None:
    from sciona.atoms.dl.video_temporal.atoms import sample_frame_indices

    result = sample_frame_indices(total_frames=10, target_fps=1.0, video_fps=2.0)
    assert result.tolist() == [0, 2, 4, 6, 8]


def test_sample_frame_indices_allows_upsampling_with_clipped_duplicates() -> None:
    from sciona.atoms.dl.video_temporal.atoms import sample_frame_indices

    result = sample_frame_indices(total_frames=3, target_fps=4.0, video_fps=2.0)
    assert result.tolist() == [0, 0, 1, 2, 2, 2]


def test_uniform_sample_indices_spans_video() -> None:
    from sciona.atoms.dl.video_temporal.atoms import uniform_sample_indices

    assert uniform_sample_indices(total_frames=10, n_samples=4).tolist() == [0, 3, 6, 9]
    assert uniform_sample_indices(total_frames=10, n_samples=1).tolist() == [0]


def test_temporal_pooling_aggregates_over_frame_axis() -> None:
    from sciona.atoms.dl.video_temporal.atoms import temporal_max_pool, temporal_mean_pool

    frame_predictions = np.array([[0.5, 0.1], [0.9, 0.3]], dtype=np.float64)
    np.testing.assert_allclose(temporal_mean_pool(frame_predictions), np.array([0.7, 0.2]))
    np.testing.assert_allclose(temporal_max_pool(frame_predictions), np.array([0.9, 0.3]))


def test_temporal_attention_pool_is_stable_and_shape_preserving_over_time() -> None:
    from sciona.atoms.dl.video_temporal.atoms import temporal_attention_pool

    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    weights = np.eye(2, dtype=np.float64)
    result = temporal_attention_pool(x, weights, weights, weights)
    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))
    assert result[0, 0] > result[0, 1]
    assert result[1, 1] > result[1, 0]


def test_temporal_median_filter_removes_single_frame_spike() -> None:
    from sciona.atoms.dl.video_temporal.atoms import temporal_median_filter

    signal = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float64)
    result = temporal_median_filter(signal, kernel_size=3)
    np.testing.assert_allclose(result, np.zeros_like(signal))


def test_sliding_windows_returns_read_only_strided_windows() -> None:
    from sciona.atoms.dl.video_temporal.atoms import sliding_windows

    sequence = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = sliding_windows(sequence, window_size=2, stride=1)
    np.testing.assert_allclose(result, np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    assert not result.flags.writeable


def test_sliding_windows_supports_stride() -> None:
    from sciona.atoms.dl.video_temporal.atoms import sliding_windows

    result = sliding_windows(np.arange(6.0), window_size=3, stride=2)
    np.testing.assert_allclose(result, np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]))


def test_stack_adjacent_frames_edge_pads_center_window() -> None:
    from sciona.atoms.dl.video_temporal.atoms import stack_adjacent_frames

    frames = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
    result = stack_adjacent_frames(frames, center_idx=0, num_adjacent=3)
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[:, :, 0], frames[0])
    np.testing.assert_array_equal(result[:, :, 1], frames[0])
    np.testing.assert_array_equal(result[:, :, 2], frames[1])


def test_temporal_unroll_repeats_block_predictions() -> None:
    from sciona.atoms.dl.video_temporal.atoms import temporal_unroll

    aggregated = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float64)
    group_sizes = np.array([2, 3], dtype=np.int64)
    result = temporal_unroll(aggregated, group_sizes)
    expected = np.array(
        [[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [2.0, 20.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected)


def test_contract_rejects_even_median_kernel() -> None:
    from sciona.atoms.dl.video_temporal.atoms import temporal_median_filter

    with pytest.raises(icontract.ViolationError):
        temporal_median_filter(np.array([1.0, 2.0, 3.0]), kernel_size=2)
