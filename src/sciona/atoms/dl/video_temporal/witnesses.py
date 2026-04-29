"""Ghost witnesses for video temporal atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sample_frame_indices(
    total_frames: int,
    target_fps: float,
    video_fps: float,
) -> AbstractArray:
    """Mirror FPS-based frame index selection."""
    return AbstractArray()


def witness_uniform_sample_indices(total_frames: int, n_samples: int) -> AbstractArray:
    """Mirror deterministic uniform frame index selection."""
    return AbstractArray()


def witness_temporal_mean_pool(frame_predictions: AbstractArray) -> AbstractArray:
    """Mirror temporal mean pooling as a feature vector."""
    return AbstractArray()


def witness_temporal_max_pool(frame_predictions: AbstractArray) -> AbstractArray:
    """Mirror temporal max pooling as a feature vector."""
    return AbstractArray()


def witness_temporal_attention_pool(
    x: AbstractArray,
    wq: AbstractArray,
    wk: AbstractArray,
    wv: AbstractArray,
) -> AbstractArray:
    """Mirror self-attention over the temporal axis."""
    return AbstractArray()


def witness_temporal_median_filter(predictions: AbstractArray, kernel_size: int) -> AbstractArray:
    """Mirror shape-preserving temporal median filtering."""
    return predictions


def witness_sliding_windows(
    sequence: AbstractArray,
    window_size: int,
    stride: int = 1,
) -> AbstractArray:
    """Mirror strided extraction of overlapping sequence windows."""
    return AbstractArray()


def witness_stack_adjacent_frames(
    frames: AbstractArray,
    center_idx: int,
    num_adjacent: int,
) -> AbstractArray:
    """Mirror channel stacking of neighboring frames."""
    return AbstractArray()


def witness_temporal_unroll(
    aggregated: AbstractArray,
    group_sizes: AbstractArray,
) -> AbstractArray:
    """Mirror temporal repetition back to finer timesteps."""
    return AbstractArray()

