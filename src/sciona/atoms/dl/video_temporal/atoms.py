"""Pure video timeline and temporal aggregation primitives.

These atoms compute deterministic frame indices, temporal pooling, local
sequence windows, 2.5D frame stacks, and timestep unrolling without touching
video decoders, files, random number generators, or model state.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.ndimage import median_filter

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sample_frame_indices,
    witness_sliding_windows,
    witness_stack_adjacent_frames,
    witness_temporal_attention_pool,
    witness_temporal_max_pool,
    witness_temporal_mean_pool,
    witness_temporal_median_filter,
    witness_temporal_unroll,
    witness_uniform_sample_indices,
)


def _finite_float_matrix(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0 and np.all(np.isfinite(array)))


@register_atom(witness_sample_frame_indices)
@icontract.require(lambda total_frames: total_frames > 0, "total_frames must be positive")
@icontract.require(lambda target_fps: target_fps > 0.0 and np.isfinite(target_fps), "target_fps must be positive")
@icontract.require(lambda video_fps: video_fps > 0.0 and np.isfinite(video_fps), "video_fps must be positive")
@icontract.ensure(lambda result: result.ndim == 1 and result.size >= 1, "at least one frame index is returned")
@icontract.ensure(lambda total_frames, result: bool(np.all((result >= 0) & (result < total_frames))), "indices stay in range")
def sample_frame_indices(
    total_frames: int,
    target_fps: float,
    video_fps: float,
) -> NDArray[np.int64]:
    """Select native frame indices that best approximate a target FPS.

    The atom maps deterministic target timestamps back onto the native frame
    grid, rounds to the closest frame, and clips boundary drift.
    """
    sample_count = max(1, int(np.ceil(float(total_frames) * target_fps / video_fps)))
    target_times = np.arange(sample_count, dtype=np.float64) / target_fps
    native_positions = target_times * video_fps
    return np.rint(native_positions).astype(np.int64).clip(0, total_frames - 1)


@register_atom(witness_uniform_sample_indices)
@icontract.require(lambda total_frames: total_frames > 0, "total_frames must be positive")
@icontract.require(lambda n_samples: n_samples > 0, "n_samples must be positive")
@icontract.require(lambda total_frames, n_samples: n_samples <= total_frames, "n_samples must not exceed total_frames")
@icontract.ensure(lambda n_samples, result: result.shape == (n_samples,), "one index per requested sample is returned")
@icontract.ensure(lambda total_frames, result: bool(np.all((result >= 0) & (result < total_frames))), "indices stay in range")
@icontract.ensure(lambda result: bool(np.all(result[1:] >= result[:-1])), "indices are monotone")
def uniform_sample_indices(total_frames: int, n_samples: int) -> NDArray[np.int64]:
    """Select uniformly spaced frame indices over a complete video timeline."""
    if n_samples == 1:
        return np.array([0], dtype=np.int64)
    positions = np.linspace(0, total_frames - 1, num=n_samples, dtype=np.float64)
    return np.rint(positions).astype(np.int64)


@register_atom(witness_temporal_mean_pool)
@icontract.require(lambda frame_predictions: _finite_float_matrix(frame_predictions), "frame_predictions must be a non-empty 2D finite array")
@icontract.ensure(lambda frame_predictions, result: result.shape == (np.asarray(frame_predictions).shape[1],), "features are pooled over time")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "pooled features must be finite")
def temporal_mean_pool(frame_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average frame-level predictions or embeddings across the temporal axis."""
    return np.mean(np.asarray(frame_predictions, dtype=np.float64), axis=0)


@register_atom(witness_temporal_max_pool)
@icontract.require(lambda frame_predictions: _finite_float_matrix(frame_predictions), "frame_predictions must be a non-empty 2D finite array")
@icontract.ensure(lambda frame_predictions, result: result.shape == (np.asarray(frame_predictions).shape[1],), "features are pooled over time")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "pooled features must be finite")
def temporal_max_pool(frame_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Take per-feature maximum activations across a video timeline."""
    return np.max(np.asarray(frame_predictions, dtype=np.float64), axis=0)


@register_atom(witness_temporal_attention_pool)
@icontract.require(lambda x: _finite_float_matrix(x), "x must be a non-empty 2D finite array")
@icontract.require(lambda x, wq: np.asarray(wq).shape[0] == np.asarray(x).shape[1], "Wq input dimension must match x")
@icontract.require(lambda x, wk: np.asarray(wk).shape[0] == np.asarray(x).shape[1], "Wk input dimension must match x")
@icontract.require(lambda x, wv: np.asarray(wv).shape[0] == np.asarray(x).shape[1], "Wv input dimension must match x")
@icontract.require(lambda wq, wk, wv: np.asarray(wq).shape == np.asarray(wk).shape == np.asarray(wv).shape, "projection matrices must share shape")
@icontract.require(lambda wq: np.asarray(wq).ndim == 2 and np.all(np.isfinite(wq)), "Wq must be a finite 2D matrix")
@icontract.require(lambda wk: np.asarray(wk).ndim == 2 and np.all(np.isfinite(wk)), "Wk must be a finite 2D matrix")
@icontract.require(lambda wv: np.asarray(wv).ndim == 2 and np.all(np.isfinite(wv)), "Wv must be a finite 2D matrix")
@icontract.ensure(lambda x, wv, result: result.shape == (np.asarray(x).shape[0], np.asarray(wv).shape[1]), "attention output keeps time and projects features")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "attention output must be finite")
def temporal_attention_pool(
    x: NDArray[np.float64],
    wq: NDArray[np.float64],
    wk: NDArray[np.float64],
    wv: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply stable scaled dot-product temporal self-attention."""
    sequence = np.asarray(x, dtype=np.float64)
    query = sequence @ np.asarray(wq, dtype=np.float64)
    key = sequence @ np.asarray(wk, dtype=np.float64)
    value = sequence @ np.asarray(wv, dtype=np.float64)
    scale = np.sqrt(float(query.shape[1]))
    scores = (query @ key.T) / scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ value


@register_atom(witness_temporal_median_filter)
@icontract.require(lambda predictions: np.asarray(predictions).ndim == 1 and np.asarray(predictions).size > 0, "predictions must be a non-empty 1D array")
@icontract.require(lambda predictions: np.all(np.isfinite(np.asarray(predictions, dtype=np.float64))), "predictions must be finite")
@icontract.require(lambda kernel_size: kernel_size > 0 and kernel_size % 2 == 1, "kernel_size must be positive and odd")
@icontract.ensure(lambda predictions, result: result.shape == np.asarray(predictions).shape, "filter preserves sequence shape")
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))), "filtered values must be finite")
def temporal_median_filter(predictions: NDArray[np.float64], kernel_size: int) -> NDArray[np.float64]:
    """Suppress isolated temporal spikes with an odd-width 1D median filter."""
    return median_filter(np.asarray(predictions, dtype=np.float64), size=kernel_size, mode="nearest")


@register_atom(witness_sliding_windows)
@icontract.require(lambda sequence: np.asarray(sequence).ndim == 1 and np.asarray(sequence).size > 0, "sequence must be a non-empty 1D array")
@icontract.require(lambda sequence: np.all(np.isfinite(np.asarray(sequence, dtype=np.float64))), "sequence must be finite")
@icontract.require(lambda window_size: window_size > 0, "window_size must be positive")
@icontract.require(lambda stride: stride > 0, "stride must be positive")
@icontract.require(lambda sequence, window_size: window_size <= np.asarray(sequence).size, "window_size must fit within sequence")
@icontract.ensure(
    lambda sequence, window_size, stride, result: result.shape == ((np.asarray(sequence).size - window_size) // stride + 1, window_size),
    "windows must follow the requested size and stride",
)
@icontract.ensure(lambda result: not result.flags.writeable, "window view must be read-only")
def sliding_windows(
    sequence: NDArray[np.float64],
    window_size: int,
    stride: int = 1,
) -> NDArray[np.float64]:
    """Extract read-only overlapping windows from a temporal sequence."""
    windows = sliding_window_view(np.asarray(sequence, dtype=np.float64), window_shape=window_size)[::stride]
    windows.flags.writeable = False
    return windows


@register_atom(witness_stack_adjacent_frames)
@icontract.require(lambda frames: np.asarray(frames).ndim == 3, "frames must have shape (time, height, width)")
@icontract.require(lambda frames: np.asarray(frames).shape[0] > 0, "frames must contain at least one timestep")
@icontract.require(lambda frames: np.issubdtype(np.asarray(frames).dtype, np.integer), "frames must contain integer pixels")
@icontract.require(lambda frames, center_idx: 0 <= center_idx < np.asarray(frames).shape[0], "center_idx must address an existing frame")
@icontract.require(lambda num_adjacent: num_adjacent > 0, "num_adjacent must be positive")
@icontract.ensure(lambda frames, num_adjacent, result: result.shape == (np.asarray(frames).shape[1], np.asarray(frames).shape[2], num_adjacent), "stacked frame shape is height-width-channels")
def stack_adjacent_frames(
    frames: NDArray[np.uint8],
    center_idx: int,
    num_adjacent: int,
) -> NDArray[np.uint8]:
    """Stack clipped neighboring grayscale frames along the channel axis."""
    array = np.asarray(frames)
    offsets = np.arange(num_adjacent, dtype=np.int64) - (num_adjacent // 2)
    indices = np.clip(center_idx + offsets, 0, array.shape[0] - 1)
    return np.moveaxis(array[indices], 0, -1).astype(array.dtype, copy=False)


@register_atom(witness_temporal_unroll)
@icontract.require(lambda aggregated: np.asarray(aggregated).ndim >= 1 and np.asarray(aggregated).shape[0] > 0, "aggregated must have a non-empty leading axis")
@icontract.require(lambda group_sizes: np.asarray(group_sizes).ndim == 1 and np.asarray(group_sizes).size > 0, "group_sizes must be a non-empty 1D array")
@icontract.require(lambda aggregated, group_sizes: np.asarray(aggregated).shape[0] == np.asarray(group_sizes).shape[0], "leading dimension must match group_sizes")
@icontract.require(lambda group_sizes: bool(np.all(np.asarray(group_sizes, dtype=np.int64) > 0)), "group sizes must be positive")
@icontract.ensure(lambda group_sizes, result: result.shape[0] == int(np.sum(np.asarray(group_sizes, dtype=np.int64))), "leading axis expands by group-size sum")
def temporal_unroll(
    aggregated: NDArray[np.float64],
    group_sizes: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Repeat aggregated temporal predictions back to timestep granularity."""
    return np.repeat(np.asarray(aggregated, dtype=np.float64), np.asarray(group_sizes, dtype=np.int64), axis=0)

