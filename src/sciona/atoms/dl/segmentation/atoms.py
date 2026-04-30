"""Pure segmentation mask post-processing and morphology operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
import re

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.draw import polygon as draw_polygon
from skimage.segmentation import watershed

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dense_crf_2d,
    witness_dilate_mask,
    witness_erode_mask,
    witness_false_color_composite,
    witness_fill_holes,
    witness_filter_components_by_area,
    witness_mask_to_rle,
    witness_morphological_close,
    witness_morphological_open,
    witness_rle_to_mask,
    witness_smooth_contour,
    witness_watershed_instances,
    witness_wkt_to_mask,
)


def _is_binary(mask: NDArray[np.bool_] | NDArray[np.integer]) -> bool:
    values = np.asarray(mask)
    return bool(values.ndim >= 2 and np.all((values == 0) | (values == 1)))


def _binary_structure(kernel_size: int) -> NDArray[np.bool_]:
    return np.ones((int(kernel_size), int(kernel_size)), dtype=bool)


def _restore_mask_dtype(result: NDArray[np.bool_], mask: NDArray[np.bool_] | NDArray[np.integer]) -> NDArray[np.bool_] | NDArray[np.integer]:
    source = np.asarray(mask)
    if source.dtype == np.bool_:
        return np.asarray(result, dtype=np.bool_)
    return np.asarray(result, dtype=source.dtype)


def _same_2d_shape(*arrays: NDArray[np.float64] | NDArray[np.integer] | NDArray[np.bool_]) -> bool:
    shapes = [np.asarray(array).shape for array in arrays]
    return bool(len(set(shapes)) == 1 and len(shapes[0]) == 2)


def _positive_shape(shape: tuple[int, int]) -> bool:
    return bool(len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) > 0)


def _valid_rle(rle: Sequence[int]) -> bool:
    if len(rle) % 2 != 0:
        return False
    values = np.asarray(rle, dtype=np.int64)
    return bool(values.size == 0 or (np.all(values > 0) and np.all(values[1::2] >= 0)))


def _points_valid(points: NDArray[np.float64]) -> bool:
    values = np.asarray(points, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] == 2 and np.all(np.isfinite(values)))


def _band_mapping_valid(bands: Mapping[str, NDArray[np.float64]]) -> bool:
    if not {"red", "green", "blue"}.issubset(set(bands)):
        return False
    shapes = [np.asarray(bands[key]).shape for key in ("red", "green", "blue")]
    try:
        finite = all(np.asarray(bands[key], dtype=np.float64).ndim == 2 and np.all(np.isfinite(bands[key])) for key in ("red", "green", "blue"))
    except (TypeError, ValueError):
        return False
    return bool(finite and len(set(shapes)) == 1)


def _bounds_valid(bounds: Mapping[str, tuple[float, float]]) -> bool:
    if not {"red", "green", "blue"}.issubset(set(bounds)):
        return False
    try:
        return bool(all(np.isfinite(lo) and np.isfinite(hi) and lo < hi for lo, hi in (bounds[key] for key in ("red", "green", "blue"))))
    except (TypeError, ValueError):
        return False


def _parse_wkt_rings(wkt_string: str) -> list[NDArray[np.float64]]:
    text = wkt_string.strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError("Only POLYGON WKT is supported")
    rings: list[NDArray[np.float64]] = []
    for ring_text in re.findall(r"\(([^()]+)\)", text):
        coords: list[tuple[float, float]] = []
        for raw_pair in ring_text.split(","):
            parts = raw_pair.strip().split()
            if len(parts) < 2:
                raise ValueError("Invalid WKT coordinate pair")
            coords.append((float(parts[0]), float(parts[1])))
        if len(coords) < 3:
            raise ValueError("WKT polygon rings need at least three points")
        rings.append(np.asarray(coords, dtype=np.float64))
    if not rings:
        raise ValueError("No polygon rings found")
    return rings


def _world_to_pixel(points: NDArray[np.float64], transform: tuple[float, float, float, float, float, float]) -> NDArray[np.float64]:
    a, b, c, d, e, f = transform
    matrix = np.array([[a, b], [d, e]], dtype=np.float64)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-12:
        raise ValueError("transform is singular")
    shifted = np.column_stack((points[:, 0] - c, points[:, 1] - f))
    col_row = shifted @ np.linalg.inv(matrix).T
    return np.column_stack((col_row[:, 1], col_row[:, 0]))


def _rdp(points: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    if points.shape[0] <= 2:
        return points.copy()
    start = points[0]
    end = points[-1]
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length == 0.0:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        distances = np.abs(segment[0] * (start[1] - points[:, 1]) - segment[1] * (start[0] - points[:, 0])) / length
    index = int(np.argmax(distances))
    if distances[index] > float(epsilon):
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return np.vstack((left[:-1], right))
    return np.vstack((start, end))


@register_atom(witness_morphological_close)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.require(lambda kernel_size: kernel_size > 0, "kernel_size must be positive")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "closed mask must preserve shape")
@icontract.ensure(lambda result: _is_binary(result), "closed mask must be binary")
def morphological_close(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    kernel_size: int,
) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Apply binary dilation followed by erosion to close small gaps in a mask."""
    result = ndimage.binary_closing(np.asarray(mask).astype(bool), structure=_binary_structure(kernel_size))
    return _restore_mask_dtype(result, mask)


@register_atom(witness_morphological_open)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.require(lambda kernel_size: kernel_size > 0, "kernel_size must be positive")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "opened mask must preserve shape")
@icontract.ensure(lambda result: _is_binary(result), "opened mask must be binary")
def morphological_open(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    kernel_size: int,
) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Apply binary erosion followed by dilation to remove small foreground artifacts."""
    result = ndimage.binary_opening(np.asarray(mask).astype(bool), structure=_binary_structure(kernel_size))
    return _restore_mask_dtype(result, mask)


@register_atom(witness_dilate_mask)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.require(lambda iterations: iterations > 0, "iterations must be positive")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "dilated mask must preserve shape")
@icontract.ensure(lambda result: _is_binary(result), "dilated mask must be binary")
def dilate_mask(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    iterations: int,
) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Expand foreground regions outward by repeated binary dilation."""
    result = ndimage.binary_dilation(np.asarray(mask).astype(bool), iterations=int(iterations))
    return _restore_mask_dtype(result, mask)


@register_atom(witness_erode_mask)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.require(lambda iterations: iterations > 0, "iterations must be positive")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "eroded mask must preserve shape")
@icontract.ensure(lambda result: _is_binary(result), "eroded mask must be binary")
def erode_mask(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    iterations: int,
) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Shrink foreground regions inward by repeated binary erosion."""
    result = ndimage.binary_erosion(np.asarray(mask).astype(bool), iterations=int(iterations))
    return _restore_mask_dtype(result, mask)


@register_atom(witness_fill_holes)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "filled mask must preserve shape")
@icontract.ensure(lambda result, mask: np.sum(result) >= np.sum(mask), "hole filling cannot reduce foreground area")
def fill_holes(mask: NDArray[np.bool_] | NDArray[np.integer]) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Fill enclosed background holes inside foreground components."""
    result = ndimage.binary_fill_holes(np.asarray(mask).astype(bool))
    return _restore_mask_dtype(result, mask)


@register_atom(witness_filter_components_by_area)
@icontract.require(lambda mask: _is_binary(mask), "mask must be a binary 2D array")
@icontract.require(lambda min_area: min_area >= 0, "min_area must be nonnegative")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "filtered mask must preserve shape")
@icontract.ensure(lambda result, mask: np.sum(result) <= np.sum(mask), "area filtering cannot add foreground pixels")
def filter_components_by_area(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    min_area: int,
) -> NDArray[np.bool_] | NDArray[np.integer]:
    """Remove connected foreground components smaller than the requested pixel area."""
    binary = np.asarray(mask).astype(bool)
    labels, count = ndimage.label(binary)
    if count == 0:
        return _restore_mask_dtype(binary, mask)
    sizes = ndimage.sum(binary, labels=labels, index=np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = np.asarray(sizes) >= int(min_area)
    return _restore_mask_dtype(keep[labels], mask)


@register_atom(witness_dense_crf_2d)
@icontract.require(lambda image: np.asarray(image).ndim == 3 and np.asarray(image).shape[2] == 3 and np.asarray(image).dtype == np.uint8, "image must be uint8 RGB")
@icontract.require(lambda unary: np.asarray(unary).ndim == 3 and np.all(np.isfinite(unary)), "unary must be finite class probability tensor")
@icontract.require(lambda image, unary: np.asarray(unary).shape[1:] == np.asarray(image).shape[:2], "unary spatial shape must match image")
@icontract.require(lambda sxy: np.isfinite(float(sxy)) and sxy > 0.0, "sxy must be positive")
@icontract.require(lambda srgb: np.isfinite(float(srgb)) and srgb > 0.0, "srgb must be positive")
@icontract.require(lambda compat: np.isfinite(float(compat)) and compat >= 0.0, "compat must be nonnegative")
@icontract.require(lambda iterations: iterations > 0, "iterations must be positive")
@icontract.ensure(lambda result, image: result.shape == np.asarray(image).shape[:2], "CRF output must preserve image height and width")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "CRF output must contain integer labels")
def dense_crf_2d(
    image: NDArray[np.uint8],
    unary: NDArray[np.float64],
    sxy: float,
    srgb: float,
    compat: float,
    iterations: int,
) -> NDArray[np.int64]:
    """Refine class probabilities with optional DenseCRF and return hard labels."""
    probabilities = np.asarray(unary, dtype=np.float64)
    if importlib.util.find_spec("pydensecrf") is not None:
        import pydensecrf.densecrf as dcrf  # type: ignore[import-not-found]
        from pydensecrf.utils import unary_from_softmax  # type: ignore[import-not-found]

        classes, height, width = probabilities.shape
        crf = dcrf.DenseCRF2D(width, height, classes)
        crf.setUnaryEnergy(unary_from_softmax(np.clip(probabilities, 1e-12, 1.0)))
        crf.addPairwiseBilateral(sxy=float(sxy), srgb=float(srgb), rgbim=np.asarray(image, dtype=np.uint8), compat=float(compat))
        refined = np.asarray(crf.inference(int(iterations)), dtype=np.float64).reshape(classes, height, width)
        return np.argmax(refined, axis=0).astype(np.int64)
    return np.argmax(probabilities, axis=0).astype(np.int64)


@register_atom(witness_watershed_instances)
@icontract.require(lambda distance_map: np.asarray(distance_map).ndim == 2 and np.all(np.isfinite(distance_map)), "distance_map must be finite 2D")
@icontract.require(lambda markers: np.asarray(markers).ndim == 2 and np.all(np.asarray(markers) >= 0), "markers must be nonnegative integer 2D")
@icontract.require(lambda mask: _is_binary(mask), "mask must be binary")
@icontract.require(lambda distance_map, markers, mask: _same_2d_shape(distance_map, markers, mask), "distance_map, markers, and mask must share shape")
@icontract.ensure(lambda result, distance_map: result.shape == np.asarray(distance_map).shape, "watershed labels must preserve shape")
@icontract.ensure(lambda result, mask: np.all(result[np.asarray(mask) == 0] == 0), "labels must not spill outside mask")
def watershed_instances(
    distance_map: NDArray[np.float64],
    markers: NDArray[np.integer],
    mask: NDArray[np.bool_] | NDArray[np.integer],
) -> NDArray[np.int64]:
    """Split a semantic foreground mask into instance labels using watershed flooding."""
    labels = watershed(-np.asarray(distance_map, dtype=np.float64), np.asarray(markers, dtype=np.int64), mask=np.asarray(mask).astype(bool))
    return np.asarray(labels, dtype=np.int64)


@register_atom(witness_mask_to_rle)
@icontract.require(lambda mask: _is_binary(mask), "mask must be binary")
@icontract.ensure(lambda result: len(result) % 2 == 0, "RLE must contain start-length pairs")
@icontract.ensure(lambda result: all(value > 0 for value in result), "RLE starts and lengths must be positive")
def mask_to_rle(mask: NDArray[np.bool_] | NDArray[np.integer]) -> list[int]:
    """Encode a binary mask as one-based column-major run-length pairs."""
    pixels = np.asarray(mask).astype(np.uint8).T.reshape(-1)
    padded = np.concatenate(([0], pixels, [0]))
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return [int(value) for value in runs]


@register_atom(witness_rle_to_mask)
@icontract.require(lambda rle: _valid_rle(rle), "rle must contain positive start-length pairs")
@icontract.require(lambda shape: _positive_shape(shape), "shape must be two positive integers")
@icontract.ensure(lambda result, shape: result.shape == shape, "decoded mask must match shape")
@icontract.ensure(lambda result: _is_binary(result), "decoded mask must be binary")
def rle_to_mask(rle: Sequence[int], shape: tuple[int, int]) -> NDArray[np.uint8]:
    """Decode one-based column-major run-length pairs into a binary mask."""
    height, width = int(shape[0]), int(shape[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    values = np.asarray(rle, dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]
    for start, length in zip(starts, lengths, strict=True):
        flat[int(start) : int(start + length)] = 1
    return flat.reshape((width, height)).T.astype(np.uint8)


@register_atom(witness_smooth_contour)
@icontract.require(lambda points: _points_valid(points), "points must have shape (N, 2) and be finite")
@icontract.require(lambda epsilon: np.isfinite(float(epsilon)) and epsilon >= 0.0, "epsilon must be finite and nonnegative")
@icontract.ensure(lambda result, points: result.shape[1] == 2 and result.shape[0] <= np.asarray(points).shape[0], "smoothed contour must be no longer than input")
def smooth_contour(points: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    """Simplify an ordered contour with the Ramer-Douglas-Peucker algorithm."""
    return _rdp(np.asarray(points, dtype=np.float64), float(epsilon)).astype(np.float64)


@register_atom(witness_wkt_to_mask)
@icontract.require(lambda wkt_string: isinstance(wkt_string, str) and len(wkt_string.strip()) > 0, "wkt_string must be non-empty")
@icontract.require(lambda image_shape: _positive_shape(image_shape), "image_shape must be two positive integers")
@icontract.require(lambda transform: len(transform) == 6 and all(np.isfinite(value) for value in transform), "transform must be a finite six-value affine tuple")
@icontract.ensure(lambda result, image_shape: result.shape == image_shape, "rasterized mask must match image_shape")
@icontract.ensure(lambda result: _is_binary(result), "rasterized mask must be binary")
def wkt_to_mask(
    wkt_string: str,
    image_shape: tuple[int, int],
    transform: tuple[float, float, float, float, float, float],
) -> NDArray[np.uint8]:
    """Rasterize a simple POLYGON WKT geometry into a binary mask."""
    output = np.zeros((int(image_shape[0]), int(image_shape[1])), dtype=np.uint8)
    rings = _parse_wkt_rings(wkt_string)
    for ring_index, ring in enumerate(rings):
        row_col = _world_to_pixel(ring, transform)
        rr, cc = draw_polygon(row_col[:, 0], row_col[:, 1], shape=output.shape)
        if ring_index == 0:
            output[rr, cc] = 1
        else:
            output[rr, cc] = 0
    return output


@register_atom(witness_false_color_composite)
@icontract.require(lambda bands: _band_mapping_valid(bands), "bands must contain finite 2D red, green, and blue arrays with identical shape")
@icontract.require(lambda bounds: _bounds_valid(bounds), "bounds must contain strict red, green, and blue ranges")
@icontract.require(lambda gamma: np.isfinite(float(gamma)) and gamma > 0.0, "gamma must be positive")
@icontract.ensure(lambda result, bands: result.shape == (np.asarray(bands["red"]).shape[0], np.asarray(bands["red"]).shape[1], 3), "false-color image must be HxWx3")
@icontract.ensure(lambda result: result.dtype == np.uint8, "false-color image must be uint8")
def false_color_composite(
    bands: Mapping[str, NDArray[np.float64]],
    bounds: Mapping[str, tuple[float, float]],
    gamma: float = 1.0,
) -> NDArray[np.uint8]:
    """Stretch red, green, and blue band arrays into an 8-bit false-color image."""
    channels: list[NDArray[np.float64]] = []
    for key in ("red", "green", "blue"):
        lo, hi = bounds[key]
        scaled = np.clip((np.asarray(bands[key], dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
        channels.append(np.power(scaled, 1.0 / float(gamma)))
    return np.round(np.stack(channels, axis=-1) * 255.0).astype(np.uint8)
