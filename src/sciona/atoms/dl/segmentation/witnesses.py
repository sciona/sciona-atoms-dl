"""Ghost witnesses for segmentation post-processing atoms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sciona.ghost.abstract import AbstractArray


def _check_2d(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(values.shape[0]), int(values.shape[1])


def _check_same_shape(left: AbstractArray, right: AbstractArray, name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} must match shape")


def witness_morphological_close(mask: AbstractArray, kernel_size: int) -> AbstractArray:
    """Describe shape-preserving binary closing."""
    _check_2d(mask, "mask")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    return AbstractArray(shape=mask.shape, dtype=mask.dtype, min_val=0.0, max_val=1.0)


def witness_morphological_open(mask: AbstractArray, kernel_size: int) -> AbstractArray:
    """Describe shape-preserving binary opening."""
    return witness_morphological_close(mask, kernel_size)


def witness_dilate_mask(mask: AbstractArray, iterations: int) -> AbstractArray:
    """Describe shape-preserving dilation."""
    _check_2d(mask, "mask")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return AbstractArray(shape=mask.shape, dtype=mask.dtype, min_val=0.0, max_val=1.0)


def witness_erode_mask(mask: AbstractArray, iterations: int) -> AbstractArray:
    """Describe shape-preserving erosion."""
    return witness_dilate_mask(mask, iterations)


def witness_fill_holes(mask: AbstractArray) -> AbstractArray:
    """Describe shape-preserving hole filling."""
    _check_2d(mask, "mask")
    return AbstractArray(shape=mask.shape, dtype=mask.dtype, min_val=0.0, max_val=1.0)


def witness_filter_components_by_area(mask: AbstractArray, min_area: int) -> AbstractArray:
    """Describe shape-preserving area filtering."""
    _check_2d(mask, "mask")
    if min_area < 0:
        raise ValueError("min_area must be nonnegative")
    return AbstractArray(shape=mask.shape, dtype=mask.dtype, min_val=0.0, max_val=1.0)


def witness_dense_crf_2d(
    image: AbstractArray,
    unary: AbstractArray,
    sxy: float,
    srgb: float,
    compat: float,
    iterations: int,
) -> AbstractArray:
    """Describe CRF hard-label output."""
    del compat
    if len(image.shape) != 3 or int(image.shape[2]) != 3:
        raise ValueError("image must be HxWx3")
    if len(unary.shape) != 3 or tuple(unary.shape[1:]) != tuple(image.shape[:2]):
        raise ValueError("unary spatial shape must match image")
    if sxy <= 0 or srgb <= 0 or iterations <= 0:
        raise ValueError("CRF parameters out of range")
    return AbstractArray(shape=tuple(image.shape[:2]), dtype="int64", min_val=0.0)


def witness_watershed_instances(distance_map: AbstractArray, markers: AbstractArray, mask: AbstractArray) -> AbstractArray:
    """Describe shape-preserving watershed labels."""
    _check_2d(distance_map, "distance_map")
    _check_same_shape(distance_map, markers, "markers")
    _check_same_shape(distance_map, mask, "mask")
    return AbstractArray(shape=distance_map.shape, dtype="int64", min_val=0.0)


def witness_mask_to_rle(mask: AbstractArray) -> AbstractArray:
    """Describe one-dimensional RLE pairs."""
    height, width = _check_2d(mask, "mask")
    return AbstractArray(shape=(height * width * 2,), dtype="int64", min_val=1.0)


def witness_rle_to_mask(rle: Sequence[int], shape: tuple[int, int]) -> AbstractArray:
    """Describe decoded binary mask shape."""
    if len(rle) % 2 != 0:
        raise ValueError("rle must contain pairs")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must be positive 2D")
    return AbstractArray(shape=(int(shape[0]), int(shape[1])), dtype="uint8", min_val=0.0, max_val=1.0)


def witness_smooth_contour(points: AbstractArray, epsilon: float) -> AbstractArray:
    """Describe contour simplification."""
    if len(points.shape) != 2 or int(points.shape[1]) != 2:
        raise ValueError("points must have shape (N, 2)")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    return AbstractArray(shape=points.shape, dtype="float64")


def witness_wkt_to_mask(
    wkt_string: str,
    image_shape: tuple[int, int],
    transform: tuple[float, float, float, float, float, float],
) -> AbstractArray:
    """Describe a mask made from polygon text."""
    del wkt_string, transform
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("image_shape must be positive 2D")
    return AbstractArray(shape=(int(image_shape[0]), int(image_shape[1])), dtype="uint8", min_val=0.0, max_val=1.0)


def witness_false_color_composite(
    bands: Mapping[str, AbstractArray],
    bounds: Mapping[str, tuple[float, float]],
    gamma: float = 1.0,
) -> AbstractArray:
    """Describe HxWx3 false-color output."""
    del bounds
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not {"red", "green", "blue"}.issubset(set(bands)):
        raise ValueError("bands must contain red, green, and blue")
    height, width = _check_2d(bands["red"], "red")
    for key in ("green", "blue"):
        if tuple(bands[key].shape) != (height, width):
            raise ValueError("band shapes must match")
    return AbstractArray(shape=(height, width, 3), dtype="uint8", min_val=0.0, max_val=255.0)
