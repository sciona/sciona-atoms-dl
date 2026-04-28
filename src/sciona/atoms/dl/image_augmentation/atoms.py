"""Pure image augmentation, preprocessing, and test-time averaging atoms.

The atoms in this family use numpy and scipy only. Random choices are kept
outside the functions: callers pass bounding boxes, offsets, transform names,
or random draws explicitly so the same inputs always produce the same outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_affine_transform_centered,
    witness_ben_graham_retinal_preprocess,
    witness_brightness_contrast_apply,
    witness_compose_augmentations,
    witness_cutmix_apply,
    witness_cutout_apply,
    witness_flip_apply,
    witness_fold_ensemble_average,
    witness_grayscale_convert_apply,
    witness_gridmask_apply,
    witness_hue_saturation_shift,
    witness_min_max_scale,
    witness_mixup_apply,
    witness_normalize_imagenet,
    witness_normalize_per_image,
    witness_random_crop_resize_apply,
    witness_resize_and_pad_apply,
    witness_ten_crop_batch,
    witness_tta_10crop_average,
    witness_tta_geometric_average,
)


BBox = tuple[int, int, int, int]
ImageTransform = Callable[[NDArray[np.float64]], NDArray[np.float64]]
ModelFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _bbox_valid(bbox: BBox, height: int, width: int) -> bool:
    y0, x0, y1, x1 = bbox
    return 0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width


def _resize_to_shape(
    image: NDArray[np.float64],
    target_size: tuple[int, int],
    order: int,
) -> NDArray[np.float64]:
    target_h, target_w = target_size
    zoom: tuple[float, ...]
    if image.ndim == 2:
        zoom = (target_h / image.shape[0], target_w / image.shape[1])
    else:
        zoom = (target_h / image.shape[0], target_w / image.shape[1]) + (1.0,) * (
            image.ndim - 2
        )
    resized = np.asarray(ndimage.zoom(image, zoom, order=order), dtype=np.float64)
    slices = (slice(0, target_h), slice(0, target_w)) + (slice(None),) * (
        resized.ndim - 2
    )
    cropped = resized[slices]
    pad_h = max(0, target_h - cropped.shape[0])
    pad_w = max(0, target_w - cropped.shape[1])
    if pad_h == 0 and pad_w == 0:
        return cropped
    pad_spec = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (cropped.ndim - 2)
    return np.pad(cropped, pad_spec, mode="edge").astype(np.float64)


def _rgb_to_hsv(image: NDArray[np.float64]) -> NDArray[np.float64]:
    rgb = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    active = delta > 0.0
    red_max = active & (maxc == r)
    green_max = active & (maxc == g)
    blue_max = active & (maxc == b)
    hue = np.where(red_max, ((g - b) / np.where(active, delta, 1.0)) % 6.0, hue)
    hue = np.where(green_max, ((b - r) / np.where(active, delta, 1.0)) + 2.0, hue)
    hue = np.where(blue_max, ((r - g) / np.where(active, delta, 1.0)) + 4.0, hue)
    hue = (hue / 6.0) % 1.0
    saturation = np.where(maxc > 0.0, delta / np.where(maxc > 0.0, maxc, 1.0), 0.0)
    return np.stack([hue, saturation, maxc], axis=-1)


def _hsv_to_rgb(image: NDArray[np.float64]) -> NDArray[np.float64]:
    hsv = np.asarray(image, dtype=np.float64)
    h = hsv[..., 0] % 1.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, 1.0)
    sector = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - sector
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    selector = sector % 6
    out = np.empty(hsv.shape, dtype=np.float64)
    out[..., 0] = np.select(
        [selector == 0, selector == 1, selector == 2, selector == 3, selector == 4],
        [v, q, p, p, t],
        default=v,
    )
    out[..., 1] = np.select(
        [selector == 0, selector == 1, selector == 2, selector == 3, selector == 4],
        [t, v, v, q, p],
        default=p,
    )
    out[..., 2] = np.select(
        [selector == 0, selector == 1, selector == 2, selector == 3, selector == 4],
        [p, p, t, v, v],
        default=q,
    )
    return out


def _invert_geometric_transform(
    prediction: NDArray[np.float64],
    transform_code: str,
) -> NDArray[np.float64]:
    if transform_code == "identity":
        return prediction
    if transform_code == "hflip":
        return np.flip(prediction, axis=1)
    if transform_code == "vflip":
        return np.flip(prediction, axis=0)
    if transform_code == "rot90":
        return np.rot90(prediction, k=-1, axes=(0, 1))
    if transform_code == "rot180":
        return np.rot90(prediction, k=2, axes=(0, 1))
    if transform_code == "rot270":
        return np.rot90(prediction, k=1, axes=(0, 1))
    raise ValueError(f"unsupported transform_code: {transform_code}")


@register_atom(witness_cutmix_apply)
@icontract.require(lambda image_a, image_b: image_a.shape == image_b.shape)
@icontract.require(lambda image_a: image_a.ndim >= 2)
@icontract.require(lambda label_a, label_b: label_a.shape == label_b.shape)
@icontract.require(lambda image_a, bbox: _bbox_valid(bbox, image_a.shape[0], image_a.shape[1]))
@icontract.ensure(lambda image_a, result: result[0].shape == image_a.shape)
@icontract.ensure(lambda label_a, result: result[1].shape == label_a.shape)
def cutmix_apply(
    image_a: NDArray[np.float64],
    image_b: NDArray[np.float64],
    label_a: NDArray[np.float64],
    label_b: NDArray[np.float64],
    bbox: BBox,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replace a rectangular region with pixels from another sample.

    The label is mixed by the exact replaced area, so clipped or edge boxes
    produce the same label ratio as the pixels that were actually copied.
    """
    y0, x0, y1, x1 = bbox
    mixed = np.array(image_a, dtype=np.float64, copy=True)
    mixed[y0:y1, x0:x1, ...] = np.asarray(image_b, dtype=np.float64)[y0:y1, x0:x1, ...]
    image_area = float(image_a.shape[0] * image_a.shape[1])
    cut_area = float((y1 - y0) * (x1 - x0))
    lam = 1.0 - cut_area / image_area
    mixed_label = lam * np.asarray(label_a, dtype=np.float64) + (1.0 - lam) * np.asarray(
        label_b, dtype=np.float64
    )
    return mixed, mixed_label


@register_atom(witness_cutout_apply)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda image, bbox: _bbox_valid(bbox, image.shape[0], image.shape[1]))
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def cutout_apply(
    image: NDArray[np.float64],
    bbox: BBox,
    fill_value: float,
) -> NDArray[np.float64]:
    """Fill a rectangular region with a constant value."""
    y0, x0, y1, x1 = bbox
    result = np.array(image, dtype=np.float64, copy=True)
    result[y0:y1, x0:x1, ...] = fill_value
    return result


@register_atom(witness_gridmask_apply)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda d: d > 0)
@icontract.require(lambda keep_ratio: 0.0 <= keep_ratio <= 1.0)
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def gridmask_apply(
    image: NDArray[np.float64],
    d: int,
    keep_ratio: float,
    delta_x: int,
    delta_y: int,
    fill_value: float,
) -> NDArray[np.float64]:
    """Apply a repeated square occlusion pattern to an image."""
    height, width = image.shape[:2]
    yy = (np.arange(height)[:, None] + delta_y) % d
    xx = (np.arange(width)[None, :] + delta_x) % d
    drop_size = int(round((1.0 - keep_ratio) * d))
    drop = (yy < drop_size) & (xx < drop_size)
    mask = ~drop
    while mask.ndim < image.ndim:
        mask = mask[..., None]
    return np.where(mask, np.asarray(image, dtype=np.float64), fill_value).astype(np.float64)


@register_atom(witness_mixup_apply)
@icontract.require(lambda image_a, image_b: image_a.shape == image_b.shape)
@icontract.require(lambda label_a, label_b: label_a.shape == label_b.shape)
@icontract.require(lambda lam: 0.0 <= lam <= 1.0)
@icontract.ensure(lambda image_a, result: result[0].shape == image_a.shape)
@icontract.ensure(lambda label_a, result: result[1].shape == label_a.shape)
def mixup_apply(
    image_a: NDArray[np.float64],
    image_b: NDArray[np.float64],
    label_a: NDArray[np.float64],
    label_b: NDArray[np.float64],
    lam: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute convex pixel and label blends between two samples."""
    mixed_image = lam * np.asarray(image_a, dtype=np.float64) + (1.0 - lam) * np.asarray(
        image_b, dtype=np.float64
    )
    mixed_label = lam * np.asarray(label_a, dtype=np.float64) + (1.0 - lam) * np.asarray(
        label_b, dtype=np.float64
    )
    return mixed_image.astype(np.float64), mixed_label.astype(np.float64)


@register_atom(witness_flip_apply)
@icontract.require(lambda image: image.ndim >= 1)
@icontract.require(lambda image, axis: -image.ndim <= axis < image.ndim)
@icontract.ensure(lambda image, result: result.shape == image.shape)
def flip_apply(image: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Reverse an image or tensor along one axis."""
    return np.asarray(np.flip(image, axis=axis), dtype=np.float64)


@register_atom(witness_random_crop_resize_apply)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda image, bbox: _bbox_valid(bbox, image.shape[0], image.shape[1]))
@icontract.require(lambda target_size: target_size[0] > 0 and target_size[1] > 0)
@icontract.require(lambda order: 0 <= order <= 5)
@icontract.ensure(lambda target_size, result: result.shape[:2] == target_size)
def random_crop_resize_apply(
    image: NDArray[np.float64],
    bbox: BBox,
    target_size: tuple[int, int],
    order: int,
) -> NDArray[np.float64]:
    """Crop a known region and resize it to fixed spatial dimensions."""
    y0, x0, y1, x1 = bbox
    crop = np.asarray(image, dtype=np.float64)[y0:y1, x0:x1, ...]
    return _resize_to_shape(crop, target_size, order)


@register_atom(witness_affine_transform_centered)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda scale: scale > 0.0)
@icontract.require(lambda order: 0 <= order <= 5)
@icontract.ensure(lambda image, result: result.shape == image.shape)
def affine_transform_centered(
    image: NDArray[np.float64],
    angle_degrees: float,
    scale: float,
    dx: float,
    dy: float,
    order: int,
) -> NDArray[np.float64]:
    """Rotate, scale, and translate around the image center."""
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians) * scale
    sin_a = math.sin(radians) * scale
    forward = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    inverse = np.linalg.inv(forward)
    center = (np.array(image.shape[:2], dtype=np.float64) - 1.0) / 2.0
    translation = np.array([dy, dx], dtype=np.float64)
    offset = center - inverse @ (center + translation)

    source = np.asarray(image, dtype=np.float64)
    if source.ndim == 2:
        return np.asarray(
            ndimage.affine_transform(
                source,
                inverse,
                offset=offset,
                output_shape=source.shape,
                order=order,
                mode="constant",
                cval=0.0,
            ),
            dtype=np.float64,
        )

    channels = [
        ndimage.affine_transform(
            source[..., channel],
            inverse,
            offset=offset,
            output_shape=source.shape[:2],
            order=order,
            mode="constant",
            cval=0.0,
        )
        for channel in range(source.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.float64)


@register_atom(witness_brightness_contrast_apply)
@icontract.require(lambda alpha: alpha >= 0.0)
@icontract.require(lambda clip_min, clip_max: clip_min < clip_max)
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda clip_min, clip_max, result: bool(np.all((result >= clip_min) & (result <= clip_max))))
def brightness_contrast_apply(
    image: NDArray[np.float64],
    alpha: float,
    beta: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> NDArray[np.float64]:
    """Apply a clipped linear brightness and contrast adjustment."""
    adjusted = alpha * np.asarray(image, dtype=np.float64) + beta
    return np.clip(adjusted, clip_min, clip_max).astype(np.float64)


@register_atom(witness_hue_saturation_shift)
@icontract.require(lambda image: image.ndim >= 3 and image.shape[-1] == 3)
@icontract.require(lambda image: bool(np.all((image >= 0.0) & (image <= 1.0))))
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))))
def hue_saturation_shift(
    image: NDArray[np.float64],
    hue_shift: float,
    sat_shift: float,
) -> NDArray[np.float64]:
    """Shift hue circularly and saturation additively in RGB images."""
    hsv = _rgb_to_hsv(image)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] + sat_shift, 0.0, 1.0)
    return np.clip(_hsv_to_rgb(hsv), 0.0, 1.0).astype(np.float64)


@register_atom(witness_grayscale_convert_apply)
@icontract.require(lambda image: image.ndim >= 3 and image.shape[-1] == 3)
@icontract.ensure(lambda image, result: result.shape[:-1] == image.shape[:-1] and result.shape[-1] == 1)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def grayscale_convert_apply(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB images to a one-channel luminance image."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = np.tensordot(np.asarray(image, dtype=np.float64), weights, axes=([-1], [0]))
    return gray[..., None].astype(np.float64)


@register_atom(witness_ben_graham_retinal_preprocess)
@icontract.require(lambda image: image.ndim >= 3 and image.shape[-1] == 3)
@icontract.require(lambda sigma: sigma > 0.0)
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 255.0))))
def ben_graham_retinal_preprocess(
    image: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Enhance retinal detail by subtracting a broad blurred background."""
    source = np.asarray(image, dtype=np.float64)
    blurred = ndimage.gaussian_filter(source, sigma=(sigma, sigma, 0.0))
    return np.clip(4.0 * source - 4.0 * blurred + 128.0, 0.0, 255.0).astype(np.float64)


@register_atom(witness_tta_geometric_average)
@icontract.require(lambda predictions, transform_codes: predictions.shape[0] == len(transform_codes))
@icontract.require(lambda predictions: predictions.ndim >= 3)
@icontract.require(lambda transform_codes: all(code in {"identity", "hflip", "vflip", "rot90", "rot180", "rot270"} for code in transform_codes))
@icontract.ensure(lambda predictions, result: result.shape == predictions.shape[1:])
def tta_geometric_average(
    predictions: NDArray[np.float64],
    transform_codes: tuple[str, ...],
) -> NDArray[np.float64]:
    """Undo geometric test-time transforms and average aligned predictions."""
    aligned = [
        _invert_geometric_transform(np.asarray(predictions[index], dtype=np.float64), code)
        for index, code in enumerate(transform_codes)
    ]
    return np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float64)


@register_atom(witness_ten_crop_batch)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda image, crop_size: 0 < crop_size <= image.shape[0] and crop_size <= image.shape[1])
@icontract.ensure(lambda crop_size, result: result.shape[0] == 10 and result.shape[1] == crop_size and result.shape[2] == crop_size)
def ten_crop_batch(image: NDArray[np.float64], crop_size: int) -> NDArray[np.float64]:
    """Return center, corner, and horizontal-flip crops for classification."""
    height, width = image.shape[:2]
    y_mid = (height - crop_size) // 2
    x_mid = (width - crop_size) // 2
    coords = (
        (0, 0),
        (0, width - crop_size),
        (height - crop_size, 0),
        (height - crop_size, width - crop_size),
        (y_mid, x_mid),
    )
    crops = [
        np.asarray(image, dtype=np.float64)[y : y + crop_size, x : x + crop_size, ...]
        for y, x in coords
    ]
    crops.extend([np.flip(crop, axis=1) for crop in crops])
    return np.stack(crops, axis=0).astype(np.float64)


@register_atom(witness_tta_10crop_average)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda image, crop_size: 0 < crop_size <= image.shape[0] and crop_size <= image.shape[1])
@icontract.ensure(lambda result: result.ndim >= 1)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def tta_10crop_average(
    image: NDArray[np.float64],
    crop_size: int,
    model_func: ModelFunc,
) -> NDArray[np.float64]:
    """Average model outputs over ten deterministic classification crops."""
    crops = ten_crop_batch(image, crop_size)
    predictions = [np.asarray(model_func(crop), dtype=np.float64) for crop in crops]
    return np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float64)


@register_atom(witness_fold_ensemble_average)
@icontract.require(lambda predictions_matrix: predictions_matrix.ndim == 3)
@icontract.require(lambda method: method in {"arithmetic", "geometric", "rank"})
@icontract.require(lambda predictions_matrix, method: method != "geometric" or bool(np.all(predictions_matrix >= 0.0)))
@icontract.ensure(lambda predictions_matrix, result: result.shape == predictions_matrix.shape[1:])
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def fold_ensemble_average(
    predictions_matrix: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Average fold predictions using arithmetic, geometric, or rank scores."""
    preds = np.asarray(predictions_matrix, dtype=np.float64)
    if method == "arithmetic":
        return np.mean(preds, axis=0).astype(np.float64)
    if method == "geometric":
        return np.exp(np.mean(np.log(np.maximum(preds, 1e-12)), axis=0)).astype(np.float64)

    ranks = np.argsort(np.argsort(-preds, axis=2), axis=2).astype(np.float64)
    n_classes = preds.shape[2]
    rank_scores = (float(n_classes) - ranks) / float(n_classes)
    return np.mean(rank_scores, axis=0).astype(np.float64)


@register_atom(witness_normalize_imagenet)
@icontract.require(lambda image: image.ndim >= 3 and image.shape[-1] == 3)
@icontract.require(lambda mean, std: len(mean) == 3 and len(std) == 3)
@icontract.require(lambda std: all(value > 0.0 for value in std))
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def normalize_imagenet(
    image: NDArray[np.float64],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Standardize color channels with supplied means and spread values."""
    mean_arr = np.array(mean, dtype=np.float64)
    std_arr = np.array(std, dtype=np.float64)
    return ((np.asarray(image, dtype=np.float64) - mean_arr) / std_arr).astype(np.float64)


@register_atom(witness_normalize_per_image)
@icontract.require(lambda image: image.size > 0)
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def normalize_per_image(
    image: NDArray[np.float64],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Normalize one image by its own mean and standard deviation."""
    source = np.asarray(image, dtype=np.float64)
    std = float(np.std(source))
    if std <= eps:
        return np.zeros_like(source, dtype=np.float64)
    return ((source - float(np.mean(source))) / std).astype(np.float64)


@register_atom(witness_min_max_scale)
@icontract.require(lambda image: image.size > 0)
@icontract.ensure(lambda image, result: result.shape == image.shape)
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))))
def min_max_scale(
    image: NDArray[np.float64],
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Scale values into the unit interval with safe constant handling."""
    source = np.asarray(image, dtype=np.float64)
    min_value = float(np.min(source))
    max_value = float(np.max(source))
    denom = max_value - min_value
    if denom <= eps:
        return np.zeros_like(source, dtype=np.float64)
    return ((source - min_value) / denom).astype(np.float64)


@register_atom(witness_resize_and_pad_apply)
@icontract.require(lambda image: image.ndim >= 2)
@icontract.require(lambda target_size: target_size[0] > 0 and target_size[1] > 0)
@icontract.require(lambda order: 0 <= order <= 5)
@icontract.ensure(lambda target_size, result: result.shape[:2] == target_size)
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def resize_and_pad_apply(
    image: NDArray[np.float64],
    target_size: tuple[int, int],
    pad_value: float,
    order: int = 1,
) -> NDArray[np.float64]:
    """Resize while preserving aspect ratio, then pad to the target size."""
    source = np.asarray(image, dtype=np.float64)
    target_h, target_w = target_size
    scale = min(target_h / source.shape[0], target_w / source.shape[1])
    new_h = max(1, min(target_h, int(round(source.shape[0] * scale))))
    new_w = max(1, min(target_w, int(round(source.shape[1] * scale))))
    resized = _resize_to_shape(source, (new_h, new_w), order)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right)] + [
        (0, 0)
        for _ in range(source.ndim - 2)
    ]
    return np.pad(resized, pad_spec, mode="constant", constant_values=pad_value).astype(
        np.float64
    )


@register_atom(witness_compose_augmentations)
@icontract.require(lambda transforms, probabilities, random_values: len(transforms) == len(probabilities) == len(random_values))
@icontract.require(lambda probabilities: all(0.0 <= value <= 1.0 for value in probabilities))
@icontract.require(lambda random_values: all(0.0 <= value <= 1.0 for value in random_values))
@icontract.ensure(lambda result: bool(np.all(np.isfinite(result))))
def compose_augmentations(
    image: NDArray[np.float64],
    transforms: Sequence[ImageTransform],
    probabilities: Sequence[float],
    random_values: Sequence[float],
) -> NDArray[np.float64]:
    """Apply configured transforms when their explicit random draws pass."""
    result = np.asarray(image, dtype=np.float64)
    for transform, probability, draw in zip(transforms, probabilities, random_values):
        if draw < probability:
            result = np.asarray(transform(result), dtype=np.float64)
    return result
