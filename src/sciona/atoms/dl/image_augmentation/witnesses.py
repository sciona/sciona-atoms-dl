"""Ghost witnesses for image augmentation atoms."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sciona.ghost.abstract import AbstractArray


ImageWitnessTransform = Callable[[AbstractArray], AbstractArray]
ModelWitnessFunc = Callable[[AbstractArray], AbstractArray]


def witness_cutmix_apply(
    image_a: AbstractArray,
    image_b: AbstractArray,
    label_a: AbstractArray,
    label_b: AbstractArray,
    bbox: tuple[int, int, int, int],
) -> tuple[AbstractArray, AbstractArray]:
    """Witness for patch replacement with area-weighted labels."""
    return image_a, label_a


def witness_cutout_apply(
    image: AbstractArray,
    bbox: tuple[int, int, int, int],
    fill_value: float,
) -> AbstractArray:
    """Witness for rectangular constant filling."""
    return image


def witness_gridmask_apply(
    image: AbstractArray,
    d: int,
    keep_ratio: float,
    delta_x: int,
    delta_y: int,
    fill_value: float,
) -> AbstractArray:
    """Witness for repeated square masking."""
    return image


def witness_mixup_apply(
    image_a: AbstractArray,
    image_b: AbstractArray,
    label_a: AbstractArray,
    label_b: AbstractArray,
    lam: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Witness for convex image and label mixing."""
    return image_a, label_a


def witness_flip_apply(image: AbstractArray, axis: int) -> AbstractArray:
    """Witness for an axis reversal."""
    return image


def witness_random_crop_resize_apply(
    image: AbstractArray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int],
    order: int,
) -> AbstractArray:
    """Witness for crop followed by fixed-size resize."""
    return image


def witness_affine_transform_centered(
    image: AbstractArray,
    angle_degrees: float,
    scale: float,
    dx: float,
    dy: float,
    order: int,
) -> AbstractArray:
    """Witness for centered affine resampling."""
    return image


def witness_brightness_contrast_apply(
    image: AbstractArray,
    alpha: float,
    beta: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> AbstractArray:
    """Witness for clipped linear color adjustment."""
    return image


def witness_hue_saturation_shift(
    image: AbstractArray,
    hue_shift: float,
    sat_shift: float,
) -> AbstractArray:
    """Witness for RGB hue and saturation changes."""
    return image


def witness_grayscale_convert_apply(image: AbstractArray) -> AbstractArray:
    """Witness for RGB to one-channel luminance conversion."""
    return image


def witness_ben_graham_retinal_preprocess(
    image: AbstractArray,
    sigma: float,
) -> AbstractArray:
    """Witness for retinal high-pass preprocessing."""
    return image


def witness_tta_geometric_average(
    predictions: AbstractArray,
    transform_codes: tuple[str, ...],
) -> AbstractArray:
    """Witness for inverse-aligning and averaging TTA predictions."""
    return predictions


def witness_ten_crop_batch(image: AbstractArray, crop_size: int) -> AbstractArray:
    """Witness for ten deterministic crops."""
    return image


def witness_tta_10crop_average(
    image: AbstractArray,
    crop_size: int,
    model_func: ModelWitnessFunc,
) -> AbstractArray:
    """Witness for averaging ten-crop model outputs."""
    return image


def witness_fold_ensemble_average(
    predictions_matrix: AbstractArray,
    method: str,
) -> AbstractArray:
    """Witness for averaging fold predictions."""
    return predictions_matrix


def witness_normalize_imagenet(
    image: AbstractArray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> AbstractArray:
    """Witness for per-channel standardization."""
    return image


def witness_normalize_per_image(
    image: AbstractArray,
    eps: float = 1e-8,
) -> AbstractArray:
    """Witness for one-image standardization."""
    return image


def witness_min_max_scale(
    image: AbstractArray,
    eps: float = 1e-8,
) -> AbstractArray:
    """Witness for safe unit-interval scaling."""
    return image


def witness_resize_and_pad_apply(
    image: AbstractArray,
    target_size: tuple[int, int],
    pad_value: float,
    order: int = 1,
) -> AbstractArray:
    """Witness for aspect-preserving resize and padding."""
    return image


def witness_compose_augmentations(
    image: AbstractArray,
    transforms: Sequence[ImageWitnessTransform],
    probabilities: Sequence[float],
    random_values: Sequence[float],
) -> AbstractArray:
    """Witness for sequential explicit-probability transforms."""
    return image
