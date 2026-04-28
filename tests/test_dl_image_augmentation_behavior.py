from __future__ import annotations

import numpy as np


def test_image_augmentation_atoms_import() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import (
        affine_transform_centered,
        ben_graham_retinal_preprocess,
        brightness_contrast_apply,
        compose_augmentations,
        cutmix_apply,
        cutout_apply,
        flip_apply,
        fold_ensemble_average,
        grayscale_convert_apply,
        gridmask_apply,
        hue_saturation_shift,
        min_max_scale,
        mixup_apply,
        normalize_imagenet,
        normalize_per_image,
        random_crop_resize_apply,
        resize_and_pad_apply,
        ten_crop_batch,
        tta_10crop_average,
        tta_geometric_average,
    )

    assert callable(cutmix_apply)
    assert callable(cutout_apply)
    assert callable(gridmask_apply)
    assert callable(mixup_apply)
    assert callable(flip_apply)
    assert callable(random_crop_resize_apply)
    assert callable(affine_transform_centered)
    assert callable(brightness_contrast_apply)
    assert callable(hue_saturation_shift)
    assert callable(grayscale_convert_apply)
    assert callable(ben_graham_retinal_preprocess)
    assert callable(tta_geometric_average)
    assert callable(ten_crop_batch)
    assert callable(tta_10crop_average)
    assert callable(fold_ensemble_average)
    assert callable(normalize_imagenet)
    assert callable(normalize_per_image)
    assert callable(min_max_scale)
    assert callable(resize_and_pad_apply)
    assert callable(compose_augmentations)


def test_cutmix_replaces_patch_and_weights_label() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import cutmix_apply

    a = np.zeros((4, 4, 1), dtype=np.float64)
    b = np.ones((4, 4, 1), dtype=np.float64)
    label_a = np.array([1.0, 0.0])
    label_b = np.array([0.0, 1.0])
    mixed, label = cutmix_apply(a, b, label_a, label_b, (1, 1, 3, 3))
    assert np.all(mixed[1:3, 1:3, 0] == 1.0)
    np.testing.assert_allclose(label, np.array([0.75, 0.25]))


def test_cutout_and_gridmask_keep_shape() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import cutout_apply, gridmask_apply

    image = np.ones((4, 4, 1), dtype=np.float64)
    cut = cutout_apply(image, (1, 1, 3, 3), fill_value=0.0)
    assert np.all(cut[1:3, 1:3, 0] == 0.0)
    masked = gridmask_apply(image, d=2, keep_ratio=0.5, delta_x=0, delta_y=0, fill_value=0.0)
    assert masked.shape == image.shape
    assert np.min(masked) == 0.0


def test_mixup_and_flip() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import flip_apply, mixup_apply

    a = np.zeros((2, 2), dtype=np.float64)
    b = np.ones((2, 2), dtype=np.float64)
    mixed, label = mixup_apply(a, b, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.25)
    assert np.all(mixed == 0.75)
    np.testing.assert_allclose(label, np.array([0.25, 0.75]))

    asymmetric = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(flip_apply(asymmetric, axis=1), np.array([[2.0, 1.0], [4.0, 3.0]]))


def test_crop_resize_resize_pad_and_affine_shapes() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import (
        affine_transform_centered,
        random_crop_resize_apply,
        resize_and_pad_apply,
    )

    image = np.arange(6 * 8, dtype=np.float64).reshape(6, 8)
    cropped = random_crop_resize_apply(image, (1, 2, 5, 6), (3, 3), order=1)
    assert cropped.shape == (3, 3)
    padded = resize_and_pad_apply(image, (10, 10), pad_value=0.0, order=1)
    assert padded.shape == (10, 10)
    rotated = affine_transform_centered(image, angle_degrees=0.0, scale=1.0, dx=0.0, dy=0.0, order=1)
    np.testing.assert_allclose(rotated, image)


def test_color_atoms() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import (
        brightness_contrast_apply,
        grayscale_convert_apply,
        hue_saturation_shift,
    )

    black = np.zeros((2, 2, 3), dtype=np.float64)
    bright = brightness_contrast_apply(black, alpha=1.0, beta=0.2)
    assert np.all(bright == 0.2)

    red = np.zeros((1, 1, 3), dtype=np.float64)
    red[0, 0, 0] = 1.0
    shifted = hue_saturation_shift(red, hue_shift=0.5, sat_shift=0.0)
    np.testing.assert_allclose(shifted[0, 0], np.array([0.0, 1.0, 1.0]), atol=1e-6)

    gray = grayscale_convert_apply(red)
    assert gray.shape == (1, 1, 1)
    assert gray[0, 0, 0] == np.float64(0.299)


def test_preprocessing_and_normalization_atoms() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import (
        ben_graham_retinal_preprocess,
        min_max_scale,
        normalize_imagenet,
        normalize_per_image,
    )

    image = np.full((4, 4, 3), 100.0, dtype=np.float64)
    retinal = ben_graham_retinal_preprocess(image, sigma=1.0)
    assert retinal.shape == image.shape
    assert np.all((retinal >= 0.0) & (retinal <= 255.0))

    norm = normalize_imagenet(np.ones((1, 1, 3)), (0.5, 0.5, 0.5), (0.5, 0.25, 0.125))
    np.testing.assert_allclose(norm[0, 0], np.array([1.0, 2.0, 4.0]))

    per = normalize_per_image(np.array([1.0, 2.0, 3.0]))
    assert abs(float(np.mean(per))) < 1e-12
    assert abs(float(np.std(per)) - 1.0) < 1e-12

    scaled = min_max_scale(np.array([-50.0, 50.0]))
    np.testing.assert_allclose(scaled, np.array([0.0, 1.0]))


def test_tta_and_ensemble_atoms() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import (
        fold_ensemble_average,
        ten_crop_batch,
        tta_10crop_average,
        tta_geometric_average,
    )

    base = np.arange(4, dtype=np.float64).reshape(2, 2)
    predictions = np.stack([base, np.flip(base, axis=1)], axis=0)
    avg = tta_geometric_average(predictions, ("identity", "hflip"))
    np.testing.assert_allclose(avg, base)

    image = np.arange(5 * 5, dtype=np.float64).reshape(5, 5)
    crops = ten_crop_batch(image, crop_size=3)
    assert crops.shape == (10, 3, 3)

    def model_func(crop: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(crop)), float(np.max(crop))], dtype=np.float64)

    pred = tta_10crop_average(image, crop_size=3, model_func=model_func)
    assert pred.shape == (2,)

    folds = np.array([
        [[0.2, 0.8], [0.6, 0.4]],
        [[0.4, 0.6], [0.8, 0.2]],
    ])
    np.testing.assert_allclose(fold_ensemble_average(folds, "arithmetic"), np.mean(folds, axis=0))
    assert fold_ensemble_average(folds, "rank").shape == (2, 2)


def test_compose_augmentations_uses_explicit_draws() -> None:
    from sciona.atoms.dl.image_augmentation.atoms import compose_augmentations

    image = np.ones((2, 2), dtype=np.float64)

    def double(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def add_three(x: np.ndarray) -> np.ndarray:
        return x + 3.0

    result = compose_augmentations(
        image,
        transforms=(double, add_three),
        probabilities=(1.0, 0.25),
        random_values=(0.0, 0.5),
    )
    assert np.all(result == 2.0)
