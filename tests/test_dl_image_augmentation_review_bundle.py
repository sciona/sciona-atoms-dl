from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.image_augmentation.cutmix_apply",
    "sciona.atoms.dl.image_augmentation.cutout_apply",
    "sciona.atoms.dl.image_augmentation.gridmask_apply",
    "sciona.atoms.dl.image_augmentation.mixup_apply",
    "sciona.atoms.dl.image_augmentation.flip_apply",
    "sciona.atoms.dl.image_augmentation.random_crop_resize_apply",
    "sciona.atoms.dl.image_augmentation.affine_transform_centered",
    "sciona.atoms.dl.image_augmentation.brightness_contrast_apply",
    "sciona.atoms.dl.image_augmentation.hue_saturation_shift",
    "sciona.atoms.dl.image_augmentation.grayscale_convert_apply",
    "sciona.atoms.dl.image_augmentation.ben_graham_retinal_preprocess",
    "sciona.atoms.dl.image_augmentation.tta_geometric_average",
    "sciona.atoms.dl.image_augmentation.ten_crop_batch",
    "sciona.atoms.dl.image_augmentation.tta_10crop_average",
    "sciona.atoms.dl.image_augmentation.fold_ensemble_average",
    "sciona.atoms.dl.image_augmentation.normalize_imagenet",
    "sciona.atoms.dl.image_augmentation.normalize_per_image",
    "sciona.atoms.dl.image_augmentation.min_max_scale",
    "sciona.atoms.dl.image_augmentation.resize_and_pad_apply",
    "sciona.atoms.dl.image_augmentation.compose_augmentations",
}


def test_image_augmentation_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_image_augmentation.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "pending"
    assert bundle["trust_readiness"] == "unreviewed"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["review_record_path"] == "data/review_bundles/dl_image_augmentation.review_bundle.json"
        for path in row["source_paths"]:
            assert (root / path).exists()
