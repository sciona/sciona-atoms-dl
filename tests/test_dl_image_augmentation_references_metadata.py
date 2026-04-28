from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "cutmix_apply",
    "cutout_apply",
    "gridmask_apply",
    "mixup_apply",
    "flip_apply",
    "random_crop_resize_apply",
    "affine_transform_centered",
    "brightness_contrast_apply",
    "hue_saturation_shift",
    "grayscale_convert_apply",
    "ben_graham_retinal_preprocess",
    "tta_geometric_average",
    "ten_crop_batch",
    "tta_10crop_average",
    "fold_ensemble_average",
    "normalize_imagenet",
    "normalize_per_image",
    "min_max_scale",
    "resize_and_pad_apply",
    "compose_augmentations",
}


def test_image_augmentation_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs_path = root / "src/sciona/atoms/dl/image_augmentation/references.json"
    registry_path = root / "data/references/registry.json"
    refs = json.loads(refs_path.read_text())
    registry = json.loads(registry_path.read_text())

    leaf_names = {key.split("@")[0].rsplit(".", 1)[-1] for key in refs["atoms"]}
    assert leaf_names == EXPECTED_ATOMS
    registered = set(registry["references"])
    for record in refs["atoms"].values():
        assert record["references"]
        for ref in record["references"]:
            assert ref["ref_id"] in registered
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"] in {"low", "medium", "high"}
            assert metadata["notes"]
