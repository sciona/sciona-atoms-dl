from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "dense_crf_2d",
    "dilate_mask",
    "erode_mask",
    "false_color_composite",
    "fill_holes",
    "filter_components_by_area",
    "mask_to_rle",
    "morphological_close",
    "morphological_open",
    "rle_to_mask",
    "smooth_contour",
    "watershed_instances",
    "wkt_to_mask",
}


def test_segmentation_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs_path = root / "src/sciona/atoms/dl/segmentation/references.json"
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

