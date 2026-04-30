from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.segmentation.dense_crf_2d",
    "sciona.atoms.dl.segmentation.dilate_mask",
    "sciona.atoms.dl.segmentation.erode_mask",
    "sciona.atoms.dl.segmentation.false_color_composite",
    "sciona.atoms.dl.segmentation.fill_holes",
    "sciona.atoms.dl.segmentation.filter_components_by_area",
    "sciona.atoms.dl.segmentation.mask_to_rle",
    "sciona.atoms.dl.segmentation.morphological_close",
    "sciona.atoms.dl.segmentation.morphological_open",
    "sciona.atoms.dl.segmentation.rle_to_mask",
    "sciona.atoms.dl.segmentation.smooth_contour",
    "sciona.atoms.dl.segmentation.watershed_instances",
    "sciona.atoms.dl.segmentation.wkt_to_mask",
}


def test_segmentation_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_segmentation.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_record_path"] == "data/review_bundles/dl_segmentation.review_bundle.json"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        for path in row["source_paths"]:
            assert (root / path).exists()
