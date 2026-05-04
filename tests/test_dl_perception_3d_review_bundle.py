from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.perception_3d.project_image_to_points",
    "sciona.atoms.dl.perception_3d.ransac_homography",
    "sciona.atoms.dl.perception_3d.rasterize_bev",
    "sciona.atoms.dl.perception_3d.voxelize_point_cloud",
}


def test_perception_3d_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_perception_3d.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    assert bundle["family"] == "dl.perception_3d"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_record_path"] == "data/review_bundles/dl_perception_3d.review_bundle.json"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        for path in row["source_paths"]:
            assert (root / path).exists()
