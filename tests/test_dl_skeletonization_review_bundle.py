from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.skeletonization.medial_axis_2d",
    "sciona.atoms.dl.skeletonization.skeleton_to_graph",
    "sciona.atoms.dl.skeletonization.skeletonize_2d",
}


def test_skeletonization_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_skeletonization.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    assert bundle["family"] == "dl.skeletonization"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_record_path"] == "data/review_bundles/dl_skeletonization.review_bundle.json"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        for path in row["source_paths"]:
            assert (root / path).exists()
