from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.detection.lung_mask_with_bone_removal",
    "sciona.atoms.dl.detection.anchor_label_mapping_with_iou_dilation",
    "sciona.atoms.dl.detection.center_feature_extraction_3d",
    "sciona.atoms.dl.detection.margin_expanded_face_crop",
    "sciona.atoms.dl.detection.face_similarity_align",
    "sciona.atoms.dl.detection.iou_matrix",
    "sciona.atoms.dl.detection.giou_matrix",
    "sciona.atoms.dl.detection.nms",
    "sciona.atoms.dl.detection.soft_nms",
    "sciona.atoms.dl.detection.wbf",
    "sciona.atoms.dl.detection.wbf_1d",
    "sciona.atoms.dl.detection.generate_anchors",
    "sciona.atoms.dl.detection.encode_boxes",
    "sciona.atoms.dl.detection.decode_boxes",
    "sciona.atoms.dl.detection.nms_1d",
    "sciona.atoms.dl.detection.masks_to_boxes",
    "sciona.atoms.dl.detection.associate_boxes",
    "sciona.atoms.dl.detection.threshold_detections",
}


def test_detection_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_detection.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["review_record_path"] == "data/review_bundles/dl_detection.review_bundle.json"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        for path in row["source_paths"]:
            assert (root / path).exists()
