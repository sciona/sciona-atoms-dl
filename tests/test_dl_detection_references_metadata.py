from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "lung_mask_with_bone_removal",
    "anchor_label_mapping_with_iou_dilation",
    "center_feature_extraction_3d",
    "margin_expanded_face_crop",
    "face_similarity_align",
    "iou_matrix",
    "giou_matrix",
    "nms",
    "soft_nms",
    "wbf",
    "wbf_1d",
    "generate_anchors",
    "encode_boxes",
    "decode_boxes",
    "nms_1d",
    "masks_to_boxes",
    "associate_boxes",
    "threshold_detections",
}


def test_detection_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs_path = root / "src/sciona/atoms/dl/detection/references.json"
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
