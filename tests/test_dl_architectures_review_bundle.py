from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.architectures.efficientnet_backbone",
    "sciona.atoms.dl.architectures.resnet_family_backbone",
    "sciona.atoms.dl.architectures.densenet_backbone",
    "sciona.atoms.dl.architectures.swin_transformer_backbone",
    "sciona.atoms.dl.architectures.unet_2d_segmentation",
    "sciona.atoms.dl.architectures.unet_1d_sequence",
    "sciona.atoms.dl.architectures.yolo_object_detector",
    "sciona.atoms.dl.architectures.whisper_asr_transformer",
    "sciona.atoms.dl.architectures.autoregressive_transformer_decoder",
    "sciona.atoms.dl.architectures.recurrent_sequence_model",
    "sciona.atoms.dl.architectures.slowfast_video_network",
    "sciona.atoms.dl.architectures.mil_attention_aggregator",
}


def test_architectures_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_architectures.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["review_record_path"] == "data/review_bundles/dl_architectures.review_bundle.json"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        for path in row["source_paths"]:
            assert (root / path).exists()

