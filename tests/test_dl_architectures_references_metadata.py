from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "efficientnet_backbone",
    "resnet_family_backbone",
    "densenet_backbone",
    "swin_transformer_backbone",
    "unet_2d_segmentation",
    "unet_1d_sequence",
    "yolo_object_detector",
    "whisper_asr_transformer",
    "autoregressive_transformer_decoder",
    "recurrent_sequence_model",
    "slowfast_video_network",
    "mil_attention_aggregator",
}


def test_architectures_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs_path = root / "src/sciona/atoms/dl/architectures/references.json"
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

