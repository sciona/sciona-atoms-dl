from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "miss_penalty_loss",
    "qwk_loss",
    "ctc_loss",
    "focal_loss",
    "lovasz_softmax_loss",
    "dice_loss",
    "crps_score",
    "contrastive_loss",
    "triplet_loss",
    "label_smoothing_ce",
    "weighted_multitask_loss",
    "multimodal_nll_loss",
    "weighted_bce_loss",
}


def test_loss_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs_path = root / "src/sciona/atoms/dl/loss/references.json"
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
