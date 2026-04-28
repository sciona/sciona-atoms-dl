from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.loss.miss_penalty_loss",
    "sciona.atoms.dl.loss.qwk_loss",
    "sciona.atoms.dl.loss.ctc_loss",
    "sciona.atoms.dl.loss.focal_loss",
    "sciona.atoms.dl.loss.lovasz_softmax_loss",
    "sciona.atoms.dl.loss.dice_loss",
    "sciona.atoms.dl.loss.crps_score",
    "sciona.atoms.dl.loss.contrastive_loss",
    "sciona.atoms.dl.loss.triplet_loss",
    "sciona.atoms.dl.loss.label_smoothing_ce",
    "sciona.atoms.dl.loss.weighted_multitask_loss",
    "sciona.atoms.dl.loss.multimodal_nll_loss",
    "sciona.atoms.dl.loss.weighted_bce_loss",
}


def test_loss_review_bundle_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "data/review_bundles/dl_loss.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "pending"
    assert bundle["trust_readiness"] == "unreviewed"
    rows = bundle["rows"]
    assert {row["atom_key"] for row in rows} == EXPECTED_ATOMS
    for source in bundle["authoritative_sources"]:
        assert (root / source["path"]).exists()
    for row in rows:
        assert row["review_record_path"] == "data/review_bundles/dl_loss.review_bundle.json"
        for path in row["source_paths"]:
            assert (root / path).exists()
