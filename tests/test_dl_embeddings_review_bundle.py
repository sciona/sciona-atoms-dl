from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "sciona.atoms.dl.embeddings.alpha_query_expansion",
    "sciona.atoms.dl.embeddings.build_faiss_flat_ip",
    "sciona.atoms.dl.embeddings.cosine_similarity_matrix",
    "sciona.atoms.dl.embeddings.embedding_delta",
    "sciona.atoms.dl.embeddings.l2_normalize",
    "sciona.atoms.dl.embeddings.pca_whiten_reduce",
    "sciona.atoms.dl.embeddings.rerank_by_distance",
}


def test_embeddings_review_bundle_covers_atoms() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "data/review_bundles/dl_embeddings.review_bundle.json").read_text())
    rows = {row["atom_name"]: row for row in bundle["rows"]}

    assert bundle["provider_repo"] == "sciona-atoms-dl"
    assert bundle["review_status"] == "reviewed"
    assert set(rows) == EXPECTED_ATOMS
    for row in rows.values():
        assert row["review_status"] == "reviewed"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"

