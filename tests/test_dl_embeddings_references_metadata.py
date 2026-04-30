from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "alpha_query_expansion",
    "build_faiss_flat_ip",
    "cosine_similarity_matrix",
    "embedding_delta",
    "l2_normalize",
    "pca_whiten_reduce",
    "rerank_by_distance",
}


def test_embeddings_references_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = json.loads((root / "src/sciona/atoms/dl/embeddings/references.json").read_text())
    registry = json.loads((root / "data/references/registry.json").read_text())

    leaf_names = {key.split("@")[0].rsplit(".", 1)[-1] for key in refs["atoms"]}
    assert leaf_names == EXPECTED_ATOMS
    registry_ids = set(registry["references"])
    for record in refs["atoms"].values():
        assert record["references"]
        for ref in record["references"]:
            assert ref["ref_id"] in registry_ids
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"] in {"low", "medium", "high"}
            assert metadata["notes"]

