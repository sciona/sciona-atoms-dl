from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_embeddings_import() -> None:
    from sciona.atoms.dl.embeddings import (
        alpha_query_expansion,
        build_faiss_flat_ip,
        cosine_similarity_matrix,
        embedding_delta,
        l2_normalize,
        pca_whiten_reduce,
        rerank_by_distance,
    )

    assert callable(alpha_query_expansion)
    assert callable(build_faiss_flat_ip)
    assert callable(cosine_similarity_matrix)
    assert callable(embedding_delta)
    assert callable(l2_normalize)
    assert callable(pca_whiten_reduce)
    assert callable(rerank_by_distance)


def test_l2_normalize_handles_zero_vectors() -> None:
    from sciona.atoms.dl.embeddings import l2_normalize

    embeddings = np.array([[3.0, 4.0], [0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    result = l2_normalize(embeddings, axis=1)
    assert np.allclose(result[0], np.array([0.6, 0.8]))
    assert np.allclose(result[1], np.array([0.0, 0.0]))
    assert np.allclose(np.linalg.norm(result[[0, 2]], axis=1), np.array([1.0, 1.0]))


def test_cosine_similarity_matrix_is_zero_safe_and_bounded() -> None:
    from sciona.atoms.dl.embeddings import cosine_similarity_matrix

    left = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    right = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = cosine_similarity_matrix(left, right)
    assert np.allclose(result, np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)


def test_alpha_query_expansion_weights_high_similarity_neighbors() -> None:
    from sciona.atoms.dl.embeddings import alpha_query_expansion

    query = np.array([1.0, 0.0], dtype=np.float64)
    neighbors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    similarities = np.array([0.9, 0.2], dtype=np.float64)
    expanded = alpha_query_expansion(query, neighbors, similarities, alpha=3.0)
    assert expanded[0] > expanded[1]
    assert np.isclose(np.linalg.norm(expanded), 1.0)


def test_pca_whiten_reduce_shapes_and_normalizes_nonzero_rows() -> None:
    from sciona.atoms.dl.embeddings import pca_whiten_reduce

    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    reduced = pca_whiten_reduce(embeddings, n_components=2)
    assert reduced.shape == (4, 2)
    assert np.all(np.isfinite(reduced))
    nonzero = np.linalg.norm(reduced, axis=1) > 1e-12
    assert np.allclose(np.linalg.norm(reduced[nonzero], axis=1), 1.0)


def test_embedding_delta_matches_vector_difference() -> None:
    from sciona.atoms.dl.embeddings import embedding_delta

    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    transformed = np.array([[2.0, 1.0], [5.0, 4.0]], dtype=np.float64)
    assert np.array_equal(embedding_delta(original, transformed), transformed - original)


def test_flat_inner_product_search_returns_sorted_scores_and_indices() -> None:
    from sciona.atoms.dl.embeddings import build_faiss_flat_ip

    references = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    scores, indices = build_faiss_flat_ip(references, queries, k=2)
    assert scores.shape == (2, 2)
    assert indices.shape == (2, 2)
    assert np.all(scores[:, 0] >= scores[:, 1])
    assert indices[0, 0] in {0, 2}
    assert indices[1, 0] in {1, 2}


def test_rerank_by_distance_sorts_candidate_ids() -> None:
    from sciona.atoms.dl.embeddings import rerank_by_distance

    query = np.array([0.0, 0.0], dtype=np.float64)
    candidates = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    candidate_ids = np.array([30, 10, 20], dtype=np.int64)
    assert np.array_equal(rerank_by_distance(query, candidates, candidate_ids, k=2), np.array([10, 20]))


def test_embedding_contracts_reject_invalid_shapes() -> None:
    from sciona.atoms.dl.embeddings import cosine_similarity_matrix, pca_whiten_reduce, rerank_by_distance

    with pytest.raises(ViolationError):
        cosine_similarity_matrix(np.ones((2, 3)), np.ones((2, 4)))

    with pytest.raises(ViolationError):
        pca_whiten_reduce(np.ones((2, 3)), n_components=3)

    with pytest.raises(ViolationError):
        rerank_by_distance(np.ones(3), np.ones((2, 2)), np.array([1, 2]), k=1)

