"""Numerically stable embedding operations for retrieval pipelines."""

from __future__ import annotations

import importlib.util

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_alpha_query_expansion,
    witness_build_faiss_flat_ip,
    witness_cosine_similarity_matrix,
    witness_embedding_delta,
    witness_l2_normalize,
    witness_pca_whiten_reduce,
    witness_rerank_by_distance,
)


def _finite_array(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.size > 0 and np.all(np.isfinite(array)))


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _valid_axis(values: NDArray[np.float64], axis: int) -> bool:
    ndim = np.asarray(values).ndim
    return bool(-ndim <= int(axis) < ndim)


def _positive_eps(eps: float) -> bool:
    return bool(np.isfinite(float(eps)) and float(eps) > 0.0)


def _cosine_result_valid(result: NDArray[np.float64], rows: int, columns: int) -> bool:
    array = np.asarray(result, dtype=np.float64)
    return bool(
        array.shape == (rows, columns)
        and np.all(np.isfinite(array))
        and np.all(array >= -1.000001)
        and np.all(array <= 1.000001)
    )


def _row_norms_safe(values: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1, keepdims=True)
    return np.maximum(norms, float(eps))


def _nonzero_norms_preserved(
    original: NDArray[np.float64],
    result: NDArray[np.float64],
    axis: int,
    eps: float,
) -> bool:
    source = np.asarray(original, dtype=np.float64)
    normalized = np.asarray(result, dtype=np.float64)
    source_norms = np.linalg.norm(source, axis=axis)
    result_norms = np.linalg.norm(normalized, axis=axis)
    mask = source_norms > float(eps)
    return bool(normalized.shape == source.shape and np.all(np.isfinite(normalized)) and np.allclose(result_norms[mask], 1.0, atol=1e-7))


def _search_result_valid(
    result: tuple[NDArray[np.float64], NDArray[np.int64]],
    n_queries: int,
    n_reference: int,
    k: int,
) -> bool:
    scores, indices = result
    score_array = np.asarray(scores, dtype=np.float64)
    index_array = np.asarray(indices, dtype=np.int64)
    sorted_scores = np.all(score_array[:, :-1] >= score_array[:, 1:]) if k > 1 else True
    return bool(
        score_array.shape == (n_queries, k)
        and index_array.shape == (n_queries, k)
        and np.all(np.isfinite(score_array))
        and np.all(index_array >= 0)
        and np.all(index_array < n_reference)
        and sorted_scores
    )


@register_atom(witness_l2_normalize)
@icontract.require(lambda embeddings: _finite_array(embeddings), "embeddings must be a finite non-empty array")
@icontract.require(lambda embeddings, axis: _valid_axis(embeddings, axis), "axis must refer to an embedding dimension")
@icontract.require(lambda eps: _positive_eps(eps), "eps must be positive and finite")
@icontract.ensure(lambda result, embeddings: result.shape == np.asarray(embeddings).shape, "normalized embeddings must preserve shape")
@icontract.ensure(lambda result, embeddings, axis, eps: _nonzero_norms_preserved(embeddings, result, axis, eps), "non-zero vectors must have unit L2 norm")
def l2_normalize(
    embeddings: NDArray[np.float64],
    axis: int = 1,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Return an out-of-place L2-normalized embedding array with zero-vector guards."""
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return (values / np.maximum(norms, float(eps))).astype(np.float64)


@register_atom(witness_cosine_similarity_matrix)
@icontract.require(lambda embeddings_a: _finite_matrix(embeddings_a), "embeddings_a must be a finite 2D matrix")
@icontract.require(lambda embeddings_b: _finite_matrix(embeddings_b), "embeddings_b must be a finite 2D matrix")
@icontract.require(lambda embeddings_a, embeddings_b: np.asarray(embeddings_a).shape[1] == np.asarray(embeddings_b).shape[1], "embedding dimensions must match")
@icontract.require(lambda eps: _positive_eps(eps), "eps must be positive and finite")
@icontract.ensure(lambda result, embeddings_a, embeddings_b: _cosine_result_valid(result, np.asarray(embeddings_a).shape[0], np.asarray(embeddings_b).shape[0]), "cosine matrix must be finite and bounded")
def cosine_similarity_matrix(
    embeddings_a: NDArray[np.float64],
    embeddings_b: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute pairwise cosine similarities by normalizing rows before matrix multiplication."""
    left = np.asarray(embeddings_a, dtype=np.float64)
    right = np.asarray(embeddings_b, dtype=np.float64)
    left_normed = left / _row_norms_safe(left, eps)
    right_normed = right / _row_norms_safe(right, eps)
    return np.clip(left_normed @ right_normed.T, -1.0, 1.0).astype(np.float64)


@register_atom(witness_alpha_query_expansion)
@icontract.require(lambda query: _finite_vector(query), "query must be a finite 1D vector")
@icontract.require(lambda retrieved_neighbors: _finite_matrix(retrieved_neighbors), "retrieved_neighbors must be a finite 2D matrix")
@icontract.require(lambda similarities: _finite_vector(similarities), "similarities must be a finite 1D vector")
@icontract.require(lambda query, retrieved_neighbors: np.asarray(query).shape[0] == np.asarray(retrieved_neighbors).shape[1], "neighbor vectors must match query dimensionality")
@icontract.require(lambda retrieved_neighbors, similarities: np.asarray(retrieved_neighbors).shape[0] == np.asarray(similarities).shape[0], "one similarity is required per neighbor")
@icontract.require(lambda alpha: np.isfinite(float(alpha)) and float(alpha) >= 0.0, "alpha must be finite and nonnegative")
@icontract.require(lambda eps: _positive_eps(eps), "eps must be positive and finite")
@icontract.ensure(lambda result, query: result.shape == np.asarray(query).shape, "expanded query must preserve query shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "expanded query must be finite")
def alpha_query_expansion(
    query: NDArray[np.float64],
    retrieved_neighbors: NDArray[np.float64],
    similarities: NDArray[np.float64],
    alpha: float = 3.0,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Expand a query by a power-weighted average of retrieved neighbor embeddings."""
    query_vector = np.asarray(query, dtype=np.float64)
    neighbors = np.asarray(retrieved_neighbors, dtype=np.float64)
    scores = np.clip(np.asarray(similarities, dtype=np.float64), 0.0, None)
    weights = scores ** float(alpha)
    if float(np.sum(weights)) <= float(eps):
        return query_vector.copy()
    weighted_neighbors = np.average(neighbors, axis=0, weights=weights)
    expanded = query_vector + weighted_neighbors
    return l2_normalize(expanded.reshape(1, -1), axis=1, eps=eps).reshape(-1)


@register_atom(witness_pca_whiten_reduce)
@icontract.require(lambda embeddings: _finite_matrix(embeddings), "embeddings must be a finite 2D matrix")
@icontract.require(lambda embeddings, n_components: 1 <= int(n_components) <= min(np.asarray(embeddings).shape), "n_components must fit the sample and feature dimensions")
@icontract.require(lambda eps: _positive_eps(eps), "eps must be positive and finite")
@icontract.ensure(lambda result, embeddings, n_components: result.shape == (np.asarray(embeddings).shape[0], int(n_components)), "PCA output must preserve rows and requested component count")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "PCA output must be finite")
def pca_whiten_reduce(
    embeddings: NDArray[np.float64],
    n_components: int,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Project embeddings with SVD PCA, whiten components, and L2-normalize rows."""
    values = np.asarray(embeddings, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(n_components)]
    projected = centered @ components.T
    scale = np.maximum(singular_values[: int(n_components)] / np.sqrt(max(values.shape[0] - 1, 1)), float(eps))
    whitened = projected / scale
    return l2_normalize(whitened, axis=1, eps=eps)


@register_atom(witness_embedding_delta)
@icontract.require(lambda original: _finite_array(original), "original must be finite")
@icontract.require(lambda transformed: _finite_array(transformed), "transformed must be finite")
@icontract.require(lambda original, transformed: np.asarray(original).shape == np.asarray(transformed).shape, "embedding arrays must have identical shape")
@icontract.ensure(lambda result, original: result.shape == np.asarray(original).shape, "delta must preserve shape")
@icontract.ensure(lambda result, original, transformed: np.allclose(result, np.asarray(transformed, dtype=np.float64) - np.asarray(original, dtype=np.float64)), "delta must equal transformed minus original")
def embedding_delta(
    original: NDArray[np.float64],
    transformed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the arithmetic direction vector from an original embedding to a transformed embedding."""
    return (np.asarray(transformed, dtype=np.float64) - np.asarray(original, dtype=np.float64)).astype(np.float64)


@register_atom(witness_build_faiss_flat_ip)
@icontract.require(lambda reference_embeddings: _finite_matrix(reference_embeddings), "reference_embeddings must be a finite 2D matrix")
@icontract.require(lambda query_embeddings: _finite_matrix(query_embeddings), "query_embeddings must be a finite 2D matrix")
@icontract.require(lambda reference_embeddings, query_embeddings: np.asarray(reference_embeddings).shape[1] == np.asarray(query_embeddings).shape[1], "query and reference dimensions must match")
@icontract.require(lambda reference_embeddings, k: 1 <= int(k) <= np.asarray(reference_embeddings).shape[0], "k must select at least one reference and no more than all references")
@icontract.require(lambda eps: _positive_eps(eps), "eps must be positive and finite")
@icontract.ensure(lambda result, reference_embeddings, query_embeddings, k: _search_result_valid(result, np.asarray(query_embeddings).shape[0], np.asarray(reference_embeddings).shape[0], int(k)), "search outputs must be sorted valid scores and indices")
def build_faiss_flat_ip(
    reference_embeddings: NDArray[np.float64],
    query_embeddings: NDArray[np.float64],
    k: int,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Run exact flat inner-product retrieval over L2-normalized embeddings.

    If the optional FAISS package is present, this creates an in-memory
    IndexFlatIP and searches it. Otherwise it uses the same exact inner-product
    calculation in NumPy, preserving the atom's input/output contract.
    """
    references = l2_normalize(np.asarray(reference_embeddings, dtype=np.float32), axis=1, eps=eps).astype(np.float32)
    queries = l2_normalize(np.asarray(query_embeddings, dtype=np.float32), axis=1, eps=eps).astype(np.float32)
    top_k = int(k)

    if importlib.util.find_spec("faiss") is not None:
        import faiss  # type: ignore[import-not-found]

        index = faiss.IndexFlatIP(references.shape[1])
        index.add(references)
        scores, indices = index.search(queries, top_k)
        return scores.astype(np.float64), indices.astype(np.int64)

    scores = queries @ references.T
    partition = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1, kind="mergesort")
    sorted_indices = np.take_along_axis(partition, order, axis=1)
    sorted_scores = np.take_along_axis(scores, sorted_indices, axis=1)
    return sorted_scores.astype(np.float64), sorted_indices.astype(np.int64)


@register_atom(witness_rerank_by_distance)
@icontract.require(lambda query: _finite_vector(query), "query must be a finite 1D vector")
@icontract.require(lambda candidates: _finite_matrix(candidates), "candidates must be a finite 2D matrix")
@icontract.require(lambda candidate_ids: np.asarray(candidate_ids).ndim == 1, "candidate_ids must be 1D")
@icontract.require(lambda query, candidates: np.asarray(query).shape[0] == np.asarray(candidates).shape[1], "candidate dimensions must match query")
@icontract.require(lambda candidates, candidate_ids: np.asarray(candidates).shape[0] == np.asarray(candidate_ids).shape[0], "one candidate_id is required per candidate row")
@icontract.require(lambda candidate_ids, k: 1 <= int(k) <= np.asarray(candidate_ids).shape[0], "k must be in candidate range")
@icontract.ensure(lambda result, k: result.shape == (int(k),), "reranked output must contain k ids")
def rerank_by_distance(
    query: NDArray[np.float64],
    candidates: NDArray[np.float64],
    candidate_ids: NDArray[np.int64],
    k: int,
) -> NDArray[np.int64]:
    """Return candidate IDs sorted by ascending Euclidean distance to the query vector."""
    query_vector = np.asarray(query, dtype=np.float64)
    candidate_matrix = np.asarray(candidates, dtype=np.float64)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    distances = np.linalg.norm(candidate_matrix - query_vector.reshape(1, -1), axis=1)
    order = np.argsort(distances, kind="mergesort")[: int(k)]
    return ids[order].astype(np.int64)

