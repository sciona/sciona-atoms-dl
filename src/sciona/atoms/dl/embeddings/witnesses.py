"""Ghost witnesses for embedding retrieval atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(values.shape[0]), int(values.shape[1])


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    return int(values.shape[0])


def witness_l2_normalize(
    embeddings: AbstractArray,
    axis: int = 1,
    eps: float = 1e-12,
) -> AbstractArray:
    """Describe output with the same shape as the input."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not -len(embeddings.shape) <= axis < len(embeddings.shape):
        raise ValueError("axis out of range")
    return AbstractArray(shape=embeddings.shape, dtype=embeddings.dtype, min_val=-1.0, max_val=1.0)


def witness_cosine_similarity_matrix(
    embeddings_a: AbstractArray,
    embeddings_b: AbstractArray,
    eps: float = 1e-12,
) -> AbstractArray:
    """Describe pairwise cosine similarity output."""
    del eps
    rows_a, dim_a = _check_matrix(embeddings_a, "embeddings_a")
    rows_b, dim_b = _check_matrix(embeddings_b, "embeddings_b")
    if dim_a != dim_b:
        raise ValueError("embedding dimensions must match")
    return AbstractArray(shape=(rows_a, rows_b), dtype="float64", min_val=-1.0, max_val=1.0)


def witness_alpha_query_expansion(
    query: AbstractArray,
    retrieved_neighbors: AbstractArray,
    similarities: AbstractArray,
    alpha: float = 3.0,
    eps: float = 1e-12,
) -> AbstractArray:
    """Describe query expansion preserving query dimensionality."""
    del eps
    query_dim = _check_vector(query, "query")
    neighbor_count, neighbor_dim = _check_matrix(retrieved_neighbors, "retrieved_neighbors")
    if _check_vector(similarities, "similarities") != neighbor_count:
        raise ValueError("one similarity is required per neighbor")
    if neighbor_dim != query_dim:
        raise ValueError("neighbor vectors must match query dimensionality")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    return AbstractArray(shape=query.shape, dtype=query.dtype)


def witness_pca_whiten_reduce(
    embeddings: AbstractArray,
    n_components: int,
    eps: float = 1e-12,
) -> AbstractArray:
    """Describe PCA whitening followed by reduced row embeddings."""
    del eps
    rows, columns = _check_matrix(embeddings, "embeddings")
    if not 1 <= n_components <= min(rows, columns):
        raise ValueError("n_components must fit the sample and feature dimensions")
    return AbstractArray(shape=(rows, int(n_components)), dtype="float64")


def witness_embedding_delta(
    original: AbstractArray,
    transformed: AbstractArray,
) -> AbstractArray:
    """Describe shape-preserving embedding delta computation."""
    if original.shape != transformed.shape:
        raise ValueError("embedding arrays must have identical shape")
    return AbstractArray(shape=original.shape, dtype="float64")


def witness_build_faiss_flat_ip(
    reference_embeddings: AbstractArray,
    query_embeddings: AbstractArray,
    k: int,
    eps: float = 1e-12,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe exact flat inner-product retrieval output."""
    del eps
    reference_count, reference_dim = _check_matrix(reference_embeddings, "reference_embeddings")
    query_count, query_dim = _check_matrix(query_embeddings, "query_embeddings")
    if reference_dim != query_dim:
        raise ValueError("query and reference dimensions must match")
    if not 1 <= k <= reference_count:
        raise ValueError("k must be in reference range")
    return (
        AbstractArray(shape=(query_count, int(k)), dtype="float64", min_val=-1.0, max_val=1.0),
        AbstractArray(shape=(query_count, int(k)), dtype="int64", min_val=0.0, max_val=float(reference_count - 1)),
    )


def witness_rerank_by_distance(
    query: AbstractArray,
    candidates: AbstractArray,
    candidate_ids: AbstractArray,
    k: int,
) -> AbstractArray:
    """Describe candidate ID reranking output."""
    query_dim = _check_vector(query, "query")
    candidate_count, candidate_dim = _check_matrix(candidates, "candidates")
    if candidate_dim != query_dim:
        raise ValueError("candidate dimensions must match query")
    if _check_vector(candidate_ids, "candidate_ids") != candidate_count:
        raise ValueError("one candidate_id is required per candidate row")
    if not 1 <= k <= candidate_count:
        raise ValueError("k must be in candidate range")
    return AbstractArray(shape=(int(k),), dtype="int64")
