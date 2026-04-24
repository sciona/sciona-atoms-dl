"""Tabular deep-learning atoms in pure numpy.

Implements entity embedding lookup from the Rossmann entity-embedding
solution, exposing categorical code to embedding-vector expansion as a
framework-agnostic preprocessing primitive.

Source: rossmann-entity-embed-1st/models.py (MIT)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_entity_embedding_lookup


@register_atom(witness_entity_embedding_lookup)
@icontract.require(
    lambda codes: codes.ndim == 2,
    "codes must be a 2-D array",
)
@icontract.require(
    lambda codes, embedding_matrices: codes.shape[1] == len(embedding_matrices),
    "codes must have one column per embedding matrix",
)
@icontract.require(
    lambda codes: np.all(codes >= 0),
    "codes must be non-negative",
)
def entity_embedding_lookup(
    codes: NDArray[np.int64],
    embedding_matrices: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Look up per-field embeddings and concatenate them across fields.

    Derived from Rossmann `embed_features`, which replaces selected categorical
    fields with learned embedding vectors and concatenates them into a single
    tabular representation.
    """
    embeddings: list[NDArray[np.float64]] = []
    for field_idx, matrix in enumerate(embedding_matrices):
        field_codes = codes[:, field_idx]
        if np.any(field_codes >= matrix.shape[0]):
            raise ValueError("codes contain an index outside an embedding matrix")
        embeddings.append(matrix[field_codes])
    return np.concatenate(embeddings, axis=1)
