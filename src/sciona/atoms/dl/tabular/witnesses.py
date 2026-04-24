"""Ghost witnesses for tabular atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_entity_embedding_lookup(
    codes: AbstractArray,
    embedding_matrices: list[AbstractArray],
) -> AbstractArray:
    """Ghost witness for entity embedding lookup."""
    return codes
