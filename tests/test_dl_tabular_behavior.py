from __future__ import annotations

import numpy as np
import pytest


def test_tabular_atoms_import() -> None:
    from sciona.atoms.dl.tabular.atoms import entity_embedding_lookup

    assert callable(entity_embedding_lookup)


def test_entity_embedding_lookup_concatenates_fields() -> None:
    from sciona.atoms.dl.tabular.atoms import entity_embedding_lookup

    codes = np.array([[0, 1], [1, 0]], dtype=np.int64)
    embeddings = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[10.0], [20.0]], dtype=np.float64),
    ]
    result = entity_embedding_lookup(codes, embeddings)
    expected = np.array([[1.0, 2.0, 20.0], [3.0, 4.0, 10.0]], dtype=np.float64)
    assert np.array_equal(result, expected)


def test_entity_embedding_lookup_raises_on_out_of_range_code() -> None:
    from sciona.atoms.dl.tabular.atoms import entity_embedding_lookup

    codes = np.array([[2]], dtype=np.int64)
    embeddings = [np.array([[1.0], [2.0]], dtype=np.float64)]
    with pytest.raises(ValueError):
        entity_embedding_lookup(codes, embeddings)
