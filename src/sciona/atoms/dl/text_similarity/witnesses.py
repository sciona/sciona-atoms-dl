"""Ghost witnesses for text similarity atoms."""

from __future__ import annotations


def witness_levenshtein_distance(
    s1: str,
    s2: str,
) -> int:
    """Ghost witness for Levenshtein edit distance."""
    return 0


def witness_jaro_winkler_similarity(
    s1: str,
    s2: str,
    p: float = 0.1,
) -> float:
    """Ghost witness for Jaro-Winkler similarity."""
    return 1.0
