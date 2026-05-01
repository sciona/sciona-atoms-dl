"""Text similarity primitives in pure Python.

Implements string distance and similarity metrics for text matching
pipelines: Levenshtein edit distance and Jaro-Winkler similarity.
"""

from __future__ import annotations

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_jaro_winkler_similarity,
    witness_levenshtein_distance,
)


@register_atom(witness_levenshtein_distance)
@icontract.require(lambda s1: isinstance(s1, str), "s1 must be a string")
@icontract.require(lambda s2: isinstance(s2, str), "s2 must be a string")
@icontract.ensure(lambda result: result >= 0, "edit distance must be non-negative")
@icontract.ensure(
    lambda result, s1, s2: result <= max(len(s1), len(s2)),
    "edit distance cannot exceed the longer input",
)
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute minimum edit distance between two strings.

    Standard dynamic programming implementation with O(min(m,n)) space.
    """
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for index1, char1 in enumerate(s1, start=1):
        current = [index1]
        for index2, char2 in enumerate(s2, start=1):
            insert_cost = current[index2 - 1] + 1
            delete_cost = previous[index2] + 1
            replace_cost = previous[index2 - 1] + (char1 != char2)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return int(previous[-1])


def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    match_distance = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    for i, c1 in enumerate(s1):
        start = max(0, i - match_distance)
        stop = min(i + match_distance + 1, len(s2))
        for j in range(start, stop):
            if s2_matches[j] or c1 != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    s1_matched = [c for c, m in zip(s1, s1_matches) if m]
    s2_matched = [c for c, m in zip(s2, s2_matches) if m]
    transpositions = sum(c1 != c2 for c1, c2 in zip(s1_matched, s2_matched)) / 2.0
    return (matches / len(s1) + matches / len(s2) + (matches - transpositions) / matches) / 3.0


@register_atom(witness_jaro_winkler_similarity)
@icontract.require(lambda s1: isinstance(s1, str), "s1 must be a string")
@icontract.require(lambda s2: isinstance(s2, str), "s2 must be a string")
@icontract.require(lambda p: 0.0 <= p <= 0.25, "p must be in [0, 0.25]")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "similarity must be in [0, 1]")
def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler string similarity score in [0, 1].

    Extends Jaro similarity with a prefix bonus for strings that
    share a common prefix up to 4 characters.
    """
    jaro = _jaro_similarity(s1, s2)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 != c2:
            break
        prefix += 1
    score = jaro + prefix * p * (1.0 - jaro)
    return float(min(1.0, max(0.0, score)))
