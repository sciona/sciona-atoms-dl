"""Explicit candidate-index ranking without task-specific labels or batch policy."""
import numbers
import torch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _count(k):
    if isinstance(k,bool) or not isinstance(k,numbers.Integral) or k<0:
        raise ValueError('Nonnegative integer selection count required')
    return int(k)


def witness_highest_scoring_indices(scores: AbstractArray,candidate_indices: AbstractArray,k: int) -> AbstractArray:
    k=_count(k)
    if len(scores.shape)!=1 or len(candidate_indices.shape)!=1 or min(scores.shape+candidate_indices.shape)<0:
        raise ValueError('Known one-dimensional score and index shapes required')
    if candidate_indices.dtype!='int64':raise ValueError('Int64 candidate indices required')
    return AbstractArray(shape=(min(k,candidate_indices.shape[0]),),dtype='int64')


@register_atom(witness_highest_scoring_indices)
def highest_scoring_indices(scores: object,candidate_indices: object,k: int) -> object:
    """Return up to k candidate indices ranked by descending finite score.

    Candidates are unique int64 indices into a floating score vector on the
    same device. Empty candidates and zero k yield empty indices. Uses native
    torch.topk ordering, including its tie behavior; no stable or cross-device
    tie ordering is promised. Selection is discrete, but gathering scores by
    the returned indices preserves their ordinary autograd path. No label or
    batch-size interpretation is built in; the caller determines candidates/k.
    """
    k=_count(k)
    if (not isinstance(scores,torch.Tensor) or scores.ndim!=1 or not scores.is_floating_point()
            or not bool(torch.isfinite(scores).all())):
        raise ValueError('Finite floating score vector required')
    if (not isinstance(candidate_indices,torch.Tensor) or candidate_indices.ndim!=1
            or candidate_indices.dtype!=torch.int64 or candidate_indices.device!=scores.device
            or bool((candidate_indices<0).any()) or bool((candidate_indices>=scores.numel()).any())
            or torch.unique(candidate_indices).numel()!=candidate_indices.numel()):
        raise ValueError('Unique in-range int64 candidate indices on the score device required')
    ranked=torch.topk(scores[candidate_indices],min(k,candidate_indices.numel())).indices
    return candidate_indices[ranked]
