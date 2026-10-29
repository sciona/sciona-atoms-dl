"""Differentiable weighted log penalty with explicit threshold and offset."""
import numbers
import torch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _controls(threshold,offset):
    for value in (threshold,offset):
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not 0<value<float('inf'):
            raise ValueError('Finite positive threshold and offset required')
    if threshold>1:raise ValueError('Probability threshold must not exceed one')


def witness_thresholded_log_penalty(probabilities: AbstractArray,weights: AbstractArray,threshold: float,offset: float) -> AbstractArray:
    _controls(threshold,offset)
    if probabilities.dtype not in ('float32','float64') or len(probabilities.shape)!=2 or min(probabilities.shape)<1 or weights.shape!=probabilities.shape or weights.dtype!=probabilities.dtype:
        raise ValueError('Aligned nonempty B,K arrays with matching dtype required')
    return AbstractArray(shape=(),dtype=probabilities.dtype)


@register_atom(witness_thresholded_log_penalty)
def thresholded_log_penalty(probabilities: object,weights: object,threshold: float,offset: float) -> object:
    """Return -sum(float32(p<threshold)*weights*log(p+offset))/B/K.

    Finite B,K float32/float64 probabilities lie in [0,1]; finite weights lie in [0,1]
    and match shape, dtype and device. The strict mask has no gradient, while
    probability and weight paths retain autograd. Averages over all entries,
    including unselected entries. Inputs and control values remain unchanged.
    Offset is explicit; p+offset>1 can yield a negative contribution, so callers
    must select controls appropriate for their objective. No labels or domains
    are inferred. Float32 mask and ordered divisions preserve source arithmetic.
    """
    _controls(threshold,offset)
    if (not isinstance(probabilities,torch.Tensor) or probabilities.ndim!=2 or min(probabilities.shape)<1
            or probabilities.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(probabilities).all())
            or not bool(((probabilities>=0)&(probabilities<=1)).all())):
        raise ValueError('Nonempty finite B,K probabilities in [0,1] required')
    if (not isinstance(weights,torch.Tensor) or weights.shape!=probabilities.shape
            or weights.dtype!=probabilities.dtype or weights.device!=probabilities.device
            or not bool(torch.isfinite(weights).all()) or not bool(((weights>=0)&(weights<=1)).all())):
        raise ValueError('Matching finite weights in [0,1] required')
    shifted=probabilities+offset
    if not bool(torch.isfinite(shifted).all()) or not bool((shifted>0).all()):
        raise ValueError('Offset must give finite positive shifted probabilities in input dtype')
    mask=(probabilities<threshold).float()
    return -torch.sum(mask*weights*torch.log(shifted))/probabilities.shape[0]/probabilities.shape[1]
