"""Differentiable probability reductions with explicit caller-owned parameters."""
import torch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_noisy_or_with_baseline(probabilities: AbstractArray, baseline_logit: AbstractArray) -> AbstractArray:
    if len(probabilities.shape)!=2 or min(probabilities.shape)<1 or baseline_logit.shape!=(1,):
        raise ValueError('Nonempty B,K probabilities and a one-element baseline required')
    if probabilities.dtype!=baseline_logit.dtype:
        raise ValueError('Matching probability and baseline dtypes required')
    return AbstractArray(shape=(probabilities.shape[0],),dtype=probabilities.dtype,min_val=0,max_val=1)


@register_atom(witness_noisy_or_with_baseline)
def noisy_or_with_baseline(probabilities: object, baseline_logit: object) -> object:
    """Return 1-prod(1-p)*(1-sigmoid(b)) per row, preserving autograd.

    Inputs are nonempty finite B,K floating probability tensors in [0,1] and
    a finite one-element floating baseline logit on the same device/dtype.
    The baseline is caller-owned; this operation creates no trainable state.
    No clamping, detachment, normalization or hidden class/feature ordering.
    Probabilistic interpretation requires caller-established assumptions about
    the events; this algebra alone does not establish independence or calibration.
    """
    if (not isinstance(probabilities,torch.Tensor) or probabilities.ndim!=2
            or min(probabilities.shape)<1 or not probabilities.is_floating_point()
            or not bool(torch.isfinite(probabilities).all())
            or not bool(((probabilities>=0)&(probabilities<=1)).all())):
        raise ValueError('Nonempty finite B,K probability tensor in [0,1] required')
    if (not isinstance(baseline_logit,torch.Tensor) or baseline_logit.shape!=(1,)
            or baseline_logit.dtype!=probabilities.dtype or baseline_logit.device!=probabilities.device
            or not bool(torch.isfinite(baseline_logit).all())):
        raise ValueError('Finite matching one-element baseline logit required')
    baseline_probability=torch.sigmoid(baseline_logit)
    return 1-torch.prod(1-probabilities,dim=1)*(1-baseline_probability.expand(probabilities.size(0)))
