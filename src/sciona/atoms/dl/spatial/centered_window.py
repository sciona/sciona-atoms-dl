"""Explicit centered tensor windows without model or domain assumptions."""
import numbers
import torch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _slices(shape, axes, widths):
    axes, widths = tuple(axes), tuple(widths)
    if not axes or len(axes) != len(widths):
        raise ValueError('Matching nonempty axes and widths required')
    if any(isinstance(a, bool) or not isinstance(a, numbers.Integral) or not 0 <= a < len(shape) for a in axes):
        raise ValueError('Explicit nonnegative axes within tensor rank required')
    if len(set(axes)) != len(axes):
        raise ValueError('Unique axes required')
    slices = [slice(None)] * len(shape)
    result_shape = list(shape)
    for axis, width in zip(axes, widths):
        if isinstance(width, bool) or not isinstance(width, numbers.Integral) or not 1 <= width <= shape[axis]:
            raise ValueError('Positive integer window widths within known dimensions required')
        start = shape[axis] // 2 - width // 2
        slices[axis] = slice(start, start + width)
        result_shape[axis] = width
    return tuple(slices), tuple(result_shape)


def witness_centered_window(values: AbstractArray, axes: object, widths: object) -> AbstractArray:
    _, shape = _slices(values.shape, axes, widths)
    return AbstractArray(shape=shape, dtype=values.dtype)


@register_atom(witness_centered_window)
def centered_window(values: object, axes: object, widths: object) -> object:
    """Return a tensor view cropped around each selected axis's integer center.

    Each window starts at dimension//2-width//2 and contains exactly width
    elements. This explicitly determines even/odd alignment. Unselected axes,
    dtype, device and autograd are preserved. No padding, copying, reduction,
    channel interpretation or normalization. Returned storage aliases the input.
    """
    if not isinstance(values, torch.Tensor):
        raise ValueError('Torch tensor required')
    slices, _ = _slices(values.shape, axes, widths)
    return values[slices]
