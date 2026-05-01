"""Ghost witnesses for skeletonization atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def _check_2d(values: AbstractArray, name: str) -> None:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")


def witness_skeletonize_2d(mask: AbstractArray) -> AbstractArray:
    """Describe shape-preserving binary skeletonization."""
    _check_2d(mask, "mask")
    return AbstractArray(shape=mask.shape, dtype="bool", min_val=0.0, max_val=1.0)


def witness_medial_axis_2d(
    mask: AbstractArray,
    return_distance: AbstractScalar | None = None,
    rng_seed: AbstractScalar | None = None,
) -> AbstractArray:
    """Describe the skeleton component of a 2D medial-axis transform."""
    del return_distance, rng_seed
    _check_2d(mask, "mask")
    return AbstractArray(shape=mask.shape, dtype="bool", min_val=0.0, max_val=1.0)


def witness_skeleton_to_graph(skeleton: AbstractArray) -> AbstractScalar:
    """Describe skeleton graph extraction as producing one graph object."""
    _check_2d(skeleton, "skeleton")
    return AbstractScalar(dtype="object", min_val=1.0, max_val=1.0)
