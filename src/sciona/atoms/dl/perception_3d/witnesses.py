"""Ghost witnesses for 3D perception atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def _check_rank(values: AbstractArray, rank: int, name: str) -> None:
    if len(values.shape) != rank:
        raise ValueError(f"{name} must have rank {rank}")


def witness_voxelize_point_cloud(
    points: AbstractArray,
    voxel_size: AbstractArray,
    point_range: AbstractArray,
    max_points_per_voxel: AbstractScalar,
) -> AbstractArray:
    """Describe hard voxelization as a rank-3 point grouping transform."""
    del voxel_size, point_range, max_points_per_voxel
    _check_rank(points, 2, "points")
    return AbstractArray(shape=("num_voxels", "max_points_per_voxel", points.shape[1]), dtype=points.dtype)


def witness_project_image_to_points(
    points_3d: AbstractArray,
    image_features: AbstractArray,
    intrinsic: AbstractArray,
    extrinsic: AbstractArray,
    image_shape: AbstractScalar,
) -> AbstractArray:
    """Describe point painting as appending sampled image channels to points."""
    del intrinsic, extrinsic, image_shape
    _check_rank(points_3d, 2, "points_3d")
    _check_rank(image_features, 3, "image_features")
    return AbstractArray(shape=(points_3d.shape[0], "point_dim_plus_image_channels"), dtype=points_3d.dtype)


def witness_rasterize_bev(
    agents: AbstractArray,
    map_elements: AbstractScalar,
    ego_pose: AbstractArray,
    raster_size: AbstractScalar,
    resolution: AbstractScalar,
) -> AbstractArray:
    """Describe BEV rasterization as a two-channel image-like tensor."""
    del map_elements, ego_pose, raster_size, resolution
    _check_rank(agents, 3, "agents")
    return AbstractArray(shape=(2, "height", "width"), dtype="uint8", min_val=0.0, max_val=1.0)


def witness_ransac_homography(
    src_points: AbstractArray,
    dst_points: AbstractArray,
    threshold: AbstractScalar,
    max_iterations: AbstractScalar,
    seed: AbstractScalar,
) -> AbstractArray:
    """Describe spatial verification as returning a 3x3 transform."""
    del dst_points, threshold, max_iterations, seed
    _check_rank(src_points, 2, "src_points")
    return AbstractArray(shape=(3, 3), dtype="float64")
