"""3D perception primitives for point clouds, BEV rasters, and geometric checks."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_project_image_to_points,
    witness_ransac_homography,
    witness_rasterize_bev,
    witness_voxelize_point_cloud,
)


PointRange = tuple[float, float, float, float, float, float]
RasterSize = tuple[int, int]


def _is_points_2d(points: NDArray[np.floating]) -> bool:
    values = np.asarray(points)
    return bool(values.ndim == 2 and values.shape[1] >= 3 and np.all(np.isfinite(values)))


def _is_positive_triplet(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.shape == (3,) and np.all(np.isfinite(array)) and np.all(array > 0.0))


def _is_valid_point_range(point_range: PointRange) -> bool:
    array = np.asarray(point_range, dtype=np.float64)
    return bool(array.shape == (6,) and np.all(np.isfinite(array)) and np.all(array[:3] < array[3:]))


def _voxel_result_valid(
    result: tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]],
    points: NDArray[np.floating],
    max_points_per_voxel: int,
) -> bool:
    features, coords, counts = result
    return bool(
        features.ndim == 3
        and coords.ndim == 2
        and counts.ndim == 1
        and features.shape[0] == coords.shape[0] == counts.shape[0]
        and features.shape[1] == max_points_per_voxel
        and features.shape[2] == np.asarray(points).shape[1]
        and coords.shape[1] == 3
        and np.all(counts >= 0)
        and np.all(counts <= max_points_per_voxel)
    )


def _is_intrinsic_matrix(intrinsic: NDArray[np.floating]) -> bool:
    values = np.asarray(intrinsic)
    return bool(values.shape in {(3, 3), (3, 4)} and np.all(np.isfinite(values)))


def _is_extrinsic_matrix(extrinsic: NDArray[np.floating]) -> bool:
    values = np.asarray(extrinsic)
    return bool(values.shape == (4, 4) and np.all(np.isfinite(values)))


def _is_image_features(image_features: NDArray[np.floating]) -> bool:
    values = np.asarray(image_features)
    return bool(values.ndim == 3 and values.shape[0] > 0 and values.shape[1] > 0 and np.all(np.isfinite(values)))


def _is_image_shape(image_shape: RasterSize) -> bool:
    return bool(len(image_shape) == 2 and image_shape[0] > 0 and image_shape[1] > 0)


def _projection_result_valid(
    result: NDArray[np.floating],
    points_3d: NDArray[np.floating],
    image_features: NDArray[np.floating],
) -> bool:
    return bool(
        result.shape == (np.asarray(points_3d).shape[0], np.asarray(points_3d).shape[1] + np.asarray(image_features).shape[2])
        and np.all(np.isfinite(result))
    )


def _is_agent_tensor(agents: NDArray[np.floating]) -> bool:
    values = np.asarray(agents)
    return bool(values.ndim == 3 and values.shape[2] >= 2 and np.all(np.isfinite(values)))


def _is_polyline_list(map_elements: Sequence[NDArray[np.floating]]) -> bool:
    for polyline in map_elements:
        values = np.asarray(polyline)
        if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
            return False
    return True


def _is_ego_pose(ego_pose: NDArray[np.floating]) -> bool:
    values = np.asarray(ego_pose)
    return bool(values.shape == (3,) and np.all(np.isfinite(values)))


def _is_raster_size(raster_size: RasterSize) -> bool:
    return bool(len(raster_size) == 2 and raster_size[0] > 0 and raster_size[1] > 0)


def _bev_result_valid(result: NDArray[np.uint8], raster_size: RasterSize) -> bool:
    return bool(result.shape == (2, raster_size[0], raster_size[1]) and np.all((result == 0) | (result == 1)))


def _is_point_pairs(src_points: NDArray[np.floating], dst_points: NDArray[np.floating]) -> bool:
    src = np.asarray(src_points)
    dst = np.asarray(dst_points)
    return bool(src.shape == dst.shape and src.ndim == 2 and src.shape[0] >= 4 and src.shape[1] == 2 and np.all(np.isfinite(src)) and np.all(np.isfinite(dst)))


def _homography_result_valid(result: tuple[NDArray[np.float64], NDArray[np.bool_]], count: int) -> bool:
    matrix, inliers = result
    return bool(matrix.shape == (3, 3) and inliers.shape == (count,) and inliers.dtype == np.bool_ and np.all(np.isfinite(matrix)))


def _homogeneous(points: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.column_stack([np.asarray(points, dtype=np.float64), np.ones(points.shape[0], dtype=np.float64)])


def _normalize_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = (transform @ _homogeneous(points).T).T[:, :2]
    return normalized, transform


def _compute_homography(src_points: NDArray[np.float64], dst_points: NDArray[np.float64]) -> NDArray[np.float64]:
    src_norm, src_transform = _normalize_points(src_points)
    dst_norm, dst_transform = _normalize_points(dst_points)
    rows: list[list[float]] = []
    for (x_coord, y_coord), (u_coord, v_coord) in zip(src_norm, dst_norm):
        rows.append([-x_coord, -y_coord, -1.0, 0.0, 0.0, 0.0, u_coord * x_coord, u_coord * y_coord, u_coord])
        rows.append([0.0, 0.0, 0.0, -x_coord, -y_coord, -1.0, v_coord * x_coord, v_coord * y_coord, v_coord])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = vh[-1].reshape(3, 3)
    denormalized = np.linalg.inv(dst_transform) @ matrix @ src_transform
    if abs(float(denormalized[2, 2])) > 1e-12:
        denormalized = denormalized / denormalized[2, 2]
    return denormalized


def _apply_homography(matrix: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    projected = (matrix @ _homogeneous(points).T).T
    denom = projected[:, 2:3]
    safe = np.where(np.abs(denom) > 1e-12, denom, np.nan)
    return projected[:, :2] / safe


def _world_to_pixel(
    points: NDArray[np.floating],
    ego_pose: NDArray[np.floating],
    raster_size: RasterSize,
    resolution: float,
) -> NDArray[np.int64]:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    ego = np.asarray(ego_pose, dtype=np.float64)
    shifted = xy - ego[:2]
    yaw = float(ego[2])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x_ego = cos_yaw * shifted[:, 0] + sin_yaw * shifted[:, 1]
    y_ego = -sin_yaw * shifted[:, 0] + cos_yaw * shifted[:, 1]
    height, width = raster_size
    cols = np.rint((width - 1) / 2.0 + x_ego / resolution).astype(np.int64)
    rows = np.rint((height - 1) / 2.0 - y_ego / resolution).astype(np.int64)
    return np.column_stack([rows, cols])


def _mark_pixel(canvas: NDArray[np.uint8], row: int, col: int) -> None:
    if 0 <= row < canvas.shape[0] and 0 <= col < canvas.shape[1]:
        canvas[row, col] = 1


def _draw_line(canvas: NDArray[np.uint8], start: NDArray[np.int64], end: NDArray[np.int64]) -> None:
    row0, col0 = int(start[0]), int(start[1])
    row1, col1 = int(end[0]), int(end[1])
    steps = max(abs(row1 - row0), abs(col1 - col0))
    if steps == 0:
        _mark_pixel(canvas, row0, col0)
        return
    for step in range(steps + 1):
        alpha = step / steps
        row = int(round(row0 + alpha * (row1 - row0)))
        col = int(round(col0 + alpha * (col1 - col0)))
        _mark_pixel(canvas, row, col)


@register_atom(witness_voxelize_point_cloud)
@icontract.require(lambda points: _is_points_2d(points), "points must be a finite N x D array with xyz columns")
@icontract.require(lambda voxel_size: _is_positive_triplet(voxel_size), "voxel_size must contain three positive finite widths")
@icontract.require(lambda point_range: _is_valid_point_range(point_range), "point_range must be finite xyz mins followed by larger xyz maxes")
@icontract.require(lambda max_points_per_voxel: max_points_per_voxel > 0, "max_points_per_voxel must be positive")
@icontract.ensure(lambda result, points, max_points_per_voxel: _voxel_result_valid(result, points, max_points_per_voxel), "voxel tensors must have consistent shapes and counts")
def voxelize_point_cloud(
    points: NDArray[np.floating],
    voxel_size: tuple[float, float, float],
    point_range: PointRange,
    max_points_per_voxel: int,
) -> tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]]:
    """Group finite point-cloud rows into deterministic hard voxels.

    Points outside `point_range` are dropped. For each occupied voxel, points
    are stored in first-observed order up to `max_points_per_voxel`; overflow
    points contribute only to the saturated count.
    """
    values = np.asarray(points)
    mins = np.asarray(point_range[:3], dtype=np.float64)
    maxs = np.asarray(point_range[3:], dtype=np.float64)
    widths = np.asarray(voxel_size, dtype=np.float64)
    grid = np.floor((maxs - mins) / widths).astype(np.int64)
    coords = np.floor((values[:, :3].astype(np.float64) - mins) / widths).astype(np.int64)
    valid = np.all((coords >= 0) & (coords < grid), axis=1)
    valid_points = values[valid]
    valid_coords = coords[valid]

    if valid_points.shape[0] == 0:
        return (
            np.zeros((0, max_points_per_voxel, values.shape[1]), dtype=values.dtype),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    hashes = valid_coords[:, 0] * (grid[1] * grid[2]) + valid_coords[:, 1] * grid[2] + valid_coords[:, 2]
    unique_hashes, first_indices, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    del unique_hashes
    order = np.argsort(first_indices)
    old_to_new = np.empty(order.shape[0], dtype=np.int64)
    old_to_new[order] = np.arange(order.shape[0], dtype=np.int64)
    voxel_ids = old_to_new[inverse]

    voxel_count = int(order.shape[0])
    voxel_features = np.zeros((voxel_count, max_points_per_voxel, values.shape[1]), dtype=values.dtype)
    fill_counts = np.zeros(voxel_count, dtype=np.int64)
    for point, voxel_id in zip(valid_points, voxel_ids):
        slot = int(fill_counts[voxel_id])
        if slot < max_points_per_voxel:
            voxel_features[voxel_id, slot] = point
        fill_counts[voxel_id] += 1

    voxel_coords = valid_coords[first_indices[order]].astype(np.int64)
    num_points = np.minimum(fill_counts, max_points_per_voxel).astype(np.int64)
    return voxel_features, voxel_coords, num_points


@register_atom(witness_project_image_to_points)
@icontract.require(lambda points_3d: _is_points_2d(points_3d), "points_3d must be a finite N x D array with xyz columns")
@icontract.require(lambda image_features: _is_image_features(image_features), "image_features must be a finite H x W x C tensor")
@icontract.require(lambda intrinsic: _is_intrinsic_matrix(intrinsic), "intrinsic must be finite 3x3 or 3x4 camera matrix")
@icontract.require(lambda extrinsic: _is_extrinsic_matrix(extrinsic), "extrinsic must be a finite 4x4 transform")
@icontract.require(lambda image_shape: _is_image_shape(image_shape), "image_shape must be positive height and width")
@icontract.ensure(lambda result, points_3d, image_features: _projection_result_valid(result, points_3d, image_features), "painted points must append one vector per point")
def project_image_to_points(
    points_3d: NDArray[np.floating],
    image_features: NDArray[np.floating],
    intrinsic: NDArray[np.floating],
    extrinsic: NDArray[np.floating],
    image_shape: RasterSize,
) -> NDArray[np.floating]:
    """Append bilinearly sampled image features to projected 3D points.

    Points are transformed with `extrinsic`, projected with `intrinsic`, and
    sampled only when the projected pixel lands inside `image_shape` with
    positive camera depth. Invalid projections receive zero image features.
    """
    points = np.asarray(points_3d, dtype=np.float64)
    features = np.asarray(image_features, dtype=np.float64)
    height, width = image_shape
    camera_points = (np.asarray(extrinsic, dtype=np.float64) @ np.column_stack([points[:, :3], np.ones(points.shape[0])]).T).T
    intrinsic_matrix = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic_matrix.shape == (3, 3):
        projected = (intrinsic_matrix @ camera_points[:, :3].T).T
    else:
        projected = (intrinsic_matrix @ camera_points.T).T

    sampled = np.zeros((points.shape[0], features.shape[2]), dtype=np.float64)
    depth = camera_points[:, 2]
    denom = projected[:, 2]
    valid = (depth > 1e-12) & (np.abs(denom) > 1e-12)
    pixels = projected[:, :2] / denom[:, None]
    x_coords = pixels[:, 0]
    y_coords = pixels[:, 1]
    valid &= (x_coords >= 0.0) & (x_coords <= width - 1) & (y_coords >= 0.0) & (y_coords <= height - 1)

    for index in np.flatnonzero(valid):
        x_coord = float(x_coords[index])
        y_coord = float(y_coords[index])
        x0 = int(np.floor(x_coord))
        y0 = int(np.floor(y_coord))
        x1 = min(x0 + 1, features.shape[1] - 1)
        y1 = min(y0 + 1, features.shape[0] - 1)
        x_weight = x_coord - x0
        y_weight = y_coord - y0
        top = (1.0 - x_weight) * features[y0, x0] + x_weight * features[y0, x1]
        bottom = (1.0 - x_weight) * features[y1, x0] + x_weight * features[y1, x1]
        sampled[index] = (1.0 - y_weight) * top + y_weight * bottom

    return np.concatenate([points, sampled], axis=1)


@register_atom(witness_rasterize_bev)
@icontract.require(lambda agents: _is_agent_tensor(agents), "agents must be a finite A x T x K tensor with xy columns")
@icontract.require(lambda map_elements: _is_polyline_list(map_elements), "map_elements must be finite polylines with xy columns")
@icontract.require(lambda ego_pose: _is_ego_pose(ego_pose), "ego_pose must be finite x, y, yaw")
@icontract.require(lambda raster_size: _is_raster_size(raster_size), "raster_size must contain positive height and width")
@icontract.require(lambda resolution: resolution > 0.0 and np.isfinite(resolution), "resolution must be a positive finite metres-per-pixel value")
@icontract.ensure(lambda result, raster_size: _bev_result_valid(result, raster_size), "BEV raster must be a binary two-channel image")
def rasterize_bev(
    agents: NDArray[np.floating],
    map_elements: Sequence[NDArray[np.floating]],
    ego_pose: NDArray[np.floating],
    raster_size: RasterSize,
    resolution: float,
) -> NDArray[np.uint8]:
    """Rasterize agent tracks and map polylines into an ego-centered BEV image.

    Channel 0 contains agent trajectory segments. Channel 1 contains static
    map polylines. Coordinates are translated and yaw-rotated by `ego_pose`
    before conversion to pixels at the requested resolution.
    """
    height, width = raster_size
    raster = np.zeros((2, height, width), dtype=np.uint8)
    for trajectory in np.asarray(agents, dtype=np.float64):
        pixels = _world_to_pixel(trajectory[:, :2], ego_pose, raster_size, resolution)
        for start, end in zip(pixels, pixels[1:]):
            _draw_line(raster[0], start, end)
        if pixels.shape[0] == 1:
            _mark_pixel(raster[0], int(pixels[0, 0]), int(pixels[0, 1]))

    for polyline in map_elements:
        points = np.asarray(polyline, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        pixels = _world_to_pixel(points[:, :2], ego_pose, raster_size, resolution)
        for start, end in zip(pixels, pixels[1:]):
            _draw_line(raster[1], start, end)
        if pixels.shape[0] == 1:
            _mark_pixel(raster[1], int(pixels[0, 0]), int(pixels[0, 1]))
    return raster


@register_atom(witness_ransac_homography)
@icontract.require(lambda src_points, dst_points: _is_point_pairs(src_points, dst_points), "src_points and dst_points must be finite paired N x 2 arrays with N >= 4")
@icontract.require(lambda threshold: threshold > 0.0 and np.isfinite(threshold), "threshold must be positive and finite")
@icontract.require(lambda max_iterations: max_iterations > 0, "max_iterations must be positive")
@icontract.ensure(lambda result, src_points: _homography_result_valid(result, np.asarray(src_points).shape[0]), "homography and inlier mask must have valid shapes")
def ransac_homography(
    src_points: NDArray[np.floating],
    dst_points: NDArray[np.floating],
    threshold: float,
    max_iterations: int,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Estimate a planar homography while rejecting geometric outliers.

    Each RANSAC iteration samples four correspondences, fits a normalized DLT
    homography, and scores reprojection error. The final matrix is refit on
    the best inlier set when at least four inliers are available.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_matrix = _compute_homography(src[:4], dst[:4])
    best_inliers = np.zeros(src.shape[0], dtype=np.bool_)
    best_count = -1

    for _ in range(max_iterations):
        sample = rng.choice(src.shape[0], size=4, replace=False)
        try:
            matrix = _compute_homography(src[sample], dst[sample])
            projected = _apply_homography(matrix, src)
        except np.linalg.LinAlgError:
            continue
        errors = np.linalg.norm(projected - dst, axis=1)
        errors = np.where(np.isfinite(errors), errors, np.inf)
        inliers = errors <= threshold
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_matrix = matrix
            best_inliers = inliers.astype(np.bool_)

    if int(np.sum(best_inliers)) >= 4:
        best_matrix = _compute_homography(src[best_inliers], dst[best_inliers])
    return best_matrix.astype(np.float64), best_inliers.astype(np.bool_)
