from __future__ import annotations

import numpy as np

from sciona.atoms.dl.perception_3d import (
    project_image_to_points,
    ransac_homography,
    rasterize_bev,
    voxelize_point_cloud,
)


def test_voxelize_point_cloud_assigns_and_truncates_points() -> None:
    points = np.array(
        [
            [0.1, 0.1, 0.1, 10.0],
            [0.2, 0.2, 0.2, 11.0],
            [0.3, 0.3, 0.3, 12.0],
            [1.1, 0.1, 0.1, 20.0],
            [5.0, 5.0, 5.0, 99.0],
        ],
        dtype=np.float64,
    )

    features, coords, counts = voxelize_point_cloud(
        points,
        voxel_size=(1.0, 1.0, 1.0),
        point_range=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
        max_points_per_voxel=2,
    )

    assert coords.tolist() == [[0, 0, 0], [1, 0, 0]]
    assert counts.tolist() == [2, 1]
    np.testing.assert_allclose(features[0, :, 3], [10.0, 11.0])
    np.testing.assert_allclose(features[1, 0], points[3])


def test_project_image_to_points_appends_bilinear_features() -> None:
    yy, xx = np.mgrid[0:4, 0:4]
    image_features = (xx + yy).astype(np.float64)[..., None]
    points = np.array([[1.5, 1.5, 1.0, 7.0], [6.0, 6.0, 1.0, 8.0]], dtype=np.float64)

    painted = project_image_to_points(
        points,
        image_features,
        intrinsic=np.eye(3, dtype=np.float64),
        extrinsic=np.eye(4, dtype=np.float64),
        image_shape=(4, 4),
    )

    assert painted.shape == (2, 5)
    np.testing.assert_allclose(painted[0, :4], points[0])
    np.testing.assert_allclose(painted[0, 4], 3.0)
    np.testing.assert_allclose(painted[1, 4], 0.0)


def test_rasterize_bev_draws_agents_and_map_polylines() -> None:
    agents = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    map_elements = [np.array([[0.0, -1.0], [0.0, 1.0]], dtype=np.float64)]

    raster = rasterize_bev(
        agents,
        map_elements,
        ego_pose=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        raster_size=(9, 9),
        resolution=1.0,
    )

    assert raster.shape == (2, 9, 9)
    assert raster[0, 4, 4] == 1
    assert raster[0, 4, 6] == 1
    assert int(np.sum(raster[1])) >= 3


def test_ransac_homography_recovers_translation_and_rejects_outliers() -> None:
    inlier_src = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.5],
            [-1.0, 1.5],
        ],
        dtype=np.float64,
    )
    inlier_dst = inlier_src + np.array([2.0, -1.0], dtype=np.float64)
    outlier_src = np.array([[4.0, 4.0], [5.0, 4.0], [4.0, 5.0]], dtype=np.float64)
    outlier_dst = np.array([[-10.0, 1.0], [-9.0, 2.0], [-8.0, 1.0]], dtype=np.float64)
    src = np.vstack([inlier_src, outlier_src])
    dst = np.vstack([inlier_dst, outlier_dst])

    matrix, inliers = ransac_homography(src, dst, threshold=1e-6, max_iterations=200, seed=3)

    projected = (matrix @ np.column_stack([inlier_src, np.ones(inlier_src.shape[0])]).T).T
    projected = projected[:, :2] / projected[:, 2:3]
    np.testing.assert_allclose(projected, inlier_dst, atol=1e-6)
    assert np.all(inliers[: inlier_src.shape[0]])
    assert not np.any(inliers[inlier_src.shape[0] :])
