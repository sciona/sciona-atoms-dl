"""Morphological skeletonization and road-centerline graph extraction atoms."""

from __future__ import annotations

from collections.abc import Iterable

import icontract
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.morphology import medial_axis, skeletonize

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_medial_axis_2d,
    witness_skeleton_to_graph,
    witness_skeletonize_2d,
)


Pixel = tuple[int, int]


def _is_binary_2d(mask: NDArray[np.bool_] | NDArray[np.integer]) -> bool:
    values = np.asarray(mask)
    return bool(values.ndim == 2 and np.all((values == 0) | (values == 1)))


def _binary(values: NDArray[np.bool_] | NDArray[np.integer]) -> NDArray[np.bool_]:
    return np.asarray(values).astype(bool)


def _subset_of_mask(result: NDArray[np.bool_], mask: NDArray[np.bool_] | NDArray[np.integer]) -> bool:
    return bool(np.all(~np.asarray(result, dtype=bool) | _binary(mask)))


def _medial_result_valid(
    result: NDArray[np.bool_] | tuple[NDArray[np.bool_], NDArray[np.float64]],
    mask: NDArray[np.bool_] | NDArray[np.integer],
    return_distance: bool,
) -> bool:
    shape = np.asarray(mask).shape
    if return_distance:
        if not isinstance(result, tuple) or len(result) != 2:
            return False
        skeleton, distance = result
        return bool(
            np.asarray(skeleton).shape == shape
            and np.asarray(distance).shape == shape
            and _subset_of_mask(np.asarray(skeleton, dtype=bool), mask)
            and np.all(np.asarray(distance, dtype=np.float64) >= 0.0)
        )
    return bool(not isinstance(result, tuple) and np.asarray(result).shape == shape and _subset_of_mask(np.asarray(result, dtype=bool), mask))


def _neighbor_pixels(pixel: Pixel, shape: tuple[int, int]) -> Iterable[Pixel]:
    row, col = pixel
    height, width = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < height and 0 <= cc < width:
                yield (rr, cc)


def _foreground_neighbors(skeleton: NDArray[np.bool_], pixel: Pixel) -> list[Pixel]:
    return [other for other in _neighbor_pixels(pixel, skeleton.shape) if bool(skeleton[other])]


def _edge_weight(path: list[Pixel]) -> float:
    total = 0.0
    for left, right in zip(path, path[1:]):
        total += float(np.hypot(right[0] - left[0], right[1] - left[1]))
    return total


def _add_edge_if_new(
    graph: nx.MultiGraph,
    seen_segments: set[frozenset[Pixel]],
    start_node: int,
    end_node: int,
    path: list[Pixel],
) -> None:
    segments = {frozenset((left, right)) for left, right in zip(path, path[1:])}
    if start_node == end_node:
        seen_segments.update(segments)
        return
    if not segments or segments.issubset(seen_segments):
        return
    seen_segments.update(segments)
    graph.add_edge(
        start_node,
        end_node,
        pts=np.asarray(path, dtype=np.int64),
        weight=_edge_weight(path),
    )


@register_atom(witness_skeletonize_2d)
@icontract.require(lambda mask: _is_binary_2d(mask), "mask must be a binary 2D array")
@icontract.ensure(lambda result, mask: result.shape == np.asarray(mask).shape, "skeleton must preserve shape")
@icontract.ensure(lambda result, mask: _subset_of_mask(result, mask), "skeleton cannot add foreground pixels")
def skeletonize_2d(mask: NDArray[np.bool_] | NDArray[np.integer]) -> NDArray[np.bool_]:
    """Thin a binary 2D mask to a one-pixel-wide skeleton.

    The atom uses scikit-image's deterministic Zhang method, which is the
    standard choice for 2D road-mask centerline extraction.
    """
    return np.asarray(skeletonize(_binary(mask), method="zhang"), dtype=np.bool_)


@register_atom(witness_medial_axis_2d)
@icontract.require(lambda mask: _is_binary_2d(mask), "mask must be a binary 2D array")
@icontract.require(lambda rng_seed: rng_seed >= 0, "rng_seed must be nonnegative")
@icontract.ensure(lambda result, mask, return_distance: _medial_result_valid(result, mask, return_distance), "medial axis output must preserve mask shape")
def medial_axis_2d(
    mask: NDArray[np.bool_] | NDArray[np.integer],
    return_distance: bool = False,
    rng_seed: int = 42,
) -> NDArray[np.bool_] | tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Compute the 2D medial axis with deterministic tie-breaking.

    Passing an explicit seed removes scikit-image's default random tie-breaker
    from the atom boundary. With `return_distance=True`, the distance map is
    returned with the skeleton for local width analysis.
    """
    result = medial_axis(
        _binary(mask),
        return_distance=return_distance,
        rng=int(rng_seed),
    )
    if return_distance:
        skeleton, distance = result
        return np.asarray(skeleton, dtype=np.bool_), np.asarray(distance, dtype=np.float64)
    return np.asarray(result, dtype=np.bool_)


@register_atom(witness_skeleton_to_graph)
@icontract.require(lambda skeleton: _is_binary_2d(skeleton), "skeleton must be a binary 2D array")
@icontract.require(lambda skeleton: np.sum(skeleton) > 0, "skeleton must contain foreground pixels")
@icontract.ensure(lambda result: isinstance(result, nx.MultiGraph), "result must be a NetworkX MultiGraph")
@icontract.ensure(lambda result: all("pts" in data and "o" in data for _, data in result.nodes(data=True)), "nodes must include pixel sets and centroids")
@icontract.ensure(lambda result: all("pts" in data and "weight" in data and data["weight"] >= 0.0 for _, _, data in result.edges(data=True)), "edges must include paths and lengths")
def skeleton_to_graph(skeleton: NDArray[np.bool_] | NDArray[np.integer]) -> nx.MultiGraph:
    """Convert a binary skeleton image into a pixel-centerline graph.

    Pixels with degree other than two become graph nodes. Adjacent node pixels
    are merged into junction clusters; degree-two chains become weighted edges
    carrying their ordered pixel paths.
    """
    binary = _binary(skeleton)
    neighbor_count = ndimage.convolve(
        binary.astype(np.int16),
        np.ones((3, 3), dtype=np.int16),
        mode="constant",
        cval=0,
    ) - binary.astype(np.int16)
    node_mask = binary & (neighbor_count != 2)
    labels, count = ndimage.label(node_mask, structure=np.ones((3, 3), dtype=np.int8))

    graph = nx.MultiGraph()
    pixel_to_node: dict[Pixel, int] = {}
    for label in range(1, int(count) + 1):
        pts = np.argwhere(labels == label).astype(np.int64)
        node_id = label - 1
        centroid = np.mean(pts, axis=0)
        graph.add_node(node_id, pts=pts, o=centroid)
        for row, col in pts:
            pixel_to_node[(int(row), int(col))] = node_id

    seen_segments: set[frozenset[Pixel]] = set()
    for start_pixel, start_node in sorted(pixel_to_node.items()):
        for first_step in _foreground_neighbors(binary, start_pixel):
            first_segment = frozenset((start_pixel, first_step))
            if first_segment in seen_segments:
                continue
            if first_step in pixel_to_node:
                _add_edge_if_new(
                    graph,
                    seen_segments,
                    start_node,
                    pixel_to_node[first_step],
                    [start_pixel, first_step],
                )
                continue

            previous = start_pixel
            current = first_step
            path = [start_pixel, current]
            while current not in pixel_to_node:
                candidates = [pixel for pixel in _foreground_neighbors(binary, current) if pixel != previous]
                if not candidates:
                    break
                next_pixel = candidates[0]
                previous, current = current, next_pixel
                path.append(current)
                if len(path) > int(binary.size) + 1:
                    break
            if current in pixel_to_node:
                _add_edge_if_new(
                    graph,
                    seen_segments,
                    start_node,
                    pixel_to_node[current],
                    path,
                )

    return graph
