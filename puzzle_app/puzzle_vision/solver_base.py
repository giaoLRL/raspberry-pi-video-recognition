"""Core solver infrastructure: error types, dataclasses, and utilities.

This module provides the shared foundation that all solver modules build on:
``SolveError``, ``AssemblyCandidate``, shape-alignment helpers, template
validation, and ``UnknownPuzzleSolver`` for autonomous rigid-piece assembly.
"""

# 求解器基础设施：异常类型、数据类、辅助函数、UnknownPuzzleSolver

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .detector import PieceObservation
from .geometry import (
    best_cyclic_alignment,
    compose_transforms,
    edge_alignment_transforms,
    edge_lengths,
    invert_transform,
    is_proper_rotation,
    min_area_rectangle,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    polygon_intersection_area,
    polygons_overlap,
    rotation_matrix_row,
    transform_angle_deg,
    transform_points,
    wrap_angle_deg,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SolveError(RuntimeError):
    # 求解器异常（无法生成有效的拼装方案）
    """Raised when the solver cannot produce a valid assembly plan."""


# ---------------------------------------------------------------------------
# Helpers (backward-compatible wrappers)
# ---------------------------------------------------------------------------


def _polygon_intersection_area(
    first: np.ndarray,
    second: np.ndarray,
    pixels_per_mm: float = 2.0,
) -> float:
    # 多边形交集面积（向后兼容的薄封装）
    """Backward-compatible wrapper used by solver diagnostics."""
    return polygon_intersection_area(
        first,
        second,
        pixels_per_mm=pixels_per_mm,
    )


def _validate_fixed_template(
    templates: list[dict[str, Any]], fixed_cfg: dict[str, Any]
) -> dict[str, Any]:
    # 验证固定模板是否完整平铺目标矩形
    target_size = np.asarray(
        fixed_cfg.get("target_size_mm", [100.0, 60.0]),
        dtype=np.float64,
    )
    if target_size.shape != (2,) or np.any(target_size <= 0):
        raise SolveError("Fixed target_size_mm must be [width, height]")
    tolerance = 1.0
    total_area = 0.0
    polygons = [item["polygon"] for item in templates]
    for polygon in polygons:
        if (
            np.min(polygon[:, 0]) < -tolerance
            or np.min(polygon[:, 1]) < -tolerance
            or np.max(polygon[:, 0]) > target_size[0] + tolerance
            or np.max(polygon[:, 1]) > target_size[1] + tolerance
        ):
            raise SolveError("A fixed template piece lies outside the target rectangle")
        total_area += polygon_area(polygon)

    maximum_overlap = 0.0
    for first, second in itertools.combinations(polygons, 2):
        if cv2.isContourConvex(first.astype(np.float32)) and cv2.isContourConvex(
            second.astype(np.float32)
        ):
            overlap = float(
                cv2.intersectConvexConvex(
                    first.astype(np.float32), second.astype(np.float32)
                )[0]
            )
            maximum_overlap = max(maximum_overlap, overlap)
    rectangle_area = float(target_size[0] * target_size[1])
    coverage_ratio = total_area / max(rectangle_area, 1e-9)
    corners = np.asarray(
        [
            [0.0, 0.0],
            [target_size[0], 0.0],
            target_size,
            [0.0, target_size[1]],
        ],
        dtype=np.float32,
    )
    corners_covered = all(
        any(
            cv2.pointPolygonTest(
                polygon.astype(np.float32).reshape(-1, 1, 2),
                tuple(map(float, corner)),
                False,
            )
            >= 0
            for polygon in polygons
        )
        for corner in corners
    )
    if (
        abs(total_area - rectangle_area) > tolerance
        or maximum_overlap > tolerance
        or not corners_covered
    ):
        raise SolveError(
            "Fixed template does not completely tile its target rectangle "
            f"(coverage={coverage_ratio:.4f}, overlap={maximum_overlap:.2f}, "
            f"corners_covered={corners_covered})"
        )
    return {
        "target_size_mm": np.round(target_size, 3).tolist(),
        "target_coverage_ratio": round(coverage_ratio, 6),
        "target_corners_covered": corners_covered,
        "maximum_target_overlap_mm2": round(maximum_overlap, 6),
    }


def _sample_polygon(polygon: np.ndarray, count: int = 80) -> np.ndarray:
    # 沿多边形周长均匀采样点
    p = normalize_winding(polygon)
    lengths = edge_lengths(p)
    perimeter = float(np.sum(lengths))
    positions = np.linspace(0.0, perimeter, count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    result = []
    for distance in positions:
        edge = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(p) - 1)
        fraction = (distance - cumulative[edge]) / max(lengths[edge], 1e-9)
        result.append(p[edge] + fraction * (p[(edge + 1) % len(p)] - p[edge]))
    return np.asarray(result, dtype=np.float64)


def _shape_alignment(
    template: np.ndarray,
    observed: np.ndarray,
    sample_count: int = 80,
) -> tuple[np.ndarray, np.ndarray, float]:
    # 将模板形状刚性对齐到观测形状
    if len(template) == len(observed):
        r, t, error, _ = best_cyclic_alignment(template, observed)
        return r, t, error

    src = _sample_polygon(template, count=sample_count)
    dst = _sample_polygon(observed, count=sample_count)
    # Sampling is uniform by perimeter; best_cyclic_alignment already checks
    # every possible starting position.
    r, t, error, _ = best_cyclic_alignment(src, dst)
    return r, t, error


# ---------------------------------------------------------------------------
# Rounded card corner detection (shared by solver_card and UnknownPuzzleSolver)
# ---------------------------------------------------------------------------


def _card_rounded_corner_frames(
    polygon: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    # 检测拼图块上的圆角卡角特征
    """Locate short chords that approximate an original rounded card corner.

    A rounded corner is observed as a short chord between two long, nearly
    perpendicular card-boundary edges.  The two adjacent long edges are outer
    card edges and must never be paired as cut seams.
    """

    points = np.asarray(polygon, dtype=np.float64)
    count = len(points)
    if count < 4:
        return []
    minimum_chord = float(config.get("card_rounded_chord_min_mm", 2.0))
    maximum_chord = float(config.get("card_rounded_chord_max_mm", 8.0))
    maximum_error = float(
        config.get("card_rounded_corner_angle_error_deg", 25.0)
    )
    frames: list[dict[str, Any]] = []
    for chord_index in range(count):
        chord_start = points[chord_index]
        chord_end = points[(chord_index + 1) % count]
        chord_length = float(np.linalg.norm(chord_end - chord_start))
        if not minimum_chord <= chord_length <= maximum_chord:
            continue
        incoming = chord_start - points[(chord_index - 1) % count]
        outgoing = points[(chord_index + 2) % count] - chord_end
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if min(incoming_length, outgoing_length) < maximum_chord:
            continue
        first = incoming / incoming_length
        second = outgoing / outgoing_length
        line_angle = math.degrees(
            math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
        )
        angle_error = abs(90.0 - line_angle)
        if angle_error > maximum_error:
            continue

        # Intersection of the two infinite boundary rays gives the virtual
        # sharp rectangle corner.  It is more stable than anchoring a noisy
        # rounded chord endpoint.
        matrix = np.column_stack((first, -second))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-6:
            continue
        coefficients = np.linalg.solve(matrix, chord_end - chord_start)
        virtual_corner = chord_start + coefficients[0] * first
        frames.append(
            {
                "chord_edge": chord_index,
                "outer_edges": {
                    (chord_index - 1) % count,
                    (chord_index + 1) % count,
                },
                "virtual_corner": virtual_corner,
                "incoming_direction": first,
                "outgoing_direction": second,
                "angle_error_deg": angle_error,
                "score": angle_error + 0.15 * chord_length,
            }
        )
    frames.sort(key=lambda item: float(item["score"]))
    return frames


def _card_rank_anchor_options(
    polygon: np.ndarray,
    width: float,
    height: float,
    config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    # 基于识别的牌角生成候选锚定变换
    frames = _card_rounded_corner_frames(polygon, config)
    if not frames:
        return []
    target_corners = (
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([width, height], dtype=np.float64),
    )
    tolerance = float(config.get("card_boundary_tolerance_mm", 4.0))
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for frame in frames:
        source_corner = np.asarray(
            frame["virtual_corner"], dtype=np.float64
        )
        for target_corner in target_corners:
            expected_inward = (
                np.asarray([1.0, 1.0], dtype=np.float64)
                if np.allclose(target_corner, 0.0)
                else np.asarray([-1.0, -1.0], dtype=np.float64)
            )
            for source_direction in (
                np.asarray(frame["incoming_direction"], dtype=np.float64),
                np.asarray(frame["outgoing_direction"], dtype=np.float64),
            ):
                source_angle = math.atan2(
                    float(source_direction[1]),
                    float(source_direction[0]),
                )
                for target_angle in (
                    0.0,
                    math.pi / 2.0,
                    math.pi,
                    -math.pi / 2.0,
                ):
                    rotation = rotation_matrix_row(
                        target_angle - source_angle
                    )
                    translation = target_corner - source_corner @ rotation
                    placed = transform_points(
                        polygon, rotation, translation
                    )
                    centroid_direction = (
                        polygon_centroid(placed) - target_corner
                    )
                    if np.any(
                        centroid_direction * expected_inward <= 0.0
                    ):
                        continue
                    outside = (
                        max(0.0, -float(np.min(placed[:, 0])))
                        + max(0.0, float(np.max(placed[:, 0])) - width)
                        + max(0.0, -float(np.min(placed[:, 1])))
                        + max(0.0, float(np.max(placed[:, 1])) - height)
                    )
                    if outside > tolerance:
                        continue
                    score = (
                        10.0 * outside
                        + float(frame["score"])
                    )
                    candidates.append((rotation, translation, score))
    unique: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]
    ] = {}
    for rotation, translation, score in candidates:
        key = (
            int(round(transform_angle_deg(rotation) * 10.0)),
            int(round(float(translation[0]) * 10.0)),
            int(round(float(translation[1]) * 10.0)),
        )
        previous = unique.get(key)
        if previous is None or score < previous[2]:
            unique[key] = (rotation, translation, score)
    return sorted(unique.values(), key=lambda item: item[2])


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AssemblyCandidate:
    # 拼装候选解（含变换、匹配、评分）
    transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    matches: list[tuple[int, int, int, int]]
    geometry_score: float
    long_side_mm: float
    short_side_mm: float
    fill_ratio: float
    box: np.ndarray
    theta: float
    overlap_area_mm2: float = 0.0
    gap_area_mm2: float = 0.0
    overlap_ratio: float = 0.0
    gap_ratio: float = 0.0
    texture_score: float = 0.0
    card_rank_score: float = 0.0
    card_rank_corner_distance_mm: float | None = None
    total_score: float = 0.0


# ---------------------------------------------------------------------------
# UnknownPuzzleSolver
# ---------------------------------------------------------------------------


class UnknownPuzzleSolver:
    # 未知拼图刚性搜索求解器（边对接+DFS）
    """Rigid edge-docking search for 1–4 pieces forming a rectangle.

    The solver precomputes all pairwise edge dockings, then runs a depth-first
    search over piece placements.  It does not introduce reflection or scaling
    — every transform is a proper rotation and translation.
    """

    def __init__(
        self,
        observations: list[PieceObservation],
        config: dict[str, Any],
        rectified_image: np.ndarray | None = None,
        pixels_per_mm: float = 4.0,
        use_texture: bool = False,
        pose_hints: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] | None = None,
        initial_transforms: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] | None = None,
    ):
        # 初始化：预计算所有成对边对接选项
        self.observations = observations
        self.polygons = [item.polygon_mm for item in observations]
        self.centroids = [
            polygon_centroid(polygon) for polygon in self.polygons
        ]
        self.config = config
        self.image = rectified_image
        self.ppm = float(pixels_per_mm)
        self.use_texture = use_texture and rectified_image is not None
        self.lab_image = (
            cv2.cvtColor(rectified_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            if self.use_texture and rectified_image is not None
            else None
        )
        self.nodes = 0
        self.candidates: list[AssemblyCandidate] = []
        self.visited: set[tuple[Any, ...]] = set()
        self.identity = np.eye(2, dtype=np.float64)
        self.zero = np.zeros(2, dtype=np.float64)
        self.allow_partial_edges = False
        self.deadline = float("inf")
        self.timed_out = False
        self.card_mode = bool(config.get("card_mode", False))
        rank_piece = config.get("card_rank_piece_index")
        rank_center = config.get("card_rank_center_mm")
        self.card_rank_piece_index = (
            int(rank_piece) if rank_piece is not None else None
        )
        self.card_rank_center_mm = (
            np.asarray(rank_center, dtype=np.float64)
            if rank_center is not None
            else None
        )
        self.initial_transforms = {
            int(index): (
                np.asarray(transform[0], dtype=np.float64),
                np.asarray(transform[1], dtype=np.float64),
            )
            for index, transform in (initial_transforms or {}).items()
        }
        configured_card_size = config.get("card_target_size_mm")
        self.card_target_size = (
            np.asarray(configured_card_size, dtype=np.float64)
            if configured_card_size is not None
            else None
        )
        self.card_corner_frames = (
            [
                _card_rounded_corner_frames(polygon, config)
                for polygon in self.polygons
            ]
            if self.card_mode
            else [[] for _ in self.polygons]
        )
        self.excluded_solver_edges = [
            {
                edge
                for frame in frames
                for edge in frame["outer_edges"]
            }
            for frames in self.card_corner_frames
        ]
        self.pose_hints = pose_hints or {}
        self.hint_centroids = {
            index: transform_points(
                self.centroids[index][None, :],
                transform[0],
                transform[1],
            )[0]
            for index, transform in self.pose_hints.items()
        }
        self.pair_options = self._precompute_pair_options()

    def _precompute_pair_options(
        self,
    ) -> dict[
        tuple[int, int, bool],
        list[
            tuple[
                float,
                int,
                int,
                np.ndarray,
                np.ndarray,
            ]
        ],
    ]:
        # 预计算每对拼图块的所有可能边对接
        """Precompute every rigid edge docking once.

        The previous depth-first search rebuilt the same edge lengths,
        rotations and translations at every node.  With at most four
        five-sided pieces, all pairwise docking options are small enough to
        cache and later compose with the placed piece transform.
        """

        result: dict[
            tuple[int, int, bool],
            list[
                tuple[
                    float,
                    int,
                    int,
                    np.ndarray,
                    np.ndarray,
                ]
            ],
        ] = {}
        lengths = [edge_lengths(polygon) for polygon in self.polygons]
        tolerance = float(self.config["edge_length_tolerance_mm"])
        relative = float(self.config["edge_length_relative_tolerance"])
        minimum_solver_edge = float(
            self.config.get("minimum_solver_edge_mm", 8.0)
        )
        for new_index, new_polygon in enumerate(self.polygons):
            for placed_index, placed_polygon in enumerate(self.polygons):
                if new_index == placed_index:
                    continue
                exact: list[
                    tuple[
                        float,
                        int,
                        int,
                        np.ndarray,
                        np.ndarray,
                    ]
                ] = []
                partial: list[
                    tuple[
                        float,
                        int,
                        int,
                        np.ndarray,
                        np.ndarray,
                    ]
                ] = []
                for new_edge_index in range(len(new_polygon)):
                    if (
                        self.card_mode
                        and new_edge_index
                        in self.excluded_solver_edges[new_index]
                    ):
                        continue
                    source_edge = np.asarray(
                        [
                            new_polygon[new_edge_index],
                            new_polygon[
                                (new_edge_index + 1) % len(new_polygon)
                            ],
                        ],
                        dtype=np.float64,
                    )
                    for placed_edge_index in range(len(placed_polygon)):
                        if (
                            self.card_mode
                            and placed_edge_index
                            in self.excluded_solver_edges[placed_index]
                        ):
                            continue
                        destination_edge = np.asarray(
                            [
                                placed_polygon[placed_edge_index],
                                placed_polygon[
                                    (placed_edge_index + 1)
                                    % len(placed_polygon)
                                ],
                            ],
                            dtype=np.float64,
                        )
                        new_length = float(
                            lengths[new_index][new_edge_index]
                        )
                        placed_length = float(
                            lengths[placed_index][placed_edge_index]
                        )
                        if (
                            new_length < minimum_solver_edge
                            or placed_length < minimum_solver_edge
                        ):
                            continue
                        length_error = abs(new_length - placed_length)
                        allowed = max(
                            tolerance,
                            relative * min(new_length, placed_length),
                        )
                        options = edge_alignment_transforms(
                            source_edge,
                            destination_edge,
                            tolerance,
                            relative,
                            float(
                                self.config.get(
                                    "minimum_partial_remainder_mm",
                                    18.0,
                                )
                            ),
                        )
                        for r, t, overlap in options:
                            item = (
                                -overlap + 0.05 * length_error,
                                new_edge_index,
                                placed_edge_index,
                                r,
                                t,
                            )
                            partial.append(item)
                            if length_error <= allowed:
                                exact.append(item)
                exact_limit = int(
                    self.config.get("max_pair_options_exact", 18)
                )
                partial_limit = int(
                    self.config.get("max_pair_options_partial", 28)
                )
                result[(new_index, placed_index, False)] = sorted(
                    exact, key=lambda item: item[0]
                )[:exact_limit]
                result[(new_index, placed_index, True)] = sorted(
                    partial, key=lambda item: item[0]
                )[:partial_limit]
        return result

    def _state_key(
        self, transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[Any, ...]:
        # 生成搜索状态哈希键（去重）
        key: list[Any] = []
        for index in sorted(transforms):
            r, t = transforms[index]
            centroid = transform_points(
                self.centroids[index][None, :], r, t
            )[0]
            key.extend(
                [
                    index,
                    round(transform_angle_deg(r) / 2.0),
                    round(float(centroid[0])),
                    round(float(centroid[1])),
                ]
            )
        return tuple(key)

    def _placed_polygon(
        self, index: int, transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        # 获取已放置拼图块的世界坐标多边形
        r, t = transforms[index]
        return transform_points(self.polygons[index], r, t)

    def _valid_new_polygon(
        self,
        polygon: np.ndarray,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        # 检查新多边形是否与已放置块合法共存
        # During search, independently detected corners can make two truly
        # matching seams overlap by a thin noisy sliver.  Keep those candidates
        # and let the completed-layout union score decide; the final motion
        # plan still uses the stricter overlap_tolerance_mm.
        tolerance = float(
            self.config.get(
                "search_overlap_tolerance_mm",
                self.config["overlap_tolerance_mm"],
            )
        )
        for index in transforms:
            if polygons_overlap(
                polygon, self._placed_polygon(index, transforms), tolerance
            ):
                return False
        if self.card_target_size is not None:
            tolerance = float(
                self.config.get("card_boundary_tolerance_mm", 4.0)
            )
            outside = (
                max(0.0, -float(np.min(polygon[:, 0])))
                + max(
                    0.0,
                    float(np.max(polygon[:, 0]))
                    - float(self.card_target_size[0]),
                )
                + max(0.0, -float(np.min(polygon[:, 1])))
                + max(
                    0.0,
                    float(np.max(polygon[:, 1]))
                    - float(self.card_target_size[1]),
                )
            )
            if outside > tolerance:
                return False
        all_polygons = [
            self._placed_polygon(index, transforms) for index in transforms
        ] + [polygon]
        _, long_side, short_side, _ = min_area_rectangle(all_polygons)
        slack = float(self.config.get("search_dimension_slack_mm", 8.0))
        if (
            long_side > float(self.config["max_width_mm"]) + slack
            or short_side > float(self.config["max_height_mm"]) + slack
        ):
            return False
        # The minimum enclosing rectangle cannot become smaller after adding
        # more pieces.  Reject a sparse partial layout that could never reach
        # the configured minimum fill ratio.
        total_area = sum(item.area_mm2 for item in self.observations)
        search_fill = float(
            self.config.get("search_minimum_fill_ratio", 0.72)
        )
        return (
            long_side * short_side
            <= total_area / max(search_fill, 1e-6)
        )

    def _evaluate(
        self,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
        matches: list[tuple[int, int, int, int]],
    ) -> None:
        # 评估完整拼装：计算矩形贴合度、重叠、间隙等评分
        polygons = [
            self._placed_polygon(index, transforms)
            for index in range(len(self.polygons))
        ]
        box, long_side, short_side, theta = min_area_rectangle(polygons)
        if self.card_target_size is not None:
            card_width = float(self.card_target_size[0])
            card_height = float(self.card_target_size[1])
            long_side = max(card_width, card_height)
            short_side = min(card_width, card_height)
            rectangle_area = max(card_width * card_height, 1e-6)
        else:
            rectangle_area = max(long_side * short_side, 1e-6)
        pieces_area = sum(item.area_mm2 for item in self.observations)
        fill_ratio = pieces_area / rectangle_area
        if fill_ratio > float(
            self.config.get("maximum_candidate_fill_ratio", 1.08)
        ):
            return

        # Area alone cannot distinguish a true tiling from a layout with one
        # large overlap and an equally large empty corner.  Measure every
        # pairwise intersection and score the actual union against the
        # enclosing rectangle.  With at most four pieces this global check is
        # cheap, and it prevents a locally attractive edge match from winning
        # before the remaining pieces have formed a rectangle.
        overlap_area = sum(
            _polygon_intersection_area(
                first,
                second,
                pixels_per_mm=float(
                    self.config.get(
                        "candidate_overlap_pixels_per_mm", 1.0
                    )
                ),
            )
            for first, second in itertools.combinations(polygons, 2)
        )
        estimated_union_area = max(0.0, pieces_area - overlap_area)
        gap_area = max(0.0, rectangle_area - estimated_union_area)
        overfill_area = max(0.0, estimated_union_area - rectangle_area)
        overlap_ratio = overlap_area / rectangle_area
        gap_ratio = gap_area / rectangle_area

        min_w = float(self.config["min_width_mm"])
        max_w = float(self.config["max_width_mm"])
        min_h = float(self.config["min_height_mm"])
        max_h = float(self.config["max_height_mm"])
        dimension_penalty = (
            max(0.0, min_w - long_side)
            + max(0.0, long_side - max_w)
            + max(0.0, min_h - short_side)
            + max(0.0, short_side - max_h)
        )
        if dimension_penalty > 12.0:
            return
        # A missing region counts once; an overlap counts twice because the
        # same paper area both covers the wrong location and leaves a hole
        # elsewhere.  This score is zero only for a non-overlapping, full
        # rectangular tiling of a legal target size.
        layout_error = (
            gap_area + overfill_area + 2.0 * overlap_area
        ) / rectangle_area
        score = 100.0 * layout_error + 1.5 * dimension_penalty
        if self.card_target_size is not None:
            all_points = np.vstack(polygons)
            extent_error = (
                abs(float(np.min(all_points[:, 0])))
                + abs(float(np.min(all_points[:, 1])))
                + abs(
                    float(np.max(all_points[:, 0]))
                    - float(self.card_target_size[0])
                )
                + abs(
                    float(np.max(all_points[:, 1]))
                    - float(self.card_target_size[1])
                )
            )
            if extent_error > float(
                self.config.get("card_maximum_extent_error_mm", 12.0)
            ):
                return
            score += float(
                self.config.get("card_extent_score_weight", 2.0)
            ) * extent_error
        if self.card_mode:
            expected_aspect = float(
                self.config.get("card_aspect_ratio", 88.0 / 57.0)
            )
            aspect_error = abs(
                long_side / max(short_side, 1e-6) - expected_aspect
            )
            if aspect_error > float(
                self.config.get("card_maximum_aspect_error", 0.22)
            ):
                return
            score += float(
                self.config.get("card_aspect_score_weight", 65.0)
            ) * aspect_error
        rank_score = 0.0
        rank_corner_distance: float | None = None
        if (
            self.card_mode
            and self.card_rank_piece_index is not None
            and self.card_rank_center_mm is not None
            and self.card_rank_piece_index in transforms
        ):
            rank_r, rank_t = transforms[self.card_rank_piece_index]
            rank_point = transform_points(
                self.card_rank_center_mm[None, :], rank_r, rank_t
            )[0]
            rank_corner_distance = float(
                np.min(np.linalg.norm(box - rank_point, axis=1))
            )
            expected_inset = float(
                self.config.get("card_rank_corner_distance_mm", 10.5)
            )
            free_tolerance = float(
                self.config.get("card_rank_corner_tolerance_mm", 5.0)
            )
            rank_score = max(
                0.0,
                abs(rank_corner_distance - expected_inset)
                - free_tolerance,
            )
        candidate = AssemblyCandidate(
            transforms={
                index: (value[0].copy(), value[1].copy())
                for index, value in transforms.items()
            },
            matches=list(matches),
            geometry_score=score,
            long_side_mm=long_side,
            short_side_mm=short_side,
            fill_ratio=fill_ratio,
            box=box,
            theta=theta,
            overlap_area_mm2=overlap_area,
            gap_area_mm2=gap_area,
            overlap_ratio=overlap_ratio,
            gap_ratio=gap_ratio,
            card_rank_score=rank_score,
            card_rank_corner_distance_mm=rank_corner_distance,
        )
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda item: item.geometry_score)
        maximum = int(self.config["max_final_candidates"])
        if len(self.candidates) > maximum:
            self.candidates = self.candidates[:maximum]

    def _search(
        self,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
        matches: list[tuple[int, int, int, int]],
    ) -> None:
        # 深度优先搜索：逐步扩展拼装状态
        if self.candidates:
            best = self.candidates[0]
            if (
                best.geometry_score
                <= float(
                    self.config.get("early_stop_geometry_score", 2.5)
                )
                and best.overlap_ratio
                <= float(
                    self.config.get("early_stop_overlap_ratio", 0.003)
                )
                and best.fill_ratio
                >= float(
                    self.config.get("early_stop_fill_ratio", 0.97)
                )
            ):
                return
        if time.perf_counter() >= self.deadline:
            self.timed_out = True
            return
        if self.nodes >= int(self.config["max_search_nodes"]):
            return
        self.nodes += 1
        state_key = self._state_key(transforms)
        if state_key in self.visited:
            return
        self.visited.add(state_key)
        if len(transforms) == len(self.polygons):
            self._evaluate(transforms, matches)
            return

        unplaced = [index for index in range(len(self.polygons)) if index not in transforms]
        generated: list[
            tuple[float, int, int, int, np.ndarray, np.ndarray, int]
        ] = []
        for new_index in unplaced:
            for placed_index, placed_transform in transforms.items():
                for (
                    option_score,
                    new_edge_index,
                    placed_edge_index,
                    local_r,
                    local_t,
                ) in self.pair_options.get(
                    (
                        new_index,
                        placed_index,
                        self.allow_partial_edges,
                    ),
                    [],
                ):
                    r, t = compose_transforms(
                        local_r,
                        local_t,
                        placed_transform[0],
                        placed_transform[1],
                    )
                    generated.append(
                        (
                            option_score,
                            new_index,
                            new_edge_index,
                            placed_edge_index,
                            r,
                            t,
                            placed_index,
                        )
                    )

        if not self.pose_hints:
            # Preserve the thoroughly regression-tested autonomous ordering.
            generated.sort(key=lambda item: item[0])
        else:
            # A demonstrated layout provides stronger information than edge
            # length alone. Rank rigid dockings by their relative-pose error
            # and partial-assembly compactness.
            placed_polygons = [
                self._placed_polygon(index, transforms)
                for index in transforms
            ]
            placed_area = sum(
                self.observations[index].area_mm2
                for index in transforms
            )
            total_piece_area = sum(
                item.area_mm2 for item in self.observations
            )
            ranked = []
            for item in generated:
                (
                    option_score,
                    new_index,
                    _,
                    _,
                    r,
                    t,
                    placed_index,
                ) = item
                transformed = transform_points(
                    self.polygons[new_index], r, t
                )
                _, long_side, short_side, _ = min_area_rectangle(
                    placed_polygons + [transformed]
                )
                box_area = max(long_side * short_side, 1e-6)
                assembly_area = (
                    placed_area
                    + self.observations[new_index].area_mm2
                )
                quick_score = (
                    option_score
                    + 0.018
                    * max(0.0, box_area - assembly_area)
                    + 0.055
                    * max(0.0, box_area - total_piece_area)
                    + 4.0
                    * (
                        max(
                            0.0,
                            long_side
                            - float(self.config["max_width_mm"]),
                        )
                        + max(
                            0.0,
                            short_side
                            - float(self.config["max_height_mm"]),
                        )
                    )
                )
                placed_r, placed_t = transforms[placed_index]
                actual_new = transform_points(
                    self.centroids[new_index][None, :],
                    r,
                    t,
                )[0]
                actual_placed = transform_points(
                    self.centroids[placed_index][None, :],
                    placed_r,
                    placed_t,
                )[0]
                hint_placed_r, _ = self.pose_hints[
                    placed_index
                ]
                actual_relative = (
                    actual_new - actual_placed
                ) @ placed_r.T
                hint_relative = (
                    self.hint_centroids[new_index]
                    - self.hint_centroids[placed_index]
                ) @ hint_placed_r.T
                angle_error = abs(
                    wrap_angle_deg(
                        transform_angle_deg(r)
                        - transform_angle_deg(placed_r)
                        - transform_angle_deg(
                            self.pose_hints[new_index][0]
                        )
                        + transform_angle_deg(hint_placed_r)
                    )
                )
                hint_deviation = float(
                    np.linalg.norm(
                        actual_relative - hint_relative
                    )
                ) + 0.18 * angle_error
                quick_score += float(
                    self.config.get("pose_hint_weight", 0.0)
                ) * hint_deviation
                ranked.append((quick_score, item))
            generated = [
                item
                for _, item in sorted(
                    ranked, key=lambda value: value[0]
                )
            ]
            # The demonstrated relative poses make hundreds of distant edge
            # dockings provably unhelpful as first choices.  Keep a generous
            # deterministic beam at each DFS node; the final geometry checks
            # remain unchanged.
            generated = generated[
                : int(
                    self.config.get(
                        "pose_hint_branch_limit", 36
                    )
                )
            ]
        local_seen: set[tuple[Any, ...]] = set()
        for (
            _,
            new_index,
            new_edge_index,
            placed_edge_index,
            r,
            t,
            placed_index,
        ) in generated:
            centroid = transform_points(
                self.centroids[new_index][None, :], r, t
            )[0]
            placement_key = (
                new_index,
                round(transform_angle_deg(r)),
                round(float(centroid[0])),
                round(float(centroid[1])),
            )
            if placement_key in local_seen:
                continue
            local_seen.add(placement_key)
            transformed_polygon = transform_points(
                self.polygons[new_index], r, t
            )
            if not self._valid_new_polygon(transformed_polygon, transforms):
                continue
            next_transforms = dict(transforms)
            next_transforms[new_index] = (r, t)
            next_matches = matches + [
                (
                    placed_index,
                    placed_edge_index,
                    new_index,
                    new_edge_index,
                )
            ]
            self._search(next_transforms, next_matches)
            if self.nodes >= int(self.config["max_search_nodes"]):
                break

    def _sample_lab(self, points_mm: np.ndarray) -> np.ndarray:
        # 从校正图像中采样Lab颜色
        assert self.lab_image is not None
        x = (points_mm[:, 0] * self.ppm).astype(np.float32).reshape(-1, 1)
        y = (points_mm[:, 1] * self.ppm).astype(np.float32).reshape(-1, 1)
        return cv2.remap(
            self.lab_image,
            x,
            y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1, 3)

    def _best_anchor(self, allow_partial: bool) -> int:
        # 选择分支因子最小的拼图块作为搜索锚点
        """Choose the piece that creates the fewest first-level branches.

        Every complete assembly can be expressed relative to any piece, so a
        more distinctive anchor cannot remove a valid solution.  It does avoid
        spending the real-time budget on a large piece whose several similar
        outer edges all attract accidental matches.
        """

        counts: list[tuple[int, int]] = []
        for anchor in range(len(self.polygons)):
            branch_count = sum(
                len(
                    self.pair_options.get(
                        (new_index, anchor, allow_partial), []
                    )
                )
                for new_index in range(len(self.polygons))
                if new_index != anchor
            )
            counts.append((branch_count, anchor))
        return min(counts)[1]

    def _texture_score(self, candidate: AssemblyCandidate) -> float:
        # 计算拼装纹理一致性评分（接缝处颜色差异）
        if not self.use_texture or self.image is None:
            return 0.0
        differences: list[float] = []
        offset = float(self.config["seam_sample_offset_mm"])
        for first, first_edge, second, second_edge in candidate.matches:
            first_r, first_t = candidate.transforms[first]
            second_r, second_t = candidate.transforms[second]
            first_poly = transform_points(self.polygons[first], first_r, first_t)
            second_poly = transform_points(self.polygons[second], second_r, second_t)
            a = first_poly[first_edge]
            b = first_poly[(first_edge + 1) % len(first_poly)]
            c = second_poly[second_edge]
            d = second_poly[(second_edge + 1) % len(second_poly)]
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length < 2.0:
                continue
            unit = direction / length
            second_projection = [(float((p - a) @ unit)) for p in (c, d)]
            lower = max(1.0, min(second_projection))
            upper = min(length - 1.0, max(second_projection))
            if upper - lower < 2.0:
                continue
            count = max(8, int((upper - lower) / 2.0))
            seam = a + np.linspace(lower, upper, count)[:, None] * unit
            normal = np.array([-unit[1], unit[0]], dtype=np.float64)
            first_centroid = polygon_centroid(first_poly)
            second_centroid = polygon_centroid(second_poly)
            if np.dot(first_centroid - np.mean(seam, axis=0), normal) < 0:
                first_normal = -normal
            else:
                first_normal = normal
            if np.dot(second_centroid - np.mean(seam, axis=0), normal) < 0:
                second_normal = -normal
            else:
                second_normal = normal
            first_assembly_samples = seam + first_normal * offset
            second_assembly_samples = seam + second_normal * offset
            first_inv_r, first_inv_t = invert_transform(first_r, first_t)
            second_inv_r, second_inv_t = invert_transform(second_r, second_t)
            first_source = transform_points(
                first_assembly_samples, first_inv_r, first_inv_t
            )
            second_source = transform_points(
                second_assembly_samples, second_inv_r, second_inv_t
            )
            first_lab = self._sample_lab(first_source)
            second_lab = self._sample_lab(second_source)
            delta = np.linalg.norm(first_lab - second_lab, axis=1)
            differences.extend(delta.tolist())
        if not differences:
            return 50.0
        # Robust mean: discard the largest 15% of edge-shadow/outlier samples.
        values = np.sort(np.asarray(differences, dtype=np.float64))
        keep = max(1, int(len(values) * 0.85))
        return float(np.mean(values[:keep]) / 10.0)

    def solve(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # 主求解入口：精确搜索→部分边搜索→回退锚点搜索
        if not 1 <= len(self.polygons) <= 4:
            raise SolveError("Unknown mode supports one to four pieces")
        search_started = time.perf_counter()
        total_budget = float(self.config.get("max_search_seconds", 8.0))
        exact_budget = min(
            total_budget,
            float(self.config.get("exact_search_seconds", 1.5)),
        )
        self.deadline = search_started + exact_budget
        # Most manufactured puzzles have one-to-one internal edges. Search that
        # small space first. If it cannot form a convincing rectangle, enable
        # partial long-edge matches for T-junction layouts.
        self.allow_partial_edges = False
        exact_anchor = (
            min(self.initial_transforms)
            if self.initial_transforms
            else self._best_anchor(False)
        )
        self._search(
            (
                dict(self.initial_transforms)
                if self.initial_transforms
                else {exact_anchor: (self.identity, self.zero)}
            ),
            [],
        )
        exact_nodes = self.nodes
        exact_timed_out = self.timed_out
        best_exact = (
            min((item.geometry_score for item in self.candidates), default=float("inf"))
        )
        partial_searched = False
        partial_timed_out = False
        if not self.candidates or best_exact > float(
            self.config.get("exact_search_accept_score", 15.0)
        ):
            partial_searched = True
            self.allow_partial_edges = True
            self.visited.clear()
            self.nodes = 0
            self.timed_out = False
            self.deadline = search_started + total_budget
            # If exact docking already produced a near-valid rectangle, keep
            # the same root and only add T-junction options.  Changing the root
            # here can send the time-limited search into a completely
            # different, much less useful branch order.
            partial_anchor = (
                exact_anchor
                if self.initial_transforms
                or (
                    self.candidates
                    and best_exact
                    <= float(
                        self.config.get(
                            "partial_reuse_exact_anchor_score", 16.0
                        )
                    )
                )
                else self._best_anchor(True)
            )
            self._search(
                (
                    dict(self.initial_transforms)
                    if self.initial_transforms
                    else {partial_anchor: (self.identity, self.zero)}
                ),
                [],
            )
            partial_timed_out = self.timed_out
        else:
            partial_anchor = None
        self.nodes += exact_nodes
        self.timed_out = partial_timed_out if partial_searched else exact_timed_out

        # A DFS rooted at one piece is geometrically complete but its ordering
        # matters under a hard real-time deadline.  Retry from piece 1 only
        # when the first pass has not produced an acceptable rectangle.  This
        # keeps easy frames fast and gives ambiguous shapes a different search
        # ordering without introducing reflections or random results.
        minimum_fill = float(
            self.config.get("minimum_accepted_fill_ratio", 0.90)
        )
        maximum_score = float(
            self.config.get("maximum_accepted_geometry_score", 16.0)
        )
        provisional_ok = any(
            item.fill_ratio >= minimum_fill
            and item.geometry_score <= maximum_score
            for item in self.candidates
        )
        fallback_anchors: list[int] = []
        fallback_timed_out = False
        primary_nodes = self.nodes
        used_anchors = {exact_anchor}
        if partial_anchor is not None:
            used_anchors.add(partial_anchor)
        if not provisional_ok and not self.initial_transforms:
            remaining_anchors = [
                index
                for index in range(len(self.polygons))
                if index not in used_anchors
            ]
            # Under a deadline, DFS ordering depends on the root piece.  Spend
            # the fallback budget across every untried anchor rather than only
            # piece 1.  The order is deterministic and therefore an unchanged
            # source scene still produces the same target layout.
            fallback_budget = float(
                self.config.get("fallback_search_seconds", 2.5)
            )
            fallback_started = time.perf_counter()
            fallback_nodes = 0
            for position, fallback_anchor in enumerate(
                remaining_anchors
            ):
                remaining_time = (
                    fallback_budget
                    - (time.perf_counter() - fallback_started)
                )
                remaining_count = len(remaining_anchors) - position
                if remaining_time <= 0.05:
                    fallback_timed_out = True
                    break
                fallback_anchors.append(fallback_anchor)
                self.allow_partial_edges = True
                self.visited.clear()
                self.nodes = 0
                self.timed_out = False
                self.deadline = (
                    time.perf_counter()
                    + remaining_time / max(1, remaining_count)
                )
                self._search(
                    {
                        fallback_anchor: (
                            self.identity,
                            self.zero,
                        )
                    },
                    [],
                )
                fallback_nodes += self.nodes
                fallback_timed_out = (
                    fallback_timed_out or self.timed_out
                )
                provisional_ok = any(
                    item.fill_ratio >= minimum_fill
                    and item.geometry_score <= maximum_score
                    for item in self.candidates
                )
                if provisional_ok:
                    break
            self.nodes = primary_nodes + fallback_nodes
            self.timed_out = (
                fallback_timed_out
                or partial_timed_out
                or exact_timed_out
            )
        if not self.candidates:
            raise SolveError(
                "No rectangular assembly was found. Check polygon vertices, "
                "edge-length calibration, or increase solver tolerances."
            )
        for candidate in self.candidates:
            candidate.texture_score = self._texture_score(candidate)
            candidate.total_score = candidate.geometry_score + float(
                self.config["texture_weight"]
            ) * candidate.texture_score + float(
                self.config.get("card_rank_score_weight", 0.2)
            ) * candidate.card_rank_score
        self.candidates.sort(key=lambda item: item.total_score)
        best = self.candidates[0]

        # Rotate the long side horizontal, then centre the rectangle in the
        # configured lower A4 target zone.
        global_r = rotation_matrix_row(-best.theta)
        assembled_polygons = [
            transform_points(
                self.polygons[index],
                best.transforms[index][0],
                best.transforms[index][1],
            )
            for index in range(len(self.polygons))
        ]
        rotated = [polygon @ global_r for polygon in assembled_polygons]
        all_points = np.vstack(rotated)
        lower = np.min(all_points, axis=0)
        upper = np.max(all_points, axis=0)
        size = upper - lower
        target_orientation = str(
            self.config.get("target_orientation", "landscape")
        ).lower()
        rotate_to_portrait = (
            target_orientation == "portrait"
            and size[0] >= size[1]
        )
        rotate_to_landscape = (
            target_orientation != "portrait"
            and size[0] < size[1]
        )
        if rotate_to_portrait or rotate_to_landscape:
            extra = rotation_matrix_row(-math.pi / 2)
            global_r = global_r @ extra
            rotated = [polygon @ extra for polygon in rotated]
            all_points = np.vstack(rotated)
            lower = np.min(all_points, axis=0)
            upper = np.max(all_points, axis=0)
            size = upper - lower

        zone = np.asarray(self.config["target_zone_mm"], dtype=np.float64)
        zone_center = np.array(
            [(zone[0] + zone[2]) * 0.5, (zone[1] + zone[3]) * 0.5],
            dtype=np.float64,
        )
        target_lower = zone_center - size * 0.5
        global_t = target_lower - lower
        target_center = target_lower + size * 0.5
        clearance = float(self.config.get("placement_clearance_mm", 0.0))

        plan: list[dict[str, Any]] = []
        for index, observation in enumerate(self.observations):
            local_r, local_t = best.transforms[index]
            target_r, target_t = compose_transforms(
                local_r, local_t, global_r, global_t
            )
            if not is_proper_rotation(target_r):
                raise SolveError("Mirrored autonomous-piece transforms are forbidden")
            target_polygon = transform_points(
                observation.polygon_mm, target_r, target_t
            )
            if clearance > 0:
                direction = polygon_centroid(target_polygon) - target_center
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    shift = direction / norm * clearance
                    target_t = target_t + shift
                    target_polygon = target_polygon + shift
            place = transform_points(
                observation.pickup_mm[None, :], target_r, target_t
            )[0]
            plan.append(
                {
                    "piece_id": observation.id,
                    "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                    "place_mm": np.round(place, 3).tolist(),
                    "rotate_deg": round(
                        wrap_angle_deg(transform_angle_deg(target_r)), 3
                    ),
                    "mirrored": False,
                    "target_polygon_mm": np.round(target_polygon, 3).tolist(),
                }
            )

        # The rectangle and every piece must share exactly the same A4-paper
        # coordinate offset.  Small contour/shadow errors otherwise leave a
        # visible outer-edge gap even though the matched seam is correct.
        # Snap only a piece extremum already close to an outer side; this adds
        # translation only and can never mirror or scale a piece.
        snap_tolerance = float(
            self.config.get("boundary_snap_tolerance_mm", 0.0)
        )
        target_upper = target_lower + size
        if snap_tolerance > 0.0:
            for item_index, item in enumerate(plan):
                polygon = np.asarray(
                    item["target_polygon_mm"], dtype=np.float64
                )
                shift = np.zeros(2, dtype=np.float64)
                for axis in range(2):
                    options: list[float] = []
                    lower_delta = float(
                        target_lower[axis] - np.min(polygon[:, axis])
                    )
                    upper_delta = float(
                        target_upper[axis] - np.max(polygon[:, axis])
                    )
                    if abs(lower_delta) <= snap_tolerance:
                        options.append(lower_delta)
                    if abs(upper_delta) <= snap_tolerance:
                        options.append(upper_delta)
                    if options:
                        shift[axis] = min(options, key=abs)
                requested_shift = shift.copy()
                if np.any(np.abs(shift) > 1e-9):
                    # Do not close an outer-edge gap by creating a real
                    # inter-piece overlap.  Reduce the correction until it is
                    # compatible with every other measured outline.
                    for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
                        candidate_shift = requested_shift * fraction
                        candidate = polygon + candidate_shift
                        conflicts = False
                        for other_index, other in enumerate(plan):
                            if other_index == item_index:
                                continue
                            other_polygon = np.asarray(
                                other["target_polygon_mm"], dtype=np.float64
                            )
                            if polygons_overlap(
                                candidate,
                                other_polygon,
                                float(self.config["overlap_tolerance_mm"]),
                            ):
                                conflicts = True
                                break
                        if not conflicts:
                            shift = candidate_shift
                            break
                if np.any(np.abs(shift) > 1e-9):
                    polygon += shift
                    place = np.asarray(item["place_mm"], dtype=np.float64) + shift
                    item["place_mm"] = np.round(place, 3).tolist()
                    item["target_polygon_mm"] = np.round(
                        polygon, 3
                    ).tolist()
                item["boundary_snap_mm"] = np.round(shift, 3).tolist()
                item["boundary_snap_requested_mm"] = np.round(
                    requested_shift, 3
                ).tolist()

        # Camera shadows and hand-cut fuzz can make two correctly docked
        # measured contours overlap by a thin sliver.  The task explicitly
        # forbids overlap, so card mode applies a final translation-only
        # relief.  It never rotates, scales or mirrors a fragment.  A tiny
        # visible gap is preferable to an unsafe robot command that stacks
        # paper.
        overlap_relief = [
            np.zeros(2, dtype=np.float64) for _ in plan
        ]
        if self.card_mode and len(plan) > 1:
            accepted_pair_overlap = float(
                self.config.get(
                    "maximum_accepted_pair_overlap_mm2", 0.5
                )
            )
            step = float(
                self.config.get("card_overlap_relief_step_mm", 0.15)
            )
            maximum_shift = float(
                self.config.get("card_overlap_relief_max_shift_mm", 2.0)
            )
            maximum_iterations = int(
                self.config.get("card_overlap_relief_iterations", 40)
            )

            def relief_cost(
                polygons: list[np.ndarray],
            ) -> float:
                overlap_cost = sum(
                    max(
                        0.0,
                        _polygon_intersection_area(first, second)
                        - accepted_pair_overlap,
                    )
                    for first, second in itertools.combinations(
                        polygons, 2
                    )
                )
                outside_cost = sum(
                    max(0.0, target_lower[0] - float(np.min(poly[:, 0])))
                    + max(0.0, float(np.max(poly[:, 0])) - target_upper[0])
                    + max(0.0, target_lower[1] - float(np.min(poly[:, 1])))
                    + max(0.0, float(np.max(poly[:, 1])) - target_upper[1])
                    for poly in polygons
                )
                # The target bounds are estimated from shadowed camera
                # contours.  Treat them as a soft guide in card mode: forcing
                # every noisy vertex inside that estimate can leave a thin
                # physical overlap at a seam.  A sub-millimetre outward
                # relief/gap is safer for the robot and does not alter the
                # fragment rotation, scale or handedness.
                outside_weight = float(
                    self.config.get(
                        "card_overlap_relief_outside_weight", 0.5
                    )
                )
                return overlap_cost + outside_weight * outside_cost

            # First use an exact separating-translation jump.  Repeating
            # 0.1 mm trial moves is unnecessarily expensive on an RDK and can
            # stop at a local minimum when three fragments meet.  Candidate
            # directions come from the pair-centre vector and every polygon
            # edge normal; a short binary search finds the smallest rigid
            # translation that removes that pair's measured sliver.  No
            # rotation, scale or reflection is introduced.
            direct_iterations = int(
                self.config.get(
                    "card_overlap_relief_direct_iterations", 4
                )
            )
            direct_binary_steps = int(
                self.config.get(
                    "card_overlap_relief_binary_steps", 6
                )
            )

            def available_distance(
                displacement: np.ndarray, direction: np.ndarray
            ) -> float:
                projection = float(displacement @ direction)
                remainder = (
                    projection * projection
                    + maximum_shift * maximum_shift
                    - float(displacement @ displacement)
                )
                if remainder <= 0.0:
                    return 0.0
                return max(0.0, -projection + math.sqrt(remainder))

            for _ in range(direct_iterations):
                polygons = [
                    np.asarray(
                        item["target_polygon_mm"], dtype=np.float64
                    )
                    for item in plan
                ]
                overlapping_pairs = [
                    (
                        _polygon_intersection_area(
                            polygons[first_index],
                            polygons[second_index],
                        ),
                        first_index,
                        second_index,
                    )
                    for first_index, second_index in itertools.combinations(
                        range(len(plan)), 2
                    )
                ]
                area, first_index, second_index = max(
                    overlapping_pairs, default=(0.0, 0, 0)
                )
                if area <= accepted_pair_overlap:
                    break
                center_delta = (
                    polygon_centroid(polygons[first_index])
                    - polygon_centroid(polygons[second_index])
                )
                directions: list[np.ndarray] = []
                edge_directions: list[np.ndarray] = []
                norm = float(np.linalg.norm(center_delta))
                if norm > 1e-8:
                    directions.append(center_delta / norm)
                for polygon in (
                    polygons[first_index],
                    polygons[second_index],
                ):
                    for edge_index in range(len(polygon)):
                        edge = (
                            polygon[(edge_index + 1) % len(polygon)]
                            - polygon[edge_index]
                        )
                        normal = np.asarray(
                            [-edge[1], edge[0]], dtype=np.float64
                        )
                        normal_norm = float(np.linalg.norm(normal))
                        if normal_norm <= 1e-8:
                            continue
                        normal /= normal_norm
                        if float(normal @ center_delta) < 0.0:
                            normal *= -1.0
                        if not any(
                            abs(float(normal @ existing)) > 0.999
                            for existing in directions
                        ):
                            directions.append(normal)
                            edge_directions.append(normal)
                first_contour = polygons[first_index].astype(
                    np.float32
                ).reshape(-1, 1, 2)
                second_contour = polygons[second_index].astype(
                    np.float32
                ).reshape(-1, 1, 2)
                if (
                    edge_directions
                    and cv2.isContourConvex(first_contour)
                    and cv2.isContourConvex(second_contour)
                ):
                    # For two convex polygons the minimum-translation
                    # separating axis is one of their edge normals.  Testing
                    # only that axis replaces dozens of equivalent RDK-side
                    # binary searches without changing the result.
                    def projected_overlap(axis: np.ndarray) -> float:
                        first_projection = (
                            polygons[first_index] @ axis
                        )
                        second_projection = (
                            polygons[second_index] @ axis
                        )
                        return float(
                            min(
                                np.max(first_projection),
                                np.max(second_projection),
                            )
                            - max(
                                np.min(first_projection),
                                np.min(second_projection),
                            )
                        )

                    directions = [
                        min(edge_directions, key=projected_overlap)
                    ]

                current_cost = relief_cost(polygons)
                direct_move: tuple[
                    int, np.ndarray, int | None, np.ndarray | None
                ] | None = None
                direct_cost = current_cost
                for direction in directions:
                    first_available = available_distance(
                        overlap_relief[first_index], direction
                    )
                    second_available = available_distance(
                        overlap_relief[second_index], -direction
                    )
                    variants = (
                        (
                            first_index,
                            direction,
                            first_available,
                            None,
                            None,
                        ),
                        (
                            second_index,
                            -direction,
                            second_available,
                            None,
                            None,
                        ),
                        (
                            first_index,
                            direction * 0.5,
                            2.0
                            * min(first_available, second_available),
                            second_index,
                            -direction * 0.5,
                        ),
                    )
                    for (
                        move_index,
                        move_direction,
                        maximum_distance,
                        other_index,
                        other_direction,
                    ) in variants:
                        if maximum_distance <= 1e-6:
                            continue

                        def pair_area(distance: float) -> float:
                            first = polygons[first_index].copy()
                            second = polygons[second_index].copy()
                            if move_index == first_index:
                                first += move_direction * distance
                            else:
                                second += move_direction * distance
                            if (
                                other_index == first_index
                                and other_direction is not None
                            ):
                                first += other_direction * distance
                            elif (
                                other_index == second_index
                                and other_direction is not None
                            ):
                                second += other_direction * distance
                            return _polygon_intersection_area(first, second)

                        if (
                            pair_area(maximum_distance)
                            > accepted_pair_overlap
                        ):
                            continue
                        lower_distance = 0.0
                        upper_distance = maximum_distance
                        for _ in range(direct_binary_steps):
                            middle = (
                                lower_distance + upper_distance
                            ) * 0.5
                            if (
                                pair_area(middle)
                                > accepted_pair_overlap
                            ):
                                lower_distance = middle
                            else:
                                upper_distance = middle
                        distance = upper_distance + 0.02
                        move = move_direction * distance
                        other_move = (
                            other_direction * distance
                            if other_direction is not None
                            else None
                        )
                        candidate_polygons = [
                            polygon.copy() for polygon in polygons
                        ]
                        candidate_polygons[move_index] += move
                        if (
                            other_index is not None
                            and other_move is not None
                        ):
                            candidate_polygons[other_index] += other_move
                        cost = relief_cost(candidate_polygons)
                        if cost + 1e-8 < direct_cost:
                            direct_cost = cost
                            direct_move = (
                                move_index,
                                move,
                                other_index,
                                other_move,
                            )
                if direct_move is None:
                    break
                move_index, move, other_index, other_move = direct_move
                move_targets = [(move_index, move)]
                if other_index is not None and other_move is not None:
                    move_targets.append((other_index, other_move))
                for item_index, displacement in move_targets:
                    polygon = np.asarray(
                        plan[item_index]["target_polygon_mm"],
                        dtype=np.float64,
                    )
                    place = np.asarray(
                        plan[item_index]["place_mm"], dtype=np.float64
                    )
                    plan[item_index]["target_polygon_mm"] = np.round(
                        polygon + displacement, 3
                    ).tolist()
                    plan[item_index]["place_mm"] = np.round(
                        place + displacement, 3
                    ).tolist()
                    overlap_relief[item_index] += displacement

            for _ in range(maximum_iterations):
                polygons = [
                    np.asarray(
                        item["target_polygon_mm"], dtype=np.float64
                    )
                    for item in plan
                ]
                current_cost = relief_cost(polygons)
                if current_cost <= 1e-6:
                    break
                best_move: tuple[
                    int, np.ndarray, int | None, np.ndarray | None
                ] | None = None
                best_cost = current_cost
                for first_index, second_index in itertools.combinations(
                    range(len(plan)), 2
                ):
                    area = _polygon_intersection_area(
                        polygons[first_index], polygons[second_index]
                    )
                    if area <= accepted_pair_overlap:
                        continue
                    directions: list[np.ndarray] = []
                    center_delta = (
                        polygon_centroid(polygons[first_index])
                        - polygon_centroid(polygons[second_index])
                    )
                    norm = float(np.linalg.norm(center_delta))
                    if norm > 1e-6:
                        directions.append(center_delta / norm)
                    for angle_index in range(16):
                        angle = angle_index * math.pi / 8.0
                        directions.append(
                            np.asarray(
                                [math.cos(angle), math.sin(angle)],
                                dtype=np.float64,
                            )
                        )
                    for direction in directions:
                        variants = (
                            (
                                first_index,
                                direction * step,
                                None,
                                None,
                            ),
                            (
                                second_index,
                                -direction * step,
                                None,
                                None,
                            ),
                            (
                                first_index,
                                direction * step * 0.5,
                                second_index,
                                -direction * step * 0.5,
                            ),
                        )
                        for (
                            move_index,
                            move,
                            other_index,
                            other_move,
                        ) in variants:
                            if (
                                np.linalg.norm(
                                    overlap_relief[move_index] + move
                                )
                                > maximum_shift
                            ):
                                continue
                            if (
                                other_index is not None
                                and other_move is not None
                                and np.linalg.norm(
                                    overlap_relief[other_index]
                                    + other_move
                                )
                                > maximum_shift
                            ):
                                continue
                            candidate_polygons = [
                                polygon.copy() for polygon in polygons
                            ]
                            candidate_polygons[move_index] += move
                            if (
                                other_index is not None
                                and other_move is not None
                            ):
                                candidate_polygons[other_index] += other_move
                            cost = relief_cost(candidate_polygons)
                            if cost + 1e-8 < best_cost:
                                best_cost = cost
                                best_move = (
                                    move_index,
                                    move,
                                    other_index,
                                    other_move,
                                )
                if best_move is None:
                    break
                move_index, move, other_index, other_move = best_move
                move_targets = [(move_index, move)]
                if other_index is not None and other_move is not None:
                    move_targets.append((other_index, other_move))
                for item_index, displacement in move_targets:
                    polygon = np.asarray(
                        plan[item_index]["target_polygon_mm"],
                        dtype=np.float64,
                    )
                    polygon += displacement
                    place = np.asarray(
                        plan[item_index]["place_mm"], dtype=np.float64
                    )
                    place += displacement
                    plan[item_index]["target_polygon_mm"] = np.round(
                        polygon, 3
                    ).tolist()
                    plan[item_index]["place_mm"] = np.round(
                        place, 3
                    ).tolist()
                    overlap_relief[item_index] += displacement
            for item, displacement in zip(plan, overlap_relief):
                item["overlap_relief_mm"] = np.round(
                    displacement, 3
                ).tolist()

        maximum_target_overlap = 0.0
        target_strictly_non_overlapping = True
        target_polygons = [
            np.asarray(item["target_polygon_mm"], dtype=np.float32)
            for item in plan
        ]
        overlap_tolerance = float(self.config["overlap_tolerance_mm"])
        for first, second in itertools.combinations(target_polygons, 2):
            overlap_area = _polygon_intersection_area(first, second)
            maximum_target_overlap = max(maximum_target_overlap, overlap_area)
            if polygons_overlap(first, second, overlap_tolerance):
                target_strictly_non_overlapping = False
        # The overlap raster includes the shared seam itself (roughly half a
        # square millimetre per millimetre of seam at 2 px/mm), plus thin
        # segmentation slivers.  Accept only a bounded contour uncertainty;
        # the global gap/overlap score still rejects genuinely stacked pieces.
        accepted_overlap_area = float(
            self.config.get("maximum_accepted_pair_overlap_mm2", 80.0)
        )
        target_non_overlapping = (
            maximum_target_overlap <= accepted_overlap_area
        )
        minimum_fill = float(
            self.config.get("minimum_accepted_fill_ratio", 0.90)
        )
        maximum_score = float(
            self.config.get("maximum_accepted_geometry_score", 12.0)
        )
        solution_accepted = (
            best.fill_ratio >= minimum_fill
            and best.geometry_score <= maximum_score
            and target_non_overlapping
        )
        # Preserve the rigid-motion result for control/verification, while
        # straightening only those displayed vertices that are within the
        # contour tolerance of an already established rectangle side.  This
        # compensates edge shadows and lens/perspective residuals; it does not
        # change pick/place coordinates, scale, chirality, or rotation.
        for item in plan:
            measured = np.asarray(
                item["target_polygon_mm"], dtype=np.float64
            )
            display = measured.copy()
            for vertex in display:
                if abs(vertex[0] - target_lower[0]) <= snap_tolerance:
                    vertex[0] = target_lower[0]
                elif abs(vertex[0] - target_upper[0]) <= snap_tolerance:
                    vertex[0] = target_upper[0]
                if abs(vertex[1] - target_lower[1]) <= snap_tolerance:
                    vertex[1] = target_lower[1]
                elif abs(vertex[1] - target_upper[1]) <= snap_tolerance:
                    vertex[1] = target_upper[1]
            item["measured_target_polygon_mm"] = np.round(
                measured, 3
            ).tolist()
            item["target_polygon_mm"] = np.round(display, 3).tolist()
        return plan, {
            "search_nodes": self.nodes,
            "exact_anchor_piece": self.observations[exact_anchor].id,
            "partial_anchor_piece": (
                self.observations[partial_anchor].id
                if partial_anchor is not None
                else None
            ),
            "fallback_anchor_piece": (
                self.observations[fallback_anchors[0]].id
                if fallback_anchors
                else None
            ),
            "fallback_anchor_pieces": [
                self.observations[index].id
                for index in fallback_anchors
            ],
            "fallback_search_timed_out": fallback_timed_out,
            "search_timed_out": self.timed_out,
            "partial_edge_search": partial_searched,
            "exact_stage_timed_out": exact_timed_out,
            "candidate_count": len(self.candidates),
            "geometry_score": round(best.geometry_score, 5),
            "assembly_gap_mm2": round(best.gap_area_mm2, 5),
            "assembly_overlap_mm2": round(best.overlap_area_mm2, 5),
            "assembly_gap_ratio": round(best.gap_ratio, 6),
            "assembly_overlap_ratio": round(best.overlap_ratio, 6),
            "texture_score": round(best.texture_score, 5),
            "card_rank_score": round(best.card_rank_score, 5),
            "card_rank_corner_distance_mm": (
                round(best.card_rank_corner_distance_mm, 3)
                if best.card_rank_corner_distance_mm is not None
                else None
            ),
            "total_score": round(best.total_score, 5),
            "target_size_mm": np.round(size, 3).tolist(),
            "target_origin_mm": np.round(target_lower, 3).tolist(),
            "target_orientation": target_orientation,
            "fill_ratio": round(best.fill_ratio, 6),
            "solution_accepted": solution_accepted,
            "motion_model": "rotation_and_translation_only",
            "mirror_allowed": False,
            "solution_quality": (
                "high"
                if solution_accepted
                and best.fill_ratio >= 0.94
                and best.geometry_score <= 8.0
                else "usable"
                if solution_accepted
                else "rejected"
            ),
            "placement_clearance_mm": clearance,
            "boundary_snap_tolerance_mm": snap_tolerance,
            "target_outline_denoised": True,
            "target_non_overlapping": target_non_overlapping,
            "target_strictly_non_overlapping": (
                target_strictly_non_overlapping
            ),
            "maximum_accepted_pair_overlap_mm2": accepted_overlap_area,
            "maximum_target_overlap_mm2": round(maximum_target_overlap, 5),
        }
