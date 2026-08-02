"""Taught-layout solver — match observations to a previously demonstrated layout.

The taught layout is a fast path for a known set of 2–4 physical pieces.
Matching uses proper rigid rotations only.  When the demonstrated layout does
not fill its bounding rectangle, the solver falls back to guided rigid search
using the demonstrated relative poses as a hint.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .detector import PieceObservation
from .geometry import (
    compose_transforms,
    invert_transform,
    is_proper_rotation,
    polygon_area,
    polygons_overlap,
    transform_angle_deg,
    transform_points,
    wrap_angle_deg,
)
from .solver_base import (
    SolveError,
    UnknownPuzzleSolver,
    _polygon_intersection_area,
    _shape_alignment,
)


def solve_taught(
    observations: list[PieceObservation],
    layout: dict[str, Any],
    unknown_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match observations to a previously demonstrated rectangle.

    The match uses proper rigid rotations only.  A taught layout is a fast
    path for a known set of two to four physical pieces and also prevents an
    unchanged scene from jumping between several geometrically similar
    autonomous solutions.
    """

    base_templates = [
        {
            "id": item["id"],
            "polygon": np.asarray(item["vertices_mm"], dtype=np.float64),
        }
        for item in layout.get("pieces", [])
    ]
    if len(observations) != len(base_templates) or not base_templates:
        raise SolveError("Taught layout piece count does not match")

    target_size = np.asarray(
        layout.get("target_size_mm"), dtype=np.float64
    )
    if target_size.shape != (2,) or np.any(target_size <= 0):
        raise SolveError("Taught target_size_mm must be [width, height]")

    reflected_templates = [
        {
            "id": item["id"],
            # This is a global mirror of the demonstrated *layout*.  Matching
            # and moving every observed piece below still uses only a proper
            # rotation and translation.  It handles the common case where all
            # pieces were turned over together between demonstration and run.
            "polygon": np.column_stack(
                [
                    target_size[0] - item["polygon"][:, 0],
                    item["polygon"][:, 1],
                ]
            ),
        }
        for item in base_templates
    ]

    variant_results: list[dict[str, Any]] = []
    for variant_name, variant_templates in (
        ("demonstrated", base_templates),
        ("globally_reflected", reflected_templates),
    ):
        variant_alignments: dict[
            tuple[int, int],
            tuple[np.ndarray, np.ndarray, float],
        ] = {}
        variant_costs = np.full(
            (len(observations), len(variant_templates)),
            1e6,
            dtype=np.float64,
        )
        for obs_index, observation in enumerate(observations):
            for template_index, template in enumerate(
                variant_templates
            ):
                r, t, error = _shape_alignment(
                    template["polygon"],
                    observation.polygon_mm,
                    sample_count=36,
                )
                if not is_proper_rotation(r):
                    continue
                area_ratio = observation.area_mm2 / max(
                    polygon_area(template["polygon"]), 1e-6
                )
                variant_costs[obs_index, template_index] = (
                    error
                    + abs(
                        math.log(max(area_ratio, 1e-6))
                    )
                    * 8.0
                    + abs(
                        len(observation.polygon_mm)
                        - len(template["polygon"])
                    )
                    * 1.5
                )
                variant_alignments[
                    (obs_index, template_index)
                ] = (r, t, error)
        variant_assignment = min(
            itertools.permutations(
                range(len(variant_templates))
            ),
            key=lambda value: sum(
                variant_costs[index, value[index]]
                for index in range(len(observations))
            ),
        )
        variant_results.append(
            {
                "name": variant_name,
                "templates": variant_templates,
                "alignments": variant_alignments,
                "costs": variant_costs,
                "assignment": variant_assignment,
                "total_cost": float(
                    sum(
                        variant_costs[index][
                            variant_assignment[index]
                        ]
                        for index in range(len(observations))
                    )
                ),
            }
        )
    selected_variant = min(
        variant_results,
        key=lambda item: item["total_cost"],
    )
    layout_variant = str(selected_variant["name"])
    templates = selected_variant["templates"]
    alignments = selected_variant["alignments"]
    costs = selected_variant["costs"]
    assignment = selected_variant["assignment"]
    match_errors = [
        alignments[(index, assignment[index])][2]
        for index in range(len(observations))
    ]
    maximum_error = max(match_errors, default=float("inf"))
    error_limit = float(layout.get("max_match_error_mm", 12.0))
    if maximum_error > error_limit:
        raise SolveError(
            f"Taught-layout match error is too large "
            f"({maximum_error:.2f} mm > {error_limit:.2f} mm)"
        )

    # A demonstration is often made with several millimetres of clearance so
    # the operator can show the intended adjacency without manually achieving
    # a perfect final fit.  Such a capture is a topology/orientation hint, not
    # an already valid motion target.  When its nominal bounding rectangle is
    # not sufficiently filled, run the normal rigid, no-reflection solver but
    # rank edge dockings by their agreement with the demonstrated relative
    # poses.  The returned layout must still pass the same independent
    # rectangle, overlap, and fill checks as a fully autonomous solution.
    demonstrated_fill = sum(
        polygon_area(item["polygon"]) for item in templates
    ) / max(float(np.prod(target_size)), 1e-6)
    if demonstrated_fill < float(
        unknown_cfg.get("minimum_accepted_fill_ratio", 0.9)
    ):
        pose_hints: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        for obs_index, template_index in enumerate(assignment):
            current_r, current_t, _ = alignments[
                (obs_index, template_index)
            ]
            pose_hints[obs_index] = invert_transform(
                current_r, current_t
            )
        guided_config = dict(unknown_cfg)
        guided_config["pose_hint_weight"] = float(
            unknown_cfg.get("pose_hint_weight", 1.8)
        )
        guided_config["minimum_partial_remainder_mm"] = float(
            unknown_cfg.get(
                "guided_minimum_partial_remainder_mm", 6.0
            )
        )
        guided_config["max_search_seconds"] = float(
            unknown_cfg.get(
                "guided_search_seconds",
                unknown_cfg.get("max_search_seconds", 2.5),
            )
        )
        guided_config["fallback_search_seconds"] = float(
            unknown_cfg.get(
                "guided_fallback_search_seconds",
                unknown_cfg.get("fallback_search_seconds", 2.5),
            )
        )
        plan, info = UnknownPuzzleSolver(
            observations,
            guided_config,
            pose_hints=pose_hints,
        ).solve()
        info["solver_method"] = "taught_guided_rigid_search"
        info["taught_layout_name"] = layout.get(
            "name", "unnamed"
        )
        info["taught_layout_variant"] = layout_variant
        info["taught_layout_variant_costs"] = {
            str(item["name"]): round(
                float(item["total_cost"]), 4
            )
            for item in variant_results
        }
        info["taught_assignment_cost"] = round(
            float(
                sum(
                    costs[index, assignment[index]]
                    for index in range(len(observations))
                )
            ),
            4,
        )
        info["taught_max_match_error_mm"] = round(
            maximum_error, 4
        )
        info["taught_demonstrated_fill_ratio"] = round(
            demonstrated_fill, 6
        )
        return plan, info

    zone = np.asarray(unknown_cfg["target_zone_mm"], dtype=np.float64)
    zone_center = np.asarray(
        [(zone[0] + zone[2]) * 0.5, (zone[1] + zone[3]) * 0.5],
        dtype=np.float64,
    )
    target_origin = zone_center - target_size * 0.5
    plan: list[dict[str, Any]] = []
    target_polygons: list[np.ndarray] = []
    for obs_index, template_index in enumerate(assignment):
        observation = observations[obs_index]
        template = templates[template_index]
        current_r, current_t, match_error = alignments[
            (obs_index, template_index)
        ]
        inverse_r, inverse_t = invert_transform(current_r, current_t)
        if not (
            is_proper_rotation(current_r)
            and is_proper_rotation(inverse_r)
        ):
            raise SolveError("Mirrored taught-piece transforms are forbidden")
        target_polygon = template["polygon"] + target_origin
        measured_target = transform_points(
            observation.polygon_mm,
            inverse_r,
            inverse_t + target_origin,
        )
        place = transform_points(
            observation.pickup_mm[None, :],
            inverse_r,
            inverse_t + target_origin,
        )[0]
        target_polygons.append(target_polygon)
        plan.append(
            {
                "piece_id": observation.id,
                "template_id": template["id"],
                "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                "place_mm": np.round(place, 3).tolist(),
                "rotate_deg": round(
                    wrap_angle_deg(-transform_angle_deg(current_r)), 3
                ),
                "mirrored": False,
                "target_polygon_mm": np.round(
                    target_polygon, 3
                ).tolist(),
                "measured_target_polygon_mm": np.round(
                    measured_target, 3
                ).tolist(),
                "match_error_mm": round(match_error, 3),
            }
        )

    maximum_overlap = 0.0
    target_non_overlapping = True
    for first, second in itertools.combinations(target_polygons, 2):
        overlap = _polygon_intersection_area(first, second)
        maximum_overlap = max(maximum_overlap, overlap)
        if polygons_overlap(
            first,
            second,
            float(unknown_cfg["overlap_tolerance_mm"]),
        ):
            target_non_overlapping = False

    pieces_area = sum(item.area_mm2 for item in observations)
    fill_ratio = pieces_area / max(float(np.prod(target_size)), 1e-6)
    dimension_ok = (
        float(unknown_cfg["min_width_mm"]) <= target_size[0]
        <= float(unknown_cfg["max_width_mm"])
        and float(unknown_cfg["min_height_mm"]) <= target_size[1]
        <= float(unknown_cfg["max_height_mm"])
    )
    fill_ok = (
        float(unknown_cfg.get("minimum_accepted_fill_ratio", 0.9))
        <= fill_ratio
        <= float(unknown_cfg.get("maximum_candidate_fill_ratio", 1.08))
    )
    solution_accepted = dimension_ok and target_non_overlapping and fill_ok
    return plan, {
        "solver_method": "taught_layout",
        "taught_layout_name": layout.get("name", "unnamed"),
        "taught_layout_variant": layout_variant,
        "taught_layout_variant_costs": {
            str(item["name"]): round(
                float(item["total_cost"]), 4
            )
            for item in variant_results
        },
        "assignment_cost": round(
            float(sum(costs[index, assignment[index]] for index in range(len(observations)))),
            4,
        ),
        "max_match_error_mm": round(maximum_error, 4),
        "match_limit_mm": error_limit,
        "target_size_mm": np.round(target_size, 3).tolist(),
        "target_origin_mm": np.round(target_origin, 3).tolist(),
        "fill_ratio": round(fill_ratio, 6),
        "geometry_score": round(maximum_error, 5),
        "solution_accepted": solution_accepted,
        "solution_quality": "taught_template_match",
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "piece_reflection_used": False,
        "target_non_overlapping": target_non_overlapping,
        "maximum_target_overlap_mm2": round(maximum_overlap, 5),
        "target_outline_denoised": True,
    }
