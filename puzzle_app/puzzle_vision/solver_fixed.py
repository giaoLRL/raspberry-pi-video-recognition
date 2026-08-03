"""Fixed-template solver — match observed pieces to a known target layout.

Each piece is matched independently via shape alignment and the optimal
permutation is selected.  Only proper rigid rotations are permitted; no
reflection or scaling is introduced.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .detector import PieceObservation
from .geometry import (
    invert_transform,
    is_proper_rotation,
    polygon_area,
    transform_angle_deg,
    transform_points,
    wrap_angle_deg,
)
from .solver_base import (
    SolveError,
    _shape_alignment,
    _validate_fixed_template,
)


def solve_fixed(
    observations: list[PieceObservation], fixed_cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 固定模板求解：用形状对齐将观察到的拼图块匹配到已知目标布局
    base_templates = [
        {
            "id": item["id"],
            "polygon": np.asarray(item["vertices_mm"], dtype=np.float64),
        }
        for item in fixed_cfg["pieces"]
    ]
    if len(observations) != len(base_templates):
        raise SolveError(
            f"Fixed mode requires {len(base_templates)} pieces, "
            f"detected {len(observations)}"
        )
    coverage_info = _validate_fixed_template(base_templates, fixed_cfg)
    target_size = np.asarray(coverage_info["target_size_mm"], dtype=np.float64)
    reflected_templates = [
        {
            "id": item["id"],
            "polygon": np.column_stack(
                [
                    target_size[0] - item["polygon"][:, 0],
                    item["polygon"][:, 1],
                ]
            ),
        }
        for item in base_templates
    ]
    _validate_fixed_template(reflected_templates, fixed_cfg)

    layout_results: list[dict[str, Any]] = []
    for layout_name, templates in (
        ("figure2", base_templates),
        ("figure2_reflected", reflected_templates),
    ):
        alignments: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, float]
        ] = {}
        costs = np.full(
            (len(observations), len(templates)), 1e6, dtype=np.float64
        )
        for obs_index, observation in enumerate(observations):
            for template_index, template in enumerate(templates):
                r, t, error = _shape_alignment(
                    template["polygon"], observation.polygon_mm
                )
                area_ratio = observation.area_mm2 / max(
                    polygon_area(template["polygon"]), 1e-6
                )
                area_penalty = abs(
                    math.log(max(area_ratio, 1e-6))
                ) * 8.0
                vertex_penalty = abs(
                    len(observation.polygon_mm)
                    - len(template["polygon"])
                ) * 1.5
                costs[obs_index, template_index] = (
                    error + area_penalty + vertex_penalty
                )
                alignments[(obs_index, template_index)] = (r, t, error)

        best_assignment: tuple[int, ...] | None = None
        best_cost = float("inf")
        for assignment in itertools.permutations(range(len(templates))):
            cost = sum(
                costs[index, assignment[index]]
                for index in range(len(observations))
            )
            if cost < best_cost:
                best_cost = float(cost)
                best_assignment = assignment
        assert best_assignment is not None
        layout_results.append(
            {
                "name": layout_name,
                "templates": templates,
                "alignments": alignments,
                "assignment": best_assignment,
                "cost": best_cost,
            }
        )

    selected = min(layout_results, key=lambda item: item["cost"])
    templates = selected["templates"]
    alignments = selected["alignments"]
    best_assignment = selected["assignment"]
    best_cost = float(selected["cost"])

    target_origin = np.asarray(fixed_cfg["target_origin_mm"], dtype=np.float64)
    plan: list[dict[str, Any]] = []
    errors = []
    for obs_index, template_index in enumerate(best_assignment):
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
            raise SolveError("Mirrored fixed-piece transforms are forbidden")
        target_polygon = template["polygon"] + target_origin
        measured_target_polygon = transform_points(
            observation.polygon_mm,
            inverse_r,
            inverse_t + target_origin,
        )
        target_pick = transform_points(
            observation.pickup_mm[None, :], inverse_r, inverse_t + target_origin
        )[0]
        rotate_deg = wrap_angle_deg(-transform_angle_deg(current_r))
        errors.append(match_error)
        plan.append(
            {
                "piece_id": observation.id,
                "template_id": template["id"],
                "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                "place_mm": np.round(target_pick, 3).tolist(),
                "rotate_deg": round(rotate_deg, 3),
                "mirrored": False,
                "target_polygon_mm": np.round(target_polygon, 3).tolist(),
                "measured_target_polygon_mm": np.round(
                    measured_target_polygon, 3
                ).tolist(),
                "match_error_mm": round(match_error, 3),
            }
        )

    maximum_error = max(errors, default=0.0)
    if maximum_error > float(fixed_cfg["max_match_error_mm"]):
        raise SolveError(
            f"Fixed-piece match error is too large ({maximum_error:.2f} mm). "
            "Check segmentation or update the fixed template vertices."
        )
    return plan, {
        "assignment_cost": round(best_cost, 4),
        "target_layout_variant": selected["name"],
        "layout_candidate_costs": {
            item["name"]: round(float(item["cost"]), 4)
            for item in layout_results
        },
        "max_match_error_mm": round(maximum_error, 4),
        "match_limit_mm": float(fixed_cfg["max_match_error_mm"]),
        **coverage_info,
        "target_origin_mm": np.round(target_origin, 3).tolist(),
        "solution_accepted": True,
        "solution_quality": "fixed_template_match",
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "piece_reflection_used": False,
        "target_non_overlapping": True,
        "maximum_target_overlap_mm2": coverage_info[
            "maximum_target_overlap_mm2"
        ],
    }
