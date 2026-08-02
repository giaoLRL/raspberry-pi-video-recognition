"""Playing-card solver — assemble 2–4 rounded-corner card fragments.

Every fragment owns one of the original card corners.  Candidate motions anchor
one fragment to each target corner using only proper rotations.  Card-rank
recognition provides a strong orientation prior when available.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import cv2
import numpy as np

from .card_recognition import recognize_card_marks
from .detector import PieceObservation
from .geometry import (
    compose_transforms,
    is_proper_rotation,
    polygon_area,
    polygon_centroid,
    polygons_overlap,
    rotation_matrix_row,
    transform_angle_deg,
    transform_points,
    wrap_angle_deg,
)
from .solver_base import (
    AssemblyCandidate,
    SolveError,
    UnknownPuzzleSolver,
    _card_rank_anchor_options,
    _card_rounded_corner_frames,
    _polygon_intersection_area,
)


def solve_card(
    observations: list[PieceObservation],
    unknown_cfg: dict[str, Any],
    rectified_image: np.ndarray | None = None,
    pixels_per_mm: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble four rounded-corner playing-card fragments.

    Every fragment owns one of the original card corners.  Candidate motions
    therefore anchor one fragment to each target corner and use only proper
    rotations.  Straight boundary support ranks the candidates; full polygon
    gap and overlap select the final result.  No reflection or scaling is
    introduced.
    """

    if not 2 <= len(observations) <= 4:
        raise SolveError("Playing-card mode requires two to four pieces")

    # A card cut does not necessarily leave exactly one rounded corner on
    # every fragment: one fragment can own two original corners while another
    # owns none.  The old one-piece/one-corner permutation therefore rejected
    # legal cuts.  Reuse the general rigid edge solver, but mark rounded-card
    # boundary edges as unavailable for seam docking and constrain the result
    # to a portrait playing-card rectangle.
    card_cfg = dict(unknown_cfg)
    card_cfg["card_mode"] = True
    card_cfg["target_orientation"] = "portrait"
    card_cfg["min_width_mm"] = float(
        unknown_cfg.get("card_minimum_long_side_mm", 78.0)
    )
    card_cfg["max_width_mm"] = float(
        unknown_cfg.get("card_maximum_long_side_mm", 100.0)
    )
    card_cfg["min_height_mm"] = float(
        unknown_cfg.get("card_minimum_short_side_mm", 45.0)
    )
    card_cfg["max_height_mm"] = float(
        unknown_cfg.get("card_maximum_short_side_mm", 68.0)
    )
    card_cfg["max_search_seconds"] = float(
        unknown_cfg.get("card_search_seconds", 6.0)
    )
    card_cfg["exact_search_seconds"] = float(
        unknown_cfg.get("card_exact_search_seconds", 1.5)
    )
    card_cfg["fallback_search_seconds"] = float(
        unknown_cfg.get("card_fallback_search_seconds", 2.0)
    )
    # Standard playing-card dimensions are a much stronger bound than the
    # broad generic field-rectangle limits.  Applying the tight bound during
    # partial search prevents the slow board from spending its deadline on
    # plausible-looking 64 x 94 mm layouts that can never be the original
    # 57 x 88 mm card.
    card_cfg["search_dimension_slack_mm"] = float(
        unknown_cfg.get("card_search_dimension_slack_mm", 1.5)
    )
    card_cfg["max_pair_options_exact"] = int(
        unknown_cfg.get("card_max_pair_options_exact", 36)
    )
    card_cfg["max_pair_options_partial"] = int(
        unknown_cfg.get("card_max_pair_options_partial", 36)
    )
    # The formal task requires genuinely non-overlapping target fragments.
    # Disable display-oriented boundary snapping for cards and make pairwise
    # polygon intersection a hard acceptance condition.
    card_cfg["boundary_snap_tolerance_mm"] = 0.0
    card_cfg["maximum_accepted_pair_overlap_mm2"] = float(
        unknown_cfg.get("card_maximum_pair_overlap_mm2", 0.5)
    )
    card_cfg["overlap_tolerance_mm"] = float(
        unknown_cfg.get("card_overlap_tolerance_mm", 0.15)
    )
    card_cfg["minimum_partial_remainder_mm"] = float(
        unknown_cfg.get("card_minimum_partial_remainder_mm", 3.0)
    )
    recognition: dict[str, Any] = {
        "rank_detected": False,
        "rank": None,
        "rank_confidence": 0.0,
    }
    if rectified_image is not None:
        recognition = recognize_card_marks(
            rectified_image,
            observations,
            pixels_per_mm,
            unknown_cfg,
        )
        if recognition.get("rank_detected", False):
            card_cfg["card_rank_piece_index"] = int(
                recognition["rank_piece_index"]
            )
            card_cfg["card_rank_center_mm"] = list(
                recognition["rank_center_mm"]
            )
        visibility = float(
            recognition.get("pattern_visibility_fraction", 0.0)
        )
        confidence = float(recognition.get("rank_confidence", 0.0))
        card_cfg["texture_weight"] = float(
            unknown_cfg.get("texture_weight", 0.3)
        ) * visibility
        card_cfg["card_rank_score_weight"] = float(
            unknown_cfg.get("card_rank_score_weight", 0.2)
        ) * confidence
    attempts: list[
        tuple[list[dict[str, Any]], dict[str, Any], bool]
    ] = []
    errors: list[str] = []
    # A reliably read corner index provides a much stronger pose anchor than
    # an arbitrary equal-length edge.  Try fixed portrait-card dimensions
    # inferred from total area first.  Each trial remains a proper rigid
    # transform and must still satisfy the non-overlap hard condition.
    rank_piece_index = recognition.get("rank_piece_index")
    if (
        recognition.get("rank_detected", False)
        and rank_piece_index is not None
        and float(recognition.get("rank_confidence", 0.0))
        >= float(
            unknown_cfg.get(
                "card_anchor_min_rank_confidence", 0.60
            )
        )
        and bool(
            recognition.get(
                "rank_rounded_corner_reference", False
            )
        )
    ):
        measured_area = sum(item.area_mm2 for item in observations)
        aspect = float(
            unknown_cfg.get("card_aspect_ratio", 88.0 / 57.0)
        )
        anchor_complete = False
        anchored_started = time.perf_counter()
        anchored_attempts = 0
        anchored_maximum_attempts = int(
            unknown_cfg.get("card_anchored_max_attempts", 2)
        )
        anchored_total_seconds = float(
            unknown_cfg.get("card_anchored_total_seconds", 1.2)
        )
        for expected_fill in unknown_cfg.get(
            "card_expected_fill_ratios", [0.98, 0.94, 0.90]
        ):
            target_area = measured_area / max(
                float(expected_fill), 1e-6
            )
            width = math.sqrt(target_area / aspect)
            height = width * aspect
            anchor_options = _card_rank_anchor_options(
                observations[int(rank_piece_index)].polygon_mm,
                width,
                height,
                unknown_cfg,
            )
            for anchor_rotation, anchor_translation, _ in anchor_options:
                if (
                    anchored_attempts >= anchored_maximum_attempts
                    or time.perf_counter() - anchored_started
                    >= anchored_total_seconds
                ):
                    break
                anchored_attempts += 1
                attempt_cfg = dict(card_cfg)
                attempt_cfg["card_target_size_mm"] = [width, height]
                attempt_cfg["card_rounded_chord_max_mm"] = 0.0
                attempt_cfg["max_search_seconds"] = float(
                    unknown_cfg.get(
                        "card_anchored_search_seconds", 1.4
                    )
                )
                attempt_cfg["exact_search_seconds"] = float(
                    unknown_cfg.get(
                        "card_anchored_exact_seconds", 0.35
                    )
                )
                try:
                    attempt_plan, attempt_information = (
                        UnknownPuzzleSolver(
                            observations,
                            attempt_cfg,
                            rectified_image=rectified_image,
                            pixels_per_mm=pixels_per_mm,
                            use_texture=rectified_image is not None,
                            initial_transforms={
                                int(rank_piece_index): (
                                    anchor_rotation,
                                    anchor_translation,
                                )
                            },
                        ).solve()
                    )
                    attempt_information[
                        "card_rank_corner_anchor_used"
                    ] = True
                    attempt_information[
                        "card_expected_fill_ratio"
                    ] = float(expected_fill)
                    attempts.append(
                        (attempt_plan, attempt_information, False)
                    )
                    if attempt_information.get(
                        "solution_accepted", False
                    ):
                        anchor_complete = True
                        break
                except SolveError as exc:
                    errors.append(str(exc))
            if anchor_complete:
                break
            if (
                anchored_attempts >= anchored_maximum_attempts
                or time.perf_counter() - anchored_started
                >= anchored_total_seconds
            ):
                break

    for exclude_rounded_edges in (
        () if any(
            item[1].get("solution_accepted", False)
            for item in attempts
        )
        else (True, False)
    ):
        attempt_cfg = dict(card_cfg)
        if not exclude_rounded_edges:
            # A short hand-cut bevel can occasionally imitate a rounded card
            # chord.  Retry without edge exclusion only when the strict pass
            # did not yield an accepted, non-overlapping rectangle.
            attempt_cfg["card_rounded_chord_max_mm"] = 0.0
        try:
            attempt_plan, attempt_information = UnknownPuzzleSolver(
                observations,
                attempt_cfg,
                rectified_image=rectified_image,
                pixels_per_mm=pixels_per_mm,
                use_texture=rectified_image is not None,
            ).solve()
            attempts.append(
                (
                    attempt_plan,
                    attempt_information,
                    exclude_rounded_edges,
                )
            )
            if attempt_information.get("solution_accepted", False):
                break
        except SolveError as exc:
            errors.append(str(exc))
    if not attempts:
        raise SolveError(
            errors[-1]
            if errors
            else "No non-overlapping playing-card assembly was found"
        )
    plan, information, excluded_edges = min(
        attempts,
        key=lambda item: (
            not bool(item[1].get("solution_accepted", False)),
            float(item[1].get("geometry_score", float("inf"))),
            float(item[1].get("total_score", float("inf"))),
        ),
    )
    information["solver_method"] = "portrait_card_rigid_search"
    information["card_aspect_ratio"] = round(
        float(unknown_cfg.get("card_aspect_ratio", 88.0 / 57.0)), 6
    )
    information["card_rounded_boundary_edges_excluded"] = excluded_edges
    if errors:
        information["card_solver_attempt_errors"] = errors
    information["card_recognition"] = recognition
    return plan, information
