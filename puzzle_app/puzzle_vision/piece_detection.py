"""Puzzle-piece segmentation — extract individual pieces from a rectified A4 sheet.

Given a ``PaperView``, this module produces a binary foreground mask and a list
of ``PieceObservation`` values, each representing one puzzle fragment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    safe_interior_point,
)
from .paper_detection import DetectionError, PaperView


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PieceObservation:
    """One detected puzzle piece with its geometry in mm coordinates."""

    id: str
    polygon_mm: np.ndarray
    contour_px: np.ndarray
    centroid_mm: np.ndarray
    pickup_mm: np.ndarray
    area_mm2: float
    perimeter_mm: float
    edge_lengths_mm: np.ndarray


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------


def _dominant_lab_background(
    image: np.ndarray,
    valid_roi: np.ndarray,
    lab_image: np.ndarray | None = None,
) -> np.ndarray:
    lab = (
        np.asarray(lab_image, dtype=np.float32)
        if lab_image is not None
        else cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    )
    samples = lab[valid_roi > 0]
    if len(samples) < 100:
        raise DetectionError("Not enough background pixels to estimate A4 colour")
    # All pieces together occupy less than half of the source half-sheet, so
    # the per-channel median is guaranteed to represent paper.  A quantised
    # histogram mode is not safe under overexposure: four uniformly white
    # pieces may fall into one bin while a lighting gradient spreads the paper
    # across several bins.
    return np.median(samples, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


def _source_roi_mask(
    paper: PaperView,
    segmentation_cfg: dict[str, Any],
    source_region: str,
) -> np.ndarray:
    image = paper.image
    ppm = paper.pixels_per_mm
    divider_px = int(round(paper.divider_y_mm * ppm))
    margin = int(round(float(segmentation_cfg["paper_margin_mm"]) * ppm))
    divider_margin = int(
        round(float(segmentation_cfg["divider_margin_mm"]) * ppm)
    )
    roi = np.zeros(image.shape[:2], dtype=np.uint8)
    if source_region == "upper":
        top = margin
        bottom = divider_px - divider_margin
    elif source_region == "lower":
        top = divider_px + divider_margin
        bottom = image.shape[0] - margin - 1
    else:
        raise ValueError(f"Unsupported source region: {source_region}")
    cv2.rectangle(
        roi,
        (margin, top),
        (image.shape[1] - margin - 1, bottom),
        255,
        -1,
    )
    return roi


# ---------------------------------------------------------------------------
# Foreground mask
# ---------------------------------------------------------------------------


def foreground_mask(
    paper: PaperView,
    paper_cfg: dict[str, Any],
    segmentation_cfg: dict[str, Any],
    background_rectified: np.ndarray | None = None,
    source_region: str = "upper",
    colour_mode: str = "combined",
    current_lab: np.ndarray | None = None,
    estimated_background_lab: np.ndarray | None = None,
) -> np.ndarray:
    image = paper.image
    ppm = paper.pixels_per_mm
    roi = _source_roi_mask(
        paper, segmentation_cfg, source_region
    )

    current_lab = (
        np.asarray(current_lab, dtype=np.float32)
        if current_lab is not None
        else cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    )
    piece_colour = segmentation_cfg.get("piece_color_bgr")
    piece_distance: np.ndarray | None = None
    piece_chroma_distance: np.ndarray | None = None
    hint_relative_foreground: np.ndarray | None = None
    if piece_colour is not None:
        colour_pixel = np.asarray(piece_colour, dtype=np.uint8).reshape(1, 1, 3)
        piece_lab = cv2.cvtColor(
            colour_pixel, cv2.COLOR_BGR2LAB
        ).reshape(3).astype(np.float32)
        piece_distance = np.linalg.norm(current_lab - piece_lab, axis=2)
        piece_chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - piece_lab[1:], axis=2
        )
    if background_rectified is not None:
        if background_rectified.shape[:2] != image.shape[:2]:
            background_rectified = cv2.resize(
                background_rectified,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        reference_lab = cv2.cvtColor(
            background_rectified, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        distance = np.linalg.norm(current_lab - reference_lab, axis=2)
        chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2
        )
        lightness_delta = current_lab[:, :, 0] - reference_lab[:, :, 0]
        background = np.median(
            reference_lab[roi > 0], axis=0
        ).astype(np.float32)
        threshold = float(
            segmentation_cfg["background_difference_threshold"]
        )
    else:
        sampling_roi = roi.copy()
        # Exclude a thin band near the divider and retain a broad region so
        # the A4 colour remains the modal colour even with four pieces.
        background = (
            np.asarray(estimated_background_lab, dtype=np.float32)
            if estimated_background_lab is not None
            else _dominant_lab_background(
                image, sampling_roi, current_lab
            )
        )
        distance = np.linalg.norm(current_lab - background, axis=2)
        chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - background[1:], axis=2
        )
        lightness_delta = current_lab[:, :, 0] - background[0]
        threshold = float(segmentation_cfg["lab_distance_threshold"])

    # Convert the two user-selected colours into a relative colour offset.
    # Applying that offset to the observed paper colour compensates for camera
    # exposure and white-balance changes, so the colour controls remain useful
    # in dim or warm lighting instead of requiring camera-exact RGB values.
    paper_colour = paper_cfg.get("color_bgr")
    if piece_colour is not None and paper_colour is not None:
        paper_pixel = np.asarray(
            paper_colour, dtype=np.uint8
        ).reshape(1, 1, 3)
        expected_paper_lab = cv2.cvtColor(
            paper_pixel, cv2.COLOR_BGR2LAB
        ).reshape(3).astype(np.float32)
        colour_delta = piece_lab - expected_paper_lab
        colour_delta = colour_delta.copy()
        colour_delta[0] *= 0.70
        adapted_piece_lab = np.clip(
            background + colour_delta, 0.0, 255.0
        )
        adapted_distance = np.linalg.norm(
            current_lab - adapted_piece_lab, axis=2
        )
        adapted_chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - adapted_piece_lab[1:], axis=2
        )
        piece_distance = np.minimum(
            piece_distance, adapted_distance
        )
        piece_chroma_distance = np.minimum(
            piece_chroma_distance, adapted_chroma_distance
        )
        relative_masks: list[np.ndarray] = []
        light_threshold = float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
        if colour_delta[0] >= 8.0:
            relative_masks.append(lightness_delta >= light_threshold)
        elif colour_delta[0] <= -8.0:
            relative_masks.append(lightness_delta <= -light_threshold)
        if float(np.linalg.norm(colour_delta[1:])) >= 8.0:
            relative_masks.append(
                adapted_chroma_distance
                <= max(
                    18.0,
                    float(
                        segmentation_cfg.get(
                            "piece_color_tolerance_chroma", 12.0
                        )
                    ),
                )
            )
        if relative_masks:
            hint_relative_foreground = np.logical_or.reduce(
                relative_masks
            )

    chroma_threshold = float(
        segmentation_cfg.get("lab_chroma_threshold", float("inf"))
    )
    if colour_mode == "hint":
        if piece_distance is None or piece_chroma_distance is None:
            foreground = np.zeros(image.shape[:2], dtype=bool)
        else:
            # Camera exposure changes L much more than a/b chroma.  A chroma
            # hint therefore remains useful for the same coloured paper under
            # different heights and lights.  Retain a broad lightness guard so
            # neutral hints such as white do not select every grey object.
            full_match = piece_distance <= float(
                segmentation_cfg.get("piece_color_tolerance_lab", 32.0)
            )
            chroma_match = piece_chroma_distance <= float(
                segmentation_cfg.get("piece_color_tolerance_chroma", 12.0)
            )
            foreground = full_match | (
                chroma_match
                & (
                    np.abs(current_lab[:, :, 0] - piece_lab[0])
                    <= 100.0
                )
            )
            if hint_relative_foreground is not None:
                foreground |= hint_relative_foreground
    elif colour_mode == "full":
        foreground = distance >= threshold
    elif colour_mode == "light":
        foreground = lightness_delta >= float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
    elif colour_mode == "dark":
        foreground = lightness_delta <= -float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
    elif colour_mode == "chroma":
        foreground = chroma_distance >= chroma_threshold
    elif colour_mode == "combined":
        foreground = (distance >= threshold) | (
            chroma_distance >= chroma_threshold
        )
    else:
        raise ValueError(f"Unsupported colour segmentation mode: {colour_mode}")
    mask = (foreground & (roi > 0)).astype(np.uint8) * 255
    open_size = max(
        1, int(round(float(segmentation_cfg["morph_open_mm"]) * ppm))
    )
    close_size = max(
        1, int(round(float(segmentation_cfg["morph_close_mm"]) * ppm))
    )
    if open_size % 2 == 0:
        open_size += 1
    if close_size % 2 == 0:
        close_size += 1
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


# ---------------------------------------------------------------------------
# Polygon approximation
# ---------------------------------------------------------------------------


def _approximate_polygon(
    contour: np.ndarray,
    ppm: float,
    segmentation_cfg: dict[str, Any],
) -> np.ndarray:
    def remove_nearly_collinear(points: np.ndarray) -> np.ndarray:
        cleaned = points.copy()
        minimum_turn = np.deg2rad(
            float(segmentation_cfg.get("minimum_corner_turn_deg", 0.0))
        )
        maximum_short_edge = float(
            segmentation_cfg.get("collinear_short_edge_mm", float("inf"))
        )
        minimum = int(segmentation_cfg["expected_min_vertices"])
        while len(cleaned) > minimum and minimum_turn > 0:
            turns: list[float] = []
            for index in range(len(cleaned)):
                incoming = cleaned[index] - cleaned[(index - 1) % len(cleaned)]
                outgoing = cleaned[(index + 1) % len(cleaned)] - cleaned[index]
                if min(
                    float(np.linalg.norm(incoming)),
                    float(np.linalg.norm(outgoing)),
                ) / ppm > maximum_short_edge:
                    turns.append(float("inf"))
                    continue
                denominator = max(
                    float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
                    1e-9,
                )
                cosine = float(
                    np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0)
                )
                turns.append(float(np.arccos(cosine)))
            index = int(np.argmin(turns))
            if turns[index] >= minimum_turn:
                break
            cleaned = np.delete(cleaned, index, axis=0)
        return cleaned

    def remove_short_edges(points: np.ndarray) -> np.ndarray:
        minimum = int(segmentation_cfg["expected_min_vertices"])
        minimum_edge = float(segmentation_cfg.get("minimum_detected_edge_mm", 0.0))
        cleaned = points.copy()
        while len(cleaned) > minimum:
            lengths = np.linalg.norm(
                np.roll(cleaned, -1, axis=0) - cleaned, axis=1
            ) / ppm
            edge_index = int(np.argmin(lengths))
            if float(lengths[edge_index]) >= minimum_edge:
                break
            first = edge_index
            second = (edge_index + 1) % len(cleaned)

            def corner_area(index: int) -> float:
                previous = cleaned[(index - 1) % len(cleaned)]
                current = cleaned[index]
                following = cleaned[(index + 1) % len(cleaned)]
                return abs(float(np.cross(current - previous, following - current)))

            remove_index = (
                first if corner_area(first) <= corner_area(second) else second
            )
            cleaned = np.delete(cleaned, remove_index, axis=0)
        return cleaned

    minimum = int(segmentation_cfg["expected_min_vertices"])
    maximum = int(segmentation_cfg["expected_max_vertices"])
    epsilon = float(segmentation_cfg["polygon_epsilon_mm"])
    epsilon_max = float(segmentation_cfg["max_polygon_epsilon_mm"])
    minimum_edge = float(segmentation_cfg.get("minimum_detected_edge_mm", 0.0))
    approximation_contour = (
        cv2.convexHull(contour)
        if bool(segmentation_cfg.get("assume_convex_pieces", False))
        else contour
    )
    while epsilon <= epsilon_max + 1e-6:
        approx = cv2.approxPolyDP(
            approximation_contour, epsilon * ppm, True
        ).reshape(-1, 2)
        if minimum <= len(approx) <= maximum:
            approx = remove_nearly_collinear(approx)
            approx = remove_short_edges(approx)
        lengths_mm = (
            np.linalg.norm(np.roll(approx, -1, axis=0) - approx, axis=1) / ppm
            if len(approx) >= 2
            else np.array([], dtype=np.float64)
        )
        if (
            minimum <= len(approx) <= maximum
            and len(lengths_mm)
            and float(np.min(lengths_mm)) >= minimum_edge
        ):
            return approx
        epsilon += 0.25
    raise DetectionError(
        f"Contour cannot be approximated as a {minimum}-to-{maximum}-vertex polygon"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_pieces(
    paper: PaperView,
    paper_cfg: dict[str, Any],
    segmentation_cfg: dict[str, Any],
    background_rectified: np.ndarray | None = None,
    source_region: str = "upper",
) -> tuple[list[PieceObservation], np.ndarray, str, str]:
    if source_region not in ("upper", "lower", "auto"):
        raise ValueError(f"Unsupported source region: {source_region}")
    regions = ("upper", "lower") if source_region == "auto" else (source_region,)
    required = int(segmentation_cfg.get("required_pieces", 0))
    candidates: list[
        tuple[list[PieceObservation], np.ndarray, str, str, float, float]
    ] = []
    current_lab = cv2.cvtColor(
        paper.image, cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    for region in regions:
        estimated_background = None
        if background_rectified is None:
            estimated_background = _dominant_lab_background(
                paper.image,
                _source_roi_mask(paper, segmentation_cfg, region),
                current_lab,
            )
        colour_modes = (
            ("hint", "light", "dark", "chroma", "full", "combined")
            if segmentation_cfg.get("piece_color_bgr") is not None
            else ("light", "dark", "chroma", "full", "combined")
        )
        for colour_mode in colour_modes:
            mask = foreground_mask(
                paper,
                paper_cfg,
                segmentation_cfg,
                background_rectified,
                region,
                colour_mode,
                current_lab=current_lab,
                estimated_background_lab=estimated_background,
            )
            observations = _observations_from_mask(
                paper, segmentation_cfg, mask
            )
            straightness_error = sum(
                abs(polygon_area(item.polygon_mm) - item.area_mm2)
                / max(item.area_mm2, 1.0)
                for item in observations
            )
            paper_margin = float(segmentation_cfg["paper_margin_mm"])
            divider_margin = float(
                segmentation_cfg["divider_margin_mm"]
            )
            if region == "upper":
                bounds = (
                    paper_margin,
                    paper_margin,
                    paper.width_mm - paper_margin,
                    paper.divider_y_mm - divider_margin,
                )
            else:
                bounds = (
                    paper_margin,
                    paper.divider_y_mm + divider_margin,
                    paper.width_mm - paper_margin,
                    paper.height_mm - paper_margin,
                )
            # A contour touching the segmentation ROI is normally a bright
            # paper edge/illumination stripe, not a freely placed fragment.
            # Penalise it instead of blindly preferring a four-component mask.
            boundary_penalty = 0.0
            clearance = 1.0
            for observation in observations:
                polygon = observation.polygon_mm
                distances = (
                    float(np.min(polygon[:, 0]) - bounds[0]),
                    float(np.min(polygon[:, 1]) - bounds[1]),
                    float(bounds[2] - np.max(polygon[:, 0])),
                    float(bounds[3] - np.max(polygon[:, 1])),
                )
                boundary_penalty += sum(
                    max(0.0, clearance - distance)
                    for distance in distances
                )
            # Inspect the mask itself as well as its simplified polygons.
            # A bright strip just outside a slightly imperfect A4 homography
            # can merge with a real piece and then simplify to vertices a few
            # millimetres inside the ROI.  The component still touches the ROI
            # boundary in the binary mask, which is a reliable way to reject
            # this false "fourth piece" while retaining nearby free pieces.
            band = max(2, int(round(0.5 * paper.pixels_per_mm)))
            left_px = max(
                0, int(round(bounds[0] * paper.pixels_per_mm))
            )
            top_px = max(
                0, int(round(bounds[1] * paper.pixels_per_mm))
            )
            right_px = min(
                mask.shape[1] - 1,
                int(round(bounds[2] * paper.pixels_per_mm)),
            )
            bottom_px = min(
                mask.shape[0] - 1,
                int(round(bounds[3] * paper.pixels_per_mm)),
            )
            boundary_pixels = (
                cv2.countNonZero(
                    mask[
                        top_px : min(bottom_px + 1, top_px + band),
                        left_px : right_px + 1,
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        max(top_px, bottom_px - band + 1) : bottom_px + 1,
                        left_px : right_px + 1,
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        top_px : bottom_px + 1,
                        left_px : min(right_px + 1, left_px + band),
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        top_px : bottom_px + 1,
                        max(left_px, right_px - band + 1) : right_px + 1,
                    ]
                )
            )
            boundary_penalty += min(
                20.0,
                float(boundary_pixels)
                / max(1.0, 2.0 * paper.pixels_per_mm),
            )
            candidates.append(
                (
                    observations,
                    mask,
                    region,
                    colour_mode,
                    straightness_error,
                    boundary_penalty,
                )
            )
            # Four valid straight-edged components are the maximum allowed by
            # the task.  Once a clean mask reaches that maximum, evaluating
            # the remaining colour modes can only duplicate the same answer.
            # This removes most of the repeated morphology/contour work on
            # RDK/K230 while preserving the full fallback sweep for 2/3-piece
            # scenes or an incomplete colour hint.
            maximum_pieces = int(
                segmentation_cfg.get("max_pieces", 4)
            )
            expected_hint_area = float(
                segmentation_cfg.get("expected_total_area_mm2", 0.0)
            )
            hint_area_ratio = (
                sum(item.area_mm2 for item in observations)
                / expected_hint_area
                if expected_hint_area > 0.0
                else 1.0
            )
            structural_card_hint = (
                expected_hint_area > 0.0
                and int(
                    segmentation_cfg.get("expected_max_vertices", 99)
                )
                <= 5
                and 0.90 <= hint_area_ratio <= 1.10
            )
            if (
                colour_mode == "hint"
                and
                (expected_hint_area <= 0.0 or structural_card_hint)
                and
                len(observations) == maximum_pieces
                and (not required or len(observations) == required)
                and straightness_error
                <= (
                    0.16
                    if structural_card_hint
                    else 0.22
                )
                and boundary_penalty <= 1.0
            ):
                return observations, mask, region, colour_mode

    expected_total_area = float(
        segmentation_cfg.get("expected_total_area_mm2", 0.0)
    )
    expected_min_ratio = float(
        segmentation_cfg.get("expected_total_area_min_ratio", 0.0)
    )
    expected_max_ratio = float(
        segmentation_cfg.get(
            "expected_total_area_max_ratio", float("inf")
        )
    )

    def candidate_area_quality(
        candidate: tuple[
            list[PieceObservation], np.ndarray, str, str, float, float
        ],
    ) -> tuple[bool, float, float]:
        if expected_total_area <= 0.0:
            return True, 0.0, 0.0
        total_area = sum(item.area_mm2 for item in candidate[0])
        ratio = total_area / expected_total_area
        error = abs(ratio - 1.0)
        # Masks produced from different colour cues naturally differ by a
        # narrow anti-aliased boundary band.  Once their total areas are all
        # within that band, exact area is weaker evidence than the task prior
        # that every fragment has three-to-five straight structural edges.
        # Treat close areas as tied and prefer the cleaner polygon fit.  This
        # prevents printed suits touching a cut edge from winning solely
        # because their bloated mask happens to be 1% nearer the nominal card
        # area.
        tie_ratio = float(
            segmentation_cfg.get("card_candidate_area_tie_ratio", 0.04)
        )
        return (
            expected_min_ratio <= ratio <= expected_max_ratio,
            -max(0.0, error - tie_ratio),
            -error,
        )

    observations, mask, detected_region, selected_mode, _, _ = max(
        candidates,
        key=lambda item: (
            not required or len(item[0]) == required,
            bool(item[0]),
            # In playing-card mode the total physical card area is known.
            # Evaluate that evidence before component count: printed ranks and
            # suits can otherwise masquerade as four tiny "pieces" and beat a
            # correct two/three-piece white-card mask.
            (
                candidate_area_quality(item)[0]
                if expected_total_area > 0.0
                else True
            ),
            (
                candidate_area_quality(item)[1]
                if expected_total_area > 0.0
                else 0.0
            ),
            len(item[0]) if not required else 0,
            -item[5],
            item[3] == "hint",
            -item[4],
            (
                candidate_area_quality(item)[2]
                if expected_total_area > 0.0
                else 0.0
            ),
            item[3] in ("light", "dark"),
            item[3] == "chroma",
            item[2] == "upper",
        ),
    )
    if not observations:
        requested = "either half" if source_region == "auto" else source_region
        raise DetectionError(
            f"No puzzle pieces were detected in the {requested} of the A4 sheet."
        )
    if required and len(observations) != required:
        raise DetectionError(
            f"Exactly {required} separate straight-edged pieces are required, "
            f"but {len(observations)} valid pieces were detected."
        )
    return observations, mask, detected_region, selected_mode


def _observations_from_mask(
    paper: PaperView,
    segmentation_cfg: dict[str, Any],
    mask: np.ndarray,
) -> list[PieceObservation]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ppm = paper.pixels_per_mm
    minimum_area = float(segmentation_cfg["min_area_mm2"]) * ppm * ppm
    maximum_area = float(segmentation_cfg["max_area_mm2"]) * ppm * ppm
    contours = [
        c
        for c in contours
        if minimum_area <= abs(cv2.contourArea(c)) <= maximum_area
    ]
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    observations: list[PieceObservation] = []
    for contour in contours:
        if len(observations) >= int(segmentation_cfg["max_pieces"]):
            break
        try:
            approx_px = _approximate_polygon(contour, ppm, segmentation_cfg)
        except DetectionError:
            continue
        polygon_mm = normalize_winding(approx_px.astype(np.float64) / ppm)
        # Preserve the measured silhouette area for rectangle-fill scoring.
        # Approximation vertices are used for edge matching, but using their
        # area would under-count wrinkled or shadowed paper corners.
        area = abs(float(cv2.contourArea(contour))) / (ppm * ppm)
        centroid = polygon_centroid(polygon_mm)
        pickup = safe_interior_point(polygon_mm)
        lengths = edge_lengths(polygon_mm)
        observations.append(
            PieceObservation(
                id=f"piece_{len(observations) + 1}",
                polygon_mm=polygon_mm,
                contour_px=contour,
                centroid_mm=centroid,
                pickup_mm=pickup,
                area_mm2=area,
                perimeter_mm=float(np.sum(lengths)),
                edge_lengths_mm=lengths,
            )
        )
    # Stable IDs should not depend on contour area: order by pickup y then x.
    observations.sort(key=lambda item: (item.pickup_mm[1], item.pickup_mm[0]))
    for index, observation in enumerate(observations, start=1):
        observation.id = f"piece_{index}"
    return observations
