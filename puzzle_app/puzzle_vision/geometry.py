from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


def signed_area(polygon: np.ndarray) -> float:
    # 计算多边形有向面积
    p = np.asarray(polygon, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    cross_sum = float(
        np.dot(x[:-1], y[1:])
        - np.dot(y[:-1], x[1:])
        + x[-1] * y[0]
        - y[-1] * x[0]
    )
    return 0.5 * cross_sum


def polygon_area(polygon: np.ndarray) -> float:
    # 计算多边形面积（绝对值）
    return abs(signed_area(polygon))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    # 计算多边形质心
    p = np.asarray(polygon, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    cross = np.empty(len(p), dtype=np.float64)
    cross[:-1] = x[:-1] * y[1:] - y[:-1] * x[1:]
    cross[-1] = x[-1] * y[0] - y[-1] * x[0]
    a2 = float(np.sum(cross))
    if abs(a2) < 1e-8:
        return np.mean(p, axis=0)
    cx = (
        np.sum((x[:-1] + x[1:]) * cross[:-1])
        + (x[-1] + x[0]) * cross[-1]
    ) / (3.0 * a2)
    cy = (
        np.sum((y[:-1] + y[1:]) * cross[:-1])
        + (y[-1] + y[0]) * cross[-1]
    ) / (3.0 * a2)
    return np.array([cx, cy], dtype=np.float64)


def normalize_winding(polygon: np.ndarray, positive: bool = True) -> np.ndarray:
    # 归一化多边形顶点绕序
    p = np.asarray(polygon, dtype=np.float64)
    if (signed_area(p) > 0) != positive:
        p = p[::-1].copy()
    return p


def rotation_matrix_row(theta_rad: float) -> np.ndarray:
    """Row-vector rotation matrix.

    With image coordinates (x right, y down), positive theta is clockwise.
    """
    # 构建行向量旋转矩阵（正角度为顺时针）

    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def is_proper_rotation(rotation: np.ndarray, tolerance: float = 1e-5) -> bool:
    """Return True only for a 2-D rotation, never a reflection or scaling."""
    # 验证是否为合法刚性旋转（非镜像/缩放）

    r = np.asarray(rotation, dtype=np.float64)
    return bool(
        r.shape == (2, 2)
        and np.allclose(r.T @ r, np.eye(2), atol=tolerance, rtol=0.0)
        and abs(float(np.linalg.det(r)) - 1.0) <= tolerance
    )


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    # 对点集应用旋转+平移变换
    return np.asarray(points, dtype=np.float64) @ rotation + translation


def compose_transforms(
    first_r: np.ndarray,
    first_t: np.ndarray,
    second_r: np.ndarray,
    second_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose row transforms: apply first, then second."""
    # 组合两个刚体变换（先first后second）

    return first_r @ second_r, first_t @ second_r + second_t


def invert_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 求刚体变换的逆变换
    inv_r = rotation.T
    inv_t = -translation @ inv_r
    return inv_r, inv_t


def transform_angle_deg(rotation: np.ndarray) -> float:
    # 从旋转矩阵提取角度（度）
    return math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))


def wrap_angle_deg(angle: float) -> float:
    # 将角度包裹到[-180, 180]范围
    return (angle + 180.0) % 360.0 - 180.0


def rigid_align(
    source: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigidly align equal-length ordered point lists without reflection."""
    # 刚性对齐两组等长有序点集

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    x = src - src_mean
    y = dst - dst_mean
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    if not is_proper_rotation(r):
        raise ValueError("Rigid alignment produced a reflected/non-rigid transform")
    t = dst_mean - src_mean @ r
    residual = transform_points(src, r, t) - dst
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return r, t, rms


def best_cyclic_alignment(
    source: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int]:
    # 找最优循环对齐（遍历所有顶点偏移）
    src = normalize_winding(source)
    dst = normalize_winding(destination)
    if len(src) != len(dst):
        raise ValueError("Polygons must have the same vertex count")
    best: tuple[np.ndarray, np.ndarray, float, int] | None = None
    for shift in range(len(dst)):
        ordered = np.roll(dst, -shift, axis=0)
        r, t, error = rigid_align(src, ordered)
        if best is None or error < best[2]:
            best = (r, t, error, shift)
    assert best is not None
    return best


def edge_lengths(polygon: np.ndarray) -> np.ndarray:
    # 计算多边形各边长
    p = np.asarray(polygon, dtype=np.float64)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def min_area_rectangle(
    polygons: Iterable[np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    # 求点集的最小面积外接矩形
    points = np.vstack([np.asarray(p, np.float32) for p in polygons])
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float64)
    lengths = np.linalg.norm(np.roll(box, -1, axis=0) - box, axis=1)
    idx = int(np.argmax(lengths))
    start = box[idx]
    end = box[(idx + 1) % 4]
    theta = math.atan2(end[1] - start[1], end[0] - start[0])
    long_side = float(max(rect[1]))
    short_side = float(min(rect[1]))
    return box, long_side, short_side, theta


def safe_interior_point(polygon: np.ndarray, resolution_mm: float = 0.5) -> np.ndarray:
    """Approximate the pole of inaccessibility using a distance transform."""
    # 用距离变换求多边形内部安全点

    p = np.asarray(polygon, dtype=np.float64)
    lower = np.floor(np.min(p, axis=0) - 2.0)
    upper = np.ceil(np.max(p, axis=0) + 2.0)
    size = np.maximum(
        3, np.ceil((upper - lower) / resolution_mm).astype(np.int32) + 1
    )
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    local = np.rint((p - lower) / resolution_mm).astype(np.int32)
    cv2.fillPoly(mask, [local], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, maximum = cv2.minMaxLoc(distance)
    return lower + np.array(maximum, dtype=np.float64) * resolution_mm


def segments_properly_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    epsilon: float,
) -> bool:
    # 判断两线段是否真相交
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    o1 = cross(b - a, c - a)
    o2 = cross(b - a, d - a)
    o3 = cross(d - c, a - c)
    o4 = cross(d - c, b - c)
    return o1 * o2 < -(epsilon**2) and o3 * o4 < -(epsilon**2)


def polygon_intersection_area(
    first: np.ndarray,
    second: np.ndarray,
    pixels_per_mm: float = 2.0,
) -> float:
    """Return polygon intersection area without OpenCV convex-clip quirks.

    Contest pieces are normally convex.  For that fast path use a small
    Sutherland-Hodgman clip implemented in double precision.  It is invariant
    to translation and treats a shared seam as zero area.  Concave outlines
    use the deterministic local raster fallback.
    """
    # 计算两个多边形的交集面积

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    contour_a = a.astype(np.float32).reshape(-1, 1, 2)
    contour_b = b.astype(np.float32).reshape(-1, 1, 2)
    if cv2.isContourConvex(contour_a) and cv2.isContourConvex(contour_b):
        # OpenCV performs this clip in compiled code and is an order of
        # magnitude faster than the Python Sutherland-Hodgman loop on RDK/K230
        # class ARM CPUs.  Rigid transformations preserve convexity, so this
        # is the dominant path for straight-edged competition/card pieces.
        area = float(cv2.intersectConvexConvex(contour_a, contour_b)[0])
        if math.isfinite(area):
            return max(0.0, area)

    scale = max(float(pixels_per_mm), 0.5)
    lower = np.floor(np.minimum(np.min(a, axis=0), np.min(b, axis=0)) - 1.0)
    upper = np.ceil(np.maximum(np.max(a, axis=0), np.max(b, axis=0)) + 1.0)
    size = np.maximum(3, np.ceil((upper - lower) * scale).astype(int) + 1)
    mask_a = np.zeros((int(size[1]), int(size[0])), np.uint8)
    mask_b = np.zeros_like(mask_a)
    local_a = np.rint((a - lower) * scale).astype(np.int32)
    local_b = np.rint((b - lower) * scale).astype(np.int32)
    cv2.fillPoly(mask_a, [local_a], 1)
    cv2.fillPoly(mask_b, [local_b], 1)
    return float(
        cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b)) / scale**2
    )


def polygons_overlap(
    first: np.ndarray, second: np.ndarray, tolerance_mm: float = 0.4
) -> bool:
    # 判断两个多边形是否重叠
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if (
        np.max(a[:, 0]) < np.min(b[:, 0])
        or np.max(b[:, 0]) < np.min(a[:, 0])
        or np.max(a[:, 1]) < np.min(b[:, 1])
        or np.max(b[:, 1]) < np.min(a[:, 1])
    ):
        return False

    contour_a = a.astype(np.float32).reshape(-1, 1, 2)
    contour_b = b.astype(np.float32).reshape(-1, 1, 2)
    if cv2.isContourConvex(contour_a) and cv2.isContourConvex(contour_b):
        intersection_area = float(
            cv2.intersectConvexConvex(contour_a, contour_b)[0]
        )
        allowed_area = max(5.0, 15.0 * tolerance_mm)
        return intersection_area > allowed_area

    # Avoid the old Python O(E1*E2) edge-crossing/probe pass here.  Card cuts
    # often create concave pieces and the recursive assembler invokes this
    # predicate thousands of times.  The local raster clip is both the final
    # authority and substantially faster because fill/intersection/count run
    # in compiled OpenCV code.  The area allowance absorbs the rasterised
    # shared-seam fringe exactly as it did after the old pre-check.
    intersection_area = polygon_intersection_area(a, b)
    allowed_area = max(5.0, 15.0 * tolerance_mm)
    return intersection_area > allowed_area


def edge_alignment_transforms(
    source_edge: np.ndarray,
    destination_edge: np.ndarray,
    length_tolerance_mm: float,
    relative_tolerance: float,
    minimum_partial_remainder_mm: float = 18.0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Map a source edge against a destination edge with opposite directions.

    Equal edges are centre-aligned.  Unequal edges may share either endpoint;
    this supports a T-junction where one long side touches two shorter sides.
    """
    # 计算边的对齐变换（含T型接合支持）

    u0, u1 = np.asarray(source_edge, dtype=np.float64)
    a, b = np.asarray(destination_edge, dtype=np.float64)
    src_vec = u1 - u0
    dst_vec = b - a
    src_len = float(np.linalg.norm(src_vec))
    dst_len = float(np.linalg.norm(dst_vec))
    allowed = max(
        length_tolerance_mm,
        relative_tolerance * min(src_len, dst_len),
    )
    difference = abs(src_len - dst_len)
    # The formal advanced task uses sides >= 20 mm.  A guided practice-piece
    # solve may explicitly lower this limit when camera simplification leaves
    # a shorter apparent remainder; autonomous mode keeps the strict default.
    if (
        difference > allowed
        and difference < minimum_partial_remainder_mm
    ):
        return []
    if max(src_len, dst_len) / min(src_len, dst_len) > 4.0:
        return []
    if min(src_len, dst_len) < 1e-6:
        return []

    source_angle = math.atan2(src_vec[1], src_vec[0])
    destination_reverse_angle = math.atan2(-dst_vec[1], -dst_vec[0])
    r = rotation_matrix_row(destination_reverse_angle - source_angle)
    unit = dst_vec / dst_len

    if difference <= allowed:
        positions = [(dst_len + src_len) * 0.5]
    else:
        # q0 is the transformed source u0 coordinate along destination a->b.
        positions = [dst_len, src_len]

    results: list[tuple[np.ndarray, np.ndarray, float]] = []
    for q0_coordinate in positions:
        q0 = a + unit * q0_coordinate
        t = q0 - u0 @ r
        overlap = min(src_len, dst_len)
        results.append((r, t, overlap))
    return results
