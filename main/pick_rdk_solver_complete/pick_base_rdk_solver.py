#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E题拼图装置：使用 pick 识别 + 原压缩包三模式矩形恢复算法

识别部分：
    保留 pick_base(9).py 的手动 A4 标定、透视校正、亮色碎片分割、
    3～5 边凸多边形拟合、面积排序和原图反投影显示。

恢复部分：
    直接调用 01_RDK_PC_完整版_含扑克牌.zip 中未改写的 puzzle_vision
    恢复模块：solve_fixed、solve_taught、solve_unknown、solve_card。
    所有放置仅允许二维旋转和平移，禁止单块镜像和缩放。

按键：
    C：重新标定 A4 四角并保存 a4_corners.json
    P：按当前模式恢复矩形，默认 AUTO 自动识别三种情况
    M：循环 AUTO / 已知固定 / 未知白色 / 扑克牌
    0：AUTO 自动模式
    1：已知固定四片模式
    2：未知白色碎片模式
    3：扑克牌图案模式
    T：显示/隐藏目标位置叠加
    D：输出识别结果、目标位置和旋转角
    A：循环显示最小面积框 / 最大面积框 / 不显示
    S：保存当前原始画面叠加图
    Q / ESC：退出

运行目录必须同时存在：
    puzzle_vision/、config.json、taught_layout.json

依赖：
    pip install opencv-python numpy
"""

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# 恢复算法原样来自压缩包。把本文件所在目录加入模块搜索路径，
# 从同目录的 puzzle_vision 包导入，不重新实现或简化求解器。
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from puzzle_vision.config import load_config
from puzzle_vision.detector import PieceObservation
from puzzle_vision.geometry import (
    edge_lengths as solver_edge_lengths,
    normalize_winding as solver_normalize_winding,
    polygon_area as solver_polygon_area,
    polygon_centroid as solver_polygon_centroid,
    safe_interior_point,
)
from puzzle_vision.solver import (
    SolveError,
    solve_card,
    solve_fixed,
    solve_taught,
    solve_unknown,
)


# ============================================================
# 参数
# ============================================================

CAMERA_INDEX = 0
MAIN_WINDOW_NAME = "Camera"

# 透视校正后的标定区域大小。
# 若标定对象是纵向A4纸，840:1188约等于210:297。
WARP_WIDTH = 840
WARP_HEIGHT = 1188

# ============================================================
# 通用面积过滤范围：兼容已知、未知和扑克牌三种情况
# ============================================================
# A4 透视校正为 840×1188，即 X/Y 均为 4 px/mm。
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PIXELS_PER_MM_X = WARP_WIDTH / A4_WIDTH_MM
PIXELS_PER_MM_Y = WARP_HEIGHT / A4_HEIGHT_MM
PIXELS_PER_MM = (PIXELS_PER_MM_X + PIXELS_PER_MM_Y) * 0.5
PIXELS_PER_CM = PIXELS_PER_MM * 10.0

# 第2项目标矩形最大约 12×9 cm，单块可能占据大部分矩形；
# 因此不能继续使用固定图2的 4.8～24 cm²窄范围。
MIN_PIECE_AREA_CM2 = 1.5
MAX_PIECE_AREA_CM2 = 115.0
MIN_PIECE_AREA_PX = int(round(MIN_PIECE_AREA_CM2 * PIXELS_PER_CM**2))
MAX_PIECE_AREA_PX = int(round(MAX_PIECE_AREA_CM2 * PIXELS_PER_CM**2))

# 面积参考框仍保留 A 键显示。
THEORETICAL_MIN_PIECE_AREA_PX = MIN_PIECE_AREA_PX
THEORETICAL_MAX_PIECE_AREA_PX = MAX_PIECE_AREA_PX

# 只识别3～5条直边的凸多边形。
MIN_SIDES = 3
MAX_SIDES = 5

# 多边形拟合基础系数。
# 如果实际轮廓顶点偏多，可适当增大；丢角时适当减小。
POLYGON_EPSILON_RATIO = 0.015

# 过滤中间分界线、细长反光等目标。
# 旋转外接矩形的短边小于此值时不认为是碎片。
MIN_PIECE_SHORT_SIDE_PX = 24.0

# 标定点保存文件。
CALIBRATION_FILE = Path("a4_corners.json")

SOLVER_CONFIG_FILE = PROJECT_DIR / "config.json"
TAUGHT_LAYOUT_FILE = PROJECT_DIR / "taught_layout.json"
MAX_DETECTED_PIECES = 4

RECOVERY_MODES = ("auto", "fixed", "unknown-white", "unknown-pattern")
RECOVERY_MODE_NAMES = {
    "auto": "AUTO自动三模式",
    "fixed": "已知固定四片",
    "unknown-white": "未知白色碎片",
    "unknown-pattern": "扑克牌图案",
}

# AUTO 模式判定参数。固定模板只有在形状误差足够小时才优先于未知求解；
# 扑克牌还会结合图案高频比例和压缩包自己的牌面识别结果。
AUTO_FIXED_STRONG_ERROR_MM = 7.0
AUTO_PATTERN_SCORE_THRESHOLD = 0.012


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DetectedPiece:
    piece_id: int
    # 对外输出坐标：标定区域左下角为原点，x向右，y向上。
    center_x: int
    center_y: int
    # OpenCV内部坐标：标定区域左上角为原点，y向下，仅用于绘图反投影。
    center_y_image: int
    area_px: float
    # 对外输出顶点：标定区域左下角为原点，x向右，y向上。
    # 顶点按逆时针排列，并从最靠左、再最靠下的顶点开始。
    vertices: list[tuple[int, int]]
    contour: np.ndarray
    polygon: np.ndarray


@dataclass
class RecoveryResult:
    requested_mode: str
    selected_mode: str
    plan: list[dict[str, Any]]
    solver_info: dict[str, Any]
    observations: list[PieceObservation]
    pattern_score: float
    solve_time_sec: float
    attempts: list[dict[str, Any]]

    @property
    def accepted(self) -> bool:
        return bool(self.solver_info.get("solution_accepted", False))


# ============================================================
# 标定区域处理
# ============================================================

def order_corners(points: np.ndarray) -> np.ndarray:
    """
    把4个点自动排列成：左上、右上、右下、左下。
    """
    points = np.asarray(points, dtype=np.float32)

    if points.shape != (4, 2):
        raise ValueError("必须提供4个角点")

    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_diff = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(coordinate_sum)]   # 左上
    ordered[1] = points[np.argmin(coordinate_diff)]  # 右上
    ordered[2] = points[np.argmax(coordinate_sum)]   # 右下
    ordered[3] = points[np.argmax(coordinate_diff)]  # 左下

    return ordered


def select_calibration_corners(frame: np.ndarray) -> np.ndarray:
    """
    直接在原始摄像头窗口中点击4个角点，不新开窗口。

    点击顺序不限：
        R：清空重选
        Enter：确认
        ESC：取消
    """
    selected_points: list[tuple[int, int]] = []

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        userdata: object,
    ) -> None:
        del flags, userdata

        if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
            selected_points.append((x, y))

    cv2.setMouseCallback(MAIN_WINDOW_NAME, mouse_callback)

    try:
        while True:
            display = frame.copy()

            # 已点击点及连线。
            for index, point in enumerate(selected_points):
                cv2.circle(display, point, 7, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    str(index + 1),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            if len(selected_points) >= 2:
                cv2.polylines(
                    display,
                    [np.asarray(selected_points, dtype=np.int32)],
                    False,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if len(selected_points) == 4:
                ordered_preview = order_corners(
                    np.asarray(selected_points, dtype=np.float32)
                )
                cv2.polylines(
                    display,
                    [np.round(ordered_preview).astype(np.int32)],
                    True,
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                f"Select 4 corners: {len(selected_points)}/4",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "R: reset   Enter: confirm   ESC: cancel",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.imshow(MAIN_WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("r"), ord("R")):
                selected_points.clear()

            elif key in (13, 10):
                if len(selected_points) == 4:
                    return order_corners(
                        np.asarray(selected_points, dtype=np.float32)
                    )

            elif key == 27:
                raise RuntimeError("已取消标定")

    finally:
        # 恢复为空回调，避免正常运行时继续记录鼠标点击。
        cv2.setMouseCallback(
            MAIN_WINDOW_NAME,
            lambda event, x, y, flags, userdata: None,
        )


def save_corners(corners: np.ndarray) -> None:
    data = {"corners": corners.tolist()}
    CALIBRATION_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def load_corners() -> Optional[np.ndarray]:
    if not CALIBRATION_FILE.exists():
        return None

    try:
        data = json.loads(
            CALIBRATION_FILE.read_text(encoding="utf-8")
        )
        corners = np.asarray(data["corners"], dtype=np.float32)

        if corners.shape != (4, 2):
            return None

        return corners

    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def build_perspective_matrices(
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回：
        camera_to_warp：原始摄像头 -> 标定平面
        warp_to_camera：标定平面 -> 原始摄像头
    """
    destination = np.array(
        [
            [0, 0],
            [WARP_WIDTH - 1, 0],
            [WARP_WIDTH - 1, WARP_HEIGHT - 1],
            [0, WARP_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    camera_to_warp = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destination,
    )

    warp_to_camera = cv2.getPerspectiveTransform(
        destination,
        corners.astype(np.float32),
    )

    return camera_to_warp, warp_to_camera


def perspective_correct(
    frame: np.ndarray,
    camera_to_warp: np.ndarray,
) -> np.ndarray:
    return cv2.warpPerspective(
        frame,
        camera_to_warp,
        (WARP_WIDTH, WARP_HEIGHT),
    )


# ============================================================
# 碎片分割
# ============================================================

def create_piece_mask(calibrated_region: np.ndarray) -> np.ndarray:
    """
    在黑色背景上分割白色或较亮的彩色碎片。
    使用HSV亮度通道，以提升对彩色碎片的兼容性。
    """
    hsv = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]

    blurred = cv2.GaussianBlur(value_channel, (5, 5), 0)

    threshold_value, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # 防止极暗画面下Otsu阈值过低，把黑纸噪声识别为碎片。
    if threshold_value < 30:
        _, mask = cv2.threshold(
            blurred,
            30,
            255,
            cv2.THRESH_BINARY,
        )

    # 去掉孤立噪声。
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    # 连接碎片边缘上的小缺口。
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    return mask


# ============================================================
# 碎片识别
# ============================================================

def calculate_area_center(
    contour: np.ndarray,
) -> Optional[tuple[int, int]]:
    """
    通过填充轮廓的图像矩计算面积中心。
    """
    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-8:
        return None

    center_x = int(round(moments["m10"] / moments["m00"]))
    center_y = int(round(moments["m01"] / moments["m00"]))

    return center_x, center_y


def approximate_straight_polygon(
    contour: np.ndarray,
) -> Optional[np.ndarray]:
    """
    将轮廓拟合成由直线组成的3～5边凸多边形。

    先取凸包消除轮廓小凹坑，再尝试不同拟合系数。
    返回形状为(N, 1, 2)的顶点数组。
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, closed=True)

    if perimeter <= 0:
        return None

    # 先尝试基础系数，再逐渐放宽，适应不同清晰度的直边轮廓。
    ratios = [
        POLYGON_EPSILON_RATIO,
        0.010,
        0.012,
        0.018,
        0.020,
        0.025,
        0.030,
        0.040,
    ]

    candidates: list[tuple[float, np.ndarray]] = []
    hull_area = max(cv2.contourArea(hull), 1.0)

    for ratio in ratios:
        polygon = cv2.approxPolyDP(
            hull,
            ratio * perimeter,
            closed=True,
        )

        side_count = len(polygon)

        if not MIN_SIDES <= side_count <= MAX_SIDES:
            continue

        if not cv2.isContourConvex(polygon):
            continue

        polygon_area = cv2.contourArea(polygon)
        area_error = abs(hull_area - polygon_area) / hull_area

        # 面积误差越小，拟合越贴近真实碎片边缘。
        candidates.append((area_error, polygon))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def convert_polygon_vertices_to_output(
    polygon: np.ndarray,
) -> list[tuple[int, int]]:
    """
    将OpenCV左上角图像坐标转换为左下角笛卡尔坐标。

    输出规则：
        1. 原点位于标定区域左下角
        2. x向右为正，y向上为正
        3. 顶点按逆时针排列
        4. 从x最小、x相同时y最小的顶点开始

    保留多边形原有相邻关系，不对顶点按坐标直接排序。
    """
    image_points = np.asarray(
        polygon,
        dtype=np.int32,
    ).reshape(-1, 2)

    output_points = [
        (
            int(point[0]),
            int((WARP_HEIGHT - 1) - point[1]),
        )
        for point in image_points
    ]

    if len(output_points) < 3:
        return output_points

    # 笛卡尔坐标下的有向面积：正值表示逆时针。
    signed_area_twice = 0.0
    for index, current in enumerate(output_points):
        next_point = output_points[(index + 1) % len(output_points)]
        signed_area_twice += (
            current[0] * next_point[1]
            - next_point[0] * current[1]
        )

    if signed_area_twice < 0:
        output_points.reverse()

    # 循环平移起点，不破坏逆时针相邻顺序。
    start_index = min(
        range(len(output_points)),
        key=lambda index: (
            output_points[index][0],
            output_points[index][1],
        ),
    )

    return (
        output_points[start_index:]
        + output_points[:start_index]
    )

def detect_pieces(
    calibrated_region: np.ndarray,
) -> tuple[list[DetectedPiece], np.ndarray]:
    """
    对整个标定区域进行碎片识别。
    """
    mask = create_piece_mask(calibrated_region)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[dict] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_PIECE_AREA_PX or area > MAX_PIECE_AREA_PX:
            continue

        # 过滤细长分界线、反光线等目标。
        rotated_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rotated_rect[1]
        short_side = min(rect_width, rect_height)

        if short_side < MIN_PIECE_SHORT_SIDE_PX:
            continue

        polygon = approximate_straight_polygon(contour)

        if polygon is None:
            continue

        center = calculate_area_center(contour)

        if center is None:
            continue

        vertices = convert_polygon_vertices_to_output(polygon)

        if len(vertices) != len(polygon):
            continue

        candidates.append(
            {
                "area": area,
                "center": center,
                "vertices": vertices,
                "contour": contour,
                "polygon": polygon,
            }
        )

    # ID按照面积从小到大排序；面积完全相同时按中心位置稳定排序。
    candidates.sort(
        key=lambda item: (
            item["area"],
            item["center"][1],
            item["center"][0],
        )
    )

    pieces: list[DetectedPiece] = []

    for index, candidate in enumerate(candidates, start=1):
        center_x_image, center_y_image = candidate["center"]

        # 坐标系转换：
        # OpenCV图像坐标以左上角为原点，y向下；
        # 对外输出改为标定区域左下角为原点，y向上。
        center_x_output = center_x_image
        center_y_output = (WARP_HEIGHT - 1) - center_y_image

        pieces.append(
            DetectedPiece(
                piece_id=index,
                center_x=center_x_output,
                center_y=center_y_output,
                center_y_image=center_y_image,
                area_px=candidate["area"],
                vertices=candidate["vertices"],
                contour=candidate["contour"],
                polygon=candidate["polygon"],
            )
        )

    return pieces, mask


# ============================================================
# pick识别结果 -> 压缩包恢复算法
# ============================================================

def load_solver_assets() -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """读取压缩包原配置和可选教学布局。"""
    config = load_config(
        SOLVER_CONFIG_FILE if SOLVER_CONFIG_FILE.exists() else None
    )
    # 固定本程序的 A4 比例和目标区域，避免相对工作目录影响配置读取。
    config["paper"]["pixels_per_mm"] = PIXELS_PER_MM
    config["paper"]["width_mm"] = A4_WIDTH_MM
    config["paper"]["height_mm"] = A4_HEIGHT_MM
    config["paper"]["divider_y_mm"] = A4_HEIGHT_MM * 0.5
    config["unknown"]["target_zone_mm"] = [
        0.0,
        A4_HEIGHT_MM * 0.5,
        A4_WIDTH_MM,
        A4_HEIGHT_MM,
    ]
    config["unknown"]["taught_layout_path"] = str(TAUGHT_LAYOUT_FILE)

    taught_layout: Optional[dict[str, Any]] = None
    if TAUGHT_LAYOUT_FILE.exists():
        try:
            taught_layout = json.loads(
                TAUGHT_LAYOUT_FILE.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            taught_layout = None
    return config, taught_layout


def detected_pieces_to_solver_observations(
    pieces: list[DetectedPiece],
) -> list[PieceObservation]:
    """只转换坐标和数据结构，不改变 pick 拟合出的多边形。"""
    observations: list[PieceObservation] = []
    for piece in pieces:
        polygon_px = np.asarray(
            piece.polygon, dtype=np.float64
        ).reshape(-1, 2)
        polygon_mm = solver_normalize_winding(
            polygon_px / PIXELS_PER_MM
        )

        centroid_mm = solver_polygon_centroid(polygon_mm)
        try:
            pickup_mm = safe_interior_point(
                polygon_mm,
                resolution_mm=0.5,
            )
        except (cv2.error, ValueError, RuntimeError):
            pickup_mm = centroid_mm.copy()

        lengths_mm = solver_edge_lengths(polygon_mm)
        area_mm2 = solver_polygon_area(polygon_mm)
        perimeter_mm = float(np.sum(lengths_mm))

        observations.append(
            PieceObservation(
                id=f"piece_{piece.piece_id}",
                polygon_mm=polygon_mm,
                contour_px=np.asarray(piece.contour, dtype=np.int32),
                centroid_mm=centroid_mm,
                pickup_mm=pickup_mm,
                area_mm2=float(area_mm2),
                perimeter_mm=perimeter_mm,
                edge_lengths_mm=lengths_mm,
            )
        )
    return observations


def estimate_pattern_score(
    calibrated_region: np.ndarray,
    pieces: list[DetectedPiece],
) -> float:
    """估计碎片内部是否有扑克牌花纹；只用于 AUTO 的尝试顺序。"""
    gray = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2GRAY)
    scores: list[float] = []
    for piece in pieces:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        polygon = np.asarray(piece.polygon, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [polygon], 255)

        # 排除边界、阴影和切割毛刺，只看碎片内部纹理。
        kernel_size = max(5, int(round(2.0 * PIXELS_PER_MM)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        inner = cv2.erode(mask, kernel, iterations=1)
        valid = inner > 0
        count = int(np.count_nonzero(valid))
        if count < 100:
            continue

        smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
        residual = cv2.absdiff(gray, smooth)
        high_frequency = float(np.mean(residual[valid] >= 13))

        edges = cv2.Canny(gray, 55, 145)
        edge_density = float(np.mean(edges[valid] > 0))

        values = gray[valid].astype(np.float32)
        p10, p90 = np.percentile(values, [10.0, 90.0])
        contrast = min(1.0, max(0.0, float(p90 - p10) / 90.0))
        scores.append(
            0.50 * high_frequency
            + 0.35 * edge_density
            + 0.15 * contrast
        )

    return float(np.median(scores)) if scores else 0.0


def _solver_attempt_record(
    mode: str,
    accepted: bool,
    elapsed: float,
    info: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "mode": mode,
        "accepted": bool(accepted),
        "time_sec": round(float(elapsed), 4),
    }
    if info is not None:
        record["quality"] = info.get("solution_quality")
        record["fill_ratio"] = info.get("fill_ratio")
        record["geometry_score"] = info.get("geometry_score")
        record["max_match_error_mm"] = info.get("max_match_error_mm")
    if error:
        record["error"] = error
    return record


def run_one_solver_mode(
    mode: str,
    observations: list[PieceObservation],
    calibrated_region: np.ndarray,
    config: dict[str, Any],
    taught_layout: Optional[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按压缩包 pipeline 的分支调用原求解器。"""
    if mode == "fixed":
        fixed_cfg = deepcopy(config["fixed"])
        return solve_fixed(observations, fixed_cfg)

    unknown_cfg = deepcopy(config["unknown"])
    unknown_cfg["target_zone_mm"] = [
        0.0,
        A4_HEIGHT_MM * 0.5,
        A4_WIDTH_MM,
        A4_HEIGHT_MM,
    ]

    if mode == "unknown-pattern":
        unknown_cfg["target_orientation"] = "portrait"
        return solve_card(
            observations,
            unknown_cfg,
            calibrated_region,
            PIXELS_PER_MM,
        )

    if mode != "unknown-white":
        raise ValueError(f"不支持的恢复模式：{mode}")

    unknown_cfg["target_orientation"] = "landscape"
    taught_error: Optional[str] = None
    if (
        bool(unknown_cfg.get("use_taught_layout", False))
        and taught_layout is not None
        and int(taught_layout.get("piece_count", 0)) == len(observations)
    ):
        try:
            plan, info = solve_taught(
                observations,
                taught_layout,
                unknown_cfg,
            )
            if bool(info.get("solution_accepted", False)):
                info["solver_path"] = "taught_layout"
                return plan, info
            raise SolveError("教学布局候选未通过矩形验收")
        except SolveError as exc:
            taught_error = str(exc)

    plan, info = solve_unknown(
        observations,
        unknown_cfg,
        calibrated_region,
        PIXELS_PER_MM,
        use_texture=False,
    )
    info["solver_path"] = "unknown_geometry"
    if taught_error is not None:
        info["taught_layout_fallback_reason"] = taught_error
    return plan, info


def solve_with_pick_recognition(
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    requested_mode: str,
) -> RecoveryResult:
    if requested_mode not in RECOVERY_MODES:
        raise RuntimeError(f"未知恢复模式：{requested_mode}")
    if not 1 <= len(pieces) <= MAX_DETECTED_PIECES:
        raise RuntimeError(
            f"恢复要求识别到1～{MAX_DETECTED_PIECES}块，当前为{len(pieces)}块"
        )

    observations = detected_pieces_to_solver_observations(pieces)
    config, taught_layout = load_solver_assets()
    pattern_score = estimate_pattern_score(calibrated_region, pieces)
    attempts: list[dict[str, Any]] = []
    successful: dict[str, tuple[list[dict[str, Any]], dict[str, Any], float]] = {}
    total_started = time.perf_counter()

    def attempt(mode: str) -> None:
        if mode in successful or any(item["mode"] == mode for item in attempts):
            return
        started = time.perf_counter()
        try:
            plan, info = run_one_solver_mode(
                mode,
                observations,
                calibrated_region,
                config,
                taught_layout,
            )
            elapsed = time.perf_counter() - started
            accepted = bool(info.get("solution_accepted", False))
            attempts.append(
                _solver_attempt_record(mode, accepted, elapsed, info=info)
            )
            if accepted:
                successful[mode] = (plan, info, elapsed)
        except (SolveError, RuntimeError, ValueError, cv2.error) as exc:
            elapsed = time.perf_counter() - started
            attempts.append(
                _solver_attempt_record(
                    mode,
                    False,
                    elapsed,
                    error=str(exc),
                )
            )

    if requested_mode != "auto":
        attempt(requested_mode)
        if requested_mode not in successful:
            detail = next(
                (item.get("error") for item in attempts if item["mode"] == requested_mode),
                None,
            )
            raise RuntimeError(
                f"{RECOVERY_MODE_NAMES[requested_mode]}恢复失败"
                + (f"：{detail}" if detail else "：候选未通过验收")
            )
        plan, info, _ = successful[requested_mode]
        return RecoveryResult(
            requested_mode=requested_mode,
            selected_mode=requested_mode,
            plan=plan,
            solver_info=info,
            observations=observations,
            pattern_score=pattern_score,
            solve_time_sec=time.perf_counter() - total_started,
            attempts=attempts,
        )

    # AUTO：有明显图案先尝试扑克牌；固定题只在四片时尝试；
    # 未知几何始终尝试。最后仍会补试尚未运行的扑克牌分支。
    if pattern_score >= AUTO_PATTERN_SCORE_THRESHOLD and 2 <= len(pieces) <= 4:
        attempt("unknown-pattern")
        card = successful.get("unknown-pattern")
        if card is not None:
            recognition = card[1].get("card_recognition", {})
            if (
                pattern_score >= AUTO_PATTERN_SCORE_THRESHOLD
                or bool(recognition.get("rank_detected", False))
            ):
                plan, info, _ = card
                return RecoveryResult(
                    requested_mode="auto",
                    selected_mode="unknown-pattern",
                    plan=plan,
                    solver_info=info,
                    observations=observations,
                    pattern_score=pattern_score,
                    solve_time_sec=time.perf_counter() - total_started,
                    attempts=attempts,
                )

    if len(pieces) == 4:
        attempt("fixed")
    attempt("unknown-white")
    if 2 <= len(pieces) <= 4:
        attempt("unknown-pattern")

    card = successful.get("unknown-pattern")
    if card is not None:
        recognition = card[1].get("card_recognition", {})
        if (
            pattern_score >= AUTO_PATTERN_SCORE_THRESHOLD
            or bool(recognition.get("rank_detected", False))
            or float(recognition.get("pattern_visibility_fraction", 0.0)) >= 0.01
        ):
            selected_mode = "unknown-pattern"
        else:
            selected_mode = ""
    else:
        selected_mode = ""

    fixed = successful.get("fixed")
    if not selected_mode and fixed is not None:
        fixed_error = float(fixed[1].get("max_match_error_mm", 1e9))
        if fixed_error <= AUTO_FIXED_STRONG_ERROR_MM:
            selected_mode = "fixed"

    if not selected_mode and "unknown-white" in successful:
        selected_mode = "unknown-white"
    if not selected_mode and fixed is not None:
        selected_mode = "fixed"
    if not selected_mode and card is not None:
        selected_mode = "unknown-pattern"

    if not selected_mode:
        details = "; ".join(
            f"{RECOVERY_MODE_NAMES.get(item['mode'], item['mode'])}: "
            f"{item.get('error', '未通过验收')}"
            for item in attempts
        )
        raise RuntimeError("三种恢复算法均失败：" + details)

    plan, info, _ = successful[selected_mode]
    return RecoveryResult(
        requested_mode="auto",
        selected_mode=selected_mode,
        plan=plan,
        solver_info=info,
        observations=observations,
        pattern_score=pattern_score,
        solve_time_sec=time.perf_counter() - total_started,
        attempts=attempts,
    )


def draw_recovery_overlay(
    result: np.ndarray,
    recovery: RecoveryResult,
    warp_to_camera: np.ndarray,
) -> None:
    colours = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 180, 255),
        (255, 100, 255),
    ]
    for item in recovery.plan:
        try:
            piece_number = int(str(item["piece_id"]).split("_")[-1])
        except (ValueError, IndexError):
            piece_number = 1
        colour = colours[(piece_number - 1) % len(colours)]
        target_mm = np.asarray(item["target_polygon_mm"], dtype=np.float64)
        target_px = target_mm * PIXELS_PER_MM
        target_camera = warp_points_to_camera(target_px, warp_to_camera)
        target_camera_int = np.round(target_camera).astype(np.int32)

        layer = result.copy()
        cv2.fillPoly(layer, [target_camera_int], colour, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.22, result, 0.78, 0.0, dst=result)
        cv2.polylines(
            result,
            [target_camera_int],
            True,
            colour,
            3,
            cv2.LINE_AA,
        )

        place_px = np.asarray(item["place_mm"], dtype=np.float64) * PIXELS_PER_MM
        place_camera = warp_points_to_camera(
            np.asarray([place_px], dtype=np.float32),
            warp_to_camera,
        )[0]
        place_camera_int = tuple(np.round(place_camera).astype(int).tolist())
        cv2.drawMarker(
            result,
            place_camera_int,
            colour,
            cv2.MARKER_TILTED_CROSS,
            18,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            f"T{piece_number} {float(item['rotate_deg']):+.1f}deg",
            (place_camera_int[0] + 8, place_camera_int[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )

    info = recovery.solver_info
    origin = info.get("target_origin_mm")
    size = info.get("target_size_mm")
    if origin is not None and size is not None:
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        if origin.shape == (2,) and size.shape == (2,):
            rectangle_mm = np.asarray(
                [
                    origin,
                    origin + [size[0], 0.0],
                    origin + size,
                    origin + [0.0, size[1]],
                ],
                dtype=np.float64,
            )
            rectangle_camera = warp_points_to_camera(
                rectangle_mm * PIXELS_PER_MM,
                warp_to_camera,
            )
            cv2.polylines(
                result,
                [np.round(rectangle_camera).astype(np.int32)],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )


# ============================================================
# 坐标反投影与原图绘制
# ============================================================

def warp_points_to_camera(
    points: np.ndarray,
    warp_to_camera: np.ndarray,
) -> np.ndarray:
    """
    把校正平面中的点映射回原始摄像头图像。

    输入可为(N, 1, 2)或(N, 2)。
    返回(N, 2)浮点坐标。
    """
    reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(reshaped, warp_to_camera)
    return mapped.reshape(-1, 2)


def draw_results_on_original(
    frame: np.ndarray,
    corners: np.ndarray,
    pieces: list[DetectedPiece],
    warp_to_camera: np.ndarray,
    area_display_mode: int,
    recovery: Optional[RecoveryResult] = None,
    show_target_overlay: bool = True,
    selected_mode: str = "auto",
) -> np.ndarray:
    """
    所有标定边框和识别结果都绘制在原始摄像头画面上。
    """
    result = frame.copy()

    # 青色：标定范围。
    cv2.polylines(
        result,
        [np.round(corners).astype(np.int32)],
        True,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    for piece in pieces:
        # 使用拟合后的多边形，不使用带噪声的原始轮廓，确保黄色边是直线。
        polygon_camera = warp_points_to_camera(
            piece.polygon,
            warp_to_camera,
        )
        polygon_camera_int = np.round(
            polygon_camera
        ).astype(np.int32)

        cv2.polylines(
            result,
            [polygon_camera_int],
            True,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        center_camera = warp_points_to_camera(
            np.array(
                [[[piece.center_x, piece.center_y_image]]],
                dtype=np.float32,
            ),
            warp_to_camera,
        )[0]

        center_camera_int = (
            int(round(float(center_camera[0]))),
            int(round(float(center_camera[1]))),
        )

        # 红色面积中心。
        cv2.circle(
            result,
            center_camera_int,
            7,
            (0, 0, 255),
            -1,
        )
        cv2.drawMarker(
            result,
            center_camera_int,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2,
        )

        # 画面只显示ID，坐标通过D键在控制台输出。
        cv2.putText(
            result,
            f"ID:{piece.piece_id}",
            (
                center_camera_int[0] + 12,
                center_camera_int[1] - 12,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if show_target_overlay and recovery is not None:
        draw_recovery_overlay(result, recovery, warp_to_camera)

    cv2.putText(
        result,
        f"Pieces: {len(pieces)}  Mode: {RECOVERY_MODE_NAMES[selected_mode]}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if recovery is not None:
        cv2.putText(
            result,
            f"RECT OK: {RECOVERY_MODE_NAMES[recovery.selected_mode]}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # A键三段循环：
    # 0：不显示
    # 1：显示最小可识别面积对应的等面积正方形
    # 2：显示最大可识别面积对应的等面积正方形
    #
    # 面积本身无法唯一决定长宽，因此这里用“等面积正方形”
    # 直观展示面积大小，并将正方形从校正平面反投影到原始画面。
    if area_display_mode in (1, 2):
        if area_display_mode == 1:
            display_area_px = MIN_PIECE_AREA_PX
            box_color = (0, 255, 0)
            box_name = "MIN AREA"
        else:
            display_area_px = MAX_PIECE_AREA_PX
            box_color = (255, 0, 255)
            box_name = "MAX AREA"

        square_side_px = float(np.sqrt(display_area_px))

        # 将参考框放在标定平面中央。
        center_x_warp = WARP_WIDTH / 2.0
        center_y_warp = WARP_HEIGHT / 2.0
        half_side = square_side_px / 2.0

        reference_square_warp = np.array(
            [
                [center_x_warp - half_side, center_y_warp - half_side],
                [center_x_warp + half_side, center_y_warp - half_side],
                [center_x_warp + half_side, center_y_warp + half_side],
                [center_x_warp - half_side, center_y_warp + half_side],
            ],
            dtype=np.float32,
        )

        reference_square_camera = warp_points_to_camera(
            reference_square_warp,
            warp_to_camera,
        )
        reference_square_camera_int = np.round(
            reference_square_camera
        ).astype(np.int32)

        # 粗线框显示对应面积。
        cv2.polylines(
            result,
            [reference_square_camera_int],
            True,
            box_color,
            4,
            cv2.LINE_AA,
        )

        # 四个角加圆点，增强可见性。
        for point in reference_square_camera_int:
            cv2.circle(
                result,
                (int(point[0]), int(point[1])),
                6,
                box_color,
                -1,
                cv2.LINE_AA,
            )

        # 标签放在框的第一条边上方。
        label_anchor = reference_square_camera_int[0]
        label_x = max(10, int(label_anchor[0]))
        label_y = max(30, int(label_anchor[1]) - 12)

        cv2.putText(
            result,
            f"{box_name}: {display_area_px} px^2",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return result


def print_detection_results(pieces: list[DetectedPiece]) -> None:
    print("\n识别结果（pick识别；左下角为原点，x向右，y向上）：")
    if not pieces:
        print("没有识别到有效碎片")
        return

    for piece in pieces:
        polygon_mm = np.asarray(piece.polygon, dtype=np.float64).reshape(-1, 2) / PIXELS_PER_MM
        lengths_cm = solver_edge_lengths(polygon_mm) / 10.0
        print(
            f"ID={piece.piece_id}, center_px=({piece.center_x}, {piece.center_y}), "
            f"area={piece.area_px:.1f}px², sides={len(piece.vertices)}, "
            f"edge_lengths_cm={[round(float(v), 3) for v in lengths_cm]}"
        )
        vertices_text = ", ".join(
            f"P{index}=({x}, {y})"
            for index, (x, y) in enumerate(piece.vertices, start=1)
        )
        print(f"  vertices_px=[{vertices_text}]")


def print_recovery_result(recovery: Optional[RecoveryResult]) -> None:
    if recovery is None:
        print("当前没有恢复方案，请先按 P")
        return

    info = recovery.solver_info
    print("\n矩形恢复方案：")
    print(
        f"  请求模式={RECOVERY_MODE_NAMES[recovery.requested_mode]}, "
        f"最终模式={RECOVERY_MODE_NAMES[recovery.selected_mode]}"
    )
    print("  算法来源=压缩包 puzzle_vision 原 solve_* 求解器")
    print("  运动限制=仅旋转和平移；无缩放、无单块镜像")
    print(
        f"  pattern_score={recovery.pattern_score:.5f}, "
        f"总求解时间={recovery.solve_time_sec:.3f}s, "
        f"验收={'通过' if recovery.accepted else '未通过'}"
    )
    for key in (
        "solution_quality",
        "solver_path",
        "target_size_mm",
        "target_origin_mm",
        "fill_ratio",
        "geometry_score",
        "assembly_gap_ratio",
        "assembly_overlap_ratio",
        "max_match_error_mm",
        "maximum_target_overlap_mm2",
    ):
        if key in info:
            print(f"  {key}={info[key]}")

    if recovery.attempts:
        print("  AUTO/模式尝试记录：")
        for attempt in recovery.attempts:
            print(f"    {attempt}")

    print("  各碎片动作（左下角原点，单位mm，逆时针为正）：")
    for item in recovery.plan:
        pick = np.asarray(item["pick_mm"], dtype=np.float64)
        place = np.asarray(item["place_mm"], dtype=np.float64)
        pick_output = np.array([pick[0], A4_HEIGHT_MM - pick[1]])
        place_output = np.array([place[0], A4_HEIGHT_MM - place[1]])
        delta = place_output - pick_output
        rotate_ccw = -float(item["rotate_deg"])
        print(
            f"    {item['piece_id']}: "
            f"吸取点=({pick_output[0]:.2f}, {pick_output[1]:.2f}), "
            f"放置点=({place_output[0]:.2f}, {place_output[1]:.2f}), "
            f"位移=({delta[0]:+.2f}, {delta[1]:+.2f}), "
            f"旋转={rotate_ccw:+.2f}°"
        )


def print_results(
    pieces: list[DetectedPiece],
    recovery: Optional[RecoveryResult],
) -> None:
    print_detection_results(pieces)
    print_recovery_result(recovery)


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            f"无法打开摄像头，CAMERA_INDEX={CAMERA_INDEX}"
        )

    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)

    corners = load_corners()
    latest_pieces: list[DetectedPiece] = []
    latest_result: Optional[np.ndarray] = None
    latest_calibrated_region: Optional[np.ndarray] = None
    latest_recovery: Optional[RecoveryResult] = None

    area_display_mode = 0
    show_target_overlay = True
    mode_index = 0

    print("按键说明：")
    print("C：重新标定4个角点")
    print("P：用当前 pick 识别结果调用压缩包矩形恢复算法")
    print("M：循环 AUTO / 已知固定 / 未知白色 / 扑克牌")
    print("0/1/2/3：直接选择 AUTO/已知/未知/扑克牌")
    print("T：显示/隐藏目标位置叠加")
    print("D：输出识别结果和机械臂放置方案")
    print("A：循环显示最小面积框 / 最大面积框 / 不显示")
    print("S：保存当前原始画面叠加结果")
    print("Q或ESC：退出")
    print("恢复求解器：直接使用同目录 puzzle_vision 包中的原始算法")
    print(
        f"输出坐标系：标定平面 {WARP_WIDTH}x{WARP_HEIGHT}，"
        "原点位于左下角，x向右，y向上，旋转逆时针为正"
    )
    print(
        f"通用识别面积范围：{MIN_PIECE_AREA_PX}～{MAX_PIECE_AREA_PX}px² "
        f"（约{MIN_PIECE_AREA_CM2:.1f}～{MAX_PIECE_AREA_CM2:.1f}cm²）"
    )

    while True:
        success, frame = camera.read()
        if not success:
            print("摄像头画面读取失败")
            break

        selected_mode = RECOVERY_MODES[mode_index]
        if corners is None:
            display = frame.copy()
            cv2.putText(
                display,
                "Press C to calibrate 4 corners",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            latest_pieces = []
            latest_calibrated_region = None
            latest_result = display
        else:
            camera_to_warp, warp_to_camera = build_perspective_matrices(corners)
            calibrated_region = perspective_correct(frame, camera_to_warp)
            pieces, _mask = detect_pieces(calibrated_region)

            display = draw_results_on_original(
                frame,
                corners,
                pieces,
                warp_to_camera,
                area_display_mode,
                recovery=latest_recovery,
                show_target_overlay=show_target_overlay,
                selected_mode=selected_mode,
            )
            latest_pieces = pieces
            latest_calibrated_region = calibrated_region.copy()
            latest_result = display

        cv2.imshow(MAIN_WINDOW_NAME, latest_result)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("c"), ord("C")):
            try:
                corners = select_calibration_corners(frame)
                save_corners(corners)
                latest_recovery = None
                print("标定完成，已保存4个角点；旧恢复方案已清空")
            except RuntimeError as error:
                print(error)

        elif key in (ord("p"), ord("P")):
            selected_mode = RECOVERY_MODES[mode_index]
            if latest_calibrated_region is None:
                print("无法恢复：尚未完成标定或没有校正图像")
                latest_recovery = None
            elif not 1 <= len(latest_pieces) <= MAX_DETECTED_PIECES:
                print(
                    f"无法恢复：当前识别到{len(latest_pieces)}块，"
                    f"需要1～{MAX_DETECTED_PIECES}块"
                )
                latest_recovery = None
            else:
                print(
                    f"\n开始恢复：模式={RECOVERY_MODE_NAMES[selected_mode]}，"
                    f"碎片数={len(latest_pieces)}"
                )
                try:
                    latest_recovery = solve_with_pick_recognition(
                        latest_pieces,
                        latest_calibrated_region,
                        selected_mode,
                    )
                    print("矩形恢复成功，压缩包求解器验收通过")
                    print_recovery_result(latest_recovery)
                except RuntimeError as error:
                    latest_recovery = None
                    print(f"矩形恢复失败：{error}")

        elif key in (ord("m"), ord("M")):
            mode_index = (mode_index + 1) % len(RECOVERY_MODES)
            latest_recovery = None
            print(
                "当前恢复模式："
                + RECOVERY_MODE_NAMES[RECOVERY_MODES[mode_index]]
            )

        elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
            mode_index = int(chr(key))
            latest_recovery = None
            print(
                "当前恢复模式："
                + RECOVERY_MODE_NAMES[RECOVERY_MODES[mode_index]]
            )

        elif key in (ord("t"), ord("T")):
            show_target_overlay = not show_target_overlay
            print("目标位置叠加：" + ("显示" if show_target_overlay else "隐藏"))

        elif key in (ord("d"), ord("D")):
            print_results(latest_pieces, latest_recovery)

        elif key in (ord("a"), ord("A")):
            area_display_mode = (area_display_mode + 1) % 3
            if area_display_mode == 1:
                print(
                    "当前显示：最小可识别面积框，"
                    f"面积={MIN_PIECE_AREA_PX}px²"
                )
            elif area_display_mode == 2:
                print(
                    "当前显示：最大可识别面积框，"
                    f"面积={MAX_PIECE_AREA_PX}px²"
                )
            else:
                print("当前显示：关闭面积参考框")

        elif key in (ord("s"), ord("S")):
            if latest_result is not None:
                output_path = "piece_detection_and_recovery.png"
                cv2.imwrite(output_path, latest_result)
                print(f"当前叠加图已保存：{output_path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
