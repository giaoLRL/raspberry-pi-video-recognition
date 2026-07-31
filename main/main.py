#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E题拼图装置：pick_base识别 + 压缩包矩形恢复算法完整版

保留 pick_base 的功能：
1. 手动标定A4四角并保存 a4_corners.json；
2. 在透视校正区域内识别3～5边凸多边形碎片；
3. ID按面积从小到大排列；
4. 识别中心、初始位置和目标位置统一使用碎片面积中心；
5. 输出坐标使用标定方框右下角为原点，x向上、y向左。

迁移的矩形恢复功能：
1. 固定四片：使用题图固定四片刚体模板匹配；
2. 自主纯色：使用边长、切边、矩形边界、面积填充和无镜像刚体搜索；
3. 自主花纹/扑克牌：先尝试竖向扑克牌恢复，再使用轮廓+纹理连续性恢复；
4. 自动模式会先严格检查固定四片，未通过时再自动判断纯色或花纹；
5. 所有恢复只允许旋转和平移，禁止镜像。

按键：
    C：重新标定4个角点
    P：自动判断固定四片 / 纯色 / 花纹并恢复矩形
    F：强制使用固定四片恢复
    W：强制使用自主纯色恢复
    G：强制使用自主花纹/扑克牌恢复
    T：显示/隐藏目标矩形和各碎片目标位置
    D：输出识别结果以及最新恢复方案
    A：循环显示最小面积框 / 最大面积框 / 不显示
    S：保存当前原始画面叠加结果
    Q / ESC：退出

运行目录必须同时包含 puzzle_vision 文件夹和 puzzle_solver_config.json。
依赖：opencv-python、numpy。
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from puzzle_vision.card_recognition import recognize_card_marks
from puzzle_vision.config import load_config
from puzzle_vision.detector import PieceObservation
from puzzle_vision.geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
)
from puzzle_vision.solver import (
    SolveError,
    solve_card,
    solve_fixed,
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
# 按题目图2尺寸比例自动计算面积过滤范围
# ============================================================
# 标定区域必须对应完整纵向A4纸。即使相机里A4纸大小变化，
# 透视校正后都会映射到 WARP_WIDTH × WARP_HEIGHT，
# 因此用“碎片实际面积 / A4纸实际面积”的比例换算像素面积。
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_AREA_CM2 = A4_WIDTH_CM * A4_HEIGHT_CM

# 题目图2四块自备碎片的理论面积分别为：
# 4.8、10.8、20.4、24.0 cm²。
SELF_PIECE_MIN_AREA_CM2 = 4.8
SELF_PIECE_MAX_AREA_CM2 = 24.0

CALIBRATED_REGION_AREA_PX = WARP_WIDTH * WARP_HEIGHT

SELF_PIECE_MIN_AREA_RATIO = (
    SELF_PIECE_MIN_AREA_CM2 / A4_AREA_CM2
)
SELF_PIECE_MAX_AREA_RATIO = (
    SELF_PIECE_MAX_AREA_CM2 / A4_AREA_CM2
)

# 没有分割误差时的理论像素面积：840×1188时为7680～38400 px²。
THEORETICAL_MIN_PIECE_AREA_PX = int(round(
    CALIBRATED_REGION_AREA_PX * SELF_PIECE_MIN_AREA_RATIO
))
THEORETICAL_MAX_PIECE_AREA_PX = int(round(
    CALIBRATED_REGION_AREA_PX * SELF_PIECE_MAX_AREA_RATIO
))

# 给阈值、阴影、反光和轮廓拟合留容差。
# 最小面积向下放宽25%，最大面积向上放宽30%。
MIN_AREA_TOLERANCE_FACTOR = 0.75
MAX_AREA_TOLERANCE_FACTOR = 1.30

# 固定四片理论阈值仍保留用于说明；实际检测范围扩大到压缩包高级题范围，
# 这样较小切片和面积较大的扑克牌碎片也不会在进入求解器前被误删。
FIXED_MIN_PIECE_AREA_PX = int(round(
    THEORETICAL_MIN_PIECE_AREA_PX * MIN_AREA_TOLERANCE_FACTOR
))
FIXED_MAX_PIECE_AREA_PX = int(round(
    THEORETICAL_MAX_PIECE_AREA_PX * MAX_AREA_TOLERANCE_FACTOR
))
ADVANCED_MIN_AREA_MM2 = 100.0
ADVANCED_MAX_AREA_MM2 = 6500.0
MIN_PIECE_AREA_PX = int(round(ADVANCED_MIN_AREA_MM2 * 16.0))
MAX_PIECE_AREA_PX = int(round(ADVANCED_MAX_AREA_MM2 * 16.0))

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


# ============================================================
# 压缩包恢复算法配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOLVER_CONFIG_FILE = SCRIPT_DIR / "puzzle_solver_config.json"
SOLVER_CONFIG = load_config(
    SOLVER_CONFIG_FILE if SOLVER_CONFIG_FILE.exists() else None
)

# A4透视图固定为840×1188，对应210×297 mm，比例恰好为4 px/mm。
PIXELS_PER_MM_X = WARP_WIDTH / 210.0
PIXELS_PER_MM_Y = WARP_HEIGHT / 297.0
PIXELS_PER_MM = (PIXELS_PER_MM_X + PIXELS_PER_MM_Y) / 2.0

# 自动模式只有固定模板最大轮廓匹配误差不超过该值，才判定为固定四片。
# 压缩包服务端的严格验收同样使用8 mm。
FIXED_AUTO_ACCEPT_ERROR_MM = 8.0

# 自动判断花纹时使用碎片内部灰度标准差和边缘比例。
PATTERN_STD_THRESHOLD = 15.0
PATTERN_EDGE_RATIO_THRESHOLD = 0.010

# 自主拼接允许1～4片；1片时直接摆正为矩形，2～4片进行边匹配搜索。
MIN_RECONSTRUCTION_PIECES = 1
MAX_RECONSTRUCTION_PIECES = 4

# ============================================================
# 实时性能参数
# ============================================================

# 碎片识别无需跟随摄像头30 FPS满速运行。限制到约12.5 Hz，
# 能显著降低RDK/PC负载，同时不会影响手动按键时使用最新稳定结果。
DETECTION_INTERVAL_SEC = 0.08

# 自动求解采用快速搜索预算。固定四片通常几十毫秒；
# 纯色/花纹自主拼接通常控制在约0.5～2秒内。
FAST_SOLVER_ENABLED = True

# 自动模式只运行最可能的一条自主拼接链。
# 若判断错误，用户可按W或G强制指定，避免自动连续穷举两套算法。
AUTO_TRY_OPPOSITE_FALLBACK = False

# 视觉花纹判断只做轻量灰度/边缘统计，不额外调用扑克牌识别器。
# 扑克牌识别仅在真正进入G/花纹求解时由solve_card执行。
ENABLE_CARD_RECOGNITION_IN_PATTERN_ESTIMATE = False


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DetectedPiece:
    piece_id: int
    # 对外输出坐标：标定区域右下角为原点，x向上，y向左。
    # 因此x范围约为0～1187，y范围约为0～839。
    center_x: int
    center_y: int
    # OpenCV内部坐标：标定区域左上角为原点，u向右、v向下。
    # 仅用于图像绘制、反投影和求解器输入，绝不能当作对外坐标输出。
    center_x_image: int
    center_y_image: int
    area_px: float
    # 对外输出顶点：标定区域右下角为原点，x向上，y向左。
    # 顶点按逆时针排列，并从x最小、再y最小的顶点开始。
    vertices: list[tuple[int, int]]
    contour: np.ndarray
    polygon: np.ndarray


@dataclass
class ReconstructionResult:
    """一次矩形恢复结果，内部坐标为A4左上角原点、单位mm。"""

    mode: str
    source_region: str
    plan: list[dict[str, Any]]
    solver_info: dict[str, Any]
    pattern_info: dict[str, Any]
    elapsed_ms: float

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
    # 只做轻量闭运算。原版7×7迭代2次会把相距很近的扑克牌碎片
    # 粘成一个轮廓，导致求解器实际只收到2～3片。
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
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
    将OpenCV左上角图像坐标转换为右下角机械坐标。

    输出规则：
        1. 原点位于标定区域右下角
        2. x轴向上为正，范围0～WARP_HEIGHT-1
        3. y轴向左为正，范围0～WARP_WIDTH-1
        4. 顶点按逆时针排列
        5. 从x最小、x相同时y最小的顶点开始

    OpenCV图像点(u, v)转换为：
        x = WARP_HEIGHT - 1 - v
        y = WARP_WIDTH  - 1 - u

    保留多边形原有相邻关系，不对顶点按坐标直接排序。
    """
    image_points = np.asarray(
        polygon,
        dtype=np.int32,
    ).reshape(-1, 2)

    output_points = [
        (
            int((WARP_HEIGHT - 1) - point[1]),
            int((WARP_WIDTH - 1) - point[0]),
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
        # OpenCV内部：(u, v)，左上角原点，u向右、v向下；
        # 对外输出：(x, y)，右下角原点，x向上、y向左。
        center_x_output = (WARP_HEIGHT - 1) - center_y_image
        center_y_output = (WARP_WIDTH - 1) - center_x_image

        pieces.append(
            DetectedPiece(
                piece_id=index,
                center_x=center_x_output,
                center_y=center_y_output,
                center_x_image=center_x_image,
                center_y_image=center_y_image,
                area_px=candidate["area"],
                vertices=candidate["vertices"],
                contour=candidate["contour"],
                polygon=candidate["polygon"],
            )
        )

    return pieces, mask


# ============================================================
# pick_base识别结果 -> 压缩包求解器输入
# ============================================================

def _piece_solver_id(piece_id: int) -> str:
    return f"piece_{piece_id}"


def _piece_id_from_solver_id(value: str) -> int:
    try:
        return int(str(value).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def pieces_to_observations(
    pieces: list[DetectedPiece],
) -> list[PieceObservation]:
    """把pick_base识别轮廓转换成压缩包求解器的毫米坐标观测。"""

    observations: list[PieceObservation] = []

    for piece in pieces:
        polygon_px = np.asarray(piece.polygon, dtype=np.float64).reshape(-1, 2)
        polygon_mm = polygon_px / PIXELS_PER_MM
        polygon_mm = normalize_winding(polygon_mm, positive=True)

        contour_px = np.asarray(piece.contour, dtype=np.int32)
        # 面积中心必须与识别输出、初始位置和目标位置完全一致。
        # 这里不用safe_interior_point；直接使用轮廓图像矩得到的面积中心。
        centroid_mm = np.array(
            [piece.center_x_image, piece.center_y_image],
            dtype=np.float64,
        ) / PIXELS_PER_MM
        pickup_mm = centroid_mm.copy()
        lengths_mm = edge_lengths(polygon_mm)

        observations.append(
            PieceObservation(
                id=_piece_solver_id(piece.piece_id),
                polygon_mm=polygon_mm,
                contour_px=contour_px,
                centroid_mm=centroid_mm,
                pickup_mm=pickup_mm,
                area_mm2=polygon_area(polygon_mm),
                perimeter_mm=float(np.sum(lengths_mm)),
                edge_lengths_mm=lengths_mm,
            )
        )

    return observations


def infer_source_region(pieces: list[DetectedPiece]) -> str:
    """根据碎片中心主要位于A4上半还是下半，选择对侧作为目标区。"""

    if not pieces:
        return "upper"
    mean_image_y = float(np.mean([piece.center_y_image for piece in pieces]))
    return "upper" if mean_image_y < WARP_HEIGHT / 2.0 else "lower"


def fixed_config_for_source(source_region: str) -> dict[str, Any]:
    cfg = deepcopy(SOLVER_CONFIG["fixed"])
    if source_region == "lower":
        target_height = float(cfg.get("target_size_mm", [100.0, 60.0])[1])
        cfg["target_origin_mm"][1] = (
            297.0 - float(cfg["target_origin_mm"][1]) - target_height
        )
    return cfg


def unknown_config_for_source(
    source_region: str,
    pattern_mode: bool,
) -> dict[str, Any]:
    """生成自主拼接配置，并在实时模式下收紧搜索预算。"""

    cfg = deepcopy(SOLVER_CONFIG["unknown"])
    cfg["use_taught_layout"] = False
    cfg["target_orientation"] = "portrait" if pattern_mode else "landscape"

    if source_region == "upper":
        cfg["target_zone_mm"] = [0.0, 148.5, 210.0, 297.0]
    else:
        cfg["target_zone_mm"] = [0.0, 0.0, 210.0, 148.5]

    if FAST_SOLVER_ENABLED:
        # 通用自主拼接：减少搜索节点、候选数和各阶段最长时间。
        # 保留几何验收阈值，因此“更快”不会通过降低验收标准换取。
        cfg["max_search_nodes"] = min(
            int(cfg.get("max_search_nodes", 18000)), 7000
        )
        cfg["max_pair_options_exact"] = min(
            int(cfg.get("max_pair_options_exact", 64)), 40
        )
        cfg["max_pair_options_partial"] = min(
            int(cfg.get("max_pair_options_partial", 64)), 40
        )
        cfg["pose_hint_branch_limit"] = min(
            int(cfg.get("pose_hint_branch_limit", 36)), 24
        )
        cfg["max_final_candidates"] = min(
            int(cfg.get("max_final_candidates", 80)), 36
        )
        cfg["max_search_seconds"] = min(
            float(cfg.get("max_search_seconds", 3.2)), 1.35
        )
        cfg["fallback_search_seconds"] = min(
            float(cfg.get("fallback_search_seconds", 2.5)), 0.75
        )
        cfg["guided_search_seconds"] = min(
            float(cfg.get("guided_search_seconds", 2.5)), 1.10
        )
        cfg["guided_fallback_search_seconds"] = min(
            float(cfg.get("guided_fallback_search_seconds", 2.5)), 0.70
        )
        cfg["exact_search_seconds"] = min(
            float(cfg.get("exact_search_seconds", 0.65)), 0.40
        )
        cfg["candidate_overlap_pixels_per_mm"] = min(
            float(cfg.get("candidate_overlap_pixels_per_mm", 2.0)), 1.25
        )

        # 扑克牌专用恢复预算。
        cfg["card_corner_option_limit"] = min(
            int(cfg.get("card_corner_option_limit", 7)), 6
        )
        cfg["card_max_pair_options_exact"] = min(
            int(cfg.get("card_max_pair_options_exact", 36)), 28
        )
        cfg["card_max_pair_options_partial"] = min(
            int(cfg.get("card_max_pair_options_partial", 36)), 28
        )
        cfg["card_search_seconds"] = min(
            float(cfg.get("card_search_seconds", 2.2)), 1.10
        )
        cfg["card_exact_search_seconds"] = min(
            float(cfg.get("card_exact_search_seconds", 0.55)), 0.32
        )
        cfg["card_fallback_search_seconds"] = min(
            float(cfg.get("card_fallback_search_seconds", 0.7)), 0.40
        )
        cfg["card_anchored_search_seconds"] = min(
            float(cfg.get("card_anchored_search_seconds", 0.75)), 0.45
        )
        cfg["card_anchored_exact_seconds"] = min(
            float(cfg.get("card_anchored_exact_seconds", 0.25)), 0.16
        )
        cfg["card_anchored_total_seconds"] = min(
            float(cfg.get("card_anchored_total_seconds", 1.2)), 0.70
        )

    return cfg


def estimate_pattern_information(
    calibrated_region: np.ndarray,
    pieces: list[DetectedPiece],
    observations: Optional[list[PieceObservation]] = None,
    include_card_recognition: bool = False,
) -> dict[str, Any]:
    """轻量判断碎片内部是否有花纹。

    性能优化：
    1. 整张图的Canny只计算一次；
    2. 每块碎片只处理自身外接框，不创建整幅A4大小的临时mask；
    3. AUTO模式不运行扑克牌字符/花色识别，避免重复高成本处理。
    """

    gray = cv2.cvtColor(calibrated_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    erosion = max(2, int(round(1.0 * PIXELS_PER_MM)))
    kernel = np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8)
    per_piece: list[dict[str, Any]] = []

    for piece in pieces:
        contour = np.asarray(piece.contour, dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        local_contour = contour.copy()
        local_contour[:, 0, 0] -= bx
        local_contour[:, 0, 1] -= by

        local_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(
            local_mask, [local_contour], -1, 255, cv2.FILLED
        )
        interior = cv2.erode(local_mask, kernel, iterations=1)
        valid = interior > 0

        gray_roi = gray[by:by + bh, bx:bx + bw]
        edge_roi = edges[by:by + bh, bx:bx + bw]
        values = gray_roi[valid]

        if values.size < 50:
            std_gray = 0.0
            edge_ratio = 0.0
        else:
            std_gray = float(np.std(values))
            edge_ratio = float(np.count_nonzero(edge_roi[valid])) / float(
                max(1, np.count_nonzero(valid))
            )

        visible = bool(
            std_gray >= PATTERN_STD_THRESHOLD
            or edge_ratio >= PATTERN_EDGE_RATIO_THRESHOLD
        )
        per_piece.append(
            {
                "piece_id": piece.piece_id,
                "std_gray": round(std_gray, 3),
                "edge_ratio": round(edge_ratio, 5),
                "pattern_visible": visible,
            }
        )

    card_info: dict[str, Any] = {"skipped_for_speed": True}
    if (
        include_card_recognition
        and ENABLE_CARD_RECOGNITION_IN_PATTERN_ESTIMATE
        and observations is not None
    ):
        try:
            card_info = recognize_card_marks(
                calibrated_region,
                observations,
                PIXELS_PER_MM,
                unknown_config_for_source(
                    infer_source_region(pieces), pattern_mode=True
                ),
            )
        except Exception as error:
            card_info = {"recognition_error": str(error)}

    visible_count = sum(1 for item in per_piece if item["pattern_visible"])
    card_visible = int(card_info.get("pattern_visible_pieces", 0) or 0)
    rank_detected = bool(card_info.get("rank_detected", False))
    is_pattern = bool(visible_count > 0 or card_visible > 0 or rank_detected)

    return {
        "is_pattern": is_pattern,
        "visible_piece_count": visible_count,
        "piece_count": len(pieces),
        "per_piece": per_piece,
        "card_recognition": card_info,
        "classification_method": "single_canny_local_roi",
    }


def _make_result(
    mode: str,
    source_region: str,
    plan: list[dict[str, Any]],
    solver_info: dict[str, Any],
    pattern_info: dict[str, Any],
    started: float,
) -> ReconstructionResult:
    return ReconstructionResult(
        mode=mode,
        source_region=source_region,
        plan=plan,
        solver_info=solver_info,
        pattern_info=pattern_info,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def solve_fixed_reconstruction(
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    strict_auto: bool,
    observations: Optional[list[PieceObservation]] = None,
) -> ReconstructionResult:
    started = time.perf_counter()
    del calibrated_region

    if len(pieces) != 4:
        raise SolveError(f"固定四片模式要求4片，当前识别到{len(pieces)}片")

    if observations is None:
        observations = pieces_to_observations(pieces)

    source_region = infer_source_region(pieces)
    fixed_cfg = fixed_config_for_source(source_region)
    plan, solver_info = solve_fixed(observations, fixed_cfg)

    maximum_error = float(solver_info.get("max_match_error_mm", math.inf))
    assignment_cost = float(solver_info.get("assignment_cost", math.inf))
    observed_total_area = float(sum(item.area_mm2 for item in observations))
    fixed_target_area = float(np.prod(fixed_cfg["target_size_mm"]))
    total_area_ratio = observed_total_area / max(fixed_target_area, 1e-9)
    solver_info["observed_total_area_ratio"] = round(total_area_ratio, 5)
    solver_info["performance_profile"] = (
        "fast_realtime" if FAST_SOLVER_ENABLED else "full"
    )

    if strict_auto and (
        maximum_error > FIXED_AUTO_ACCEPT_ERROR_MM
        or not 0.88 <= total_area_ratio <= 1.12
        or assignment_cost > 20.0
    ):
        raise SolveError(
            "四片数量满足，但未通过固定模板严格检查："
            f"最大匹配误差={maximum_error:.2f} mm，"
            f"总面积比例={total_area_ratio:.3f}，"
            f"分配代价={assignment_cost:.2f}"
        )

    pattern_info = {
        "is_pattern": False,
        "piece_count": len(pieces),
        "classification_skipped": "fixed_template_does_not_need_pattern_scan",
    }
    return _make_result(
        "fixed", source_region, plan, solver_info, pattern_info, started
    )


def solve_white_reconstruction(
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    observations: Optional[list[PieceObservation]] = None,
    pattern_info: Optional[dict[str, Any]] = None,
) -> ReconstructionResult:
    started = time.perf_counter()

    if observations is None:
        observations = pieces_to_observations(pieces)
    source_region = infer_source_region(pieces)
    if pattern_info is None:
        pattern_info = estimate_pattern_information(
            calibrated_region, pieces, observations
        )

    cfg = unknown_config_for_source(source_region, pattern_mode=False)
    plan, solver_info = solve_unknown(
        observations,
        cfg,
        calibrated_region,
        PIXELS_PER_MM,
        use_texture=False,
    )
    solver_info["performance_profile"] = (
        "fast_realtime" if FAST_SOLVER_ENABLED else "full"
    )
    if not solver_info.get("solution_accepted", False):
        raise SolveError(
            str(solver_info.get("solve_error", "纯色自主拼接结果未通过验收"))
        )
    return _make_result(
        "unknown-white", source_region, plan, solver_info, pattern_info, started
    )


def solve_pattern_reconstruction(
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    observations: Optional[list[PieceObservation]] = None,
    pattern_info: Optional[dict[str, Any]] = None,
) -> ReconstructionResult:
    """先走扑克牌专用恢复，失败后走通用轮廓+纹理自主恢复。"""

    started = time.perf_counter()
    if observations is None:
        observations = pieces_to_observations(pieces)
    source_region = infer_source_region(pieces)
    if pattern_info is None:
        pattern_info = estimate_pattern_information(
            calibrated_region, pieces, observations
        )

    cfg = unknown_config_for_source(source_region, pattern_mode=True)
    card_error: Optional[str] = None
    card_attempt: Optional[dict[str, Any]] = None

    if 2 <= len(observations) <= 4:
        try:
            card_plan, card_solver = solve_card(
                observations,
                cfg,
                calibrated_region,
                PIXELS_PER_MM,
            )
            card_attempt = card_solver
            card_solver["performance_profile"] = (
                "fast_realtime" if FAST_SOLVER_ENABLED else "full"
            )
            if card_solver.get("solution_accepted", False):
                return _make_result(
                    "unknown-pattern-card",
                    source_region,
                    card_plan,
                    card_solver,
                    pattern_info,
                    started,
                )
            card_error = str(
                card_solver.get("solve_error", "扑克牌矩形验收未通过")
            )
        except SolveError as error:
            card_error = str(error)

    # 非标准扑克牌花纹或扑克牌专用恢复失败时，再执行一次通用纹理恢复。
    # 这里不会再重复进行AUTO花纹判断。
    generic_cfg = unknown_config_for_source(source_region, pattern_mode=False)
    plan, solver_info = solve_unknown(
        observations,
        generic_cfg,
        calibrated_region,
        PIXELS_PER_MM,
        use_texture=True,
    )
    solver_info["performance_profile"] = (
        "fast_realtime" if FAST_SOLVER_ENABLED else "full"
    )
    if card_error is not None:
        solver_info["card_solver_fallback_reason"] = card_error
    if card_attempt is not None:
        # 完整候选信息可能较大，只保留关键诊断字段。
        solver_info["card_solver_attempt_summary"] = {
            key: card_attempt.get(key)
            for key in (
                "solution_accepted",
                "solve_error",
                "candidate_count",
                "geometry_score",
                "fill_ratio",
            )
            if key in card_attempt
        }
    if not solver_info.get("solution_accepted", False):
        raise SolveError(
            str(solver_info.get("solve_error", "花纹自主拼接结果未通过验收"))
        )
    return _make_result(
        "unknown-pattern", source_region, plan, solver_info, pattern_info, started
    )


def reconstruct_rectangle(
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    requested_mode: str = "auto",
) -> ReconstructionResult:
    """统一恢复入口：auto / fixed / white / pattern。

    AUTO只进行一次轻量分类，并复用观测与花纹统计，避免重复计算。
    """

    if not MIN_RECONSTRUCTION_PIECES <= len(pieces) <= MAX_RECONSTRUCTION_PIECES:
        raise SolveError(
            "矩形恢复只支持1～4片，"
            f"当前识别到{len(pieces)}片；请先检查识别轮廓"
        )

    observations = pieces_to_observations(pieces)

    if requested_mode == "fixed":
        return solve_fixed_reconstruction(
            pieces,
            calibrated_region,
            strict_auto=False,
            observations=observations,
        )
    if requested_mode == "white":
        pattern_info = estimate_pattern_information(
            calibrated_region, pieces, observations
        )
        return solve_white_reconstruction(
            pieces,
            calibrated_region,
            observations=observations,
            pattern_info=pattern_info,
        )
    if requested_mode == "pattern":
        pattern_info = estimate_pattern_information(
            calibrated_region, pieces, observations
        )
        return solve_pattern_reconstruction(
            pieces,
            calibrated_region,
            observations=observations,
            pattern_info=pattern_info,
        )
    if requested_mode != "auto":
        raise ValueError(f"不支持的恢复模式：{requested_mode}")

    fixed_error: Optional[str] = None
    if len(pieces) == 4:
        try:
            return solve_fixed_reconstruction(
                pieces,
                calibrated_region,
                strict_auto=True,
                observations=observations,
            )
        except SolveError as error:
            fixed_error = str(error)

    # 固定模板未通过后，仅进行一次轻量花纹分类。
    pattern_info = estimate_pattern_information(
        calibrated_region,
        pieces,
        observations,
        include_card_recognition=False,
    )
    preferred = "pattern" if pattern_info["is_pattern"] else "white"
    errors: list[str] = []
    if fixed_error:
        errors.append(f"固定四片：{fixed_error}")

    try:
        if preferred == "pattern":
            result = solve_pattern_reconstruction(
                pieces,
                calibrated_region,
                observations=observations,
                pattern_info=pattern_info,
            )
        else:
            result = solve_white_reconstruction(
                pieces,
                calibrated_region,
                observations=observations,
                pattern_info=pattern_info,
            )
        result.solver_info["auto_preferred_mode"] = preferred
        if errors:
            result.solver_info["auto_previous_failures"] = errors
        return result
    except SolveError as error:
        errors.append(f"{preferred}：{error}")

    if AUTO_TRY_OPPOSITE_FALLBACK:
        fallback = "white" if preferred == "pattern" else "pattern"
        try:
            if fallback == "pattern":
                result = solve_pattern_reconstruction(
                    pieces,
                    calibrated_region,
                    observations=observations,
                    pattern_info=pattern_info,
                )
            else:
                result = solve_white_reconstruction(
                    pieces,
                    calibrated_region,
                    observations=observations,
                    pattern_info=pattern_info,
                )
            result.solver_info["auto_preferred_mode"] = preferred
            result.solver_info["auto_fallback_mode"] = fallback
            result.solver_info["auto_previous_failures"] = errors
            return result
        except SolveError as error:
            errors.append(f"{fallback}：{error}")

    mode_key = "G" if preferred == "white" else "W"
    raise SolveError(
        "；".join(errors)
        + f"。为避免长时间连续穷举，AUTO未继续跑相反模式；"
        f"可按{mode_key}强制尝试另一模式"
    )


def solver_mm_to_output_px(point_mm: list[float] | np.ndarray) -> tuple[int, int]:
    """求解器左上角毫米坐标 -> 右下角机械像素坐标(x向上、y向左)。"""
    point = np.asarray(point_mm, dtype=np.float64)
    u_image_px = float(point[0]) * PIXELS_PER_MM
    v_image_px = float(point[1]) * PIXELS_PER_MM
    x_output_px = int(round((WARP_HEIGHT - 1) - v_image_px))
    y_output_px = int(round((WARP_WIDTH - 1) - u_image_px))
    return x_output_px, y_output_px


def solver_mm_to_warp_px(points_mm: np.ndarray) -> np.ndarray:
    return np.asarray(points_mm, dtype=np.float64) * PIXELS_PER_MM


def draw_reconstruction_on_original(
    result_image: np.ndarray,
    reconstruction: Optional[ReconstructionResult],
    warp_to_camera: np.ndarray,
    show_overlay: bool,
) -> np.ndarray:
    if reconstruction is None or not show_overlay:
        return result_image

    output = result_image.copy()
    all_target_points: list[np.ndarray] = []

    for item in reconstruction.plan:
        polygon_mm = np.asarray(item["target_polygon_mm"], dtype=np.float64)
        polygon_warp = solver_mm_to_warp_px(polygon_mm)
        polygon_camera = np.round(
            warp_points_to_camera(polygon_warp, warp_to_camera)
        ).astype(np.int32)
        all_target_points.append(polygon_camera)

        cv2.polylines(
            output,
            [polygon_camera],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        place_warp = solver_mm_to_warp_px(
            np.asarray(item["place_mm"], dtype=np.float64)[None, :]
        )
        place_camera = warp_points_to_camera(
            place_warp, warp_to_camera
        )[0]
        place_point = (
            int(round(float(place_camera[0]))),
            int(round(float(place_camera[1]))),
        )
        cv2.drawMarker(
            output,
            place_point,
            (255, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )

        piece_id = _piece_id_from_solver_id(item.get("piece_id", ""))
        # 求解器在左上角图像坐标中以顺时针为正；按用户约定输出：逆时针为负、顺时针为正。
        rotate_output = float(item.get("rotate_deg", 0.0))
        cv2.putText(
            output,
            f"T{piece_id} R:{rotate_output:+.1f}",
            (place_point[0] + 8, place_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if all_target_points:
        merged = np.vstack(all_target_points)
        rectangle = cv2.minAreaRect(merged.astype(np.float32))
        box = np.round(cv2.boxPoints(rectangle)).astype(np.int32)
        cv2.polylines(
            output,
            [box],
            True,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"Restore: {reconstruction.mode}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def print_reconstruction_result(
    reconstruction: Optional[ReconstructionResult],
) -> None:
    if reconstruction is None:
        print("当前没有矩形恢复方案，请先按P/F/W/G")
        return

    print("\n矩形恢复方案：")
    print(f"  mode={reconstruction.mode}")
    print(f"  source_region={reconstruction.source_region}")
    print(f"  solve_time={reconstruction.elapsed_ms:.1f} ms")
    print(
        "  quality="
        f"{reconstruction.solver_info.get('solution_quality', 'unknown')}"
    )
    if "max_match_error_mm" in reconstruction.solver_info:
        print(
            "  max_match_error_mm="
            f"{float(reconstruction.solver_info['max_match_error_mm']):.3f}"
        )
    if "fill_ratio" in reconstruction.solver_info:
        print(
            "  fill_ratio="
            f"{float(reconstruction.solver_info['fill_ratio']):.4f}"
        )

    print("  位置基准：初始位置和目标位置均为碎片面积中心")
    print("  输出坐标：右下角原点，x向上、y向左，旋转逆时针为负、顺时针为正")
    for item in sorted(
        reconstruction.plan,
        key=lambda value: _piece_id_from_solver_id(value.get("piece_id", "")),
    ):
        piece_id = _piece_id_from_solver_id(item.get("piece_id", ""))
        initial_px = solver_mm_to_output_px(item["pick_mm"])
        target_px = solver_mm_to_output_px(item["place_mm"])
        rotate_output = float(item.get("rotate_deg", 0.0))
        print(
            f"  ID={piece_id}: "
            f"初始面积中心=({initial_px[0]}, {initial_px[1]}), "
            f"目标面积中心=({target_px[0]}, {target_px[1]}), "
            f"旋转={rotate_output:+.3f}°"
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
                [[[piece.center_x_image, piece.center_y_image]]],
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

    cv2.putText(
        result,
        f"Pieces: {len(pieces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
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


def print_results(pieces: list[DetectedPiece]) -> None:
    """只输出ID和面积中心；多边形顶点仅供内部求解，不向外输出。"""
    print("\n识别结果（右下角为原点，x向上，y向左）：")

    if not pieces:
        print("没有识别到有效碎片")
        return

    for piece in pieces:
        print(
            f"ID={piece.piece_id}, "
            f"面积中心=({piece.center_x}, {piece.center_y})"
        )


# ============================================================
# 异步求解与快照
# ============================================================

def clone_pieces_for_solver(
    pieces: list[DetectedPiece],
) -> list[DetectedPiece]:
    """冻结当前识别结果，避免求解线程读取到下一帧变化的数据。"""

    return [
        DetectedPiece(
            piece_id=piece.piece_id,
            center_x=piece.center_x,
            center_y=piece.center_y,
            center_x_image=piece.center_x_image,
            center_y_image=piece.center_y_image,
            area_px=piece.area_px,
            vertices=list(piece.vertices),
            contour=np.asarray(piece.contour).copy(),
            polygon=np.asarray(piece.polygon).copy(),
        )
        for piece in pieces
    ]


def solver_worker(
    job_id: int,
    pieces: list[DetectedPiece],
    calibrated_region: np.ndarray,
    requested_mode: str,
    result_queue: "queue.Queue[tuple[int, bool, Any, float]]",
) -> None:
    """后台线程入口；只通过队列把结果传回OpenCV主线程。"""

    started = time.perf_counter()
    try:
        result = reconstruct_rectangle(
            pieces,
            calibrated_region,
            requested_mode=requested_mode,
        )
        result_queue.put(
            (job_id, True, result, time.perf_counter() - started)
        )
    except Exception as error:
        result_queue.put(
            (job_id, False, error, time.perf_counter() - started)
        )


def draw_solver_status(
    image: np.ndarray,
    requested_mode: Optional[str],
    started_at: float,
) -> np.ndarray:
    """求解期间在画面上显示状态，但不阻塞摄像头刷新。"""

    if requested_mode is None:
        return image

    output = image.copy()
    elapsed = max(0.0, time.perf_counter() - started_at)
    label = f"SOLVING {requested_mode.upper()}  {elapsed:.1f}s"

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        2,
    )
    x0, y0 = 14, 52
    cv2.rectangle(
        output,
        (x0 - 7, y0 - text_height - 9),
        (x0 + text_width + 7, y0 + baseline + 7),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        output,
        label,
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return output


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"无法打开摄像头，CAMERA_INDEX={CAMERA_INDEX}"
        )

    # 尽量减少摄像头内部缓存，避免处理负载高时看到旧画面。
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)

    corners = load_corners()
    camera_to_warp: Optional[np.ndarray] = None
    warp_to_camera: Optional[np.ndarray] = None
    if corners is not None:
        camera_to_warp, warp_to_camera = build_perspective_matrices(corners)

    latest_pieces: list[DetectedPiece] = []
    latest_result: Optional[np.ndarray] = None
    latest_calibrated_region: Optional[np.ndarray] = None
    latest_reconstruction: Optional[ReconstructionResult] = None
    show_target_overlay = True

    # 识别节流状态。
    last_detection_time = -1e9

    # 后台求解状态。
    solver_result_queue: "queue.Queue[tuple[int, bool, Any, float]]" = (
        queue.Queue()
    )
    solver_thread: Optional[threading.Thread] = None
    solver_job_id = 0
    active_solver_job_id = -1
    solving_mode: Optional[str] = None
    solving_started_at = 0.0

    # A键三段循环状态：0=不显示，1=最小，2=最大。
    area_display_mode = 0

    print("按键说明：")
    print("C：重新标定4个角点")
    print("P：自动判断固定四片 / 纯色 / 花纹并恢复矩形")
    print("F：强制固定四片恢复")
    print("W：强制自主纯色恢复")
    print("G：强制自主花纹/扑克牌恢复")
    print("T：显示/隐藏目标位置叠加")
    print("D：输出识别结果和最新恢复方案")
    print("A：循环显示最小面积框 / 最大面积框 / 不显示")
    print("S：保存原始画面叠加结果")
    print("Q或ESC：退出")
    print(
        f"输出坐标系：标定平面 {WARP_WIDTH}x{WARP_HEIGHT}，"
        "原点位于标定区域右下角，x向上(0～1187)，y向左(0～839)"
    )
    print(
        "实时优化：求解在后台线程执行，摄像头窗口不会再因P/F/W/G卡住；"
        f"碎片识别频率约={1.0 / DETECTION_INTERVAL_SEC:.1f} Hz"
    )
    print(
        "当前带容差的识别面积范围："
        f"{MIN_PIECE_AREA_PX}～{MAX_PIECE_AREA_PX} px²"
    )

    while True:
        # 先接收后台求解结果。
        while True:
            try:
                job_id, ok, payload, worker_elapsed = (
                    solver_result_queue.get_nowait()
                )
            except queue.Empty:
                break

            if job_id != active_solver_job_id:
                continue

            solver_thread = None
            solving_mode = None
            if ok:
                latest_reconstruction = payload
                print(
                    "矩形恢复成功，"
                    f"后台总耗时={worker_elapsed * 1000.0:.1f} ms"
                )
                print_reconstruction_result(latest_reconstruction)
            else:
                latest_reconstruction = None
                print(
                    "矩形恢复失败，"
                    f"后台总耗时={worker_elapsed * 1000.0:.1f} ms：{payload}"
                )

        success, frame = camera.read()
        if not success:
            print("摄像头画面读取失败")
            break

        solver_running = bool(
            solver_thread is not None and solver_thread.is_alive()
        )

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
            if camera_to_warp is None or warp_to_camera is None:
                camera_to_warp, warp_to_camera = build_perspective_matrices(
                    corners
                )

            now = time.perf_counter()
            # 求解期间把识别降到约5.5Hz，给后台搜索更多CPU；
            # 画面仍按摄像头帧率刷新。
            detection_interval = (
                max(DETECTION_INTERVAL_SEC, 0.18)
                if solver_running
                else DETECTION_INTERVAL_SEC
            )
            should_detect = (
                latest_calibrated_region is None
                or now - last_detection_time >= detection_interval
            )

            if should_detect:
                calibrated_region = perspective_correct(
                    frame,
                    camera_to_warp,
                )
                pieces, _mask = detect_pieces(calibrated_region)
                latest_pieces = pieces
                latest_calibrated_region = calibrated_region
                last_detection_time = now

            display = draw_results_on_original(
                frame,
                corners,
                latest_pieces,
                warp_to_camera,
                area_display_mode,
            )
            display = draw_reconstruction_on_original(
                display,
                latest_reconstruction,
                warp_to_camera,
                show_target_overlay,
            )
            if solver_running:
                display = draw_solver_status(
                    display,
                    solving_mode,
                    solving_started_at,
                )
            latest_result = display

        cv2.imshow(MAIN_WINDOW_NAME, latest_result)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("c"), ord("C")):
            if solver_running:
                print("当前正在求解，请等待本次求解结束后再重新标定")
                continue
            try:
                corners = select_calibration_corners(frame)
                save_corners(corners)
                camera_to_warp, warp_to_camera = build_perspective_matrices(
                    corners
                )
                latest_calibrated_region = None
                latest_pieces = []
                latest_reconstruction = None
                last_detection_time = -1e9
                print("标定完成，已保存4个角点；旧恢复方案已清空")
            except RuntimeError as error:
                print(error)

        elif key in (
            ord("p"), ord("P"),
            ord("f"), ord("F"),
            ord("w"), ord("W"),
            ord("g"), ord("G"),
        ):
            if solver_running:
                print("矩形恢复仍在计算中，已忽略重复按键")
                continue

            if latest_calibrated_region is None or not latest_pieces:
                print("无法恢复：当前没有有效标定画面或碎片")
                continue

            if key in (ord("f"), ord("F")):
                requested_mode = "fixed"
            elif key in (ord("w"), ord("W")):
                requested_mode = "white"
            elif key in (ord("g"), ord("G")):
                requested_mode = "pattern"
            else:
                requested_mode = "auto"

            # 冻结按键瞬间的识别结果，后台求解期间下一帧不会改动它。
            pieces_snapshot = clone_pieces_for_solver(latest_pieces)
            image_snapshot = latest_calibrated_region.copy()

            solver_job_id += 1
            active_solver_job_id = solver_job_id
            solving_mode = requested_mode
            solving_started_at = time.perf_counter()
            latest_reconstruction = None

            solver_thread = threading.Thread(
                target=solver_worker,
                args=(
                    active_solver_job_id,
                    pieces_snapshot,
                    image_snapshot,
                    requested_mode,
                    solver_result_queue,
                ),
                name=f"puzzle-solver-{active_solver_job_id}",
                daemon=True,
            )
            solver_thread.start()
            print(
                f"已启动后台矩形恢复：mode={requested_mode}, "
                f"pieces={len(pieces_snapshot)}；窗口可继续刷新和响应Q键"
            )

        elif key in (ord("t"), ord("T")):
            show_target_overlay = not show_target_overlay
            print(
                "目标位置叠加："
                + ("显示" if show_target_overlay else "隐藏")
            )

        elif key in (ord("d"), ord("D")):
            print_results(latest_pieces)
            if solver_running:
                print(
                    "恢复方案仍在计算："
                    f"mode={solving_mode}, "
                    f"elapsed={time.perf_counter() - solving_started_at:.2f}s"
                )
            else:
                print_reconstruction_result(latest_reconstruction)

        elif key in (ord("a"), ord("A")):
            area_display_mode = (area_display_mode + 1) % 3

            if area_display_mode == 1:
                print(
                    "当前显示：最小可识别面积框，"
                    f"面积={MIN_PIECE_AREA_PX} px²，"
                    f"等面积正方形边长约="
                    f"{np.sqrt(MIN_PIECE_AREA_PX):.1f} px"
                )
            elif area_display_mode == 2:
                print(
                    "当前显示：最大可识别面积框，"
                    f"面积={MAX_PIECE_AREA_PX} px²，"
                    f"等面积正方形边长约="
                    f"{np.sqrt(MAX_PIECE_AREA_PX):.1f} px"
                )
            else:
                print("当前显示：关闭面积参考框")

        elif key in (ord("s"), ord("S")):
            if latest_result is not None:
                output_path = "piece_detection_result.png"
                cv2.imwrite(output_path, latest_result)
                print(f"识别结果已保存：{output_path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()