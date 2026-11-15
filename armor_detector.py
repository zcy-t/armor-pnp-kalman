# -*- coding: utf-8 -*-
"""
装甲板识别与角点获取模块（作业模块 2）
方案：基于颜色的传统视觉（不依赖额外模型文件，OpenCV 即可独立运行）
    颜色二值化 -> 轮廓提取 -> 灯条筛选(细长旋转矩形) -> 灯条配对 -> 四角点排序
角点输出顺序固定为：左上 lt -> 右上 rt -> 右下 rb -> 左下 lb
该顺序与 pnp_solver 中三维模型点严格一一对应。
若课堂使用的是 YOLO 等深度模型，可使用文件末尾的 YoloArmorDetector 适配（接口相同）。
"""
import math

import cv2
import numpy as np

import config

# HSV 阈值（不同光照/相机可微调）。H:0-180, S:0-255, V:0-255
HSV_RANGE = {
    "red": [((0, 80, 80), (10, 255, 255)),
            ((170, 80, 80), (180, 255, 255))],   # 红色跨 0 度，需两段合并
    "blue": [((95, 80, 80), (130, 255, 255))],
    # 本项目 MVS 实拍装甲板为绿色 LED 灯条（H 约 60~85），
    # 暗背景下饱和度和亮度阈值放低，保证灯条不被漏检
    "green": [((35, 60, 60), (95, 255, 255))],
}


class LightBar:
    """单个灯条：由 minAreaRect 拟合，并求上下端点。"""
    __slots__ = ("center", "top", "bottom", "length", "width", "angle_deg", "box")

    def __init__(self, rect):
        self.center = np.array(rect[0], np.float32)
        (rw, rh) = rect[1]
        self.length = max(rw, rh)          # 长边
        self.width = min(rw, rh)           # 短边
        self.box = cv2.boxPoints(rect).astype(np.float32)

        # 四个顶点按 y 排序，上方两点均值=上端点，下方两点均值=下端点
        order = np.argsort(self.box[:, 1])
        upper = self.box[order[:2]].mean(axis=0)
        lower = self.box[order[2:]].mean(axis=0)
        # 倾斜灯条：y 相同的极端情况下用 x 再区分，这里取均值已足够稳健
        self.top = upper
        self.bottom = lower

        # 灯条方向与竖直方向的夹角（0 表示完全竖直）
        dx, dy = self.bottom - self.top
        self.angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))

    def __repr__(self):
        return f"Light(len={self.length:.1f}, angle={self.angle_deg:.1f})"


class Armor:
    """一块装甲板：保存两个灯条与按固定顺序排列的四角点。"""
    def __init__(self, left, right, color):
        self.left = left
        self.right = right
        self.color = color
        # 严格顺序：左上、右上、右下、左下
        self.corners = np.array(
            [left.top, right.top, right.bottom, left.bottom], dtype=np.float32)
        self.center = self.corners.mean(axis=0)

    @property
    def ordered_points(self):
        """返回 solvePnP 所需的 (4,1,2) float32 角点。"""
        return self.corners.reshape(-1, 1, 2)


class ArmorDetector:
    """传统视觉装甲板检测器。"""

    def __init__(self, color=config.ARMOR_COLOR,
                 min_area=config.LIGHT_MIN_AREA,
                 min_ratio=config.LIGHT_MIN_RATIO,
                 max_angle=config.LIGHT_MAX_ANGLE_DEG):
        self.color = color
        self.min_area = min_area
        self.min_ratio = min_ratio
        self.max_angle = max_angle
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # ------------------------------------------------------------ 二值化
    def _color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in HSV_RANGE[self.color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        # 闭运算：连接同一灯条上因反光产生的断裂
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        return mask

    # ------------------------------------------------------------ 灯条提取
    def _find_light_bars(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        lights = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            light = LightBar(rect)
            if light.length < 1e-3 or light.width < 1e-3:
                continue
            ratio = light.length / max(light.width, 1e-6)
            if ratio < self.min_ratio:
                continue
            if light.angle_deg > self.max_angle:
                continue
            lights.append(light)
        return lights

    # ------------------------------------------------------------ 灯条配对
    def _pair(self, lights):
        candidates = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                a, b = lights[i], lights[j]
                # 确定左右灯条
                if a.center[0] < b.center[0]:
                    left, right = a, b
                else:
                    left, right = b, a

                # 1) 长度差
                avg_len = (left.length + right.length) / 2.0
                len_diff = abs(left.length - right.length) / max(avg_len, 1e-6)
                if len_diff > config.PAIR_MAX_HIGH_DIFF_RATIO:
                    continue

                # 2) 两灯条自身倾斜角之差
                if abs(left.angle_deg - right.angle_deg) > \
                        config.PAIR_MAX_LIGHT_ANGLE_DIFF:
                    continue

                # 3) 中心连线与水平方向夹角
                d = right.center - left.center
                center_angle = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
                if center_angle > config.PAIR_MAX_CENTER_ANGLE_DEG:
                    continue

                # 4) 装甲板宽高比（中心距离 / 平均灯长）
                gap = float(np.linalg.norm(d))
                wh_ratio = gap / max(avg_len, 1e-6)
                if not (config.PAIR_MIN_W_H_RATIO <= wh_ratio <=
                        config.PAIR_MAX_W_H_RATIO):
                    continue

                # 评分：差异越小越合理，取负作为“越大越好”的分数
                score = -(len_diff * 2 + center_angle / 10.0 +
                          abs(left.angle_deg - right.angle_deg) / 10.0)
                candidates.append((score, left, right))

        # 贪心：按评分从高到低，且每个灯条最多使用一次
        candidates.sort(key=lambda t: t[0], reverse=True)
        used = set()
        armors = []
        for _, left, right in candidates:
            lid, rid = id(left), id(right)
            if lid in used or rid in used:
                continue
            used.add(lid)
            used.add(rid)
            armors.append(Armor(left, right, self.color))
        return armors

    # ------------------------------------------------------------ 主接口
    def detect(self, frame):
        """返回 Armor 列表（多目标场景可能有多块）。"""
        mask = self._color_mask(frame)
        lights = self._find_light_bars(mask)
        if len(lights) < 2:
            return []
        return self._pair(lights)

    # ------------------------------------------------------------ 可视化
    @staticmethod
    def draw(frame, armors, draw_mask=False, mask=None):
        for idx, arm in enumerate(armors):
            pts = arm.corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, config.COLOR_FILTER, 2)
            labels = ["LT", "RT", "RB", "LB"]
            for k, (p, lab) in enumerate(zip(pts, labels)):
                cv2.circle(frame, tuple(p), 4, config.COLOR_CORNER, -1)
                cv2.putText(frame, lab, (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            config.COLOR_CORNER, 1)
            c = arm.center.astype(int)
            cv2.putText(frame, f"armor#{idx}", (c[0] - 30, c[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        config.COLOR_FILTER, 2)
        if draw_mask and mask is not None:
            cv2.imshow("color mask", mask)
        return frame


class YoloArmorDetector:
    """YOLO 深度模型适配器（与 ArmorDetector 接口一致）。
    需要自行安装 ultralytics 并提供课堂模型权重，例如 best.pt / best.onnx。
    传统检测器已能独立完成作业，本类仅在需要复用课堂模型时启用。
    """

    def __init__(self, weights, color=config.ARMOR_COLOR, conf=0.5):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "未安装 ultralytics，请先 `pip install ultralytics` 后再使用 YOLO 模型。"
            ) from exc
        self.model = YOLO(weights)
        self.color = color
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        armors = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                # 检测框即视为装甲板矩形，按固定顺序给出四角点
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                   np.float32)
                armor = _ArmorFromCorners(corners, self.color)
                armors.append(armor)
        return armors

    @staticmethod
    def draw(frame, armors, **kwargs):
        return ArmorDetector.draw(frame, armors)


class _ArmorFromCorners:
    """YOLO 路径下仅含角点的轻量装甲板对象。"""
    def __init__(self, corners, color):
        self.corners = corners.astype(np.float32)
        self.center = corners.mean(axis=0)
        self.color = color

    @property
    def ordered_points(self):
        return self.corners.reshape(-1, 1, 2)
