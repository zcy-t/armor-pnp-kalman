# -*- coding: utf-8 -*-
"""
PnP 位姿解算模块（作业模块 3）
根据装甲板实际尺寸建立物体三维坐标系（中心为原点，z=0 平面），
调用 cv2.solvePnP()，由 2D 角点 + 相机内参/畸变，解算装甲板相对相机的
三维位置 tvec (X,Y,Z) 与姿态 rvec，并输出距离、欧拉角、重投影误差。
相机坐标系约定（OpenCV）：X 向右、Y 向下、Z 沿光轴向前，单位与建模一致(mm)。
"""
import math

import cv2
import numpy as np

import config

_PNP_FLAGS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "IPPE": cv2.SOLVEPNP_IPPE,        # 平面 4 点专用
    "EPNP": cv2.SOLVEPNP_EPNP,
}


class PoseResult:
    """一帧 PnP 解算结果。"""
    __slots__ = ("xyz", "distance", "rvec", "R", "euler_deg",
                 "reproj_error", "reproj_points", "tvec")

    def __init__(self, tvec, rvec, R, euler_deg, reproj_error, reproj_points):
        self.tvec = tvec
        self.rvec = rvec
        self.xyz = tvec.reshape(3).astype(np.float64)        # X,Y,Z (mm)
        self.distance = float(np.linalg.norm(self.xyz))      # 中心距离 (mm)
        self.R = R
        self.euler_deg = euler_deg                           # (roll, pitch, yaw) 度
        self.reproj_error = reproj_error
        self.reproj_points = reproj_points


def rotation_matrix_to_euler(R):
    """旋转矩阵 -> ZYX 欧拉角(roll, pitch, yaw)，单位度，含奇异判断。"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


class PnPSolver:
    def __init__(self, mtx, dist,
                 armor_width=config.ARMOR_WIDTH_MM,
                 armor_height=config.ARMOR_HEIGHT_MM,
                 method=config.PNP_METHOD,
                 use_distortion=config.USE_DISTORTION):
        self.mtx = mtx.astype(np.float64)
        self.dist = dist.astype(np.float64) if use_distortion else np.zeros((5, 1))
        self.method = _PNP_FLAGS[method]
        self.w = armor_width
        self.h = armor_height

        # 三维模型点，顺序与图像角点 lt, rt, rb, lb 严格对应
        hw, hh = armor_width / 2.0, armor_height / 2.0
        self.object_points = np.array([
            [-hw, -hh, 0.0],   # 左上
            [ hw, -hh, 0.0],   # 右上
            [ hw,  hh, 0.0],   # 右下
            [-hw,  hh, 0.0],   # 左下
        ], dtype=np.float32).reshape(-1, 1, 3)

    def solve(self, image_points):
        """image_points: (4,1,2) 角点；成功返回 PoseResult，失败返回 None。"""
        image_points = np.asarray(image_points, np.float32).reshape(-1, 1, 2)
        if len(image_points) != 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(self.object_points, image_points,
                                      self.mtx, self.dist, flags=self.method)
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        euler = rotation_matrix_to_euler(R)

        # 重投影验证：把 3D 点再投回图像，与实际角点比较
        reproj, _ = cv2.projectPoints(self.object_points, rvec, tvec,
                                      self.mtx, self.dist)
        err = float(np.linalg.norm(
            reproj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1).mean())
        return PoseResult(tvec.astype(np.float64), rvec, R, euler, err, reproj)

    # ------------------------------------------------------------ 绘制
    def draw_axis(self, frame, pose, axis_len=None):
        """在装甲板中心绘制三维坐标轴（X 蓝、Y 绿、Z 红）。"""
        if axis_len is None:
            axis_len = max(self.w, self.h) * 0.9
        axes = np.array([
            [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]
        ], dtype=np.float32)
        pts, _ = cv2.projectPoints(axes, pose.rvec, pose.tvec,
                                   self.mtx, self.dist)
        pts = pts.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        cv2.arrowedLine(frame, origin, tuple(pts[1]), (255, 0, 0), 2)   # X 蓝
        cv2.arrowedLine(frame, origin, tuple(pts[2]), (0, 255, 0), 2)   # Y 绿
        cv2.arrowedLine(frame, origin, tuple(pts[3]), (0, 0, 255), 2)   # Z 红
        return frame

    def draw_info(self, frame, pose, origin=(15, 90),
                  filtered_xyz=None, predicted_xyz=None):
        """实时输出 X/Y/Z、距离、欧拉角，并可叠加滤波/预测值。"""
        x, y, z = pose.xyz
        lines = [
            f"MEAS  XYZ(mm): x={x:7.1f} y={y:7.1f} z={z:7.1f} d={pose.distance:7.1f}",
        ]
        if filtered_xyz is not None:
            fx, fy, fz = filtered_xyz
            lines.append(
                f"FILT  XYZ(mm): x={fx:7.1f} y={fy:7.1f} z={fz:7.1f} "
                f"d={np.linalg.norm(filtered_xyz):7.1f}")
        if predicted_xyz is not None:
            px, py, pz = predicted_xyz
            lines.append(
                f"PRED  XYZ(mm): x={px:7.1f} y={py:7.1f} z={pz:7.1f} "
                f"d={np.linalg.norm(predicted_xyz):7.1f}")
        r, p, yw = pose.euler_deg
        lines.append(f"euler(deg): roll={r:6.1f} pitch={p:6.1f} yaw={yw:6.1f} "
                     f"reprojErr={pose.reproj_error:.2f}px")

        x0, y0 = origin
        for i, text in enumerate(lines):
            color = (255, 255, 255)
            if text.startswith("MEAS"):
                color = config.COLOR_MEASURE
            elif text.startswith("FILT"):
                color = config.COLOR_FILTER
            elif text.startswith("PRED"):
                color = config.COLOR_PREDICT
            cv2.putText(frame, text, (x0, y0 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
