# -*- coding: utf-8 -*-
"""
卡尔曼滤波与运动预测模块（作业模块 4）
状态向量: x = [px, py, pz, vx, vy, vz]^T   （6 维，匀速运动模型）
观测向量: z = [px, py, pz]^T               （3 维，来自 PnP 解算）
过程噪声: 连续白噪声加速度模型（块对角，三轴独立）
提供两种实现，接口完全一致：
    KalmanFilter3D  —— 自行实现标准卡尔曼五个方程（默认，逻辑透明）
    OpenCVKalman3D  —— 基于 cv2.KalmanFilter 的等价实现（--filter opencv）
"""
import numpy as np

import config


def build_process_noise(dt, accel_std):
    """连续白噪声加速度模型离散化，返回 6x6 过程噪声协方差 Q。
    单轴: q * [[dt^4/4, dt^3/2],
                [dt^3/2, dt^2]]      其中 q = accel_std^2
    """
    q = accel_std ** 2
    q1 = q * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                       [dt ** 3 / 2.0, dt ** 2]], np.float64)
    Q = np.zeros((6, 6), np.float64)
    for axis in range(3):
        p = axis          # 位置状态索引: 0,1,2
        v = axis + 3      # 速度状态索引: 3,4,5
        Q[np.ix_([p, v], [p, v])] = q1
    return Q


class KalmanFilter3D:
    """自行实现的三维位置匀速卡尔曼滤波器。"""

    def __init__(self,
                 measure_std=config.KF_MEASURE_NOISE,
                 accel_std=config.KF_ACCEL_NOISE,
                 init_pos_std=config.KF_INIT_POS_STD,
                 init_vel_std=config.KF_INIT_VEL_STD):
        self.accel_std = accel_std
        self.x = np.zeros((6, 1), np.float64)          # 状态后验
        self.P = np.diag([init_pos_std ** 2] * 3 +
                         [init_vel_std ** 2] * 3).astype(np.float64)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])  # 观测矩阵
        self.R = np.eye(3) * measure_std ** 2              # 测量噪声协方差
        self.initialized = False

    # ------------------------------------------------------------ 矩阵构造
    @staticmethod
    def _F(dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    # ------------------------------------------------------------ 预测
    def predict(self, dt):
        """时间更新（先验）。返回先验位置 (3,)。"""
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + build_process_noise(dt, self.accel_std)
        self.P = (self.P + self.P.T) / 2.0
        return self.x[:3].reshape(3)

    # ------------------------------------------------------------ 校正
    def update(self, measurement, dt):
        """收到一帧 PnP 测量：首帧初始化，否则先预测再校正。返回滤波位置。"""
        z = np.asarray(measurement, np.float64).reshape(3, 1)
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return self.position

        self.predict(dt)
        y = z - self.H @ self.x                              # 新息
        S = self.H @ self.P @ self.H.T + self.R              # 新息协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)             # 卡尔曼增益
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P
        self.P = (self.P + self.P.T) / 2.0
        return self.position

    # ------------------------------------------------------------ 运动预测
    def future_position(self, n_frames, dt):
        """基于当前后验位置与速度，匀速外推 n_frames 帧后的位置。"""
        if not self.initialized:
            return None
        return self.position + self.velocity * (n_frames * dt)

    def reset(self):
        self.x[:] = 0.0
        self.P = np.diag([config.KF_INIT_POS_STD ** 2] * 3 +
                         [config.KF_INIT_VEL_STD ** 2] * 3)
        self.initialized = False

    @property
    def position(self):
        return self.x[:3].reshape(3).copy()

    @property
    def velocity(self):
        return self.x[3:].reshape(3).copy()


class OpenCVKalman3D:
    """基于 cv2.KalmanFilter 的等价实现，接口与 KalmanFilter3D 相同。"""

    def __init__(self,
                 measure_std=config.KF_MEASURE_NOISE,
                 accel_std=config.KF_ACCEL_NOISE):
        import cv2
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.hstack(
            [np.eye(3), np.zeros((3, 3))]).astype(np.float32)
        self.kf.measurementNoiseCov = (
            np.eye(3) * measure_std ** 2).astype(np.float32)
        self.accel_std = accel_std
        self.initialized = False

    def _set_dynamic(self, dt):
        self.kf.transitionMatrix = KalmanFilter3D._F(dt).astype(np.float32)
        self.kf.processNoiseCov = build_process_noise(
            dt, self.accel_std).astype(np.float32)

    def predict(self, dt):
        self._set_dynamic(dt)
        p = self.kf.predict()
        return np.asarray(p[:3]).reshape(3)

    def update(self, measurement, dt):
        z = np.asarray(measurement, np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost = np.vstack(
                [z, np.zeros((3, 1), np.float32)])
            self.initialized = True
            return self.position
        self._set_dynamic(dt)
        self.kf.predict()
        self.kf.correct(z)
        return self.position

    def future_position(self, n_frames, dt):
        if not self.initialized:
            return None
        return self.position + self.velocity * (n_frames * dt)

    def reset(self):
        self.kf.statePost = np.zeros((6, 1), np.float32)
        self.initialized = False

    @property
    def position(self):
        return np.asarray(self.kf.statePost[:3]).reshape(3).copy()

    @property
    def velocity(self):
        return np.asarray(self.kf.statePost[3:]).reshape(3).copy()


def build_filter(name="custom"):
    """工厂函数：name='custom' 自行实现，'opencv' 使用 OpenCV 实现。"""
    if name == "opencv":
        return OpenCVKalman3D()
    return KalmanFilter3D()
