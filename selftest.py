# -*- coding: utf-8 -*-
"""
一键自检：无需摄像头，验证各模块算法正确性。
    1. 两种卡尔曼实现（初始化/滤波/预测/重置）
    2. 相机标定：用已知虚拟内参生成棋盘图，再标定，检查内参是否准确恢复
    3. PnP 往返：已知位姿投影后再 solvePnP，检查位置是否还原
运行：python selftest.py
"""
import numpy as np
import cv2

import config
from kalman_filter import KalmanFilter3D, OpenCVKalman3D
from camera_calibration import build_object_points, calibrate
from pnp_solver import PnPSolver

W, H = 1280, 720
MTX0 = np.array([[800, 0, W / 2], [0, 800, H / 2], [0, 0, 1]], np.float64)
DIST0 = np.zeros((5, 1))
# 自检用合成棋盘格固定 25mm 方格，与真实标定板尺寸解耦，保证自检结果稳定
SYNTH_SQUARE_MM = 25.0
rng = np.random.default_rng(0)


def test_kalman():
    for cls in (KalmanFilter3D, OpenCVKalman3D):
        f = cls()
        for _ in range(30):
            z = np.array([100.0, 200.0, 800.0]) + rng.normal(0, 3, 3)
            p = f.update(z, 1 / 30)
        assert p.shape == (3,) and np.isfinite(p).all()
        fut = f.future_position(5, 1 / 30)
        assert fut is not None and fut.shape == (3,)
        f.reset()
        assert not f.initialized
    print("[PASS] 卡尔曼滤波（custom / opencv 两实现）")


def render_board(rvec, tvec):
    """按已知位姿渲染一张棋盘格灰度图。"""
    img = np.full((H, W), 255, np.uint8)
    cols, rows, sq = config.CHESSBOARD_COLS, config.CHESSBOARD_ROWS, \
        SYNTH_SQUARE_MM
    # 形成 (cols x rows) 内角点需要 (cols+1) x (rows+1) 个方格
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 != 0:
                continue
            g = np.array([[c, r, 0], [c + 1, r, 0],
                          [c + 1, r + 1, 0], [c, r + 1, 0]],
                         np.float32) * sq
            p, _ = cv2.projectPoints(g.reshape(-1, 1, 3), rvec, tvec,
                                     MTX0, DIST0)
            cv2.fillPoly(img, [p.reshape(-1, 2).astype(np.int32)], 0)
    return img


def test_calibration():
    frames = []
    for k in range(12):
        rvec = np.array([np.radians(-8 + 16 * k / 11),
                         np.radians(-10 + 20 * k / 11),
                         np.radians(3 * np.sin(k))], np.float64)
        tvec = np.array([30 * np.sin(k), 20 * np.cos(k), 620], np.float64)
        frames.append(cv2.cvtColor(render_board(rvec, tvec), cv2.COLOR_GRAY2BGR))
    result = calibrate(frames, config.CHESSBOARD_COLS, config.CHESSBOARD_ROWS,
                       SYNTH_SQUARE_MM)
    m = result["mtx"]
    fx_err = abs(m[0, 0] - 800) / 800
    fy_err = abs(m[1, 1] - 800) / 800
    cx_err = abs(m[0, 2] - W / 2)
    cy_err = abs(m[1, 2] - H / 2)
    assert fx_err < 0.03 and fy_err < 0.03, "焦距恢复误差过大"
    assert cx_err < 8 and cy_err < 8, "主点恢复误差过大"
    assert result["reproj_error"] < 1.0, "重投影误差过大"
    print(f"[PASS] 相机标定：fx/fy 误差<3%，重投影 RMS={result['reproj_error']:.3f}px")


def test_pnp():
    solver = PnPSolver(MTX0, DIST0)
    rvec0 = np.array([0.05, -0.08, 0.02], np.float64)
    tvec0 = np.array([120.0, -40.0, 850.0], np.float64)
    proj, _ = cv2.projectPoints(solver.object_points, rvec0, tvec0,
                                 MTX0, DIST0)
    pose = solver.solve(proj)
    assert pose is not None
    err = np.linalg.norm(pose.xyz - tvec0)
    assert err < 2.0, f"PnP 位置还原误差 {err:.2f}mm"
    assert abs(pose.distance - np.linalg.norm(tvec0)) < 2.0
    print(f"[PASS] PnP 往返：位置还原误差 {err:.3f} mm，距离 {pose.distance:.1f} mm")


if __name__ == "__main__":
    test_kalman()
    test_calibration()
    test_pnp()
    print("=" * 50)
    print("全部自检通过，环境与各模块工作正常。")
