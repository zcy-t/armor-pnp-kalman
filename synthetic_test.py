# -*- coding: utf-8 -*-
"""
合成数据端到端测试（无摄像头也能验证整条流水线）
用已知的虚拟相机内参与一条已知三维轨迹，把装甲板投影成红色灯条图像，
再依次走 检测 -> PnP -> 卡尔曼 -> 预测，并与真值对比，量化滤波收益。
注意：作业最终演示必须使用真实相机数据，本脚本仅用于开发验证与生成参考截图。
运行：python synthetic_test.py
"""
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from armor_detector import ArmorDetector
from pnp_solver import PnPSolver
from kalman_filter import build_filter

# 虚拟相机
W, H = 1280, 720
FX = FY = 800.0
VIRTUAL_MTX = np.array([[FX, 0, W / 2.0],
                        [0, FY, H / 2.0],
                        [0, 0, 1.0]], np.float64)
DIST0 = np.zeros((5, 1))
N = 120
FPS = 30.0
DT = 1.0 / FPS
NOISE_PX = 2.5          # 模拟角点检测噪声（像素）
LIGHT_WIDTH_PX = 12.0   # 灯条宽度（像素）


def draw_light_bar(frame, top, bottom, width, color):
    """用 fillPoly 精确绘制以 top/bottom 为长边中心的旋转矩形灯条，
    避免 cv2.line 粗线端帽额外延伸半个线宽。"""
    top = top.astype(np.float64)
    bottom = bottom.astype(np.float64)
    d = bottom - top
    d = d / max(np.linalg.norm(d), 1e-6)
    n = np.array([-d[1], d[0]])               # 垂直于灯条的法向量
    hw = width / 2.0
    poly = np.array([
        top + n * hw,
        top - n * hw,
        bottom - n * hw,
        bottom + n * hw,
    ], np.int32)
    cv2.fillPoly(frame, [poly], color)


def main():
    rng = np.random.default_rng(42)
    detector = ArmorDetector(color="red")
    solver = PnPSolver(VIRTUAL_MTX, DIST0)
    kf = build_filter("custom")

    truth, meas_l, filt_l, pred_l = [], [], [], []
    csv_path = config.LOG_DIR / "synthetic_log.csv"
    fcsv = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(fcsv)
    writer.writerow([
        "frame", "timestamp", "detected", "lost",
        "meas_x", "meas_y", "meas_z", "meas_dist",
        "filt_x", "filt_y", "filt_z", "filt_dist",
        "pred_x", "pred_y", "pred_z",
        "vel_x", "vel_y", "vel_z", "reproj_err"])

    for i in range(N):
        t = i * DT
        # ---------------- 真值轨迹（mm）
        z = 1000.0 - 300.0 * (i / (N - 1))           # 1000 -> 700 靠近
        x = 160.0 * np.sin(2 * np.pi * t / 2.5)
        y = 60.0 * np.sin(2 * np.pi * t / 4.0)
        tvec = np.array([x, y, z], np.float64)
        rvec = np.array([np.radians(5.0),
                         np.radians(10.0 * np.sin(2 * np.pi * t / 3.0)),
                         0.0], np.float64)

        # ---------------- 投影 + 绘制带噪声的红色灯条
        pts, _ = cv2.projectPoints(solver.object_points, rvec, tvec,
                                   VIRTUAL_MTX, DIST0)
        pts = pts.reshape(4, 2) + rng.normal(0, NOISE_PX, (4, 2))
        frame = np.zeros((H, W, 3), np.uint8)
        lt, rt, rb, lb = pts
        draw_light_bar(frame, lt, lb, LIGHT_WIDTH_PX, (0, 0, 255))
        draw_light_bar(frame, rt, rb, LIGHT_WIDTH_PX, (0, 0, 255))

        # ---------------- 真实流水线
        armors = detector.detect(frame)
        meas = None
        pose = None
        if armors:
            arm = max(armors, key=lambda a: cv2.contourArea(a.corners))
            pose = solver.solve(arm.ordered_points)
            if pose is not None:
                meas = pose.xyz
                kf.update(meas, DT)
        if pose is None and kf.initialized:
            kf.predict(DT)

        filt = kf.position if kf.initialized else None
        pred = kf.future_position(config.PREDICT_FRAMES, DT) \
            if kf.initialized else None

        truth.append(tvec)
        meas_l.append(meas)
        filt_l.append(filt)
        pred_l.append(pred)

        writer.writerow([
            i, f"{t:.3f}", int(meas is not None), 0,
            *([f"{v:.2f}" for v in meas] if meas is not None else ["", "", ""]),
            f"{np.linalg.norm(meas):.2f}" if meas is not None else "",
            *([f"{v:.2f}" for v in filt] if filt is not None else ["", "", ""]),
            f"{np.linalg.norm(filt):.2f}" if filt is not None else "",
            *([f"{v:.2f}" for v in pred] if pred is not None else ["", "", ""]),
            *[f"{v:.2f}" for v in kf.velocity],
            f"{pose.reproj_error:.3f}" if pose is not None else "",
        ])

        if i in (10, 60, 110) and armors:
            ArmorDetector.draw(frame, armors)
            if pose is not None:
                solver.draw_axis(frame, pose)
            cv2.imwrite(str(config.ASSET_DIR / f"synthetic_frame_{i}.jpg"), frame)

    fcsv.close()

    # ---------------- 与真值对比
    truth = np.array(truth)
    meas_arr = np.array([m if m is not None else [np.nan] * 3 for m in meas_l])
    filt_arr = np.array([m if m is not None else [np.nan] * 3 for m in filt_l])
    pred_arr = np.array([m if m is not None else [np.nan] * 3 for m in pred_l])

    mask = ~np.isnan(meas_arr[:, 0])
    meas_rmse = np.sqrt(np.nanmean((meas_arr - truth) ** 2))
    filt_rmse = np.sqrt(np.nanmean((filt_arr - truth) ** 2))
    print("=" * 60)
    print(f"检测成功帧数: {mask.sum()}/{N}")
    print(f"相对真值 RMSE(mm)  measurement : {meas_rmse:.2f}")
    print(f"相对真值 RMSE(mm)  filtered    : {filt_rmse:.2f}")
    print(f"滤波降低误差: {(1 - filt_rmse / meas_rmse) * 100:.1f}%")
    print("=" * 60)

    # ---------------- 对比图
    t_axis = np.arange(N) * DT
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    names = ["X (mm)", "Y (mm)", "Z (mm)", "distance (mm)"]
    series_truth = [truth[:, 0], truth[:, 1], truth[:, 2],
                    np.linalg.norm(truth, axis=1)]
    series_meas = [meas_arr[:, 0], meas_arr[:, 1], meas_arr[:, 2],
                   np.linalg.norm(meas_arr, axis=1)]
    series_filt = [filt_arr[:, 0], filt_arr[:, 1], filt_arr[:, 2],
                   np.linalg.norm(filt_arr, axis=1)]
    series_pred = [pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2],
                   np.linalg.norm(pred_arr, axis=1)]
    for ax, name, a, b, c, d in zip(axes, names, series_truth, series_meas,
                                     series_filt, series_pred):
        ax.plot(t_axis, a, color="k", lw=1.6, label="ground truth")
        ax.plot(t_axis, b, color="#ff8c00", lw=1.0, label="measurement")
        ax.plot(t_axis, c, color="#00a000", lw=1.4, label="filtered")
        ax.plot(t_axis, d, color="#d62728", lw=1.0, ls="--", label="predicted")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Synthetic End-to-End: truth vs measurement vs filtered vs predicted")
    fig.tight_layout()
    out = config.LOG_DIR / "synthetic_compare.png"
    fig.savefig(out, dpi=130)
    print(f"对比图已保存: {out}")
    print(f"合成 CSV 已保存: {csv_path}（可用 plot_data.py --csv 该文件分析）")


if __name__ == "__main__":
    main()
