# -*- coding: utf-8 -*-
"""
主程序：串联完整作业流程
相机采集 -> 装甲板识别 -> 角点提取 -> PnP 位姿解算 -> 卡尔曼滤波 -> 运动状态估计与位置预测
运行：
    python main.py                                   # 默认摄像头 0，红色装甲板
    python main.py --source 0 --color red
    python main.py --source test.mp4 --color blue
    python main.py --source photo.jpg --filter opencv
按键：q 退出 | s 保存截图 | r 重置滤波器
"""
import argparse
import csv
import time

import cv2
import numpy as np

import config
from camera_calibration import load_calibration, get_undistort_maps
from armor_detector import ArmorDetector
from pnp_solver import PnPSolver
from kalman_filter import build_filter
from viz import TrendPlot

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- 输入源
def open_source(source, width, height):
    """返回 (reader, is_image)。reader 对视频/摄像头为 VideoCapture。"""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        return cap, False
    p = str(source)
    if str(source).lower().endswith(tuple(IMAGE_EXTS)):
        return p, True
    cap = cv2.VideoCapture(p)
    return cap, False


# ---------------------------------------------------------------- 相机模型
def setup_camera_model(calib_path, w, h, do_undistort):
    """返回 (maps(或None), pnp_mtx, pnp_dist, calibrated)。"""
    cal = load_calibration(calib_path)
    if cal is not None:
        mtx, dist, _ = cal
        if do_undistort:
            map1, map2, new_mtx, _ = get_undistort_maps(mtx, dist, (w, h))
            print("[OK] 已加载标定参数，启用去畸变，PnP 使用校正后内参。")
            return (map1, map2), new_mtx, np.zeros((5, 1)), True
        print("[OK] 已加载标定参数（不去畸变，PnP 直接使用原始内参+畸变）。")
        return None, mtx, dist, True

    print("[WARN] 未找到标定参数文件！使用基于分辨率的近似内参，结果仅供流程验证，")
    print(f"       请先运行 camera_calibration.py 完成标定：{config.CALIB_PARAM_PATH}")
    fx = fy = float(w)
    approx = np.array([[fx, 0, w / 2.0],
                       [0, fy, h / 2.0],
                       [0, 0, 1.0]], np.float64)
    return None, approx, np.zeros((5, 1)), False


def pick_main_armor(armors):
    """多目标时取四边形面积最大的一块作为主跟踪目标。"""
    def area(a):
        return cv2.contourArea(a.corners.astype(np.float32))
    return max(armors, key=area)


def main():
    ap = argparse.ArgumentParser(description="装甲板 PnP 位姿解算 + 卡尔曼滤波")
    ap.add_argument("--source", default=str(config.CAMERA_ID),
                    help="摄像头索引(如 0)、视频文件或图片路径")
    ap.add_argument("--color", choices=["red", "blue"], default=config.ARMOR_COLOR)
    ap.add_argument("--calib", default=str(config.CALIB_PARAM_PATH))
    ap.add_argument("--filter", choices=["custom", "opencv"], default="custom")
    ap.add_argument("--predict-frames", type=int, default=config.PREDICT_FRAMES)
    ap.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    ap.add_argument("--csv", default=str(config.CSV_PATH))
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--show-mask", action="store_true")
    args = ap.parse_args()

    reader, is_image = open_source(args.source, args.width, args.height)
    maps, pnp_mtx, pnp_dist, calibrated = setup_camera_model(
        args.calib, args.width, args.height, not args.no_undistort)

    detector = ArmorDetector(color=args.color)
    solver = PnPSolver(pnp_mtx, pnp_dist)
    kf = build_filter(args.filter)
    trend = TrendPlot()

    csv_file = open(args.csv, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame", "timestamp", "detected", "lost",
        "meas_x", "meas_y", "meas_z", "meas_dist",
        "filt_x", "filt_y", "filt_z", "filt_dist",
        "pred_x", "pred_y", "pred_z",
        "vel_x", "vel_y", "vel_z", "reproj_err"])

    prev_t = time.perf_counter()
    t0 = prev_t
    frame_id = 0
    lost = 0
    fps = 30.0

    try:
        while True:
            now = time.perf_counter()
            dt = max(now - prev_t, 1e-3)
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if is_image:
                frame = cv2.imread(reader)
                if frame is None:
                    print("无法读取图片。")
                    break
            else:
                ok, frame = reader.read()
                if not ok:
                    break

            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

            armors = detector.detect(frame)
            mask = detector._color_mask(frame) if args.show_mask else None
            meas = None
            pose = None

            if armors:
                arm = pick_main_armor(armors)
                ArmorDetector.draw(frame, armors, args.show_mask, mask)
                pose = solver.solve(arm.ordered_points)
                if pose is not None:
                    meas = pose.xyz
                    solver.draw_axis(frame, pose)
                    kf.update(meas, dt)
                    lost = 0
            if pose is None:
                lost += 1
                if kf.initialized:
                    kf.predict(dt)                 # 目标丢失：仅时间更新，协方差增大
                if lost > config.MAX_LOST_FRAMES:
                    kf.reset()
                    lost = 0

            filt = kf.position if kf.initialized else None
            pred = kf.future_position(args.predict_frames, dt) \
                if kf.initialized else None
            trend.add(meas, filt, pred)

            if pose is not None:
                solver.draw_info(frame, pose, filtered_xyz=filt, predicted_xyz=pred)
            else:
                cv2.putText(frame, f"NO ARMOR  lost={lost}", (15, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            config.COLOR_PREDICT, 2)

            # 顶部状态条
            status = f"FPS:{fps:5.1f}  filter:{args.filter}  calib:{'Y' if calibrated else 'N'}"
            cv2.putText(frame, status, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            if kf.initialized:
                v = kf.velocity
                cv2.putText(frame,
                            f"vel(mm/s): vx={v[0]:7.1f} vy={v[1]:7.1f} vz={v[2]:7.1f}",
                            (15, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2)

            writer.writerow([
                frame_id, f"{now - t0:.3f}", int(pose is not None), lost,
                *([f"{v:.2f}" for v in meas] if meas is not None else ["", "", ""]),
                f"{np.linalg.norm(meas):.2f}" if meas is not None else "",
                *([f"{v:.2f}" for v in filt] if filt is not None else ["", "", ""]),
                f"{np.linalg.norm(filt):.2f}" if filt is not None else "",
                *([f"{v:.2f}" for v in pred] if pred is not None else ["", "", ""]),
                *[f"{v:.2f}" for v in kf.velocity],
                f"{pose.reproj_error:.3f}" if pose is not None else "",
            ])

            cv2.imshow(config.WINDOW_NAME, frame)
            cv2.imshow(config.PLOT_WINDOW_NAME, trend.render())
            frame_id += 1

            key = cv2.waitKey(0 if is_image else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                sp = config.ASSET_DIR / f"snapshot_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(sp), frame)
                print(f"截图已保存: {sp}")
            if key == ord("r"):
                kf.reset()
                trend.meas.clear()
                trend.filt.clear()
                trend.pred.clear()
                print("滤波器与历史已重置。")
            if is_image:
                break
    finally:
        csv_file.close()
        if not is_image:
            reader.release()
        cv2.destroyAllWindows()
        print(f"数据已记录到: {args.csv}（共 {frame_id} 帧），"
              f"可运行 plot_data.py 生成对比分析。")


if __name__ == "__main__":
    main()
