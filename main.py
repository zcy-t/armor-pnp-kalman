# -*- coding: utf-8 -*-
"""
主程序：串联完整作业流程
相机采集 -> 装甲板识别 -> 角点提取 -> PnP 位姿解算 -> 卡尔曼滤波 -> 运动状态估计与位置预测

数据源可以是：
    1) MVS 采集的真实视频（默认 data/armor_mvs.avi）
    2) 摄像头实时画面（--source 0）
    3) 单张图片

两种运行方式：
    python main.py                                  # 实时窗口显示
    python main.py --no-show --video-out assets/demo_video.mp4   # 无界面生成演示视频

按键（窗口模式）：q 退出 | s 保存截图 | r 重置滤波器
"""
import argparse
import csv
import os
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
CSV_HEADER = [
    "frame", "timestamp", "detected", "lost",
    "meas_x", "meas_y", "meas_z", "meas_dist",
    "filt_x", "filt_y", "filt_z", "filt_dist",
    "pred_x", "pred_y", "pred_z",
    "vel_x", "vel_y", "vel_z", "reproj_err"]


# ---------------------------------------------------------------- 输入源
def open_source(source, width, height):
    """返回 (reader, is_image, first_frame)。"""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    elif str(source).lower().endswith(tuple(IMAGE_EXTS)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {source}")
        return str(source), True, img
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"视频源没有可读帧: {source}")
    return cap, False, first


def frame_iter(reader, is_image, first_frame):
    yield first_frame
    if is_image:
        return
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        yield frame


# ---------------------------------------------------------------- 相机模型
def setup_camera_model(calib_path, w, h, do_undistort):
    """返回 (maps(或None), pnp_mtx, pnp_dist, calibrated)。"""
    cal = load_calibration(calib_path)
    if cal is not None:
        mtx, dist, img_size = cal
        if tuple(img_size) != (w, h):
            print(f"[WARN] 标定分辨率 {tuple(img_size)} 与输入分辨率 {(w, h)} 不一致，"
                  f"请确认使用同一相机/分辨率采集。")
        if do_undistort:
            map1, map2, new_mtx, _ = get_undistort_maps(mtx, dist, (w, h))
            print("[OK] 已加载 MVS 实拍标定参数，启用去畸变，PnP 使用校正后内参。")
            return (map1, map2), new_mtx, np.zeros((5, 1)), True
        print("[OK] 已加载标定参数（不去畸变，PnP 直接使用原始内参+畸变）。")
        return None, mtx, dist, True

    print("[WARN] 未找到标定参数文件！使用基于分辨率的近似内参，结果仅供流程验证。")
    print(f"       请先运行 camera_calibration.py 完成标定：{config.CALIB_PARAM_PATH}")
    fx = fy = float(w)
    approx = np.array([[fx, 0, w / 2.0],
                       [0, fy, h / 2.0],
                       [0, 0, 1.0]], np.float64)
    return None, approx, np.zeros((5, 1)), False


# ---------------------------------------------------------------- 目标筛选
def pick_best_armor(armors, solver, image_area, max_reproj=6.0):
    """多候选时按 PnP 重投影误差 + 面积挑选真实装甲板，滤掉反光/误配对。

    真实装甲板是固定宽高比的矩形，solvePnP 的重投影误差会很小；
    误配对的四边形无法用该矩形模型解释，重投影误差通常 > 10 px。
    """
    best, best_score = None, -1.0
    for arm in armors:
        area = cv2.contourArea(arm.corners.astype(np.float32))
        if area < image_area * 0.02:          # 太小的候选直接丢弃
            continue
        pose = solver.solve(arm.ordered_points)
        if pose is None or pose.reproj_error > max_reproj:
            continue
        score = area / (1.0 + pose.reproj_error)
        if score > best_score:
            best, best_score = (arm, pose), score
    return best if best is not None else (None, None)


# ---------------------------------------------------------------- 显示合成
_GAMMA_LUT = {}


def gamma_brighten(frame, gamma):
    """仅用于演示画面提亮；检测与 PnP 始终使用原始帧。"""
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return frame.copy()
    key = round(float(gamma), 3)
    lut = _GAMMA_LUT.get(key)
    if lut is None:
        lut = np.array([((i / 255.0) ** gamma) * 255
                        for i in range(256)]).astype(np.uint8)
        _GAMMA_LUT[key] = lut
    return cv2.LUT(frame, lut)


def compose_canvas(disp, trend_img, frame_id, total, status):
    """主画面 + 趋势图 + 标题栏，拼成演示视频帧。"""
    fw, fh = config.DEMO_FRAME_W, config.DEMO_FRAME_H
    tw = config.DEMO_TREND_W
    hd = config.DEMO_HEADER_H
    canvas = np.full((fh + hd, fw + tw, 3), (22, 24, 28), np.uint8)
    canvas[hd:hd + fh, 0:fw] = cv2.resize(disp, (fw, fh),
                                          interpolation=cv2.INTER_AREA)
    if trend_img is not None:
        canvas[hd:hd + fh, fw:fw + tw] = cv2.resize(
            trend_img, (tw, fh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (fw + tw - 1, hd), (38, 42, 50), -1)
    cv2.putText(canvas,
                "Armor Detection  ->  PnP  ->  Kalman Filter  ->  Motion Prediction",
                (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 238, 245), 1)
    cv2.putText(canvas,
                f"{config.CAMERA_MODEL}   |   {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
                f"   |   frame {frame_id}/{total}   |   {status}",
                (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 180, 195), 1)
    cv2.line(canvas, (fw, hd), (fw, hd + fh), (90, 95, 105), 1)
    return canvas


def _text_card(canvas, lines, title, accent=(90, 200, 255)):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, h), (18, 20, 26), -1)
    cv2.rectangle(canvas, (40, 60), (w - 40, h - 60), (45, 50, 60), 2)
    cv2.putText(canvas, title, (80, 135), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, accent, 2)
    cv2.line(canvas, (80, 158), (w - 80, 158), (70, 75, 90), 2)
    y = 215
    for text, scale, color in lines:
        cv2.putText(canvas, text, (90, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1 if scale < 0.6 else 2)
        y += int(38 + scale * 18)
    return canvas


def title_card(width, height, calib_info):
    c = np.zeros((height, width, 3), np.uint8)
    lines = [
        ("OpenCV course project: armor detection / PnP pose / Kalman filter",
         0.68, (235, 238, 245)),
        ("", 0.4, (200, 200, 200)),
        (f"Camera : {config.CAMERA_MODEL}", 0.6, (180, 220, 255)),
        (f"Image  : {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
         f"   |   calibration RMS : {calib_info} px", 0.6, (180, 220, 255)),
        (f"Armor  : {config.ARMOR_WIDTH_MM:.0f} x "
         f"{config.ARMOR_HEIGHT_MM:.0f} mm   |   color : {config.ARMOR_COLOR}",
         0.6, (180, 220, 255)),
        ("", 0.4, (200, 200, 200)),
        ("Pipeline: capture -> detect -> corners -> solvePnP", 0.62, (255, 200, 120)),
        ("          -> Kalman filter -> state estimation / prediction",
         0.62, (255, 200, 120)),
        ("", 0.4, (200, 200, 200)),
        ("ORANGE = measurement   GREEN = filtered   RED = prediction",
         0.6, (170, 235, 170)),
        ("All frames are REAL continuous frames captured by MVS (no simulation)",
         0.6, (170, 235, 170)),
    ]
    return _text_card(c, lines, "ARMOR PNP + KALMAN  /  LIVE DEMO")


def summary_card(width, height, stats):
    c = np.zeros((height, width, 3), np.uint8)
    lines = [
        (f"Frames: {stats['frames']}    solved: {stats['hits']}"
         f"    detection rate: {stats['rate']:.1f}%", 0.62, (235, 238, 245)),
        (f"Distance range: {stats['dmin']:.1f} ~ {stats['dmax']:.1f} mm"
         f"   (armor position / distance changed)", 0.62, (235, 238, 245)),
        ("", 0.4, (200, 200, 200)),
        (f"Distance jitter RMS: measurement {stats['meas_jitter']:.2f} mm"
         f"  ->  filtered {stats['filt_jitter']:.2f} mm", 0.65, (120, 235, 120)),
        (f"Jitter reduced by {stats['improve']:.1f}%", 0.8, (120, 235, 120)),
        ("", 0.4, (200, 200, 200)),
        (f"Mean reprojection error: {stats['reproj']:.2f} px"
         f"   (2D corners match 3D model points)", 0.6, (180, 220, 255)),
        ("", 0.4, (200, 200, 200)),
        ("Outputs: logs/pnp_log.csv, logs/*_compare.png, assets/demo_*.jpg",
         0.6, (200, 205, 215)),
        ("Code and data: github.com/zcy-t/armor-pnp-kalman", 0.6, (200, 205, 215)),
    ]
    return _text_card(c, lines, "RESULT  /  FILTER EFFECT", (120, 235, 120))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="装甲板 PnP 位姿解算 + 卡尔曼滤波")
    ap.add_argument("--source", default=str(config.MVS_ARMOR_VIDEO),
                    help="摄像头索引(如 0)、视频文件或图片路径")
    ap.add_argument("--color", choices=["red", "blue", "green"],
                    default=config.ARMOR_COLOR)
    ap.add_argument("--calib", default=str(config.CALIB_PARAM_PATH))
    ap.add_argument("--filter", choices=["custom", "opencv"], default="custom")
    ap.add_argument("--predict-frames", type=int, default=config.PREDICT_FRAMES)
    ap.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    ap.add_argument("--csv", default=str(config.CSV_PATH))
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--show-mask", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                    help="无界面运行（服务器 / 生成演示视频）")
    ap.add_argument("--video-out", default="",
                    help="输出带标注的演示视频，如 assets/demo_video.mp4")
    ap.add_argument("--no-trend", action="store_true",
                    help="演示视频中不叠加趋势图")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="显示用伽马值，0 表示自动（视频用 config.DEMO_GAMMA）")
    ap.add_argument("--out-fps", type=float, default=config.DEMO_FPS)
    ap.add_argument("--snapshots", type=int, default=4,
                    help="保存的运行截图数量（仅 --video-out 时）")
    ap.add_argument("--max-reproj", type=float, default=10.0,
                    help="PnP 重投影误差上限(px)，用于滤除误检测")
    args = ap.parse_args()

    reader, is_image, first = open_source(args.source, args.width, args.height)
    h, w = first.shape[:2]
    maps, pnp_mtx, pnp_dist, calibrated = setup_camera_model(
        args.calib, w, h, not args.no_undistort)

    detector = ArmorDetector(color=args.color)
    solver = PnPSolver(pnp_mtx, pnp_dist)
    kf = build_filter(args.filter)
    trend = TrendPlot(width=config.DEMO_TREND_W, height=config.DEMO_FRAME_H)

    # 时间步长：视频用真实帧率，摄像头用墙钟，图片用默认帧率
    src_fps = 0.0
    if not is_image:
        src_fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0 or src_fps > 240:
        src_fps = float(config.CAMERA_FPS)
    from_camera = str(args.source).isdigit()
    dt = 1.0 / src_fps
    print(f"[Info] 输入 {w}x{h}，源帧率 {src_fps:.2f} fps，dt={dt:.5f}s，"
          f"标定={'Y' if calibrated else 'N'}")

    gamma = args.gamma if args.gamma > 0 else (
        config.DEMO_GAMMA if args.video_out else 1.0)
    make_video = bool(args.video_out)
    canvas_w = config.DEMO_FRAME_W + config.DEMO_TREND_W
    canvas_h = config.DEMO_FRAME_H + config.DEMO_HEADER_H
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_image else 1

    writer = None
    if make_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_out)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.video_out, fourcc, args.out_fps,
                                 (canvas_w, canvas_h))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {args.video_out}")
        cal = np.load(args.calib)
        rms = float(cal["reproj_error"]) if "reproj_error" in cal else 0.0
        for _ in range(int(args.out_fps * 2.0)):
            writer.write(title_card(canvas_w, canvas_h, f"{rms:.3f}"))
        print(f"[Info] 演示视频写入: {args.video_out}")

    csv_file = open(args.csv, "w", newline="", encoding="utf-8-sig")
    writer_csv = csv.writer(csv_file)
    writer_csv.writerow(CSV_HEADER)

    frame_id = 0
    hits = 0
    lost = 0
    meas_all, filt_all, reproj_all = [], [], []
    snap_every = max(1, total_frames // (args.snapshots + 1)) \
        if make_video and args.snapshots > 0 else 0
    snap_count = 0
    prev_t = time.perf_counter()

    try:
        for frame in frame_iter(reader, is_image, first):
            now = time.perf_counter()
            if from_camera:
                dt = min(max(now - prev_t, 1e-3), 0.1)
            prev_t = now

            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

            armors = detector.detect(frame)
            mask = detector._color_mask(frame) if args.show_mask else None
            arm, pose = pick_best_armor(armors, solver, w * h, args.max_reproj)
            meas = None

            # 检测用原始帧，显示用提亮帧
            disp = gamma_brighten(frame, gamma)
            if arm is not None:
                ArmorDetector.draw(disp, [arm], args.show_mask, mask)
                meas = pose.xyz
                solver.draw_axis(disp, pose)
                kf.update(meas, dt)
                lost = 0
                hits += 1
                meas_all.append(meas.copy())
                reproj_all.append(pose.reproj_error)
            else:
                lost += 1
                if kf.initialized:
                    kf.predict(dt)
                if lost > config.MAX_LOST_FRAMES:
                    kf.reset()
                    lost = 0

            filt = kf.position if kf.initialized else None
            pred = kf.future_position(args.predict_frames, dt) \
                if kf.initialized else None
            if filt is not None:
                filt_all.append(filt.copy())
            trend.add(meas, filt, pred)

            status = f"filter:{args.filter}  calib:{'Y' if calibrated else 'N'}"
            if make_video or args.no_show:
                canvas = compose_canvas(
                    disp, None if args.no_trend else trend.render(),
                    frame_id, total_frames, status)
                if pose is not None:
                    solver.draw_info(canvas, pose, filtered_xyz=filt,
                                     predicted_xyz=pred, origin=(15, 90))
                else:
                    cv2.putText(canvas, f"NO ARMOR   lost={lost}", (15, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                config.COLOR_PREDICT, 2)
                if kf.initialized:
                    v = kf.velocity
                    cv2.putText(
                        canvas,
                        f"vel(mm/s): vx={v[0]:7.1f} vy={v[1]:7.1f} vz={v[2]:7.1f}",
                        (15, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (225, 225, 225), 1)
                if writer is not None:
                    writer.write(canvas)
                if snap_every and frame_id > 0 and frame_id % snap_every == 0 \
                        and snap_count < args.snapshots:
                    sp = config.ASSET_DIR / f"demo_{snap_count:02d}_f{frame_id:03d}.jpg"
                    cv2.imwrite(str(sp), canvas)
                    snap_count += 1
            else:
                show = disp.copy()
                if pose is not None:
                    solver.draw_info(show, pose, filtered_xyz=filt,
                                     predicted_xyz=pred)
                else:
                    cv2.putText(show, f"NO ARMOR  lost={lost}", (15, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                config.COLOR_PREDICT, 2)
                cv2.putText(show, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 255, 200), 2)
                cv2.imshow(config.WINDOW_NAME, show)
                try:
                    cv2.imshow(config.PLOT_WINDOW_NAME, trend.render())
                except cv2.error:
                    pass

            writer_csv.writerow([
                frame_id, f"{frame_id * dt:.4f}", int(pose is not None), lost,
                *([f"{v:.2f}" for v in meas] if meas is not None else ["", "", ""]),
                f"{np.linalg.norm(meas):.2f}" if meas is not None else "",
                *([f"{v:.2f}" for v in filt] if filt is not None else ["", "", ""]),
                f"{np.linalg.norm(filt):.2f}" if filt is not None else "",
                *([f"{v:.2f}" for v in pred] if pred is not None else ["", "", ""]),
                *[f"{v:.2f}" for v in kf.velocity],
                f"{pose.reproj_error:.3f}" if pose is not None else "",
            ])

            if not args.no_show and not make_video:
                key = cv2.waitKey(0 if is_image else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    sp = config.ASSET_DIR / f"snapshot_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(str(sp), disp)
                    print(f"截图已保存: {sp}")
                if key == ord("r"):
                    kf.reset()
                    trend.meas.clear()
                    trend.filt.clear()
                    trend.pred.clear()
            if is_image:
                break
            frame_id += 1
    finally:
        csv_file.close()
        if writer is not None:
            m = np.array(meas_all) if meas_all else np.zeros((1, 3))
            f_ = np.array(filt_all) if filt_all else np.zeros((1, 3))
            md = np.linalg.norm(m, axis=1)
            fd = np.linalg.norm(f_, axis=1)
            mj = float(np.sqrt(np.mean(np.diff(md) ** 2))) if len(md) > 1 else 0.0
            fj = float(np.sqrt(np.mean(np.diff(fd) ** 2))) if len(fd) > 1 else 0.0
            stats = {
                "frames": frame_id, "hits": hits,
                "rate": 100.0 * hits / max(frame_id, 1),
                "dmin": float(md.min()), "dmax": float(md.max()),
                "meas_jitter": mj, "filt_jitter": fj,
                "improve": 100.0 * (1 - fj / mj) if mj > 1e-6 else 0.0,
                "reproj": float(np.mean(reproj_all)) if reproj_all else 0.0,
            }
            for _ in range(int(args.out_fps * 2.5)):
                writer.write(summary_card(canvas_w, canvas_h, stats))
            writer.release()
        if not is_image:
            reader.release()
        if not args.no_show and not make_video:
            cv2.destroyAllWindows()

    print(f"[Done] 共 {frame_id} 帧，有效解算 {hits} 帧 "
          f"({100.0 * hits / max(frame_id, 1):.1f}%)")
    print(f"[Done] 数据: {args.csv}")
    if make_video:
        print(f"[Done] 演示视频: {args.video_out}")


if __name__ == "__main__":
    main()
