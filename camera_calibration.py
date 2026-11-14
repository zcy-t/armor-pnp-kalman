# -*- coding: utf-8 -*-
"""
相机标定模块（作业模块 1）
功能：
    1. 从文件夹中的标定图片批量标定（--mode images）
    2. 连接摄像头实时抓拍标定（--mode camera）
    3. 计算相机内参矩阵 cameraMatrix 与畸变系数 distCoeffs
    4. 评估重投影误差，并将结果保存为 npz 供后续 PnP 使用
运行示例：
    python camera_calibration.py --mode images
    python camera_calibration.py --mode camera --camera 0 --max-frames 20
"""
import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np

import config

# 亚像素角点迭代终止条件
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


# ---------------------------------------------------------------- 基础工具
def build_object_points(board_cols, board_rows, square_mm):
    """生成棋盘格在标定板平面(z=0)上的三维坐标，单位 mm。
    形状 (board_cols*board_rows, 1, 3)，与 findChessboardCorners 的角点顺序一致。
    """
    pts = np.zeros((board_cols * board_rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    pts *= square_mm
    return pts.reshape(-1, 1, 3)


def find_chessboard(frame, board_cols, board_rows):
    """检测棋盘格并做亚像素精细化，返回 (ok, corners)。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (board_cols, board_rows), flags)
    if not ret:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return True, corners


# ---------------------------------------------------------------- 核心标定
def calibrate(frames, board_cols, board_rows, square_mm):
    """对给定的一组图像执行标定。
    返回 dict：mtx, dist, rvecs, tvecs, reproj_error, per_view_error, image_size, ok_count
    """
    objp = build_object_points(board_cols, board_rows, square_mm)
    obj_points = []          # 每张图的 3D 点
    img_points = []          # 每张图的 2D 角点
    image_size = None
    used = 0

    for frame in frames:
        if frame is None:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])  # (w, h)
        ok, corners = find_chessboard(frame, board_cols, board_rows)
        if ok:
            obj_points.append(objp)
            img_points.append(corners)
            used += 1

    if used < 5:
        raise RuntimeError(f"有效标定图过少（{used} 张），至少需要 5~10 张，请重新采集。")

    # 内参初值留空，由 calibrateCamera 自动估计
    mtx = np.zeros((3, 3), np.float64)
    dist = np.zeros((5, 1), np.float64)
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, mtx, dist,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )

    # 逐视图重投影误差，便于发现质量差的标定图
    per_view = []
    for i in range(len(obj_points)):
        reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        a = img_points[i].reshape(-1, 2).astype(np.float64)
        b = reproj.reshape(-1, 2).astype(np.float64)
        err = float(np.linalg.norm(a - b, axis=1).mean())
        per_view.append(err)

    return {
        "mtx": mtx,
        "dist": dist,
        "rvecs": np.array(rvecs, dtype=object),
        "tvecs": np.array(tvecs, dtype=object),
        "reproj_error": float(rms),
        "per_view_error": np.float64(per_view),
        "image_size": np.int64(image_size),
        "ok_count": used,
    }


# ---------------------------------------------------------------- 存取
def save_calibration(result, path=config.CALIB_PARAM_PATH):
    """保存标定结果到 npz。"""
    np.savez(
        path,
        mtx=result["mtx"],
        dist=result["dist"],
        reproj_error=result["reproj_error"],
        per_view_error=result["per_view_error"],
        image_size=result["image_size"],
        ok_count=result["ok_count"],
        board_cols=config.CHESSBOARD_COLS,
        board_rows=config.CHESSBOARD_ROWS,
        square_mm=config.SQUARE_SIZE_MM,
    )
    print(f"[OK] 标定参数已保存: {path}")


def load_calibration(path=config.CALIB_PARAM_PATH):
    """读取标定结果，返回 (mtx, dist, image_size)；文件不存在则返回 None。"""
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    mtx = data["mtx"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    image_size = tuple(int(v) for v in data["image_size"])
    return mtx, dist, image_size


def get_undistort_maps(mtx, dist, image_size):
    """返回 remap 所需的映射表（主程序初始化一次即可，避免每帧重复计算）。"""
    w, h = image_size
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    return map1, map2, new_mtx, roi


def undistort_frame(frame, mtx, dist):
    """单帧去畸变（便捷函数，实时场景建议用 get_undistort_maps 预计算）。"""
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


# ---------------------------------------------------------------- 模式一：文件夹
def calibrate_from_images(image_dir=config.CALIB_IMAGE_DIR):
    paths = sorted(
        glob.glob(str(Path(image_dir) / "*.jpg"))
        + glob.glob(str(Path(image_dir) / "*.png"))
        + glob.glob(str(Path(image_dir) / "*.jpeg"))
    )
    if not paths:
        raise FileNotFoundError(f"目录 {image_dir} 下没有找到标定图片(jpg/png)。")
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    print(f"共读取 {len(frames)} 张图片，开始标定...")
    result = calibrate(frames, config.CHESSBOARD_COLS, config.CHESSBOARD_ROWS,
                       config.SQUARE_SIZE_MM)
    _report(result)
    save_calibration(result)
    return result


def _report(result):
    print("=" * 60)
    print(f"有效图片数量 : {result['ok_count']}")
    print(f"整体重投影 RMS 误差: {result['reproj_error']:.4f} px")
    print("相机内参矩阵 cameraMatrix:")
    print(np.round(result["mtx"], 4))
    print("畸变系数 distCoeffs (k1,k2,p1,p2,k3):")
    print(np.round(result["dist"].ravel(), 4))
    print(f"各视图误差范围: {result['per_view_error'].min():.3f} ~ "
          f"{result['per_view_error'].max():.3f} px")
    print("=" * 60)


# ---------------------------------------------------------------- 模式二：实时抓拍
def calibrate_from_camera(camera_id=config.CAMERA_ID, max_frames=20,
                          snapshot_dir=config.CALIB_SNAPSHOT_DIR):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")

    saved_frames = []
    last_save_ts = 0.0
    print("操作: [空格/s]抓拍  [c]标定  [q]退出；建议移动标定板覆盖不同角度与距离。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            found, corners = find_chessboard(frame, config.CHESSBOARD_COLS,
                                             config.CHESSBOARD_ROWS)
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview,
                                          (config.CHESSBOARD_COLS, config.CHESSBOARD_ROWS),
                                          corners, True)
            cv2.putText(preview, f"saved: {len(saved_frames)}/{max_frames}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(preview, "SPACE:capture  c:calibrate  q:quit",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("camera calibration", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(" "), ord("s")) and found and (time.time() - last_save_ts) > 0.8:
                saved_frames.append(frame.copy())
                sp = Path(snapshot_dir) / f"shot_{len(saved_frames):02d}.jpg"
                cv2.imwrite(str(sp), frame)
                last_save_ts = time.time()
                print(f"抓拍第 {len(saved_frames)} 张 -> {sp.name}")
            elif key == ord("c"):
                if len(saved_frames) < 5:
                    print("抓拍数量不足 5 张，无法标定。")
                    continue
                result = calibrate(saved_frames, config.CHESSBOARD_COLS,
                                   config.CHESSBOARD_ROWS, config.SQUARE_SIZE_MM)
                _report(result)
                save_calibration(result)
                break
            elif key == ord("q"):
                break
            if len(saved_frames) >= max_frames:
                print("已达到目标数量，自动执行标定...")
                result = calibrate(saved_frames, config.CHESSBOARD_COLS,
                                   config.CHESSBOARD_ROWS, config.SQUARE_SIZE_MM)
                _report(result)
                save_calibration(result)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved_frames


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="OpenCV 相机标定")
    ap.add_argument("--mode", choices=["images", "camera"], default="images")
    ap.add_argument("--camera", type=int, default=config.CAMERA_ID)
    ap.add_argument("--max-frames", type=int, default=20)
    args = ap.parse_args()

    if args.mode == "images":
        calibrate_from_images()
    else:
        calibrate_from_camera(args.camera, args.max_frames)


if __name__ == "__main__":
    main()
