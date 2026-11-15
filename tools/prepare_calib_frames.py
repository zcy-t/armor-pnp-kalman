# -*- coding: utf-8 -*-
"""
从 MVS 采集的棋盘格视频中挑选不同位置/角度的帧，补充为标定图片。

作业要求「采集不同位置、不同角度的标定图像」。本项目把 MVS 客户端录制的
棋盘格视频 data/calib_mvs.avi 逐帧检测角点，再用最远点采样挑出视角差异
最大的若干帧，写入 calibration/calib_images/，随后可直接运行：
    python camera_calibration.py --mode images

运行：
    python tools/prepare_calib_frames.py --count 16
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE


def detect(gray, pattern):
    ret, c = cv2.findChessboardCorners(gray, pattern, FLAGS)
    if ret:
        return c
    ret, c = cv2.findChessboardCornersSB(gray, pattern, None)
    return c if ret else None


def feature(points, w, h):
    """用角点云的形状描述视角，用于挑选差异大的帧。"""
    p = points.reshape(-1, 2)
    return np.array([
        p[:, 0].mean() / w, p[:, 1].mean() / h,
        (p[:, 0].max() - p[:, 0].min()) / w,
        (p[:, 1].max() - p[:, 1].min()) / h,
        (p[0, 0] - p[-1, 0]) / w,
        (p[0, 1] - p[-1, 1]) / h,
    ], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(config.MVS_CALIB_VIDEO))
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--step", type=int, default=3, help="每隔几帧检测一次")
    ap.add_argument("--out", default=str(config.CALIB_IMAGE_DIR))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开棋盘格视频: {args.video}")
    pattern = (config.CHESSBOARD_COLS, config.CHESSBOARD_ROWS)
    cands = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect(gray, pattern)
            if corners is not None:
                h, w = gray.shape[:2]
                cands.append((idx, frame.copy(), feature(corners, w, h)))
        idx += 1
    cap.release()
    print(f"检测到角点的候选帧: {len(cands)} / {idx}")
    if len(cands) < 3:
        raise SystemExit("可用帧过少，请重新采集棋盘格视频")

    center = np.mean(np.stack([c[2] for c in cands]), axis=0)
    first = int(np.argmin([np.linalg.norm(c[2] - center) for c in cands]))
    chosen, remaining = [first], [k for k in range(len(cands)) if k != first]
    while remaining and len(chosen) < args.count:
        best, best_d = None, -1.0
        for k in remaining:
            d = min(np.linalg.norm(cands[k][2] - cands[j][2]) for j in chosen)
            if d > best_d:
                best_d, best = d, k
        chosen.append(best)
        remaining.remove(best)
    chosen.sort(key=lambda k: cands[k][0])

    for j, k in enumerate(chosen):
        fid, frame, _ = cands[k]
        out = os.path.join(args.out, f"mvs_video_{j:02d}_f{fid:03d}.jpg")
        cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"已写入 {len(chosen)} 张多角度标定图 -> {args.out}")
    print("帧号:", [cands[k][0] for k in chosen])


if __name__ == "__main__":
    main()
