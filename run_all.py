# -*- coding: utf-8 -*-
"""
一键复现整份作业（不需要相机，使用仓库内的 MVS 真实采集数据）：

    python run_all.py

依次执行：
    1. selftest.py                     算法自检
    2. camera_calibration.py --mode images   用 29 张 MVS 实拍图标定
    3. main.py --no-show --video-out ...    真实连续帧跑 PnP + 卡尔曼并录演示视频
    4. plot_data.py                     生成滤波前后对比图与数值结论

如需连真实相机实时运行：python main.py --source 0 --color green
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("算法自检", [sys.executable, "selftest.py"]),
    ("相机标定（MVS 实拍图）",
     [sys.executable, "camera_calibration.py", "--mode", "images"]),
    ("真实帧 PnP + 卡尔曼 + 演示视频",
     [sys.executable, "main.py", "--source", "data/armor_mvs.avi",
      "--no-show", "--video-out", "assets/demo_video.mp4"]),
    ("结果分析与对比图", [sys.executable, "plot_data.py"]),
]


def main():
    for name, cmd in STEPS:
        print("\n" + "=" * 70)
        print(f">>> {name}")
        print("=" * 70, flush=True)
        ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if ret != 0:
            print(f"[FAIL] {name} 退出码 {ret}，流水线中止。")
            sys.exit(ret)
    print("\n" + "=" * 70)
    print("全部完成。交付物：")
    print("  assets/demo_video.mp4      演示视频")
    print("  assets/demo_*.jpg          运行截图")
    print("  logs/pnp_log.csv           每帧真实解算数据")
    print("  logs/*_compare.png         滤波前后对比图")
    print("  logs/motion_analysis.png   位置/距离变化分析")
    print("  logs/analysis_summary.txt  数值结论")
    print("  calibration/camera_params.npz  相机内参与畸变系数")
    print("=" * 70)


if __name__ == "__main__":
    main()
