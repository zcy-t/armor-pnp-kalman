# -*- coding: utf-8 -*-
"""
数据分析脚本：读取 main.py 记录的 CSV，
绘制 原始测量 / 滤波 / 预测 对比图，并量化滤波前后的抖动与离散程度。
运行：python plot_data.py
产物：logs/ 下的 position_compare.png、distance_compare.png、jitter_compare.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(csv_path):
    return np.genfromtxt(csv_path, delimiter=",", names=True,
                         dtype=None, encoding="utf-8-sig")


def _series(data, col):
    return data[col].astype(np.float64)


def plot_positions(data, out):
    t = data["timestamp"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    comps = [("X", "meas_x", "filt_x", "pred_x"),
             ("Y", "meas_y", "filt_y", "pred_y"),
             ("Z", "meas_z", "filt_z", "pred_z")]
    for ax, (name, m, f, p) in zip(axes, comps):
        ax.plot(t, _series(data, m), color="#ff8c00", lw=1.2, label="measurement")
        ax.plot(t, _series(data, f), color="#00a000", lw=1.6, label="filtered")
        ax.plot(t, _series(data, p), color="#d62728", lw=1.0, ls="--", label="predicted")
        ax.set_ylabel(f"{name} (mm)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title("3D Position: Measurement vs Filtered vs Predicted")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distance(data, out):
    t = data["timestamp"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    md = _series(data, "meas_dist")
    fd = _series(data, "filt_dist")
    # 预测距离由 pred xyz 计算
    px, py, pz = _series(data, "pred_x"), _series(data, "pred_y"), _series(data, "pred_z")
    pd_ = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    ax.plot(t, md, color="#ff8c00", lw=1.2, label="measurement")
    ax.plot(t, fd, color="#00a000", lw=1.6, label="filtered")
    ax.plot(t, pd_, color="#d62728", lw=1.0, ls="--", label="predicted")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (mm)")
    ax.set_title("Armor Center Distance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_jitter(data, out):
    """相邻帧变化量（逐帧抖动）：滤波后应明显更集中在 0 附近。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (m, f, title) in zip(axes, [
        ("meas_dist", "filt_dist", "distance frame-to-frame delta (mm)"),
        ("meas_z", "filt_z", "Z frame-to-frame delta (mm)")]):
        dm = np.diff(_series(data, m))
        df = np.diff(_series(data, f))
        dm = dm[~np.isnan(dm)]
        df = df[~np.isnan(df)]
        ax.hist(dm, bins=30, alpha=0.55, color="#ff8c00", label="measurement")
        ax.hist(df, bins=30, alpha=0.55, color="#00a000", label="filtered")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def report_stats(data):
    """打印滤波前后标准差与帧间抖动 RMS。"""
    print("=" * 66)
    print(f"{'channel':<8}{'meas_std':>12}{'filt_std':>12}"
          f"{'meas_jitter_RMS':>18}{'filt_jitter_RMS':>18}")
    print("-" * 66)
    for name, m, f in [
        ("X", "meas_x", "filt_x"), ("Y", "meas_y", "filt_y"),
        ("Z", "meas_z", "filt_z"), ("DIST", "meas_dist", "filt_dist")]:
        mv, fv = _series(data, m), _series(data, f)
        mv, fv = mv[~np.isnan(mv)], fv[~np.isnan(fv)]
        mj = np.sqrt(np.mean(np.diff(mv) ** 2)) if len(mv) > 1 else np.nan
        fj = np.sqrt(np.mean(np.diff(fv) ** 2)) if len(fv) > 1 else np.nan
        print(f"{name:<8}{mv.std():>12.2f}{fv.std():>12.2f}{mj:>18.2f}{fj:>18.2f}")
    print("=" * 66)
    print("说明：std 衡量整体离散度，jitter_RMS 衡量相邻帧抖动；"
          "滤波后两项均应下降。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.CSV_PATH))
    args = ap.parse_args()
    data = load(args.csv)

    out1 = config.LOG_DIR / "position_compare.png"
    out2 = config.LOG_DIR / "distance_compare.png"
    out3 = config.LOG_DIR / "jitter_compare.png"
    plot_positions(data, out1)
    plot_distance(data, out2)
    plot_jitter(data, out3)
    report_stats(data)
    print("图表已保存：")
    for o in (out1, out2, out3):
        print(" -", o)


if __name__ == "__main__":
    main()
