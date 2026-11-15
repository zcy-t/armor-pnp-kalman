# -*- coding: utf-8 -*-
"""
数据分析脚本：读取 main.py 在 MVS 真实连续帧上记录的 CSV，
绘制 原始测量 / 滤波 / 预测 对比图，量化滤波前后的抖动与离散程度，
并输出改变装甲板位置/距离后的数据变化分析。

运行：python plot_data.py
产物：
    logs/position_compare.png   X/Y/Z 三通道测量-滤波-预测对比
    logs/distance_compare.png   中心距离对比
    logs/jitter_compare.png     帧间抖动直方图（滤波前后）
    logs/motion_analysis.png    位置/距离变化 + 丢帧时的预测保持
    logs/analysis_summary.txt   数值结论（README 引用）
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
    return np.asarray(data[col], dtype=np.float64)


def _clean(v):
    v = np.asarray(v, dtype=np.float64)
    return v[~np.isnan(v)]


def plot_positions(data, out):
    t = data["timestamp"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    comps = [("X", "meas_x", "filt_x", "pred_x"),
             ("Y", "meas_y", "filt_y", "pred_y"),
             ("Z", "meas_z", "filt_z", "pred_z")]
    for ax, (name, m, f, p) in zip(axes, comps):
        ax.plot(t, _series(data, m), color="#ff8c00", lw=1.2,
                label="measurement (PnP)")
        ax.plot(t, _series(data, f), color="#00a000", lw=1.8,
                label="filtered (Kalman)")
        ax.plot(t, _series(data, p), color="#d62728", lw=1.0, ls="--",
                label="predicted")
        ax.set_ylabel(f"{name} (mm)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title("3D Position: measurement vs filtered vs predicted "
                      "(real MVS frames)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distance(data, out):
    t = data["timestamp"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    md = _series(data, "meas_dist")
    fd = _series(data, "filt_dist")
    px, py, pz = (_series(data, "pred_x"), _series(data, "pred_y"),
                  _series(data, "pred_z"))
    pd_ = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    ax.plot(t, md, color="#ff8c00", lw=1.2, label="measurement")
    ax.plot(t, fd, color="#00a000", lw=1.8, label="filtered")
    ax.plot(t, pd_, color="#d62728", lw=1.0, ls="--", label="predicted")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (mm)")
    ax.set_title("Armor Center Distance (armor moved during capture)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_jitter(data, out):
    """相邻帧变化量（逐帧抖动）直方图：滤波后应更集中在 0 附近。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (m, f, title) in zip(axes, [
        ("meas_dist", "filt_dist", "distance frame-to-frame delta (mm)"),
        ("meas_z", "filt_z", "Z frame-to-frame delta (mm)")]):
        dm = _clean(np.diff(_series(data, m)))
        df = _clean(np.diff(_series(data, f)))
        ax.hist(dm, bins=30, alpha=0.55, color="#ff8c00", label="measurement")
        ax.hist(df, bins=30, alpha=0.55, color="#00a000", label="filtered")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_motion(data, out):
    """展示改变装甲板位置/距离时的数据变化，以及检测丢失期间的预测保持。"""
    t = data["timestamp"]
    md = _series(data, "meas_dist")
    fd = _series(data, "filt_dist")
    det = _series(data, "detected")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(t, md, "o-", ms=3, color="#ff8c00", lw=1.0, label="measurement")
    ax.plot(t, fd, color="#00a000", lw=2.0, label="filtered")
    # 检测丢失区间用灰底标出：此时只有卡尔曼预测在推进
    lost = det == 0
    ax.fill_between(t, 0, 1, where=lost, transform=ax.get_xaxis_transform(),
                    color="#888888", alpha=0.18,
                    label="armor not detected (prediction only)")
    d_clean = _clean(md)
    ax.set_ylabel("distance (mm)")
    ax.set_title(f"Changing armor position / distance: "
                 f"{d_clean.min():.0f} - {d_clean.max():.0f} mm; "
                 f"gray bands = detection gaps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    dm = np.diff(md)
    df = np.diff(fd)
    ax.plot(t[1:], dm, color="#ff8c00", lw=1.0, label="measurement delta")
    ax.plot(t[1:], df, color="#00a000", lw=1.4, label="filtered delta")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frame delta (mm)")
    ax.set_title("Frame-to-frame jitter: before vs after Kalman filter")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def report_stats(data, out_txt):
    lines = ["=" * 72,
             "MVS real-frame PnP + Kalman analysis",
             "=" * 72]
    hdr = (f"{'channel':<8}{'meas_std':>12}{'filt_std':>12}"
           f"{'meas_jitter_RMS':>18}{'filt_jitter_RMS':>18}{'drop':>10}")
    lines.append(hdr)
    lines.append("-" * 72)
    summary = {}
    for name, m, f in [
        ("X", "meas_x", "filt_x"), ("Y", "meas_y", "filt_y"),
        ("Z", "meas_z", "filt_z"), ("DIST", "meas_dist", "filt_dist")]:
        mv, fv = _clean(_series(data, m)), _clean(_series(data, f))
        mj = float(np.sqrt(np.mean(np.diff(mv) ** 2))) if len(mv) > 1 else np.nan
        fj = float(np.sqrt(np.mean(np.diff(fv) ** 2))) if len(fv) > 1 else np.nan
        drop = 100.0 * (1 - fj / mj) if mj > 1e-6 else 0.0
        summary[name] = (float(mv.std()), float(fv.std()), mj, fj, drop)
        lines.append(f"{name:<8}{mv.std():>12.2f}{fv.std():>12.2f}"
                     f"{mj:>18.2f}{fj:>18.2f}{drop:>9.1f}%")
    lines.append("-" * 72)
    md = _clean(_series(data, "meas_dist"))
    det = _series(data, "detected")
    reproj = _clean(_series(data, "reproj_err"))
    lines += [
        f"total frames            : {len(data)}",
        f"frames with PnP solution: {int(det.sum())}  "
        f"({100.0 * det.sum() / len(data):.1f}%)",
        f"distance range (mm)     : {md.min():.1f} ~ {md.max():.1f}",
        f"distance travel (mm)    : {md.max() - md.min():.1f}",
        f"mean reprojection error : {reproj.mean():.2f} px",
        "",
        "Note: measurement jitter is computed per solved frame; the filter",
        "also keeps a smooth state through detection gaps (gray bands).",
        "=" * 72,
    ]
    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.CSV_PATH))
    args = ap.parse_args()
    data = load(args.csv)

    outs = {
        "positions": config.LOG_DIR / "position_compare.png",
        "distance": config.LOG_DIR / "distance_compare.png",
        "jitter": config.LOG_DIR / "jitter_compare.png",
        "motion": config.LOG_DIR / "motion_analysis.png",
        "summary": config.LOG_DIR / "analysis_summary.txt",
    }
    plot_positions(data, outs["positions"])
    plot_distance(data, outs["distance"])
    plot_jitter(data, outs["jitter"])
    plot_motion(data, outs["motion"])
    report_stats(data, outs["summary"])
    print("\n产物：")
    for k, v in outs.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
