# -*- coding: utf-8 -*-
"""
实时数据可视化工具：用 OpenCV 自绘趋势曲线（无需额外 GUI 线程）。
同时展示原始测量(橙)、滤波结果(绿)、预测位置(红)，覆盖 X、Y、Z 与距离四行。
"""
from collections import deque

import cv2
import numpy as np

import config


def _norm(vec):
    return None if vec is None else float(np.linalg.norm(vec))


class TrendPlot:
    def __init__(self, history=config.PLOT_HISTORY, width=800, height=480):
        self.W, self.H = width, height
        self.meas = deque(maxlen=history)
        self.filt = deque(maxlen=history)
        self.pred = deque(maxlen=history)

    def add(self, meas, filt, pred):
        self.meas.append(meas)
        self.filt.append(filt)
        self.pred.append(pred)

    # ------------------------------------------------------------ 单行绘制
    def _draw_row(self, canvas, row_idx, row_h, name, comp):
        y0 = row_idx * row_h
        x_label, x_plot = 70, 84
        plot_w = self.W - x_plot - 10

        def extract(seq):
            out = []
            for v in seq:
                if v is None:
                    out.append(None)
                else:
                    out.append(_norm(v) if comp is None else float(v[comp]))
            return out

        series = [
            (extract(self.meas), config.COLOR_MEASURE, "MEAS"),
            (extract(self.filt), config.COLOR_FILTER, "FILT"),
            (extract(self.pred), config.COLOR_PREDICT, "PRED"),
        ]
        valid = [v for s, _, _ in series for v in s if v is not None]
        cv2.line(canvas, (x_plot, y0 + row_h - 18),
                 (self.W - 10, y0 + row_h - 18), (40, 40, 40), 1)
        cv2.putText(canvas, name, (10, y0 + row_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if not valid:
            return

        vmin, vmax = min(valid), max(valid)
        if vmax - vmin < 1e-6:
            vmax += 1.0
            vmin -= 1.0
        pad = (vmax - vmin) * 0.1
        vmin, vmax = vmin - pad, vmax + pad
        top, bottom = y0 + 8, y0 + row_h - 22

        def y_of(v):
            return int(bottom - (v - vmin) / (vmax - vmin) * (bottom - top))

        n = max(len(self.meas), 1)
        for vals, color, label in series:
            prev = None
            for i, v in enumerate(vals):
                if v is None:
                    prev = None
                    continue
                x = x_plot + int(i / max(n - 1, 1) * plot_w)
                p = (x, y_of(v))
                if prev is not None:
                    cv2.line(canvas, prev, p, color, 2)
                prev = p
        # 最新值文字
        last_filt = series[1][0]
        lv = next((v for v in reversed(last_filt) if v is not None), None)
        if lv is not None:
            cv2.putText(canvas, f"{lv:7.1f}", (self.W - 96, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_FILTER, 1)
        cv2.putText(canvas, f"{vmin:6.0f}", (x_plot - 60, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(canvas, f"{vmax:6.0f}", (x_plot - 60, top + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def render(self):
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.W - 1, self.H - 1), (60, 60, 60), 1)
        rows = [("X", 0), ("Y", 1), ("Z", 2), ("DIST", None)]
        row_h = self.H // len(rows)
        for i, (name, comp) in enumerate(rows):
            self._draw_row(canvas, i, row_h, name, comp)
        cv2.putText(canvas, "MEAS", (100, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, config.COLOR_MEASURE, 1)
        cv2.putText(canvas, "FILT", (170, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, config.COLOR_FILTER, 1)
        cv2.putText(canvas, "PRED", (240, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, config.COLOR_PREDICT, 1)
        return canvas
