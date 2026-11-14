# -*- coding: utf-8 -*-
"""
全局配置文件
集中管理路径、相机、棋盘格标定板、装甲板物理尺寸、灯条筛选、PnP 与卡尔曼滤波参数。
运行前请重点核对：
    1. SQUARE_SIZE_MM  —— 棋盘格单个方格的实际边长（用尺子量）
    2. ARMOR_WIDTH_MM / ARMOR_HEIGHT_MM —— 装甲板实际尺寸（用尺子量）
    3. CHESSBOARD_COLS / CHESSBOARD_ROWS —— 棋盘格内角点数量
"""
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_DIR = PROJECT_ROOT / "calibration"
CALIB_IMAGE_DIR = CALIB_DIR / "calib_images"          # 批量标定图片存放目录
CALIB_SNAPSHOT_DIR = CALIB_DIR / "snapshots"          # 实时抓拍图片存放目录
CALIB_PARAM_PATH = CALIB_DIR / "camera_params.npz"    # 标定结果文件
LOG_DIR = PROJECT_ROOT / "logs"                       # CSV 数据与分析图表
ASSET_DIR = PROJECT_ROOT / "assets"

for _d in (CALIB_IMAGE_DIR, CALIB_SNAPSHOT_DIR, LOG_DIR, ASSET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 相机
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 60

# ---------------------------------------------------------------- 棋盘格标定
# 内角点 = (格子列数 - 1, 格子行数 - 1)
# 你上传的标定板为 7 列格子 x 10 行格子，故内角点为 (6, 9)
CHESSBOARD_COLS = 6
CHESSBOARD_ROWS = 9
SQUARE_SIZE_MM = 25.0          # TODO: 单个方格实际边长(mm)，请实测后修改
CALIB_FLAGS = None             # 自适应阈值等标志，None 表示默认

# ---------------------------------------------------------------- 装甲板物理尺寸 (mm)
# RoboMaster 常见小装甲板灯条外沿约 135 x 56；大装甲板约 230 x 56
# TODO: 请用尺子实测自己装甲板后修改，PnP 精度直接取决于该尺寸
ARMOR_WIDTH_MM = 135.0
ARMOR_HEIGHT_MM = 56.0
ARMOR_COLOR = "red"            # "red" 或 "blue"

# 灯条轮廓筛选（minAreaRect 拟合结果）
LIGHT_MIN_AREA = 30.0          # 最小面积（像素），分辨率变化时需调整
LIGHT_MIN_RATIO = 1.6          # 灯条长边 / 短边 下限（灯条是细长矩形）
LIGHT_MAX_ANGLE_DEG = 35.0     # 灯条与竖直方向最大夹角（度）

# 灯条配对条件
PAIR_MAX_HIGH_DIFF_RATIO = 0.55   # 两灯条长度差 / 平均长度 上限
PAIR_MAX_CENTER_ANGLE_DEG = 12.0  # 两灯条中心连线与水平方向夹角上限
PAIR_MIN_W_H_RATIO = 1.0          # 装甲板宽 / 灯条长度 范围
PAIR_MAX_W_H_RATIO = 6.0
PAIR_MAX_LIGHT_ANGLE_DIFF = 25.0  # 两灯条自身倾斜角之差上限

# ---------------------------------------------------------------- PnP
# ITERATIVE: 通用迭代法(默认)；IPPE: 平面四点专用，4 点时更稳；EPNP: 至少 4 点
PNP_METHOD = "ITERATIVE"
USE_DISTORTION = True          # PnP 时是否传入畸变系数

# ---------------------------------------------------------------- 卡尔曼滤波
# 状态向量 x = [px, py, pz, vx, vy, vz]^T（6 维），观测 z = [px, py, pz]^T（3 维）
# 过程噪声采用“连续白噪声加速度”模型，只需一个加速度噪声标准差
KF_MEASURE_NOISE = 5.0         # 测量噪声标准差(mm)，越大越平滑、越信任预测
KF_ACCEL_NOISE = 600.0         # 过程噪声加速度标准差(mm/s^2)，越大越能跟上机动
KF_INIT_POS_STD = 10.0         # 初始位置不确定度(mm)
KF_INIT_VEL_STD = 80.0         # 初始速度不确定度(mm/s)
PREDICT_FRAMES = 5             # 向未来预测的帧数（用于运动预测展示）
MAX_LOST_FRAMES = 30           # 连续丢失超过该帧数则重置滤波器

# ---------------------------------------------------------------- 可视化
WINDOW_NAME = "Armor PnP + Kalman (q:quit s:snapshot r:reset)"
PLOT_WINDOW_NAME = "Trend: measurement / filtered / predicted"
PLOT_HISTORY = 150             # 趋势图保留的最近帧数
CSV_PATH = LOG_DIR / "pnp_log.csv"

# 颜色 (BGR)
COLOR_MEASURE = (0, 165, 255)   # 原始测量：橙色
COLOR_FILTER = (0, 255, 0)      # 滤波结果：绿色
COLOR_PREDICT = (0, 0, 255)     # 预测结果：红色
COLOR_CORNER = (255, 200, 0)    # 角点：蓝色
