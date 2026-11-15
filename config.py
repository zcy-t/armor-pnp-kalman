# -*- coding: utf-8 -*-
"""
全局配置文件
集中管理路径、相机、棋盘格标定板、装甲板物理尺寸、灯条筛选、PnP 与卡尔曼滤波参数。

本项目的真实数据全部由虚拟机中的海康 MVS 客户端 / MVS SDK 采集：
    相机型号   : MV-CS016-10UC（USB3.0 彩色工业相机）
    采集分辨率 : 1440 x 1080
    标定板     : 7x10 格棋盘格，内角点 (6, 9)，单格边长 26 mm
    装甲板     : 绿色灯条大装甲板，灯条中心距 235 mm，灯条长度 55 mm

如更换相机/标定板/装甲板，请重点核对：
    1. SQUARE_SIZE_MM  —— 棋盘格单个方格的实际边长（用尺子量）
    2. ARMOR_WIDTH_MM / ARMOR_HEIGHT_MM —— 装甲板实际尺寸（用尺子量）
    3. CHESSBOARD_COLS / CHESSBOARD_ROWS —— 棋盘格内角点数量
    4. FRAME_WIDTH / FRAME_HEIGHT —— 必须与采集分辨率一致
"""
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_DIR = PROJECT_ROOT / "calibration"
CALIB_IMAGE_DIR = CALIB_DIR / "calib_images"          # 批量标定图片存放目录
CALIB_SNAPSHOT_DIR = CALIB_DIR / "snapshots"          # 实时抓拍图片存放目录
CALIB_PARAM_PATH = CALIB_DIR / "camera_params.npz"    # 标定结果文件
LOG_DIR = PROJECT_ROOT / "logs"                       # CSV 数据与分析图表
ASSET_DIR = PROJECT_ROOT / "assets"                   # 运行截图、演示视频
DATA_DIR = PROJECT_ROOT / "data"                      # MVS 采集的原始视频
TOOLS_DIR = PROJECT_ROOT / "tools"                    # 辅助脚本

for _d in (CALIB_IMAGE_DIR, CALIB_SNAPSHOT_DIR, LOG_DIR, ASSET_DIR,
           DATA_DIR, TOOLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 相机
CAMERA_MODEL = "Hikrobot MV-CS016-10UC (MVS, VMware Ubuntu)"
CAMERA_ID = 0
FRAME_WIDTH = 1440             # 必须与 MVS 采集分辨率一致
FRAME_HEIGHT = 1080
CAMERA_FPS = 47.67             # MVS 录制视频的实际帧率

# MVS 原始素材（真实连续帧，非模拟数据）
MVS_ARMOR_VIDEO = DATA_DIR / "armor_mvs.avi"        # 装甲板移动/变距离
MVS_CALIB_VIDEO = DATA_DIR / "calib_mvs.avi"        # 棋盘格多视角视频

# ---------------------------------------------------------------- 棋盘格标定
# 内角点 = (格子列数 - 1, 格子行数 - 1)
CHESSBOARD_COLS = 6
CHESSBOARD_ROWS = 9
SQUARE_SIZE_MM = 26.0          # 实测：单个方格边长 26 mm
CALIB_FLAGS = None             # 自适应阈值等标志，None 表示默认

# ---------------------------------------------------------------- 装甲板物理尺寸 (mm)
# 由真实正面帧的 2D 宽高比（约 4.25）反推 + 实测确认：
# 两灯条中心距 235 mm，灯条长度 55 mm（RoboMaster 大装甲板标准尺寸附近）
# 三维模型点与图像角点 lt/rt/rb/lb 一一对应，尺寸错误会直接让 Z 与姿态失真
ARMOR_WIDTH_MM = 235.0
ARMOR_HEIGHT_MM = 55.0
ARMOR_COLOR = "green"          # "red" / "blue" / "green"

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
DEMO_VIDEO_PATH = ASSET_DIR / "demo_video.mp4"

# 演示视频版式（画面 + 趋势图 + 标题栏）
DEMO_FRAME_W = 960
DEMO_FRAME_H = 720
DEMO_TREND_W = 640
DEMO_HEADER_H = 56
DEMO_GAMMA = 0.62              # 仅用于显示的伽马提亮；检测始终使用原始帧
DEMO_FPS = 24.0                # 输出帧率（把 2.8s 原始片段放慢便于观察）

# 颜色 (BGR)
COLOR_MEASURE = (0, 165, 255)   # 原始测量：橙色
COLOR_FILTER = (0, 255, 0)      # 滤波结果：绿色
COLOR_PREDICT = (0, 0, 255)     # 预测结果：红色
COLOR_CORNER = (255, 200, 0)    # 角点：蓝色
