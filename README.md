# 装甲板识别 PnP 位姿解算与卡尔曼滤波（OpenCV 综合作业）

本项目使用 OpenCV 实现从二维图像中的装甲板识别，到三维空间定位，再到目标运动状态
估计与预测的完整流程：

```
相机采集 → 装甲板识别 → 角点提取 → PnP 位姿解算 → 卡尔曼滤波 → 运动状态估计与位置预测
```

对应作业四大模块：

| 模块 | 实现文件 | 核心 API |
| --- | --- | --- |
| 1. 相机标定 | `camera_calibration.py` | `findChessboardCorners` / `cornerSubPix` / `calibrateCamera` |
| 2. 装甲板识别与角点获取 | `armor_detector.py` | `inRange` / `findContours` / `minAreaRect` |
| 3. PnP 位姿解算 | `pnp_solver.py` | `solvePnP` / `Rodrigues` / `projectPoints` |
| 4. 卡尔曼滤波与运动预测 | `kalman_filter.py` | 自行实现卡尔曼，另提供 `cv2.KalmanFilter` 版本 |

---

## 一、项目结构

```
armor_pnp_kalman/
├── config.py               # 全局配置（路径、棋盘格、装甲板尺寸、噪声参数）
├── camera_calibration.py   # 模块1：相机标定（文件夹/实时抓拍两种模式）
├── armor_detector.py       # 模块2：颜色灯条法装甲板识别与四角点排序（含 YOLO 适配）
├── pnp_solver.py           # 模块3：PnP 位姿、距离、欧拉角、坐标轴绘制
├── kalman_filter.py        # 模块4：卡尔曼滤波（自实现 + OpenCV 两版本）
├── viz.py                  # 实时趋势曲线（测量/滤波/预测）
├── main.py                 # 主程序，串联完整流程，记录 CSV
├── plot_data.py            # 读取 CSV，生成对比图与统计
├── synthetic_test.py       # 合成数据端到端测试（无摄像头也能验证）
├── selftest.py             # 一键自检（算法正确性）
├── requirements.txt
├── calibration/
│   ├── calib_images/       # 批量标定图片放这里
│   ├── snapshots/          # 实时抓拍图片
│   └── camera_params.npz   # 标定结果（运行后生成）
├── logs/                   # CSV 数据与分析图表
└── assets/                 # 运行截图
```

---

## 二、环境安装

要求 Python 3.9+（已在 Python 3.14 + OpenCV 5.0 验证）。

```bash
pip install -r requirements.txt
# 或
pip install opencv-python numpy matplotlib
```

先运行一键自检，确认环境与算法正常：

```bash
python selftest.py
```

看到 “全部自检通过” 即可继续。

---

## 三、运行步骤

### 第 1 步：核对物理尺寸（重要）

打开 `config.py`，用尺子实测并修改：

- `SQUARE_SIZE_MM`：棋盘格**单个方格**的边长（mm）。
- `ARMOR_WIDTH_MM` / `ARMOR_HEIGHT_MM`：装甲板实际宽高（mm）。
- `CHESSBOARD_COLS / CHESSBOARD_ROWS`：棋盘格内角点，本项目标定板为 `(6, 9)`。

> 内角点 = 格子数 − 1。你上传的标定板为 7 列 × 10 行格子，故内角点 (6, 9)。

### 第 2 步：相机标定

**方式 A —— 文件夹批量标定（推荐）**

把不同位置、角度拍摄的棋盘格图片（建议 15~20 张）放入
`calibration/calib_images/`，然后：

```bash
python camera_calibration.py --mode images
```

**方式 B —— 实时抓拍标定**

```bash
python camera_calibration.py --mode camera --camera 0 --max-frames 20
# 空格抓拍，c 标定，q 退出；尽量覆盖画面四角与不同距离、倾角
```

标定成功后会打印内参矩阵、畸变系数与重投影误差，并保存到
`calibration/camera_params.npz`。重投影 RMS 误差一般应小于 0.5~1 px。

### 第 3 步：运行主程序

```bash
python main.py --source 0 --color red                 # 摄像头，红色装甲板
python main.py --source 0 --color blue               # 蓝色装甲板
python main.py --source test.mp4                     # 视频文件
python main.py --source photo.jpg                    # 单张图片
python main.py --filter opencv                       # 使用 OpenCV 版卡尔曼
```

运行时会弹出两个窗口：

1. 主画面：装甲板框、四角点（LT/RT/RB/LB）、三维坐标轴、测量/滤波/预测的
   XYZ 与距离、欧拉角、速度、帧率。
2. 趋势图：X、Y、Z、距离四个通道下，原始测量（橙）、滤波（绿）、预测（红）。

按键：`q` 退出 | `s` 保存截图 | `r` 重置滤波器。数据同时写入 `logs/pnp_log.csv`。

### 第 4 步：数据分析与对比

```bash
python plot_data.py
```

生成 `logs/` 下的 `position_compare.png`、`distance_compare.png`、
`jitter_compare.png`，并打印滤波前后的标准差与帧间抖动 RMS。

---

## 四、各模块原理

### 1. 相机标定

针孔相机模型将三维点投影到像素平面：

```
s · [u v 1]^T = K · [R | t] · [X Y Z 1]^T
        ┌ fx  0  cx ┐
   K =  │ 0  fy  cy │      （内参矩阵）
        └ 0   0   1 ┘
```

真实镜头存在径向畸变（k1, k2, k3）和切向畸变（p1, p2）。对棋盘格每个内角点，
其三维坐标由方格尺寸精确已知，二维坐标由图像检测并做亚像素细化，
`calibrateCamera` 通过最小化重投影误差，一次性求出内参 K 与畸变系数 D。
本项目还会逐图计算重投影误差，用于剔除质量差的标定图。

### 2. 装甲板识别与角点获取

采用不依赖额外模型的传统视觉流程（可独立运行）：

1. 转 HSV，按红/蓝色相阈值二值化（红色跨 0 度需两段合并），形态学闭运算连接断裂。
2. `findContours` 提取轮廓，`minAreaRect` 拟合旋转矩形，按细长比、面积、
   与竖直方向夹角筛选出灯条。
3. 灯条两两配对：长度差、倾斜角差、中心连线水平角、装甲板宽高比均满足条件，
   多目标时用评分贪心并保证每个灯条只使用一次。
4. 由左右灯条的上下端点，按**固定顺序**输出四角点：

```
左上 LT → 右上 RT → 右下 RB → 左下 LB
```

若需复用课堂 YOLO 模型，可使用 `armor_detector.py` 中的 `YoloArmorDetector`
（接口相同，需安装 `ultralytics` 并提供权重）。

### 3. PnP 位姿解算

以装甲板中心为原点、灯条所在平面为 z=0 建立物体坐标系，四个三维模型点与上面的
二维角点严格一一对应：

```
LT(-w/2,-h/2,0)  RT(w/2,-h/2,0)  RB(w/2,h/2,0)  LB(-w/2,h/2,0)
```

`solvePnP` 据此求出旋转向量 `rvec` 和平移向量 `tvec`：

- `tvec` 即装甲板中心在相机坐标系下的 **X、Y、Z**（单位 mm），Z 沿光轴方向。
- 中心距离 `distance = ‖tvec‖₂`。
- `Rodrigues` 把 rvec 转旋转矩阵，再分解出 roll/pitch/yaw 欧拉角。
- `projectPoints` 重投影并计算像素误差，用于自检。

相机坐标系：X 向右、Y 向下、Z 沿光轴向前。

### 4. 卡尔曼滤波与运动预测

采用匀速运动模型，状态 6 维、观测 3 维：

```
x = [px, py, pz, vx, vy, vz]^T        z = [px, py, pz]^T
```

每个周期执行标准卡尔曼五步：

```
预测:  x⁻ = F·x        P⁻ = F·P·Fᵀ + Q
校正:  y = z − H·x⁻     S = H·P⁻·Hᵀ + R
       K = P⁻·Hᵀ·S⁻¹    x = x⁻ + K·y     P = (I − K·H)·P⁻
```

- 状态转移矩阵 F 含 `F[:3,3:] = dt·I`，体现“位置 = 位置 + 速度·dt”。
- 过程噪声 Q 使用连续白噪声加速度模型离散化，`KF_ACCEL_NOISE` 越大越能跟上机动。
- 测量噪声 R 由 `KF_MEASURE_NOISE` 决定，越大越平滑。
- **运动预测**：以当前位置、速度匀速外推 N 帧：`x_pred = x + v·(N·dt)`。
- **目标丢失**：只做预测不校正，P 持续增大表示不确定度上升；丢失超过
  `MAX_LOST_FRAMES` 帧自动重置。

`kalman_filter.py` 同时提供基于 `cv2.KalmanFilter` 的 `OpenCVKalman3D`，
主程序加 `--filter opencv` 即可切换，两者结果一致。

---

## 五、滤波效果（合成数据验证）

无摄像头时可运行端到端合成测试：

```bash
python synthetic_test.py
```

该测试用已知虚拟内参和一条已知三维轨迹（含正弦机动 + 匀速靠近 + 像素级检测噪声）
生成红色灯条图像，再走完整流水线并与真值对比。一次典型结果：

- 相对真值 RMSE：测量 20.9 mm → 滤波 17.6 mm，降低约 15%。
- 距离/ Z 的帧间抖动 RMS：约 34 → 10.6，**抖动降低约 69%**。
- 滤波曲线在紧贴真值的同时显著更平滑；预测线在近帧较准，拐点处匀速外推会偏离，
  符合预期。

> 注意：作业要求最终演示必须使用真实相机的连续帧数据，合成测试仅用于开发验证。
> 含真实运动的通道，其整体标准差主要反映运动本身，应重点看 **jitter_RMS（帧间抖动）**
> 是否下降，而非整体 std。

---

## 六、作业提交指引

1. **上传 Git 仓库**

```bash
git init
git add .
git commit -m "feat: OpenCV 装甲板 PnP 位姿解算与卡尔曼滤波"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

2. **运行截图 / 演示视频**：运行 `main.py`，用 `s` 键截图，或录屏展示装甲板
   识别框、四角点、三维坐标轴以及测量/滤波/预测数据。
3. **改变位置与距离测试**：移动装甲板到不同 X/Y 位置、改变远近，观察 XYZ 与
   距离变化；运行 `plot_data.py` 对比滤波前后曲线与 jitter_RMS。

---

## 七、常见问题

- **找不到标定参数**：主程序会用近似内参并警告，仅用于流程验证，请先完成标定。
- **检测不到装甲板**：调整 `config.py` 中 `LIGHT_MIN_AREA`、`LIGHT_MIN_RATIO`
  以及 `armor_detector.py` 顶部 HSV 阈值；保证光照均匀、灯条完整入镜。
- **滤波太滞后 / 太抖**：滞后则调大 `KF_ACCEL_NOISE` 或调小 `KF_MEASURE_NOISE`；
  太抖则反向调整。
- **距离整体偏差**：核对 `ARMOR_WIDTH_MM/ARMOR_HEIGHT_MM` 是否为实测尺寸，
  并确认标定内参与当前分辨率一致。
