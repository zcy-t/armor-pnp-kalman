# 装甲板识别 → PnP 位姿解算 → 卡尔曼滤波（OpenCV 综合作业）

完整流程：

```
相机采集 → 装甲板识别 → 角点提取 → PnP 位姿解算 → 卡尔曼滤波 → 运动状态估计与位置预测
```

> **全部演示数据均为虚拟机内海康 MVS 采集的真实连续帧，不是模拟数据。**
> 相机：MV-CS016-10UC（USB3.0 工业相机，1440×1080），固定相机、移动装甲板。

---

## 一、作业要求对照

| 作业要求 | 本项目实现 | 产物 |
| --- | --- | --- |
| 1. 相机标定，采集不同位置/角度的棋盘格，保存内参与畸变 | `camera_calibration.py` + 29 张 MVS 实拍标定图（20 张角点有效） | `calibration/camera_params.npz` |
| 2. 装甲板识别与四角点、顺序正确 | `armor_detector.py`：绿色灯条 HSV → 轮廓 → `minAreaRect` → 灯条配对 → LT/RT/RB/LB | `assets/demo_video.mp4`、`assets/demo_*.jpg` |
| 3. 按实际尺寸建模，`solvePnP()` 实时输出 X/Y/Z 与距离 | `pnp_solver.py`：235×55 mm 三维模型点，与 2D 角点严格一一对应 | 每帧 `logs/pnp_log.csv` |
| 4. 卡尔曼滤波 + 短时间运动预测，展示原始/滤波/预测 | `kalman_filter.py`（自实现 6 维匀速模型，另提供 `cv2.KalmanFilter` 版），`viz.py` 趋势图 | `logs/*_compare.png` |
| 5. 实时显示识别结果、三维位置、距离与滤波后状态 | `main.py` 窗口模式 / 无界面录屏模式 | 演示视频与截图 |
| 提交 1：完整代码上传 Git 仓库 | 本仓库 | https://github.com/zcy-t/armor-pnp-kalman |
| 提交 2：运行截图或演示视频 | `assets/demo_video.mp4`（240 帧 / 10 s）、4 张运行截图 | 见 `assets/` |
| 提交 3：改变装甲板位置与距离，对比滤波前后 | 真实片段中装甲板从 656 mm 移近到 473 mm，行程 183 mm | `logs/motion_analysis.png`、`logs/analysis_summary.txt` |

---

## 二、环境

- Python 3.9+（开发验证：Python 3.14 + OpenCV 5.0；虚拟机：Python 3.12 + OpenCV 4.10）
- 依赖：`pip install -r requirements.txt`（opencv-python / numpy / matplotlib）
- 采集端：VMware Ubuntu 虚拟机 + 海康 MVS 客户端 / MVS SDK

```bash
python selftest.py        # 一键自检：卡尔曼、标定、PnP 往返，应输出「全部自检通过」
python run_all.py         # 一键复现：自检 → 标定 → 真实帧全流程 → 演示视频 → 分析图表
```

`run_all.py` 不需要摄像头，直接使用仓库内 `data/` 的 MVS 真实视频即可完整跑通，
满足「程序能够独立运行」。

---

## 三、真实数据是怎么采集的（虚拟机中的 MVS）

1. 相机通过 USB3 直通到 VMware 虚拟机（`/dev/bus/usb/...`，设备 `2bdf:0001 U3V MV-CS016-10UC`）。
2. 在虚拟机内用 **MVS 客户端**采集：
   - `calibration/calib_images/mvs_*.jpg`：13 张棋盘格抓拍（MVS 保存为 BMP，已转 JPG）；
   - `data/calib_mvs.avi`：棋盘格多视角视频，用 `tools/prepare_calib_frames.py` 按视角差异自动挑出 16 帧，共同构成 29 张标定图；
   - `data/armor_mvs.avi`：132 帧装甲板视频（47.67 fps），拍摄时移动装甲板改变位置与距离。
3. 也可用 **MVS SDK** 直接取流（`tools/mvs_capture.py`）：

```bash
# 虚拟机内
sudo mount -t fuse.vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other   # 挂载共享目录
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
python3 tools/mvs_capture.py --frames 150 --out /mnt/hgfs/mvs_shots/armor/Video_live.avi
```

> MVS 图形客户端会独占相机。若报 `0x80000203`（MV_E_ACCESS_DENIED），
> 先关闭 MVS 客户端再采集；完成后用 `/opt/MVS/bin/MVS.sh` 重新启动。
> `assets/mvs_sdk_live_check.jpg` 是本次在虚拟机内通过 MVS SDK 实时取流验证的画面。

---

## 四、分步运行

```bash
# 1) 相机标定（用仓库内 MVS 实拍图，无需相机）
python camera_calibration.py --mode images

# 2) 实时窗口运行（默认读取 data/armor_mvs.avi）
python main.py
# 按键：q 退出 | s 保存截图 | r 重置滤波器

# 3) 连真实相机
python main.py --source 0 --color green
# 或实时抓拍标定：python camera_calibration.py --mode camera --camera 0 --max-frames 20

# 4) 无界面生成演示视频（服务器 / 录屏场景）
python main.py --source data/armor_mvs.avi --no-show --video-out assets/demo_video.mp4

# 5) 生成滤波前后对比图与数值分析
python plot_data.py

# 6) 可选：OpenCV 版卡尔曼；开发期合成数据验证（不用于最终演示）
python main.py --filter opencv --no-show --video-out assets/demo_opencv.mp4
python synthetic_test.py
```

---

## 五、相机标定结果

`calibration/camera_params.npz`（1440×1080，20 张有效视角，重投影 RMS **0.819 px**）：

```
cameraMatrix = [[2805.45,    0.  , 663.17],
                [   0.  , 2811.79, 557.63],
                [   0.  ,    0.  ,   1.  ]]
distCoeffs  = [-0.371, 0.334, 0.0037, 0.009, -7.525]   # k1,k2,p1,p2,k3
```

- 各视图重投影误差 0.479 ~ 1.249 px。
- 标定图与 PnP 视频来自**同一台相机、同一分辨率**，符合「相机标定参数必须与实际相机及图像分辨率对应」。
- `main.py` 通过 `getOptimalNewCameraMatrix` + `initUndistortRectifyMap` 做去畸变，PnP 使用校正后的内参、畸变置零。
- 早期只用 5 张图时得到 `k3≈-133`，去畸变会把画面压缩到角落、识别框跑偏；补足到 20 张视角后畸变系数回归正常，这一步对结果影响很大。

---

## 六、装甲板尺寸与 2D/3D 点对应

`pnp_solver.py` 中三维模型点顺序为 **左上 → 右上 → 右下 → 左下**，与
`armor_detector.py` 输出的 `LT / RT / RB / LB` 严格一一对应：

```
[-117.5, -27.5, 0]  [ 117.5, -27.5, 0]  [ 117.5, 27.5, 0]  [-117.5, 27.5, 0]   (mm)
```

- 实测：两灯条中心距 **235 mm**，灯条长度 **55 mm**（`config.py`）。
- 尺寸校验：真实正面帧中检测出的四边形 2D 宽高比约 **4.05~4.27**，与 235/55 = 4.27 一致；
  用 235×55 建模时 `solvePnP` 的俯仰角约 -9°、重投影误差约 1 px；换成 235×45 时俯仰角变成 -29°、
  重投影误差 14 px，说明尺寸必须与实物一致。
- 误检测过滤：桌面绿色反光会被当成第二条灯条。`main.py::pick_best_armor` 用
  「面积 + PnP 重投影误差 < 10 px」筛选真实装甲板，避免错误位姿污染滤波。

---

## 七、卡尔曼滤波与运动预测

状态 `x = [px, py, pz, vx, vy, vz]^T`，观测 `z = [px, py, pz]^T`（来自每帧真实 `solvePnP` 结果）：

```
预测：x⁻ = F x,   P⁻ = F P Fᵀ + Q        F = [I  dt·I; 0  I]
校正：K = P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹,  x = x⁻ + K(z − Hx⁻),  P = (I−KH)P⁻
外推：p(t+n·dt) = p + v·n·dt
```

- `Q` 采用连续白噪声加速度模型离散化，`R = σ²I`（σ=5 mm），过程噪声加速度 600 mm/s²。
- 检测丢失时只做时间更新（`predict`），滤波器继续给出平滑状态与预测位置；连续丢失超过 30 帧才重置。
- 同时提供 `OpenCVKalman3D`（`--filter opencv`），接口完全一致。

---

## 八、演示视频与运行截图

`assets/demo_video.mp4`：1600×776，24 fps，共 240 帧（约 10 s），由真实 MVS 帧逐帧渲染：

- 片头：相机型号、分辨率、标定 RMS、装甲板尺寸、流程说明；
- 正片（132 帧）：装甲板识别框与 LT/RT/RB/LB 四角点、三维坐标轴、
  `MEAS`（橙，原始测量）/ `FILT`（绿，滤波）/ `PRED`（红，预测）的 XYZ 与距离、
  欧拉角、重投影误差、速度；右侧实时趋势图给出 X/Y/Z/距离四条曲线；
- 片尾：检出率、距离变化范围、滤波前后抖动与下降比例。

> 暗场画面在显示层做了伽马提亮（`DEMO_GAMMA=0.62`），**检测与 PnP 始终使用原始帧**，
> 不影响数据真实性。

运行截图：`assets/demo_00_f026.jpg` ~ `assets/demo_03_f104.jpg`。

---

## 九、结果分析（对应提交要求 3）

`logs/pnp_log.csv` 记录了 132 帧的真实解算数据，`logs/analysis_summary.txt` 为数值结论：

| 指标 | 数值 |
| --- | --- |
| 处理帧数 / 有效 PnP 解算帧 | 132 / **86**（65.2%） |
| 装甲板距离变化 | **473.1 ~ 656.3 mm**，行程 **183.2 mm** |
| 平均重投影误差 | 3.70 px |
| 距离帧间抖动 RMS | 测量 **11.68 mm** → 滤波 **7.23 mm**，下降 **38.1%** |
| Z 帧间抖动 RMS | 11.63 mm → 7.28 mm，下降 37.4% |
| Y 帧间抖动 RMS | 18.22 mm → 4.76 mm，下降 73.9% |

**改变装甲板位置/距离时的观察：**

1. 装甲板由远及近移动时，`meas_z` / `meas_dist` 从约 656 mm 连续下降到约 473 mm，
   X/Y 也随横向移动变化；`solvePnP` 输出的欧拉角与速度同步变化，说明位姿解算确实跟随真实运动。
2. 原始测量曲线存在明显的逐帧毛刺（灯条角点抖动、局部反光）；滤波曲线明显平滑，
   并且在装甲板部分移出画面、灯条只剩一条导致检测失败的区间（图中灰色带），
   滤波器依靠时间更新继续给出位置与速度，重新捕获目标后再快速收敛——这正是卡尔曼滤波的价值。
3. 预测曲线（红）始终领先滤波曲线一小段，体现基于估计速度的短时运动预测。

图表：`logs/position_compare.png`、`logs/distance_compare.png`、
`logs/jitter_compare.png`、`logs/motion_analysis.png`。

---

## 十、注意事项落实情况

1. ✅ 标定参数与实际相机（MV-CS016-10UC）、分辨率（1440×1080）一一对应。
2. ✅ 2D 角点与 3D 模型点顺序严格对应，装甲板尺寸按实测 235×55 mm 设置。
3. ✅ 卡尔曼滤波使用连续帧的真实 `solvePnP` 结果；`synthetic_test.py` 仅用于开发期算法验证，
   `logs/synthetic_log.csv` / `logs/synthetic_compare.png` 不参与最终演示。
4. ✅ 固定相机、移动装甲板，符合「以固定相机为基础进行运动状态估计」。
5. ✅ 程序可独立运行：`run_all.py` 无需摄像头即可复现全部产物；各模块均有注释说明作用。

---

## 十一、文件清单

```
config.py                 全局配置（真实相机/标定板/装甲板参数）
camera_calibration.py     模块1：相机标定（文件夹 / 实时抓拍 / SB 角点回退）
armor_detector.py         模块2：绿色灯条识别、灯条配对、四角点排序（含 YOLO 适配器）
pnp_solver.py             模块3：solvePnP、距离、欧拉角、重投影误差、三维坐标轴
kalman_filter.py          模块4：自实现卡尔曼 + cv2.KalmanFilter 两版本
viz.py                    OpenCV 实时趋势曲线
main.py                   主程序：实时窗口 / 无界面演示视频、CSV 记录
plot_data.py              对比图、抖动统计、位置变化分析
run_all.py                一键复现整份作业
selftest.py               算法自检
synthetic_test.py         开发期合成数据端到端测试（非最终演示）
calibration/calib_images/ 29 张 MVS 实拍标定图
calibration/camera_params.npz  相机内参、畸变系数、RMS
data/armor_mvs.avi        真实装甲板连续帧（移动/变距离）
data/calib_mvs.avi        真实棋盘格多视角视频
assets/demo_video.mp4     演示视频
assets/demo_*.jpg         运行截图
assets/mvs_sdk_live_check.jpg  虚拟机 MVS SDK 实时取流验证
logs/pnp_log.csv          每帧真实解算数据
logs/*_compare.png        滤波前后对比图
logs/motion_analysis.png  位置/距离变化分析
logs/analysis_summary.txt 数值结论
tools/mvs_capture.py      虚拟机内 MVS SDK 采集脚本
tools/prepare_calib_frames.py  从棋盘格视频挑选多角度标定帧
```

---

## 十二、常见问题

- **画面整体发黑、看不到装甲板**：检查 MVS 曝光/增益；演示层可用 `--gamma 0.6` 提亮。
- **识别框跑到画面角落**：多为标定文件分辨率不匹配或畸变系数异常，重新跑
  `python camera_calibration.py --mode images`，或临时用 `--no-undistort` 对比。
- **误检测 / 位姿跳变**：调小 `--max-reproj`；或调整 `config.py` 中 `HSV_RANGE['green']`
  与灯条配对阈值，排除桌面反光。
- **虚拟机里打不开相机**：MVS 客户端独占设备，关闭后再用 `tools/mvs_capture.py` 采集。
