# Piper Diffusion Policy 数据采集工具

这个仓库用于 Piper 机械臂真机数据采集，当前面向 push-T 类平面操作任务。采集器支持双相机预览、初始工作平面标定、XY 遥控、轨迹删除、轨迹回放，以及多格式数据保存。

核心训练定义：

```text
observation = wrist image + global image + 当前 6 关节角 qpos
action      = 下一步 6 关节角 qpos[t+1]
```

采集时人工控制方式仍然是末端执行器在固定工作平面内做 XY 运动；HDF5 主训练 action 保存的是实际反馈到的下一步关节角。

## 目录结构

```text
.
├── tools/
│   ├── initial_pose_tuner.py       # 初始姿态和工作平面调节工具
│   └── diffusion_xy_collector.py   # 数据采集 GUI 启动入口
├── piper_diffusion_collect/
│   ├── camera.py                   # OpenCV 相机采集
│   ├── robot.py                    # Piper SDK 封装
│   ├── episode_writer.py           # HDF5 / video / images 多格式写入
│   ├── xy_collector_window.py      # 采集 GUI
│   └── cli.py                      # 命令行参数
├── docs/
│   └── piper_diffusion_xy_schema.md # HDF5 schema 说明
└── piper_sdk/                      # 本地 Piper SDK
```

## 环境准备

推荐使用 conda 环境 `datacollect`：

```bash
conda activate datacollect
python -m pip install numpy h5py pyqt5 opencv-python
```

克隆仓库后需要拉取 Piper SDK 子模块：

```bash
git submodule update --init --recursive
```

不激活环境时也可以使用：

```bash
conda run -n datacollect python ...
```

## 硬件准备

默认配置：

```text
CAN: can0
wrist camera: 10
global camera: 4
dataset hz: 10
command hz: 30
```

激活 CAN：

```bash
sudo bash piper_sdk/piper_sdk/can_activate.sh can0 1000000
```

检查相机：

```bash
v4l2-ctl --list-devices
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --list-cameras \
  --list-camera-max-index 20
```

如果使用非官方 CAN 设备，运行 GUI 时通常需要加：

```bash
--no-can-judge
```

## 1. 标定初始姿态

先用 `initial_pose_tuner.py` 找到 push-T 工作平面和每个 episode 的 reset 姿态：

```bash
conda run -n datacollect python tools/initial_pose_tuner.py \
  --can can0 \
  --camera-wrist 10 \
  --camera-global 4 \
  --connect \
  --no-can-judge
```

流程：

1. 点击 `Connect`
2. 点击 `Enable`
3. 点击 `Read Current`
4. 用关节滑条调整末端到合适工作平面
5. 观察腕部相机和全局相机
6. 点击 `Save Pose`

建议保存到：

```text
initial_pose/initial_pose.json
```

该 JSON 记录：

- reset 用的 6 个关节角
- 当前末端位姿 `[x, y, z, rx, ry, rz]`
- 固定工作平面的 `Z/RX/RY/RZ`
- 采集时使用的相机编号和 CAN 配置

## 2. 采集数据

基础 HDF5 采集命令：

```bash
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --can can0 \
  --camera-wrist 10 \
  --camera-global 4 \
  --initial-pose-json initial_pose/initial_pose.json \
  --output-dir data/piper_xy \
  --connect \
  --no-can-judge
```

完整常用命令：

```bash
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --can can0 \
  --camera-wrist 10 \
  --camera-global 4 \
  --initial-pose-json initial_pose/initial_pose.json \
  --output-dir data/piper_xy \
  --hz 10 \
  --command-hz 30 \
  --xy-speed-mm-s 20 \
  --speed-percent 20 \
  --reset-duration 5 \
  --reset-hz 20 \
  --replay-reset-duration 3 \
  --replay-hz 60 \
  --disable-delay 10 \
  --connect \
  --no-can-judge
```

GUI 流程：

1. 点击 `Enable`
2. 点击 `Reset From JSON`
3. 等待 reset 完成后点击 `Read Current Pose`
4. 点击 `Start Episode`
5. 用 `W/A/S/D`、方向键或 `XY Control` 遥感圆盘控制末端 XY 运动
6. 点击 `Stop Episode` 保存
7. 轨迹录坏时点击 `Discard Episode` 删除当前或最近保存的 episode
8. 点击 `Load Replay` 选择 HDF5，再点击 `Start Replay` 回放轨迹
9. 下一个 episode 前再次 `Reset From JSON`
10. 结束时点击 `Disable After Delay`

右侧相机预览会旋转 180 度，方便人按第一视角操作。保存到数据集里的原始图像不旋转。

## 3. 多格式保存

默认只保存 HDF5：

```bash
--save-format hdf5
```

同时保存 HDF5 和视频：

```bash
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --initial-pose-json initial_pose/initial_pose.json \
  --save-format hdf5 \
  --save-format video \
  --connect \
  --no-can-judge
```

同时保存 HDF5、视频和图片序列：

```bash
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --initial-pose-json initial_pose/initial_pose.json \
  --save-format hdf5 \
  --save-format video \
  --save-format images \
  --image-format png \
  --video-codec mp4v \
  --connect \
  --no-can-judge
```

只保存视频和图片，不保存 HDF5：

```bash
conda run -n datacollect python tools/diffusion_xy_collector.py \
  --initial-pose-json initial_pose/initial_pose.json \
  --save-format video \
  --save-format images \
  --connect \
  --no-can-judge
```

注意：GUI 的 `Load Replay` / `Start Replay` 需要读取 HDF5 里的
`observations/qpos`。如果后面还要回放轨迹，至少保留 `--save-format hdf5`。

多格式输出结构示例：

```text
data/piper_xy/
├── episode_20260524_120000_0000.hdf5
└── episode_20260524_120000_0000/
    ├── metadata.json
    ├── frames.csv
    ├── videos/
    │   ├── wrist.mp4
    │   └── global.mp4
    └── images/
        ├── wrist/
        │   ├── 000000.png
        │   └── ...
        └── global/
            ├── 000000.png
            └── ...
```

`frames.csv` 保存每个数据步的：

- 主时间戳
- 机器人反馈时间戳
- 两路相机时间戳
- 相机帧 age
- 有效性 mask
- 图片文件路径
- 当前 qpos
- 当前末端位姿
- 当前 XY 控制方向

## 4. 数据格式

HDF5 核心字段：

```text
observations/images/wrist    uint8   [T, H, W, 3]
observations/images/global   uint8   [T, H, W, 3]
observations/qpos            float32 [T, 6]
observations/eef_pose        float32 [T, 6]
action                       float32 [T, 6]
actual_action                float32 [T, 6]
valid/action                 bool    [T]
time/timestamp_ns            int64   [T]
```

关键对齐关系：

```text
action[t] = observations/qpos[t + 1]
```

最后一帧没有下一步反馈，因此：

```text
valid/action[-1] = false
```

完整 schema 见：

```text
docs/piper_diffusion_xy_schema.md
```

## 5. 回放

回放读取 HDF5 里的：

```text
observations/qpos
```

流程：

1. `Load Replay` 选择 episode HDF5
2. `Start Replay`
3. 程序先用 `--replay-reset-duration` 平滑移动到轨迹第一帧关节角
4. 再按原始 `control_hz` 时间轴线性插值，并以 `--replay-hz` 高频发送 JointCtrl

默认：

```text
--replay-reset-duration 3
--replay-hz 60
```

## 6. 数据审计

已采集数据可以用简单脚本检查：

```bash
conda run -n datacollect python -c "import h5py, pathlib; print(len(list(pathlib.Path('data/piper_xy').glob('*.hdf5'))))"
```

建议重点检查：

- `control_hz` 是否为 10
- `valid/wrist_camera`、`valid/global_camera`、`valid/robot_feedback` 是否全 true
- `valid/action` 是否为 `T-1`
- `action[t]` 是否等于 `observations/qpos[t+1]`
- `time/timestamp_ns` 是否单调递增

## 7. 上传仓库前

本仓库的 `.gitignore` 默认忽略：

- `data/`
- `initial_pose/*.json`
- `__pycache__/`
- 日志、临时文件、视频和压缩包

不要把真机采集数据、私有初始姿态和本地设备路径直接提交到公开仓库。需要发布数据时，建议单独做数据集 release、使用对象存储，或上传到 Hugging Face Dataset。

初始化仓库示例：

```bash
git init
git submodule add https://github.com/agilexrobotics/piper_sdk.git piper_sdk
git submodule add https://github.com/agilexrobotics/Piper_sdk_ui.git Piper_sdk_ui
git add README.md docs tools piper_diffusion_collect .gitignore .gitmodules initial_pose/.gitkeep
git commit -m "Initial Piper diffusion data collection tools"
```

当前仓库把 `piper_sdk/` 和 `Piper_sdk_ui/` 作为 Git submodule 管理。克隆后运行
`git submodule update --init --recursive` 即可拉取 SDK 源码。

## 安全提醒

- 第一次运行请降低速度，例如 `--speed-percent 10 --xy-speed-mm-s 5`
- 机械臂运动空间内不要放无关物体、线缆或手
- `Enable` 不会让关节角归零，不要把使能当成 reset
- 每个 episode 前先 `Reset From JSON`
- 结束采集优先使用 `Disable After Delay`
- 异常运动时立即使用 GUI 的 `Emergency Stop` 或实体急停
