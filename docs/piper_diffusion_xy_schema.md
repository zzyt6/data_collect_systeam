# Piper Diffusion Joint-Action 数据集格式

Schema version: `piper_diffusion_joint_action_hdf5_v2`

每个 HDF5 文件对应一个 episode。所有 dataset 的第一维都是采样时间步 `T`。
当前格式面向 diffusion policy / ACT 这类模仿学习训练：策略输入 observation 主要是“两路图像 + 当前 6 关节角”，监督目标 action 是“下一步 6 关节角”。

采集界面仍然允许人工用键盘在工作平面内控制末端 XY 运动；这些 XY 命令会作为审计字段保存，但不再作为主训练 `action`。

## 时间对齐

- `observations/*[t]`：在时间步 `t` 采样到的观测。
- `observations/qpos[t]`：当前反馈到的 6 个关节角，作为 proprioception observation。
- `action[t]`：下一步实际反馈到的 6 个关节角，即 `observations/qpos[t + 1]`。
- `actual_action[t]`：与 `action[t]` 含义相同，也是下一步实际关节角，用于明确表示真实执行结果。
- `valid/action[-1]` 和 `valid/actual_action[-1]` 永远是 `false`，因为最后一个时间步在同一个 episode 内没有下一帧反馈可作为监督标签。
- `actions/command_delta_xy_mm[t]`：采样步内累计的人工遥控 XY 命令，只用于审计和调试。
- 如果某一帧相机或机械臂反馈无效，对应的 `valid/*` 会标记为 `false`，下游训练或清洗数据时应该使用这些 mask。

## 数据集分组

不是所有字段都需要喂给 diffusion policy。推荐训练关系是：

```text
observation = images + qpos
action      = next qpos
```

也就是：

```text
observations/images/*[t], observations/qpos[t] -> action[t] = observations/qpos[t+1]
```

### 策略输入 Observation 字段

- `observations/images/wrist`：`uint8 [T, H, W, 3]`，腕部相机 RGB 图像。
- `observations/images/global`：`uint8 [T, H, W, 3]`，全局相机 RGB 图像。
- `observations/qpos`：`float32 [T, 6]`，当前反馈到的 6 个关节角，单位为 rad。

这些是后续训练 diffusion policy 最核心的 observation。也就是说，模型输入通常就是图像和当前关节角。

### 可选 Observation / 调试状态字段

- `observations/eef_pose`：`float32 [T, 6]`，当前反馈到的末端位姿 `[x, y, z, rx, ry, rz]`。`x/y/z` 单位为 mm，`rx/ry/rz` 单位为 deg。
- `observations/state`：`float32 [T, 12]`，`eef_pose + qpos` 拼接后的状态向量。

这两个字段可以用于调试、可视化、质量检查，或者你想让策略额外使用末端位姿时作为输入。但如果目标是“图像 + 六个电机关节角 -> 六个电机关节角”，训练时可以只用 `observations/qpos`，不用 `observations/eef_pose`。

### 训练监督 Action 字段

- `action`：`float32 [T, 6]`，训练监督标签，表示下一步实际反馈到的 6 个关节角，单位为 rad。
- `valid/action`：`bool [T]`，当前时间步的 `action` 是否有效。最后一帧通常无效。

这个字段是 diffusion policy 要学习预测的目标。训练时常见用法是：

```text
obs[t] = wrist_image[t], global_image[t], qpos[t]
target action[t] = qpos[t+1]
```

如果训练 action horizon，可以从连续多帧 `action[t:t+horizon]` 取出未来关节角序列。

### 实际执行反馈字段

- `actual_action`：`float32 [T, 6]`，与 `action` 含义相同，表示下一步实际关节角，单位为 rad。
- `actions/actual_qpos_next_rad`：`float32 [T, 6]`，下一步关节反馈，单位为 rad。
- `actions/actual_joint_delta_rad`：`float32 [T, 6]`，由反馈计算出的关节角变化量，定义为 `qpos[t+1] - qpos[t]`。
- `actions/actual_delta_xy_mm`：`float32 [T, 2]`，由末端反馈计算出的实际 XY 位移。
- `actions/actual_eef_pose_next_mm_deg`：`float32 [T, 6]`，下一步末端反馈。

这些字段主要用于质量检查。例如你可以看 `actions/actual_joint_delta_rad` 是否过大，或者用 `actions/actual_delta_xy_mm` 检查末端是否真的在工作平面内移动。

### 命令审计字段

- `actions/command_delta_xy_mm`：`float32 [T, 2]`，当前采样步内累计的人工遥控 XY 命令，单位为 mm。
- `actions/command_eef_pose_mm_deg`：`float32 [T, 6]`，发送给 Piper 的完整末端目标位姿。
- `actions/command_sdk_units`：`int32 [T, 6]`，完整末端目标位姿对应的 Piper SDK 整数单位。
- `actions/command_count`：`int32 [T]`，当前采样步内累计了多少次高频命令发送。

这些字段不作为主训练 action。它们用于复现和审查人工遥控时到底给机械臂发送了什么末端命令。

### 人工控制调试字段

- `control/key_state`：`bool [T, 4]`，键盘控制状态，顺序为 `w_or_up, a_or_left, s_or_down, d_or_right`。
- `control/xy_direction`：`float32 [T, 2]`，归一化后的 XY 运动方向。

这些字段用于调试人工遥控过程。训练 diffusion policy 时通常不需要把它们作为输入。

## 时间和有效性

- `time/step`：当前时间步编号。
- `time/timestamp_ns`：主循环调度 tick 的单调时钟时间戳。
- `time/robot_timestamp_ns`：读取机械臂反馈时的单调时钟时间戳。
- `time/command_sent_timestamp_ns`：SDK 命令发送完成后的单调时钟时间戳。
- `time/camera_wrist_timestamp_ns`：腕部相机最新帧的单调时钟时间戳。
- `time/camera_global_timestamp_ns`：全局相机最新帧的单调时钟时间戳。
- `time/camera_wrist_age_s`：腕部相机帧相对当前 tick 的年龄，单位为秒。
- `time/camera_global_age_s`：全局相机帧相对当前 tick 的年龄，单位为秒。
- `valid/wrist_camera`：当前时间步是否有有效腕部相机图像。
- `valid/global_camera`：当前时间步是否有有效全局相机图像。
- `valid/robot_feedback`：当前时间步是否有有效机械臂反馈。
- `valid/action`：当前时间步是否有有效训练 action。
- `valid/actual_action`：当前时间步是否已经成功回填实际动作。
- `valid/command_sent`：当前采样步内是否至少有一次 SDK 命令成功返回且没有抛出异常。

## 文件元数据

重要 HDF5 attributes：

- `schema_version`：数据格式版本。
- `created_at`：episode 文件创建时间。
- `control_hz`：HDF5 数据采样频率。
- `command_hz`：高频遥控命令发送频率。它可以高于 `control_hz`，让人工控制更顺滑。
- `dt_seconds`：相邻两个采样时间步之间的理论间隔。
- `command_dt_seconds`：相邻两次高频命令发送之间的理论间隔。
- `alignment`：观测、训练 action 和命令审计字段的对齐说明。
- `action_units`：`action` 和 `actual_action` 的单位说明。
- `eef_pose_units`：末端位姿单位说明。
- `qpos_units`：关节角单位说明。
- `image_encoding`：图像编码说明，目前为 `RGB uint8`。
- `joint_names_json`：关节名称顺序。
- `pose_names_json`：末端位姿字段顺序。
- `key_names_json`：键盘控制字段顺序。
- `config_json`：本次采集启动参数。
- `start_pose_mm_deg_json`：episode 开始时使用的末端目标位姿。

如果启动采集时传入了 `--initial-pose-json`，还会写入：

- `initial_pose_source`：初始姿态 JSON 文件路径。
- `initial_pose_json`：由 `initial_pose_tuner.py` 保存的完整初始姿态文件内容。
- `work_plane_json`：push-T 工作平面定义。当前实现固定 `Z/RX/RY/RZ`，人工遥控只改变 `X/Y`。
- `reset_json`：用于每个 episode 前 reset 的关节初始姿态。

## 推荐训练读取方式

最小训练字段：

```text
observations/images/wrist
observations/images/global
observations/qpos
action
valid/action
```

训练时建议过滤：

```text
valid/wrist_camera == true
valid/global_camera == true
valid/robot_feedback == true
valid/action == true
```

## 推荐采集流程

1. 运行 `tools/initial_pose_tuner.py`，通过滑条调整机械臂，让末端对准 push-T 任务工作平面。
2. 点击 `Save Pose`，保存 `initial_pose.json`。该文件会记录 reset 关节角、末端位姿和工作平面固定分量。
3. 运行 `tools/diffusion_xy_collector.py` 时传入 `--initial-pose-json initial_pose.json`。
4. 每个 episode 前点击 `Reset From JSON`，让机械臂平滑回到相同初始姿态。
5. 开始 episode 后，脚本在工作平面内发送 XY 遥控命令，同时保存图像、当前 qpos、下一步 qpos action 和审计字段。
