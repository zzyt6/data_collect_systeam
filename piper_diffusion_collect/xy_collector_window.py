"""Piper XY 遥控、关节角 action 数据采集的 Qt 图形界面。

这个窗口负责完整采集状态机：连接机械臂、使能机械臂、开始 episode、
按固定定时器采样观测、用更高频率发送 XY 命令，并把下一帧关节角回填成训练 action。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import CameraCapture, parse_camera_id
from .episode_writer import EpisodeWriter
from .qt_widgets import CameraView, JoystickWidget
from .robot import PiperRobot, RobotFeedback, empty_feedback, qpos_rad_to_sdk_units


class DataCollectorWindow(QMainWindow):
    """通过键盘控制末端 XY 运动的交互式采集窗口。"""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.robot: PiperRobot | None = None
        self.connected = False
        self.enabled = False
        self.recording = False
        self.writer: EpisodeWriter | None = None
        self.last_episode_path: Path | None = None
        self.last_episode_output_paths: list[Path] = []
        self.last_episode_steps = 0
        self.replay_qpos_rad: np.ndarray | None = None
        self.replay_control_hz = float(args.hz)
        self.replay_phase = "idle"
        self.replay_start_qpos_rad: np.ndarray | None = None
        self.replay_phase_start_s = 0.0
        self.replay_step_counter = 0
        self.replay_path: Path | None = None
        self.replay_running = False
        self.target_pose: np.ndarray | None = None
        self.last_feedback: RobotFeedback | None = None
        self.pressed_keys: set[int] = set()
        self.joystick_direction = np.zeros(2, dtype=np.float32)
        self.episode_index = 0
        self.pending_command_delta_xy = np.zeros(2, dtype=np.float32)
        self.pending_command_count = 0
        self.last_command_sent_timestamp_ns = 0
        self.last_command_sent_ok = False
        self.reset_in_progress = False
        self.reset_start_qpos_rad: np.ndarray | None = None
        self.reset_target_units: np.ndarray | None = None
        self.reset_target_qpos_rad: np.ndarray | None = None
        self.reset_start_time_s = 0.0
        self.disable_pending = False
        self.disable_deadline_s = 0.0
        self.initial_pose_payload = self._load_initial_pose_payload(args.initial_pose_json)
        self.args.initial_pose_payload = self.initial_pose_payload
        self.reset_completed = self.initial_pose_payload is None
        self.zero_image = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)

        self.setWindowTitle("Piper Diffusion XY Collector")
        self.resize(1280, 820)
        self.setFocusPolicy(Qt.StrongFocus)

        self.cameras = {
            "wrist": CameraCapture(
                "wrist",
                parse_camera_id(args.camera_wrist),
                args.camera_width,
                args.camera_height,
                round(args.hz),
            ),
            "global": CameraCapture(
                "global",
                parse_camera_id(args.camera_global),
                args.camera_width,
                args.camera_height,
                round(args.hz),
            ),
        }

        # tick_timer 是数据集的权威时钟。相机线程和命令定时器独立运行，
        # 每个数据 tick 只取它们最新可用/累计的结果。
        self.tick_timer = QTimer(self)
        self.tick_timer.setTimerType(Qt.PreciseTimer)
        self.tick_timer.setInterval(max(1, round(1000.0 / args.hz)))
        self.tick_timer.timeout.connect(self._tick)

        self.command_timer = QTimer(self)
        self.command_timer.setTimerType(Qt.PreciseTimer)
        self.command_timer.setInterval(max(1, round(1000.0 / args.command_hz)))
        self.command_timer.timeout.connect(self._command_tick)

        self.reset_timer = QTimer(self)
        self.reset_timer.setTimerType(Qt.PreciseTimer)
        self.reset_timer.setInterval(max(1, round(1000.0 / args.reset_hz)))
        self.reset_timer.timeout.connect(self._reset_tick)

        self.disable_timer = QTimer(self)
        self.disable_timer.setInterval(200)
        self.disable_timer.timeout.connect(self._disable_countdown_tick)

        self.replay_timer = QTimer(self)
        self.replay_timer.setTimerType(Qt.PreciseTimer)
        self.replay_timer.setInterval(max(1, round(1000.0 / args.replay_hz)))
        self.replay_timer.timeout.connect(self._replay_tick)

        self._build_ui()
        self._connect_cameras()
        self.tick_timer.start()
        self.command_timer.start()

        if args.connect:
            QTimer.singleShot(100, self.connect_piper)

    def _load_initial_pose_payload(self, path: Path | None) -> dict | None:
        """读取初始姿态/工作平面 JSON，并用于后续 episode 元数据。"""

        if path is None:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Initial pose JSON not loaded",
                f"Cannot read {path}:\n{exc}",
            )
            return None

    def _build_ui(self) -> None:
        """组装左侧控制面板和右侧相机预览区域。"""

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.addWidget(self._build_robot_group())
        control_layout.addWidget(self._build_record_group())
        control_layout.addWidget(self._build_control_hint_group())
        control_layout.addStretch(1)

        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        self.camera_views = {
            "wrist": CameraView("Wrist Camera", rotate_preview_180=True),
            "global": CameraView("Global Camera", rotate_preview_180=True),
        }
        camera_layout.addWidget(self.camera_views["wrist"], stretch=1)
        camera_layout.addWidget(self.camera_views["global"], stretch=1)

        root.addWidget(control_panel, stretch=0)
        root.addWidget(camera_panel, stretch=1)

    def _build_robot_group(self) -> QGroupBox:
        """创建机械臂连接和运动速度控制区。"""

        group = QGroupBox("Robot")
        layout = QGridLayout(group)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_piper)
        self.enable_button = QPushButton("Enable")
        self.enable_button.clicked.connect(self.enable_piper)
        self.disable_button = QPushButton("Disable After Delay")
        self.disable_button.clicked.connect(self.disable_after_delay)
        self.read_pose_button = QPushButton("Read Current Pose")
        self.read_pose_button.clicked.connect(self.read_current_pose)
        self.reset_pose_button = QPushButton("Reset From JSON")
        self.reset_pose_button.clicked.connect(self.reset_to_initial_pose)
        self.reset_pose_button.setEnabled(self.initial_pose_payload is not None)
        self.stop_button = QPushButton("Emergency Stop")
        self.stop_button.clicked.connect(self.emergency_stop)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(self.args.speed_percent)
        self.speed_spin.setSuffix(" %")

        self.xy_speed_spin = QSpinBox()
        self.xy_speed_spin.setRange(1, 200)
        self.xy_speed_spin.setValue(round(self.args.xy_speed_mm_s))
        self.xy_speed_spin.setSuffix(" mm/s")

        self.robot_status = QLabel("Not connected")
        self.robot_status.setWordWrap(True)

        layout.addWidget(QLabel(f"CAN: {self.args.can}"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Piper speed"), 1, 0)
        layout.addWidget(self.speed_spin, 1, 1)
        layout.addWidget(QLabel("XY speed"), 2, 0)
        layout.addWidget(self.xy_speed_spin, 2, 1)
        layout.addWidget(self.connect_button, 3, 0)
        layout.addWidget(self.enable_button, 3, 1)
        layout.addWidget(self.read_pose_button, 4, 0)
        layout.addWidget(self.reset_pose_button, 4, 1)
        layout.addWidget(self.disable_button, 5, 0, 1, 2)
        layout.addWidget(self.stop_button, 6, 0, 1, 2)
        layout.addWidget(self.robot_status, 7, 0, 1, 2)
        return group

    def _build_record_group(self) -> QGroupBox:
        """创建 episode 输出目录控制和录制状态显示。"""

        group = QGroupBox("Episode")
        layout = QGridLayout(group)

        self.output_label = QLabel(str(self.args.output_dir))
        self.output_label.setWordWrap(True)
        initial_pose_label_text = (
            str(self.args.initial_pose_json)
            if self.initial_pose_payload is not None
            else "Not loaded"
        )
        self.initial_pose_label = QLabel(initial_pose_label_text)
        self.initial_pose_label.setWordWrap(True)
        browse_button = QPushButton("Output Dir")
        browse_button.clicked.connect(self.choose_output_dir)

        self.start_button = QPushButton("Start Episode")
        self.start_button.clicked.connect(self.start_episode)
        self.stop_episode_button = QPushButton("Stop Episode")
        self.stop_episode_button.clicked.connect(self.stop_episode)
        self.stop_episode_button.setEnabled(False)
        self.discard_episode_button = QPushButton("Discard Episode")
        self.discard_episode_button.clicked.connect(self.discard_episode)
        self.discard_episode_button.setEnabled(False)
        self.load_replay_button = QPushButton("Load Replay")
        self.load_replay_button.clicked.connect(self.load_replay_episode)
        self.start_replay_button = QPushButton("Start Replay")
        self.start_replay_button.clicked.connect(self.start_replay)
        self.start_replay_button.setEnabled(False)
        self.stop_replay_button = QPushButton("Stop Replay")
        self.stop_replay_button.clicked.connect(self.stop_replay)
        self.stop_replay_button.setEnabled(False)

        self.require_cameras_checkbox = QCheckBox("Require both cameras")
        self.require_cameras_checkbox.setChecked(not self.args.allow_missing_camera)

        self.record_status = QLabel("Idle")
        self.record_status.setWordWrap(True)

        layout.addWidget(browse_button, 0, 0)
        layout.addWidget(self.output_label, 0, 1)
        layout.addWidget(QLabel("Initial pose"), 1, 0)
        layout.addWidget(self.initial_pose_label, 1, 1)
        layout.addWidget(QLabel(f"Sample Hz: {self.args.hz:g}"), 2, 0, 1, 2)
        layout.addWidget(QLabel(f"Command Hz: {self.args.command_hz:g}"), 3, 0, 1, 2)
        layout.addWidget(self.require_cameras_checkbox, 4, 0, 1, 2)
        layout.addWidget(self.start_button, 5, 0)
        layout.addWidget(self.stop_episode_button, 5, 1)
        layout.addWidget(self.discard_episode_button, 6, 0, 1, 2)
        layout.addWidget(self.load_replay_button, 7, 0)
        layout.addWidget(self.start_replay_button, 7, 1)
        layout.addWidget(self.stop_replay_button, 8, 0, 1, 2)
        layout.addWidget(self.record_status, 9, 0, 1, 2)
        return group

    def _build_control_hint_group(self) -> QGroupBox:
        """创建简洁的按键说明和末端位姿反馈面板。"""

        group = QGroupBox("XY Control")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "W/Up: +X\n"
            "S/Down: -X\n"
            "D/Right: +Y\n"
            "A/Left: -Y\n\n"
            "Drag the joystick for mouse XY control."
        )
        hint.setWordWrap(True)
        self.joystick = JoystickWidget()
        self.joystick.direction_changed.connect(self._on_joystick_direction)
        self.pose_label = QLabel("Pose: not loaded")
        self.pose_label.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.joystick)
        layout.addWidget(self.pose_label)
        return group

    def _connect_cameras(self) -> None:
        """启动相机线程，并把图像/状态信号连接到预览组件。"""

        for name, camera in self.cameras.items():
            camera.frame_ready.connect(self._on_frame_ready)
            camera.status.connect(lambda text, n=name: self.camera_views[n].set_status(text))
            camera.status.connect(self.statusBar().showMessage)
            camera.start()

    def connect_piper(self) -> None:
        """连接 Piper CAN 接口，并读取当前初始位姿。"""

        if self.connected:
            return
        self.robot_status.setText("Connecting...")
        QApplication.processEvents()
        try:
            self.robot = PiperRobot(self.args.can, no_can_judge=self.args.no_can_judge)
            self.robot.connect()
        except Exception as exc:
            self.robot = None
            self.robot_status.setText("Connect failed")
            self._error("Connect failed", str(exc))
            return

        self.connected = True
        self.connect_button.setEnabled(False)
        self.robot_status.setText("Connected. Read pose, then Enable.")
        QTimer.singleShot(300, self.read_current_pose)

    def enable_piper(self) -> None:
        """在开始录制 episode 前使能机械臂。"""

        if self.robot is None:
            return
        self.robot_status.setText("Enabling...")
        QApplication.processEvents()
        try:
            ok = self.robot.enable(self.args.enable_timeout)
        except Exception as exc:
            self._error("Enable failed", str(exc))
            ok = False
        self.enabled = ok
        if ok:
            self.reset_completed = self.initial_pose_payload is None
            self.robot_status.setText(
                "Enabled. Joint angles are unchanged; use Reset From JSON before recording."
            )
        else:
            self.robot_status.setText("Enable timeout")

    def disable_after_delay(self) -> None:
        """停止控制命令，等待指定时间后再失能机械臂。"""

        if self.robot is None or not self.enabled:
            return
        self._reset_joystick()
        if self.replay_running:
            self.stop_replay()
        if self.recording:
            self.stop_episode()
        self.reset_timer.stop()
        self.reset_in_progress = False
        self.disable_pending = True
        self.disable_deadline_s = time.monotonic() + float(self.args.disable_delay)
        self.robot_status.setText(f"Will disable in {self.args.disable_delay:.1f}s")
        self.disable_timer.start()

    def _disable_countdown_tick(self) -> None:
        """更新失能倒计时，到点后真正调用 DisablePiper。"""

        if not self.disable_pending:
            self.disable_timer.stop()
            return
        remain = self.disable_deadline_s - time.monotonic()
        if remain > 0:
            self.robot_status.setText(f"Disabling in {remain:.1f}s")
            return
        self.disable_timer.stop()
        self.disable_pending = False
        self.reset_completed = False
        if self.robot is None:
            return
        try:
            self.robot.disable()
        except Exception as exc:
            self._error("Disable failed", str(exc))
            return
        self.enabled = False
        self.robot_status.setText("Disabled")

    def read_current_pose(self) -> None:
        """把当前反馈位姿加载为后续命令目标的基准。"""

        feedback = self._read_feedback()
        if not feedback.valid:
            self.robot_status.setText("Could not read current pose")
            return
        self.last_feedback = feedback
        self.target_pose = feedback.eef_pose_mm_deg.copy()
        self._apply_initial_pose_fixed_plane()
        self._update_pose_label(feedback)
        self.robot_status.setText("Current pose loaded")

    def _apply_initial_pose_fixed_plane(self) -> None:
        """如果加载了初始姿态文件，就把目标位姿固定到同一个工作平面。"""

        if self.target_pose is None or self.initial_pose_payload is None:
            return
        work_plane = self.initial_pose_payload.get("work_plane", {})
        fixed = work_plane.get("fixed_components", {})
        mapping = (
            ("z_mm", 2),
            ("rx_deg", 3),
            ("ry_deg", 4),
            ("rz_deg", 5),
        )
        for key, index in mapping:
            value = fixed.get(key)
            if value is not None:
                self.target_pose[index] = float(value)

    def reset_to_initial_pose(self) -> None:
        """从当前关节角缓慢插值到初始姿态 JSON 里的 reset 姿态。"""

        if self.initial_pose_payload is None:
            self._error("No initial pose", "Load --initial-pose-json before using reset.")
            return
        if not self.connected or self.robot is None:
            self._error("Not connected", "Connect Piper before reset.")
            return
        if not self.enabled:
            self._error("Not enabled", "Enable Piper before reset.")
            return
        if self.replay_running:
            self._error("Replay running", "Stop replay before reset.")
            return

        joint_units = self._reset_joint_units_from_initial_pose()
        if joint_units is None:
            self._error("Invalid reset pose", "Initial pose JSON has no usable reset joint values.")
            return

        feedback = self._read_feedback()
        if not feedback.valid:
            self._error("No feedback", "Cannot read current joints before reset.")
            return

        self.reset_start_qpos_rad = feedback.qpos_rad.copy()
        self.reset_target_units = joint_units
        self.reset_target_qpos_rad = self._reset_joint_radians_from_initial_pose(joint_units)
        self.reset_start_time_s = time.monotonic()
        self.reset_in_progress = True
        self.reset_completed = False
        self.target_pose = None
        self._reset_joystick()
        self._reset_command_accumulator()
        self.robot_status.setText(
            f"Smooth reset started: {self.args.reset_duration:.1f}s"
        )
        self.reset_timer.start()

    def _reset_tick(self) -> None:
        """reset 过程中按固定频率发送关节插值命令。"""

        if (
            self.robot is None
            or not self.enabled
            or not self.reset_in_progress
            or self.reset_start_qpos_rad is None
            or self.reset_target_qpos_rad is None
            or self.reset_target_units is None
        ):
            self.reset_timer.stop()
            self.reset_in_progress = False
            return

        elapsed = time.monotonic() - self.reset_start_time_s
        duration = max(0.1, float(self.args.reset_duration))
        alpha = min(1.0, elapsed / duration)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        if alpha >= 1.0:
            joint_units = self.reset_target_units
        else:
            qpos = (
                self.reset_start_qpos_rad
                + (self.reset_target_qpos_rad - self.reset_start_qpos_rad) * smooth_alpha
            )
            joint_units = qpos_rad_to_sdk_units(qpos)

        try:
            self.robot.send_joint_pose_units(joint_units, int(self.speed_spin.value()))
        except Exception as exc:
            self.reset_timer.stop()
            self.reset_in_progress = False
            self._error("Reset failed", str(exc))
            return

        if alpha >= 1.0:
            self.reset_timer.stop()
            self.reset_in_progress = False
            self.reset_completed = True
            self.robot_status.setText("Reset finished. Reading pose soon...")
            QTimer.singleShot(600, self.read_current_pose)
        else:
            self.robot_status.setText(f"Resetting... {smooth_alpha * 100.0:.0f}%")

    def _reset_joint_units_from_initial_pose(self) -> np.ndarray | None:
        """从初始姿态 JSON 中提取 Piper SDK 关节单位。"""

        if self.initial_pose_payload is None:
            return None
        reset = self.initial_pose_payload.get("reset", {})
        units = reset.get("piper_sdk_joint_units")
        if units is not None:
            arr = np.asarray(units, dtype=np.int32)
            return arr if arr.shape == (6,) else None

        joints_rad = reset.get("joints_rad")
        if joints_rad is None:
            joints_rad = self.initial_pose_payload.get("joints_rad")
        if joints_rad is None:
            return None
        arr = np.asarray(joints_rad, dtype=np.float64)
        if arr.shape != (6,):
            return None
        sdk_units_per_rad = 1000.0 * 180.0 / math.pi
        return np.round(arr * sdk_units_per_rad).astype(np.int32)

    def _reset_joint_radians_from_initial_pose(self, joint_units: np.ndarray) -> np.ndarray:
        """读取 reset 目标的弧度关节角，用于插值。"""

        if self.initial_pose_payload is None:
            return np.asarray(joint_units, dtype=np.float64) / (1000.0 * 180.0 / math.pi)
        reset = self.initial_pose_payload.get("reset", {})
        joints_rad = reset.get("joints_rad")
        if joints_rad is None:
            joints_rad = self.initial_pose_payload.get("joints_rad")
        if joints_rad is not None:
            arr = np.asarray(joints_rad, dtype=np.float64)
            if arr.shape == (6,):
                return arr
        return np.asarray(joint_units, dtype=np.float64) / (1000.0 * 180.0 / math.pi)

    def emergency_stop(self) -> None:
        """发送 Piper 急停命令，并在界面状态中标记为未使能。"""

        if self.robot is None:
            return
        if self.replay_running:
            self.stop_replay()
        self._reset_joystick()
        try:
            self.robot.emergency_stop()
        except Exception as exc:
            self._error("Emergency stop failed", str(exc))
        self.enabled = False
        self.robot_status.setText("Emergency stop sent")

    def choose_output_dir(self) -> None:
        """让用户选择后续 episode 文件的保存目录。"""

        directory = QFileDialog.getExistingDirectory(self, "Choose output directory", str(self.args.output_dir))
        if not directory:
            return
        self.args.output_dir = Path(directory)
        self.output_label.setText(str(self.args.output_dir))

    def start_episode(self) -> None:
        """打开新的 HDF5 文件，并开始把定时器 tick 追加进去。"""

        if self.recording:
            return
        if self.replay_running:
            self._error("Replay running", "Stop replay before starting a new episode.")
            return
        if self.reset_in_progress:
            self._error("Reset in progress", "Wait until smooth reset finishes before recording.")
            return
        if self.disable_pending:
            self._error("Disable pending", "Cancel is not supported; wait until Piper is disabled.")
            return
        if self.initial_pose_payload is not None and not self.reset_completed:
            self._error("Reset required", "Click Reset From JSON and wait until it finishes before recording.")
            return
        if not self.connected or self.robot is None:
            self._error("Not connected", "Connect Piper before starting an episode.")
            return
        if not self.enabled:
            self._error("Not enabled", "Enable Piper before starting an episode.")
            return
        if self.target_pose is None:
            self.read_current_pose()
        if self.target_pose is None:
            self._error("Pose not loaded", "Read current pose before starting.")
            return
        self._apply_initial_pose_fixed_plane()

        wrist = self.cameras["wrist"].latest()
        global_cam = self.cameras["global"].latest()
        if self.require_cameras_checkbox.isChecked() and (wrist is None or global_cam is None):
            self._error("Camera not ready", "Both cameras must have valid frames before recording.")
            return

        episode_path = self._next_episode_path()
        try:
            self.writer = EpisodeWriter(
                episode_path,
                self.args,
                (self.args.camera_height, self.args.camera_width, 3),
                self.target_pose,
            )
        except Exception as exc:
            self.writer = None
            self._error("Cannot create episode", str(exc))
            return

        self.recording = True
        self.last_episode_path = None
        self.last_episode_output_paths = []
        self.last_episode_steps = 0
        self._reset_command_accumulator()
        self.start_button.setEnabled(False)
        self.stop_episode_button.setEnabled(True)
        self.discard_episode_button.setEnabled(True)
        self.record_status.setText(f"Recording: {episode_path}")
        self.setFocus()

    def stop_episode(self) -> None:
        """结束当前 episode，并关闭对应的 HDF5 文件。"""

        if not self.recording:
            return
        self.recording = False
        path = None
        output_paths: list[Path] = []
        steps = 0
        if self.writer is not None:
            path = self.writer.path
            output_paths = list(self.writer.output_paths)
            steps = self.writer.count
            self.writer.close()
            self.writer = None
        self.last_episode_path = path
        self.last_episode_output_paths = output_paths
        self.last_episode_steps = steps
        self.start_button.setEnabled(True)
        self.stop_episode_button.setEnabled(False)
        self.discard_episode_button.setEnabled(path is not None)
        self.record_status.setText(f"Saved {steps} steps: {', '.join(str(p) for p in output_paths) or path}")

    def load_replay_episode(self) -> None:
        """选择一个已保存的 HDF5 episode，并读取其中的 qpos 轨迹。"""

        if self.recording:
            self._error("Recording", "Stop or discard the current episode before loading replay.")
            return
        if self.replay_running:
            self._error("Replay running", "Stop replay before loading another episode.")
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose episode to replay",
            str(self.args.output_dir),
            "HDF5 files (*.hdf5 *.h5);;All files (*)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            with h5py.File(path, "r") as f:
                if "observations/qpos" not in f:
                    raise ValueError("Dataset observations/qpos is missing.")
                qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
                control_hz = float(f.attrs.get("control_hz", self.args.hz))
        except Exception as exc:
            self._error("Load replay failed", str(exc))
            return

        if qpos.ndim != 2 or qpos.shape[1] != 6:
            self._error("Load replay failed", f"Expected observations/qpos shape [T, 6], got {qpos.shape}.")
            return
        valid_rows = np.isfinite(qpos).all(axis=1)
        qpos = qpos[valid_rows]
        if len(qpos) == 0:
            self._error("Load replay failed", "No valid qpos rows found.")
            return

        self.replay_qpos_rad = qpos
        self.replay_control_hz = control_hz
        self.replay_phase = "idle"
        self.replay_step_counter = 0
        self.replay_path = path
        self.replay_timer.setInterval(max(1, round(1000.0 / float(self.args.replay_hz))))
        self.start_replay_button.setEnabled(True)
        self.stop_replay_button.setEnabled(False)
        self.record_status.setText(f"Loaded replay {len(qpos)} steps @ {control_hz:g} Hz: {path}")

    def start_replay(self) -> None:
        """先平滑移动到轨迹首帧，再插值回放 qpos 轨迹。"""

        if self.replay_qpos_rad is None or len(self.replay_qpos_rad) == 0:
            self._error("Replay not loaded", "Load an episode before replay.")
            return
        if self.recording:
            self._error("Recording", "Stop or discard the current episode before replay.")
            return
        if self.reset_in_progress:
            self._error("Reset in progress", "Wait until reset finishes before replay.")
            return
        if self.disable_pending:
            self._error("Disable pending", "Wait until Piper is disabled.")
            return
        if self.robot is None or not self.connected:
            self._error("Not connected", "Connect Piper before replay.")
            return
        if not self.enabled:
            self._error("Not enabled", "Enable Piper before replay.")
            return
        feedback = self._read_feedback()
        if not feedback.valid:
            self._error("No feedback", "Cannot read current joints before replay.")
            return

        answer = QMessageBox.question(
            self,
            "Start replay",
            (
                "Replay will first move to the first recorded joint pose, then send "
                "interpolated joint angles directly to Piper.\n\n"
                f"{self.replay_path}\n\n"
                f"Steps: {len(self.replay_qpos_rad)}\n"
                f"Reset before replay: {self.args.replay_reset_duration:.1f}s\n"
                f"Replay send rate: {self.args.replay_hz:g} Hz"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.pressed_keys.clear()
        self._reset_joystick()
        self._reset_command_accumulator()
        self.replay_start_qpos_rad = feedback.qpos_rad.copy()
        self.replay_phase_start_s = time.monotonic()
        self.replay_phase = "pre_reset"
        self.replay_step_counter = 0
        self.replay_running = True
        self.start_button.setEnabled(False)
        self.stop_episode_button.setEnabled(False)
        self.discard_episode_button.setEnabled(False)
        self.load_replay_button.setEnabled(False)
        self.start_replay_button.setEnabled(False)
        self.stop_replay_button.setEnabled(True)
        self.replay_timer.start()
        self.record_status.setText(f"Replay pre-reset: {self.replay_path}")

    def stop_replay(self) -> None:
        """停止当前关节轨迹回放。"""

        self.replay_timer.stop()
        self.replay_running = False
        self._reset_joystick()
        self.start_button.setEnabled(True)
        self.stop_episode_button.setEnabled(False)
        self.discard_episode_button.setEnabled(self.last_episode_path is not None)
        self.load_replay_button.setEnabled(True)
        self.start_replay_button.setEnabled(self.replay_qpos_rad is not None)
        self.stop_replay_button.setEnabled(False)
        self.replay_phase = "idle"
        self.record_status.setText(f"Replay stopped after {self.replay_step_counter} commands")

    def _replay_tick(self) -> None:
        """高频发送插值后的回放关节角。"""

        if not self.replay_running:
            self.replay_timer.stop()
            return
        if self.robot is None or self.replay_qpos_rad is None or self.replay_start_qpos_rad is None:
            self.stop_replay()
            return
        now = time.monotonic()

        if self.replay_phase == "pre_reset":
            duration = max(0.01, float(self.args.replay_reset_duration))
            alpha = min(1.0, (now - self.replay_phase_start_s) / duration)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            qpos = self.replay_start_qpos_rad + (
                self.replay_qpos_rad[0] - self.replay_start_qpos_rad
            ) * smooth_alpha
            status = f"Replay pre-reset {smooth_alpha * 100.0:.0f}%"
            if alpha >= 1.0:
                self.replay_phase = "trajectory"
                self.replay_phase_start_s = now
                self.replay_step_counter = 0
        elif self.replay_phase == "trajectory":
            elapsed = max(0.0, now - self.replay_phase_start_s)
            sample_position = elapsed * max(1e-6, self.replay_control_hz)
            last_index = len(self.replay_qpos_rad) - 1
            if sample_position >= last_index:
                qpos = self.replay_qpos_rad[last_index]
                self._send_replay_qpos(qpos)
                self._finish_replay()
                return
            lower = int(math.floor(sample_position))
            upper = min(lower + 1, last_index)
            alpha = float(sample_position - lower)
            qpos = self.replay_qpos_rad[lower] * (1.0 - alpha) + self.replay_qpos_rad[upper] * alpha
            status = f"Replaying {sample_position:.1f}/{last_index}"
        else:
            self.stop_replay()
            return

        try:
            self._send_replay_qpos(qpos)
        except Exception as exc:
            self.replay_timer.stop()
            self.replay_running = False
            self.replay_phase = "idle"
            self._error("Replay failed", str(exc))
            self.start_button.setEnabled(True)
            self.load_replay_button.setEnabled(True)
            self.start_replay_button.setEnabled(True)
            self.stop_replay_button.setEnabled(False)
            self.discard_episode_button.setEnabled(self.last_episode_path is not None)
            return

        self.replay_step_counter += 1
        self.record_status.setText(f"{status}: {self.replay_path}")

    def _send_replay_qpos(self, qpos_rad: np.ndarray) -> None:
        """发送一帧关节回放命令。"""

        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        self.robot.send_joint_pose_units(
            qpos_rad_to_sdk_units(qpos_rad),
            int(self.speed_spin.value()),
        )

    def _finish_replay(self) -> None:
        """回放正常结束后恢复 UI 状态。"""

        self.replay_timer.stop()
        self.replay_running = False
        self.replay_phase = "idle"
        self.start_button.setEnabled(True)
        self.load_replay_button.setEnabled(True)
        self.start_replay_button.setEnabled(True)
        self.stop_replay_button.setEnabled(False)
        self.discard_episode_button.setEnabled(self.last_episode_path is not None)
        self.record_status.setText(f"Replay finished: {self.replay_path}")
        self.read_current_pose()

    def discard_episode(self) -> None:
        """丢弃当前正在录制或最近刚保存的 episode 文件。"""

        path = None
        output_paths: list[Path] = []
        steps = 0
        if self.recording:
            if self.writer is not None:
                path = self.writer.path
                output_paths = list(self.writer.output_paths)
                steps = self.writer.count
            prompt = f"Discard current recording?\n\n{path}\n\nSteps recorded: {steps}"
        else:
            path = self.last_episode_path
            output_paths = list(self.last_episode_output_paths)
            steps = self.last_episode_steps
            if path is None:
                self.record_status.setText("No episode to discard")
                self.discard_episode_button.setEnabled(False)
                return
            prompt = f"Delete last saved episode?\n\n{path}\n\nSteps saved: {steps}"

        answer = QMessageBox.question(
            self,
            "Discard episode",
            prompt,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        if self.recording:
            self.recording = False
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None

        deleted_paths: list[Path] = []
        for output_path in output_paths or ([path] if path is not None else []):
            if output_path is None or not output_path.exists():
                continue
            try:
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
                deleted_paths.append(output_path)
            except Exception as exc:
                self._error("Discard failed", str(exc))
                return

        self.last_episode_path = None
        self.last_episode_output_paths = []
        self.last_episode_steps = 0
        self._reset_joystick()
        self._reset_command_accumulator()
        self.start_button.setEnabled(True)
        self.stop_episode_button.setEnabled(False)
        self.discard_episode_button.setEnabled(False)
        status = "Discarded and deleted" if deleted_paths else "Discarded"
        self.record_status.setText(f"{status}: {path}")

    def _reset_command_accumulator(self) -> None:
        """清空当前数据采样步内累计的高频命令。"""

        self.pending_command_delta_xy[:] = 0.0
        self.pending_command_count = 0
        self.last_command_sent_timestamp_ns = 0
        self.last_command_sent_ok = False

    def _reset_joystick(self) -> None:
        """清空鼠标遥感输入，避免释放事件丢失后继续运动。"""

        self.joystick_direction[:] = 0.0
        if hasattr(self, "joystick"):
            self.joystick.reset()

    def _next_episode_path(self) -> Path:
        """返回一个带时间戳且尚不存在的 episode 文件路径。"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        while True:
            path = Path(self.args.output_dir) / f"episode_{timestamp}_{self.episode_index:04d}.hdf5"
            self.episode_index += 1
            if not path.exists() and not path.with_suffix("").exists():
                return path

    def _command_tick(self) -> None:
        """按较高频率发送 XY 命令，让人工遥控更顺滑。"""

        if (
            self.robot is None
            or not self.connected
            or not self.enabled
            or not self.recording
            or self.replay_running
            or self.reset_in_progress
            or self.disable_pending
            or self.target_pose is None
        ):
            return

        dt = 1.0 / float(self.args.command_hz)
        direction = self._xy_direction()
        command_delta_xy = direction * float(self.xy_speed_spin.value()) * dt
        if not np.any(command_delta_xy):
            return
        self.target_pose[:2] = self.target_pose[:2] + command_delta_xy
        self._apply_initial_pose_fixed_plane()
        self._apply_workspace_limits()

        try:
            self.last_command_sent_timestamp_ns = self.robot.send_end_pose(
                self.target_pose,
                int(self.speed_spin.value()),
            )
            self.last_command_sent_ok = True
            self.pending_command_delta_xy += command_delta_xy.astype(np.float32)
            self.pending_command_count += 1
        except Exception as exc:
            self.last_command_sent_ok = False
            self.statusBar().showMessage(f"Command failed: {exc}", 3000)

    def _tick(self) -> None:
        """采集一个固定频率样本，并保存 XY 命令审计字段。

        这里的顺序是刻意设计的：
        1. 读取机械臂反馈，作为 obs[t]。
        2. 取最新相机帧，作为 obs[t]。
        3. 读取上一个采样周期内高频 XY 命令，作为审计字段。
        4. 把所有值追加到 episode writer；writer 会用下一帧 qpos 回填 action[t]。
        """

        if self.robot is None or not self.connected:
            return

        feedback = self._read_feedback()
        if feedback.valid:
            self.last_feedback = feedback
            self._update_pose_label(feedback)
            if self.target_pose is None:
                self.target_pose = feedback.eef_pose_mm_deg.copy()

        if not self.recording or self.writer is None or self.target_pose is None:
            return

        timestamp_ns = time.monotonic_ns()
        direction = self._xy_direction()
        command_delta_xy = self.pending_command_delta_xy.copy()
        command_count = self.pending_command_count
        command_sent_timestamp_ns = self.last_command_sent_timestamp_ns
        command_sent = self.last_command_sent_ok and command_count > 0
        command_pose = self.target_pose.copy()
        self._reset_command_accumulator()

        if not feedback.valid:
            feedback = empty_feedback()

        key_state = np.asarray(
            [
                Qt.Key_W in self.pressed_keys or Qt.Key_Up in self.pressed_keys,
                Qt.Key_A in self.pressed_keys or Qt.Key_Left in self.pressed_keys,
                Qt.Key_S in self.pressed_keys or Qt.Key_Down in self.pressed_keys,
                Qt.Key_D in self.pressed_keys or Qt.Key_Right in self.pressed_keys,
            ],
            dtype=np.bool_,
        )
        self.writer.append(
            timestamp_ns=timestamp_ns,
            wrist=self.cameras["wrist"].latest(),
            global_cam=self.cameras["global"].latest(),
            feedback=feedback,
            command_delta_xy=command_delta_xy.astype(np.float32),
            command_pose=command_pose.astype(np.float32),
            command_sent_timestamp_ns=command_sent_timestamp_ns,
            command_sent=command_sent,
            command_count=command_count,
            key_state=key_state,
            xy_direction=direction.astype(np.float32),
            zero_image=self.zero_image,
        )
        self.record_status.setText(f"Recording steps: {self.writer.count}")

    def _read_feedback(self) -> RobotFeedback:
        """读取机械臂反馈；失败时返回无效样本。"""

        if self.robot is None:
            return empty_feedback()
        return self.robot.read_feedback()

    def _xy_direction(self) -> np.ndarray:
        """把当前按键和鼠标遥感输入合成为归一化 XY 方向向量。"""

        dx = 0.0
        dy = 0.0
        if Qt.Key_W in self.pressed_keys or Qt.Key_Up in self.pressed_keys:
            dx += 1.0
        if Qt.Key_S in self.pressed_keys or Qt.Key_Down in self.pressed_keys:
            dx -= 1.0
        if Qt.Key_D in self.pressed_keys or Qt.Key_Right in self.pressed_keys:
            dy += 1.0
        if Qt.Key_A in self.pressed_keys or Qt.Key_Left in self.pressed_keys:
            dy -= 1.0
        vec = np.asarray([dx, dy], dtype=np.float32) + self.joystick_direction
        norm = float(np.linalg.norm(vec))
        if norm > 1.0:
            vec /= norm
        return vec

    def _on_joystick_direction(self, x: float, y: float) -> None:
        """接收鼠标遥感方向，供高频命令定时器读取。"""

        self.joystick_direction[:] = (x, y)

    def _apply_workspace_limits(self) -> None:
        """把命令目标限制在用户可选的 XY 工作空间范围内。"""

        if self.target_pose is None:
            return
        if self.args.x_min is not None:
            self.target_pose[0] = max(self.args.x_min, self.target_pose[0])
        if self.args.x_max is not None:
            self.target_pose[0] = min(self.args.x_max, self.target_pose[0])
        if self.args.y_min is not None:
            self.target_pose[1] = max(self.args.y_min, self.target_pose[1])
        if self.args.y_max is not None:
            self.target_pose[1] = min(self.args.y_max, self.target_pose[1])

    def _update_pose_label(self, feedback: RobotFeedback) -> None:
        """在界面中显示最新末端反馈位姿。"""

        pose = feedback.eef_pose_mm_deg
        self.pose_label.setText(
            f"EEF: X={pose[0]:.1f} mm, Y={pose[1]:.1f} mm, Z={pose[2]:.1f} mm, "
            f"RX={pose[3]:.1f}, RY={pose[4]:.1f}, RZ={pose[5]:.1f}"
        )

    def _on_frame_ready(self, name: str, image: QImage) -> None:
        """根据相机信号更新对应预览组件。"""

        self.camera_views[name].set_image(image)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        """记录非自动重复的按键按下事件，用于连续 XY 控制。"""

        if not event.isAutoRepeat():
            self.pressed_keys.add(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        """从当前运动按键集合中移除已释放的按键。"""

        if not event.isAutoRepeat():
            self.pressed_keys.discard(event.key())
        super().keyReleaseEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API 方法名。
        """停止定时器、关闭文件、释放相机并断开 Piper。"""

        self.tick_timer.stop()
        self.command_timer.stop()
        self.reset_timer.stop()
        self.disable_timer.stop()
        self.replay_timer.stop()
        if self.recording:
            self.stop_episode()
        for camera in self.cameras.values():
            camera.stop()
        for camera in self.cameras.values():
            camera.join(timeout=1.0)
        if self.robot is not None:
            try:
                if self.args.standby_on_exit:
                    self.robot.standby()
                if self.args.disable_on_exit:
                    self.robot.disable()
                self.robot.disconnect()
            except Exception:
                pass
        event.accept()

    def _error(self, title: str, message: str) -> None:
        """在 GUI 事件处理函数中弹出错误对话框。"""

        QMessageBox.critical(self, title, message)
