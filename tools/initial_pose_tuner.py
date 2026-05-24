#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""带双路实时相机预览的 Piper 初始姿态调节工具。

该工具用于在采集 diffusion policy 示教数据前，交互式寻找可复现的起始姿态。
它通过关节模式控制 Piper，并在同一个 PyQt 窗口中显示两路 OpenCV 相机画面。

cd /home/gx4070/Desktop/arm-datasets-collect
sudo bash piper_sdk/piper_sdk/can_activate.sh can0 1000000
python tools/initial_pose_tuner.py --can can0 --camera-wrist 10 --camera-global 4 --connect


python train.py \
  --config-name=train_diffusion_unet_piper_zarr_real_image_workspace \
  training.device=cuda:0 \
  logging.mode=offline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except Exception as exc:  # pragma: no cover - 运行环境检查。
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - 运行环境检查。
    print(
        "PyQt5 is required. Install it in your conda environment with:\n"
        "  python -m pip install pyqt5",
        file=sys.stderr,
    )
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SDK_ROOT = REPO_ROOT / "piper_sdk"
if (LOCAL_SDK_ROOT / "piper_sdk" / "__init__.py").exists():
    sys.path.insert(0, str(LOCAL_SDK_ROOT))

try:
    from piper_sdk import C_PiperInterface_V2
except Exception as exc:  # pragma: no cover - 运行时会在对话框中显示。
    C_PiperInterface_V2 = None
    PIPER_IMPORT_ERROR = exc
else:
    PIPER_IMPORT_ERROR = None


DEG_PER_RAD = 180.0 / math.pi
SDK_UNITS_PER_RAD = 1000.0 * DEG_PER_RAD
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class JointSpec:
    name: str
    min_rad: float
    max_rad: float
    default_rad: float = 0.0


JOINT_SPECS = (
    JointSpec("J1", -2.6179, 2.6179, 0.0),
    JointSpec("J2", 0.0, 3.14, 0.0),
    JointSpec("J3", -2.967, 0.0, 0.0),
    JointSpec("J4", -1.745, 1.745, 0.0),
    JointSpec("J5", -1.22, 1.22, 0.0),
    JointSpec("J6", -2.09439, 2.09439, 0.0),
)


def rad_to_slider(rad: float) -> int:
    return int(round(rad * 1000.0))


def slider_to_rad(value: int) -> float:
    return value / 1000.0


def rad_to_sdk_joint(rad: float) -> int:
    return int(round(rad * SDK_UNITS_PER_RAD))


def sdk_joint_to_rad(value: int) -> float:
    return (value / 1000.0) / DEG_PER_RAD


def sdk_pose_to_mm_deg(values: list[int]) -> list[float]:
    return [value / 1000.0 for value in values]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_camera_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def frame_to_qimage(frame) -> QImage:
    if frame.ndim == 2:
        if frame.dtype != "uint8":
            normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = normalized.astype("uint8")
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


def open_video_capture(camera_id: int | str):
    attempts = [None, cv2.CAP_V4L2]
    last_cap = None
    for backend in attempts:
        cap = cv2.VideoCapture(camera_id) if backend is None else cv2.VideoCapture(camera_id, backend)
        last_cap = cap
        if cap.isOpened():
            return cap
        cap.release()
    return last_cap


def print_camera_probe(max_index: int = 16) -> None:
    if cv2 is None:
        print(f"OpenCV import failed: {CV2_IMPORT_ERROR}", file=sys.stderr)
        return

    print("OpenCV camera probe:")
    for idx in range(max_index):
        cap = open_video_capture(idx)
        opened = cap.isOpened()
        ok = False
        shape = None
        dtype = None
        if opened:
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = frame.shape
                dtype = frame.dtype
        cap.release()
        print(f"  {idx}: open={opened} read={ok} shape={shape} dtype={dtype}")


class CameraWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    status = pyqtSignal(str)

    def __init__(
        self,
        camera_id: int | str,
        width: int,
        height: int,
        fps: int,
        name: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self._running = False
        self._cap = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def join(self, timeout: float = 1.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run(self) -> None:
        if cv2 is None:
            self.status.emit(f"{self.name}: OpenCV import failed: {CV2_IMPORT_ERROR}")
            return

        self._running = True
        cap = open_video_capture(self.camera_id)
        self._cap = cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            self.status.emit(f"{self.name}: cannot open camera {self.camera_id}")
            return

        self.status.emit(f"{self.name}: camera {self.camera_id} opened")
        sleep_s = 1.0 / max(self.fps, 1) if self.fps > 0 else 0.01

        while self._running:
            ok, frame = cap.read()
            if ok and frame is not None:
                try:
                    self.frame_ready.emit(frame_to_qimage(frame))
                except Exception as exc:
                    self.status.emit(f"{self.name}: convert frame failed: {exc}")
            else:
                self.status.emit(f"{self.name}: frame read failed")
                time.sleep(0.1)
            time.sleep(sleep_s)

        cap.release()
        self._cap = None
        self.status.emit(f"{self.name}: camera released")


class CameraView(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 300)
        self.image_label.setStyleSheet(
            "QLabel { background: #171717; color: #d6d6d6; border: 1px solid #333; }"
        )
        self.status_label = QLabel("Waiting")
        self.status_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.status_label)
        self._image: QImage | None = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API 方法名。
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class JointRow(QObject):
    changed = pyqtSignal()

    def __init__(self, spec: JointSpec, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(rad_to_slider(spec.min_rad), rad_to_slider(spec.max_rad))
        self.slider.setSingleStep(5)
        self.slider.setPageStep(50)
        self.slider.setValue(rad_to_slider(spec.default_rad))

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.005)
        self.spin.setRange(spec.min_rad, spec.max_rad)
        self.spin.setValue(spec.default_rad)
        self.spin.setSuffix(" rad")
        self.spin.setMinimumWidth(110)

        self.deg_label = QLabel("0.0 deg")
        self.deg_label.setMinimumWidth(75)
        self._syncing = False

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self._update_degree_label()

    def value_rad(self) -> float:
        return float(self.spin.value())

    def set_value_rad(self, value: float, emit_changed: bool = True) -> None:
        value = clamp(value, self.spec.min_rad, self.spec.max_rad)
        self._syncing = True
        self.slider.setValue(rad_to_slider(value))
        self.spin.setValue(value)
        self._syncing = False
        self._update_degree_label()
        if emit_changed:
            self.changed.emit()

    def _slider_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(slider_to_rad(value))
        self._syncing = False
        self._update_degree_label()
        self.changed.emit()

    def _spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(rad_to_slider(value))
        self._syncing = False
        self._update_degree_label()
        self.changed.emit()

    def _update_degree_label(self) -> None:
        self.deg_label.setText(f"{self.value_rad() * DEG_PER_RAD:.1f} deg")


class InitialPoseTuner(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.piper = None
        self.connected = False
        self.enabled = False
        self.feedback_loaded = False
        self.latest_feedback_joints_rad: list[float] | None = None
        self.latest_end_pose_mm_deg: list[float] | None = None
        self.last_command_time = 0.0
        self.pending_send = False
        self.disable_pending = False
        self.disable_deadline_s = 0.0
        self.camera_workers: list[CameraWorker] = []

        self.setWindowTitle("Piper Initial Pose Tuner")
        self.resize(1280, 780)

        self.command_timer = QTimer(self)
        self.command_timer.setInterval(max(10, int(args.command_interval_ms)))
        self.command_timer.timeout.connect(self._flush_pending_command)

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(250)
        self.status_timer.timeout.connect(self._refresh_feedback)

        self.disable_timer = QTimer(self)
        self.disable_timer.setInterval(200)
        self.disable_timer.timeout.connect(self._disable_countdown_tick)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.addWidget(self._build_connection_group())
        control_layout.addWidget(self._build_joint_group(), stretch=1)
        control_layout.addWidget(self._build_feedback_group())
        control_layout.addStretch(1)

        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        self.wrist_camera_view = CameraView("Camera 1 - Wrist")
        self.global_camera_view = CameraView("Camera 2 - Global")
        camera_layout.addWidget(self.wrist_camera_view, stretch=1)
        camera_layout.addWidget(self.global_camera_view, stretch=1)

        root.addWidget(control_panel, stretch=0)
        root.addWidget(camera_panel, stretch=1)

        self._set_controls_enabled(False)
        self._start_cameras()
        if args.connect:
            QTimer.singleShot(100, self.connect_piper)

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Piper")
        layout = QGridLayout(group)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_piper)
        self.enable_button = QPushButton("Enable")
        self.enable_button.clicked.connect(self.enable_piper)
        self.disable_button = QPushButton("Disable After Delay")
        self.disable_button.clicked.connect(self.disable_piper)
        self.stop_button = QPushButton("Emergency Stop")
        self.stop_button.clicked.connect(self.emergency_stop)

        self.auto_send_checkbox = QCheckBox("Auto send while sliding")
        self.auto_send_checkbox.setChecked(not self.args.manual_send)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(self.args.speed)
        self.speed_spin.setSuffix(" %")

        self.status_label = QLabel("Not connected")
        self.status_label.setWordWrap(True)

        layout.addWidget(QLabel(f"CAN: {self.args.can}"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Speed"), 1, 0)
        layout.addWidget(self.speed_spin, 1, 1)
        layout.addWidget(self.auto_send_checkbox, 2, 0, 1, 2)
        layout.addWidget(self.connect_button, 3, 0)
        layout.addWidget(self.enable_button, 3, 1)
        layout.addWidget(self.disable_button, 4, 0)
        layout.addWidget(self.stop_button, 4, 1)
        layout.addWidget(self.status_label, 5, 0, 1, 2)
        return group

    def _build_joint_group(self) -> QGroupBox:
        group = QGroupBox("Joint Target")
        layout = QGridLayout(group)
        self.joint_rows: list[JointRow] = []

        for row, spec in enumerate(JOINT_SPECS):
            joint = JointRow(spec, self)
            joint.changed.connect(self._joint_changed)
            self.joint_rows.append(joint)

            range_label = QLabel(f"{spec.min_rad:.3f} .. {spec.max_rad:.3f}")
            range_label.setMinimumWidth(115)
            layout.addWidget(QLabel(spec.name), row, 0)
            layout.addWidget(joint.slider, row, 1)
            layout.addWidget(joint.spin, row, 2)
            layout.addWidget(joint.deg_label, row, 3)
            layout.addWidget(range_label, row, 4)

        self.read_current_button = QPushButton("Read Current")
        self.read_current_button.clicked.connect(self.read_current_joint_pose)
        self.send_button = QPushButton("Send Target")
        self.send_button.clicked.connect(self.send_current_target)
        self.zero_button = QPushButton("All Zero")
        self.zero_button.clicked.connect(self.set_all_zero)
        self.save_button = QPushButton("Save Pose")
        self.save_button.clicked.connect(self.save_pose)

        button_row = len(JOINT_SPECS)
        layout.addWidget(self.read_current_button, button_row, 0)
        layout.addWidget(self.send_button, button_row, 1)
        layout.addWidget(self.zero_button, button_row, 2)
        layout.addWidget(self.save_button, button_row, 3, 1, 2)
        return group

    def _build_feedback_group(self) -> QGroupBox:
        group = QGroupBox("Feedback")
        layout = QVBoxLayout(group)
        self.feedback_label = QLabel("No feedback yet")
        self.feedback_label.setWordWrap(True)
        layout.addWidget(self.feedback_label)
        return group

    def _set_controls_enabled(self, enabled: bool) -> None:
        is_disabling = self.disable_pending
        motion_enabled = enabled and (
            self.feedback_loaded or self.args.allow_send_without_read
        ) and not is_disabling
        self.enable_button.setEnabled(self.connected and not self.enabled and not is_disabling)
        self.disable_button.setEnabled(self.connected and self.enabled and not is_disabling)
        self.stop_button.setEnabled(self.connected)
        self.read_current_button.setEnabled(self.connected and not is_disabling)
        self.send_button.setEnabled(motion_enabled)
        self.zero_button.setEnabled(motion_enabled)
        self.save_button.setEnabled(True)
        self.auto_send_checkbox.setEnabled(motion_enabled)
        self.speed_spin.setEnabled(motion_enabled)
        for joint in self.joint_rows:
            joint.slider.setEnabled(motion_enabled)
            joint.spin.setEnabled(motion_enabled)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def connect_piper(self) -> None:
        if self.connected:
            self._set_status("Already connected")
            return
        if C_PiperInterface_V2 is None:
            self._show_error(
                "Cannot import piper_sdk",
                f"piper_sdk import failed: {PIPER_IMPORT_ERROR}",
            )
            return

        self._set_status("Connecting...")
        QApplication.processEvents()
        try:
            self.piper = C_PiperInterface_V2(
                self.args.can,
                judge_flag=not self.args.no_can_judge,
            )
            self.piper.ConnectPort()
        except Exception as exc:
            self.piper = None
            self._set_status("Connect failed")
            self._show_error("Connect failed", str(exc))
            return

        self.connected = True
        self.connect_button.setEnabled(False)
        self._set_status("Connected. Loading current joint feedback...")
        self._set_controls_enabled(False)
        self.status_timer.start()
        QTimer.singleShot(300, lambda: self._load_current_joint_pose(show_error=False))

    def enable_piper(self) -> None:
        if self.piper is None:
            return
        if self.disable_pending:
            self._set_status("Disable is pending. Wait until it finishes.")
            return

        self._set_status("Enabling...")
        QApplication.processEvents()
        ok = False
        deadline = time.time() + max(0.1, self.args.enable_timeout)
        while time.time() < deadline:
            try:
                ok = bool(self.piper.EnablePiper())
            except Exception as exc:
                self._show_error("Enable failed", str(exc))
                break
            if ok:
                break
            QApplication.processEvents()
            time.sleep(0.02)

        self.enabled = ok
        if ok:
            if not self.feedback_loaded:
                self._load_current_joint_pose(show_error=False)
            if self.feedback_loaded or self.args.allow_send_without_read:
                self._set_status("Enabled. Sliders can move the arm.")
            else:
                self._set_status("Enabled. Click Read Current before moving.")
            self._set_controls_enabled(True)
            self.command_timer.start()
        else:
            self._set_status("Enable timeout. Check arm mode, CAN, and errors.")
            self._set_controls_enabled(False)

    def disable_piper(self) -> None:
        if self.piper is None or not self.enabled:
            return
        self.pending_send = False
        self.command_timer.stop()
        self.disable_pending = True
        self.disable_deadline_s = time.monotonic() + float(self.args.disable_delay)
        self._set_status(f"Will disable in {self.args.disable_delay:.1f}s")
        self._set_controls_enabled(False)
        self.disable_timer.start()

    def _disable_countdown_tick(self) -> None:
        """更新失能倒计时，到点后真正调用 DisablePiper。"""

        if not self.disable_pending:
            self.disable_timer.stop()
            return
        remain = self.disable_deadline_s - time.monotonic()
        if remain > 0:
            self._set_status(f"Disabling in {remain:.1f}s")
            return

        self.disable_timer.stop()
        self.disable_pending = False
        try:
            self.piper.DisablePiper()
        except Exception as exc:
            self._show_error("Disable failed", str(exc))
            if self.enabled:
                self.command_timer.start()
            self._set_controls_enabled(self.enabled)
            return
        self.enabled = False
        self._set_status("Disabled")
        self._set_controls_enabled(False)

    def emergency_stop(self) -> None:
        if self.piper is None:
            return
        self.disable_pending = False
        self.disable_timer.stop()
        try:
            self.piper.EmergencyStop(0x01)
        except Exception as exc:
            self._show_error("Emergency stop failed", str(exc))
        self.enabled = False
        self.command_timer.stop()
        self._set_status("Emergency stop sent")
        self._set_controls_enabled(False)

    def _joint_changed(self) -> None:
        if not self.enabled or not self.auto_send_checkbox.isChecked():
            return
        self.pending_send = True

    def _flush_pending_command(self) -> None:
        if self.pending_send:
            self.pending_send = False
            self.send_current_target()

    def current_joint_radians(self) -> list[float]:
        return [joint.value_rad() for joint in self.joint_rows]

    def send_current_target(self) -> None:
        if not self.enabled or self.piper is None:
            return
        if not self.feedback_loaded and not self.args.allow_send_without_read:
            self._show_error(
                "Current pose not loaded",
                "Click Read Current first so untouched joints keep their current angles.",
            )
            return
        now = time.time()
        min_interval = max(0.005, self.args.command_interval_ms / 1000.0)
        if now - self.last_command_time < min_interval:
            self.pending_send = True
            return

        joints_rad = self.current_joint_radians()
        joints_sdk = [rad_to_sdk_joint(value) for value in joints_rad]
        try:
            self.piper.MotionCtrl_2(0x01, 0x01, int(self.speed_spin.value()), 0x00)
            self.piper.JointCtrl(*joints_sdk)
        except Exception as exc:
            self._show_error("Send target failed", str(exc))
            self.pending_send = False
            return

        self.last_command_time = now
        self._set_status(
            "Sent: "
            + ", ".join(
                f"{spec.name}={value:.3f}"
                for spec, value in zip(JOINT_SPECS, joints_rad)
            )
            + " rad"
        )

    def read_current_joint_pose(self) -> None:
        self._load_current_joint_pose(show_error=True)

    def _load_current_joint_pose(self, show_error: bool) -> bool:
        joints = self._read_current_joint_radians(show_error)
        if joints is None:
            return False

        for joint, value in zip(self.joint_rows, joints):
            joint.set_value_rad(value, emit_changed=False)
        self.latest_feedback_joints_rad = joints
        self.latest_end_pose_mm_deg = (
            self._read_current_end_pose_mm_deg(show_error=False)
            or self.latest_end_pose_mm_deg
        )
        self.feedback_loaded = True
        self._set_status("Loaded current joint feedback into sliders")
        self._set_controls_enabled(self.enabled)
        return True

    def _read_current_joint_radians(self, show_error: bool) -> list[float] | None:
        if self.piper is None:
            return None
        try:
            msg = self.piper.GetArmJointMsgs()
        except Exception as exc:
            if show_error:
                self._show_error("Read current pose failed", str(exc))
            return None

        joint_state = getattr(msg, "joint_state", None)
        if joint_state is None:
            if show_error:
                self._show_error("Read current pose failed", f"Unexpected joint message: {msg}")
            return None

        values = [
            getattr(joint_state, f"joint_{idx}", None)
            for idx in range(1, len(JOINT_SPECS) + 1)
        ]
        if any(value is None for value in values):
            if show_error:
                self._show_error("Read current pose failed", f"Unexpected joint state: {joint_state}")
            return None

        return [sdk_joint_to_rad(int(raw)) for raw in values]

    def _read_current_end_pose_mm_deg(self, show_error: bool) -> list[float] | None:
        if self.piper is None:
            return None
        try:
            msg = self.piper.GetArmEndPoseMsgs()
        except Exception as exc:
            if show_error:
                self._show_error("Read end pose failed", str(exc))
            return None

        end_pose = getattr(msg, "end_pose", None)
        if end_pose is None:
            if show_error:
                self._show_error("Read end pose failed", f"Unexpected end pose message: {msg}")
            return None

        raw_values = [
            getattr(end_pose, "X_axis", None),
            getattr(end_pose, "Y_axis", None),
            getattr(end_pose, "Z_axis", None),
            getattr(end_pose, "RX_axis", None),
            getattr(end_pose, "RY_axis", None),
            getattr(end_pose, "RZ_axis", None),
        ]
        if any(value is None for value in raw_values):
            if show_error:
                self._show_error("Read end pose failed", f"Unexpected end pose: {end_pose}")
            return None
        return sdk_pose_to_mm_deg([int(value) for value in raw_values])

    def set_all_zero(self) -> None:
        for joint in self.joint_rows:
            joint.set_value_rad(0.0, emit_changed=False)
        if self.enabled and self.auto_send_checkbox.isChecked():
            self.send_current_target()

    def save_pose(self) -> None:
        default_path = REPO_ROOT / "initial_pose.json"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save initial pose",
            str(default_path),
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return

        feedback_joints_rad = (
            self._read_current_joint_radians(show_error=False)
            or self.latest_feedback_joints_rad
            or self.current_joint_radians()
        )
        slider_target_joints_rad = self.current_joint_radians()
        end_pose_mm_deg = (
            self._read_current_end_pose_mm_deg(show_error=False)
            or self.latest_end_pose_mm_deg
        )
        if end_pose_mm_deg is None:
            self._show_error(
                "Cannot save work plane",
                "Current end pose is required. Connect Piper and click Read Current before saving.",
            )
            return

        fixed_components = {
            "z_mm": end_pose_mm_deg[2],
            "rx_deg": end_pose_mm_deg[3],
            "ry_deg": end_pose_mm_deg[4],
            "rz_deg": end_pose_mm_deg[5],
        }
        payload = {
            "schema_version": "piper_initial_pose_v1",
            "purpose": "push_t_work_plane_and_reset_pose",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "can": self.args.can,
            "camera_wrist": self.args.camera_wrist,
            "camera_global": self.args.camera_global,
            "joint_order": [spec.name for spec in JOINT_SPECS],
            "joints_rad": feedback_joints_rad,
            "joints_deg": [value * DEG_PER_RAD for value in feedback_joints_rad],
            "piper_sdk_joint_units": [rad_to_sdk_joint(value) for value in feedback_joints_rad],
            "slider_target_joints_rad": slider_target_joints_rad,
            "end_pose_order": list(POSE_NAMES),
            "end_pose_mm_deg": end_pose_mm_deg,
            "piper_sdk_end_pose_units": [int(round(value * 1000.0)) for value in end_pose_mm_deg],
            "work_plane": {
                "frame": "piper_base",
                "control_axes": ["x_mm", "y_mm"],
                "fixed_components": fixed_components,
                "origin_eef_pose_mm_deg": end_pose_mm_deg,
                "teleop_command": "delta_xy_mm_per_command",
                "policy_action": "next_qpos_rad",
                "description": (
                    "diffusion_xy_collector teleop only changes X/Y and keeps "
                    "Z/RX/RY/RZ fixed to this initial end pose. The main "
                    "training action is the next measured 6-joint qpos."
                ),
            },
            "reset": {
                "joint_order": [spec.name for spec in JOINT_SPECS],
                "joints_rad": feedback_joints_rad,
                "joints_deg": [value * DEG_PER_RAD for value in feedback_joints_rad],
                "piper_sdk_joint_units": [rad_to_sdk_joint(value) for value in feedback_joints_rad],
                "end_pose_mm_deg": end_pose_mm_deg,
            },
        }

        path = Path(filename)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._show_error("Save pose failed", str(exc))
            return
        self._set_status(f"Saved pose: {path}")

    def _refresh_feedback(self) -> None:
        if self.piper is None:
            return
        try:
            joint_msg = self.piper.GetArmJointMsgs()
            pose_msg = self.piper.GetArmEndPoseMsgs()
        except Exception:
            return

        joint_state = getattr(joint_msg, "joint_state", None)
        end_pose = getattr(pose_msg, "end_pose", None)
        lines: list[str] = []

        if joint_state is not None:
            raw_joints = [
                getattr(joint_state, f"joint_{idx}", None)
                for idx in range(1, len(JOINT_SPECS) + 1)
            ]
            if all(value is not None for value in raw_joints):
                joints = [sdk_joint_to_rad(int(value)) for value in raw_joints]
                self.latest_feedback_joints_rad = joints
                lines.append(
                    "Joint fb rad: "
                    + ", ".join(
                        f"{spec.name}={value:.3f}"
                        for spec, value in zip(JOINT_SPECS, joints)
                    )
                )

        if end_pose is not None:
            pose_values = {
                "X": getattr(end_pose, "X_axis", None),
                "Y": getattr(end_pose, "Y_axis", None),
                "Z": getattr(end_pose, "Z_axis", None),
                "RX": getattr(end_pose, "RX_axis", None),
                "RY": getattr(end_pose, "RY_axis", None),
                "RZ": getattr(end_pose, "RZ_axis", None),
            }
            if all(value is not None for value in pose_values.values()):
                self.latest_end_pose_mm_deg = sdk_pose_to_mm_deg(
                    [int(value) for value in pose_values.values()]
                )
                lines.append(
                    "End pose: "
                    f"X={pose_values['X'] / 1000.0:.1f} mm, "
                    f"Y={pose_values['Y'] / 1000.0:.1f} mm, "
                    f"Z={pose_values['Z'] / 1000.0:.1f} mm, "
                    f"RX={pose_values['RX'] / 1000.0:.1f} deg, "
                    f"RY={pose_values['RY'] / 1000.0:.1f} deg, "
                    f"RZ={pose_values['RZ'] / 1000.0:.1f} deg"
                )

        if lines:
            self.feedback_label.setText("\n".join(lines))

    def _start_cameras(self) -> None:
        camera_specs = [
            (
                parse_camera_id(self.args.camera_wrist),
                "Wrist",
                self.wrist_camera_view,
            ),
            (
                parse_camera_id(self.args.camera_global),
                "Global",
                self.global_camera_view,
            ),
        ]

        for camera_id, name, view in camera_specs:
            worker = CameraWorker(
                camera_id,
                self.args.camera_width,
                self.args.camera_height,
                self.args.camera_fps,
                name,
            )
            worker.frame_ready.connect(view.set_image)
            worker.status.connect(view.set_status)
            worker.status.connect(self._camera_status_to_main)
            self.camera_workers.append(worker)
            worker.start()

    def _camera_status_to_main(self, text: str) -> None:
        if "cannot open" in text or "failed" in text:
            self.statusBar().showMessage(text, 4000)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API 方法名。
        self.command_timer.stop()
        self.status_timer.stop()
        self.disable_timer.stop()

        for worker in self.camera_workers:
            worker.stop()
        for worker in self.camera_workers:
            worker.join(timeout=1.0)

        if self.piper is not None:
            try:
                if self.args.standby_on_exit:
                    self.piper.MotionCtrl_2(0x00, 0x01, 0, 0x00)
                if self.args.disable_on_exit:
                    delay = max(0.0, float(self.args.disable_delay))
                    if delay > 0:
                        self._set_status(f"Closing: disabling in {delay:.1f}s")
                        QApplication.processEvents()
                        time.sleep(delay)
                    self.piper.DisablePiper()
                self.piper.DisconnectPort()
            except Exception:
                pass
        event.accept()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune Piper initial joint pose while viewing two cameras."
    )
    parser.add_argument("--can", default="can0", help="SocketCAN interface name.")
    parser.add_argument(
        "--camera-wrist",
        default="10",
        help="Wrist camera index or device path. Default: 10.",
    )
    parser.add_argument(
        "--camera-global",
        default="4",
        help="Global camera index or device path. Default: 4.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe /dev/video indexes with OpenCV, then exit.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=20,
        help="Piper move speed percentage used for slider commands.",
    )
    parser.add_argument(
        "--command-interval-ms",
        type=int,
        default=50,
        help="Minimum interval between joint commands while dragging sliders.",
    )
    parser.add_argument(
        "--enable-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for EnablePiper to report success.",
    )
    parser.add_argument(
        "--disable-delay",
        type=float,
        default=10.0,
        help="Seconds to wait before DisablePiper. Default: 10.",
    )
    parser.add_argument(
        "--manual-send",
        action="store_true",
        help="Do not send while sliding; use the Send Target button.",
    )
    parser.add_argument(
        "--allow-send-without-read",
        action="store_true",
        help="Allow slider commands before current joint feedback is loaded.",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to Piper automatically after opening the UI.",
    )
    parser.add_argument(
        "--no-can-judge",
        action="store_true",
        help="Pass judge_flag=False to the Piper SDK for non-official CAN devices.",
    )
    parser.add_argument(
        "--disable-on-exit",
        action="store_true",
        help="Disable Piper when closing the window.",
    )
    parser.add_argument(
        "--standby-on-exit",
        action="store_true",
        help="Send standby mode when closing the window.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.disable_delay < 0:
        print("--disable-delay must be >= 0", file=sys.stderr)
        return 2
    if cv2 is None:
        print(
            "OpenCV is required for camera display. Install it in datacollect with:\n"
            "  conda activate datacollect\n"
            "  python -m pip install opencv-python\n"
            f"\nOriginal import error: {CV2_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 2
    if args.list_cameras:
        print_camera_probe()
        return 0

    app = QApplication(sys.argv)
    window = InitialPoseTuner(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
