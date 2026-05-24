"""Piper SDK 封装和反馈格式转换工具。

数采系统内部统一使用物理单位：毫米、角度和弧度。
本模块是 Piper SDK 整数单位和这些物理单位之间的边界。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .constants import DEG_PER_RAD
from .paths import ensure_local_piper_sdk


ensure_local_piper_sdk()

try:
    from piper_sdk import C_PiperInterface_V2
except Exception as exc:  # pragma: no cover - 运行时会在 GUI 中显示。
    C_PiperInterface_V2 = None
    PIPER_IMPORT_ERROR = exc
else:
    PIPER_IMPORT_ERROR = None


@dataclass
class RobotFeedback:
    """在某一个单调时钟时刻采样到的机械臂反馈。"""

    eef_pose_mm_deg: np.ndarray
    qpos_rad: np.ndarray
    timestamp_ns: int
    valid: bool


def empty_feedback(timestamp_ns: int = 0) -> RobotFeedback:
    """创建一个无效反馈样本，数据字段用 NaN 填充。"""

    return RobotFeedback(
        eef_pose_mm_deg=np.full(6, np.nan, dtype=np.float32),
        qpos_rad=np.full(6, np.nan, dtype=np.float32),
        timestamp_ns=timestamp_ns,
        valid=False,
    )


def sdk_joint_to_rad(value: int) -> float:
    """把 Piper 关节反馈从 0.001 度单位转换成弧度。"""

    return (value / 1000.0) / DEG_PER_RAD


def qpos_rad_to_sdk_units(qpos_rad: np.ndarray) -> np.ndarray:
    """把 6 个弧度制关节角转换成 Piper JointCtrl 使用的整数单位。"""

    qpos = np.asarray(qpos_rad, dtype=np.float64)
    if qpos.shape != (6,):
        raise ValueError(f"Expected 6 joint angles, got shape {qpos.shape}")
    return np.round(qpos * 1000.0 * DEG_PER_RAD).astype(np.int32)


def eef_pose_msg_to_array(msg: object) -> np.ndarray:
    """把 Piper 末端反馈转换成 [x,y,z,rx,ry,rz]。

    Piper 返回的 x/y/z 单位是 0.001 mm，rx/ry/rz 单位是 0.001 度。
    """

    end_pose = getattr(msg, "end_pose")
    return np.asarray(
        [
            getattr(end_pose, "X_axis") / 1000.0,
            getattr(end_pose, "Y_axis") / 1000.0,
            getattr(end_pose, "Z_axis") / 1000.0,
            getattr(end_pose, "RX_axis") / 1000.0,
            getattr(end_pose, "RY_axis") / 1000.0,
            getattr(end_pose, "RZ_axis") / 1000.0,
        ],
        dtype=np.float32,
    )


def joint_msg_to_array(msg: object) -> np.ndarray:
    """把 Piper 关节反馈消息转换成 6 个关节角，单位为弧度。"""

    joint_state = getattr(msg, "joint_state")
    return np.asarray(
        [
            sdk_joint_to_rad(int(getattr(joint_state, f"joint_{idx}")))
            for idx in range(1, 7)
        ],
        dtype=np.float32,
    )


def pose_to_sdk_units(pose_mm_deg: np.ndarray) -> np.ndarray:
    """把 [mm, deg] 的末端目标位姿转换成 Piper 的 0.001 单位整数。"""

    return np.round(pose_mm_deg * 1000.0).astype(np.int32)


class PiperRobot:
    """GUI 使用的 C_PiperInterface_V2 简单封装。"""

    def __init__(self, can_name: str, no_can_judge: bool = False) -> None:
        if C_PiperInterface_V2 is None:
            raise ImportError(f"Cannot import piper_sdk: {PIPER_IMPORT_ERROR}")
        self.interface = C_PiperInterface_V2(can_name, judge_flag=not no_can_judge)

    def connect(self) -> None:
        """打开 CAN 端口，并启动 Piper SDK 的接收线程。"""

        self.interface.ConnectPort()

    def enable(self, timeout_s: float) -> bool:
        """使能机械臂，并轮询 SDK 反馈直到成功或超时。"""

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if bool(self.interface.EnablePiper()):
                return True
            time.sleep(0.02)
        return False

    def read_feedback(self) -> RobotFeedback:
        """读取一次尽量同步的机械臂反馈样本。

        SDK 内部有自己的接收线程。这里连续读取最新的末端位姿和关节状态，
        并用当前单调时钟给这次采样打时间戳。
        """

        timestamp_ns = time.monotonic_ns()
        try:
            eef_pose = eef_pose_msg_to_array(self.interface.GetArmEndPoseMsgs())
            qpos = joint_msg_to_array(self.interface.GetArmJointMsgs())
        except Exception:
            return empty_feedback(timestamp_ns)
        return RobotFeedback(eef_pose_mm_deg=eef_pose, qpos_rad=qpos, timestamp_ns=timestamp_ns, valid=True)

    def send_end_pose(self, pose_mm_deg: np.ndarray, speed_percent: int) -> int:
        """发送 MOVE P 末端位姿命令，并返回发送完成时刻的时间戳。"""

        command_units = pose_to_sdk_units(pose_mm_deg)
        self.interface.MotionCtrl_2(0x01, 0x00, int(speed_percent), 0x00)
        self.interface.EndPoseCtrl(*[int(value) for value in command_units])
        return time.monotonic_ns()

    def send_joint_pose_units(self, joint_units: np.ndarray, speed_percent: int) -> int:
        """发送关节空间 reset 命令，并返回发送完成时刻的时间戳。"""

        units = np.asarray(joint_units, dtype=np.int32)
        if units.shape != (6,):
            raise ValueError(f"Expected 6 joint units, got shape {units.shape}")
        self.interface.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
        self.interface.JointCtrl(*[int(value) for value in units])
        return time.monotonic_ns()

    def emergency_stop(self) -> None:
        self.interface.EmergencyStop(0x01)

    def standby(self) -> None:
        self.interface.MotionCtrl_2(0x00, 0x00, 0, 0x00)

    def disable(self) -> None:
        self.interface.DisablePiper()

    def disconnect(self) -> None:
        self.interface.DisconnectPort()
