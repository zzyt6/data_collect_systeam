"""数采模块共享的数据集常量。

这些名称会写入 HDF5 属性，方便下游训练代码确认文件来自预期的数据格式。
"""

from __future__ import annotations

import math


SCHEMA_VERSION = "piper_diffusion_joint_action_hdf5_v2"
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")
KEY_NAMES = ("w_or_up", "a_or_left", "s_or_down", "d_or_right")
DEG_PER_RAD = 180.0 / math.pi
