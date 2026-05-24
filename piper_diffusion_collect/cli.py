"""Piper 数据采集命令行入口。

tools/ 里的可执行脚本故意保持很薄；argparse 放在这里，方便测试或未来其他启动器
复用同一套默认参数，而不需要导入脚本文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from .camera import list_cameras
from .paths import REPO_ROOT
from .xy_collector_window import DataCollectorWindow


def build_arg_parser() -> argparse.ArgumentParser:
    """构建采集器命令行参数。

    默认值偏保守：较低 XY 速度、中等控制频率、压缩 RGB 图像，
    并且需要显式指定才会自动连接机械臂。
    """

    parser = argparse.ArgumentParser(
        description="Collect Piper XY-teleop episodes with joint-angle diffusion actions."
    )
    parser.add_argument("--can", default="can0", help="SocketCAN interface name.")
    parser.add_argument("--camera-wrist", default="10", help="Wrist camera index or path. Default: 10.")
    parser.add_argument("--camera-global", default="4", help="Global camera index or path. Default: 4.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Dataset sampling frequency for observations and joint-action labels.",
    )
    parser.add_argument(
        "--command-hz",
        type=float,
        default=30.0,
        help="High-rate XY teleop command send frequency.",
    )
    parser.add_argument("--xy-speed-mm-s", type=float, default=20.0, help="Keyboard XY speed.")
    parser.add_argument("--speed-percent", type=int, default=20, help="Piper speed percentage for EndPoseCtrl.")
    parser.add_argument("--reset-duration", type=float, default=5.0, help="Seconds for smooth reset to initial pose.")
    parser.add_argument("--reset-hz", type=float, default=20.0, help="Joint interpolation command frequency during reset.")
    parser.add_argument("--replay-reset-duration", type=float, default=3.0, help="Seconds to move to first replay qpos before playback.")
    parser.add_argument("--replay-hz", type=float, default=60.0, help="Joint command send frequency during interpolated replay.")
    parser.add_argument("--disable-delay", type=float, default=10.0, help="Seconds to wait before disabling Piper.")
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "piper_xy")
    parser.add_argument(
        "--initial-pose-json",
        type=Path,
        default=None,
        help="Initial pose/work-plane JSON saved by initial_pose_tuner.py.",
    )
    parser.add_argument(
        "--image-compression",
        choices=("none", "lzf", "gzip"),
        default="lzf",
        help="HDF5 image compression.",
    )
    parser.add_argument(
        "--save-format",
        dest="save_formats",
        action="append",
        choices=("hdf5", "video", "images"),
        default=None,
        help=(
            "Episode output format. Repeat to save multiple formats, e.g. "
            "--save-format hdf5 --save-format video. Default: hdf5."
        ),
    )
    parser.add_argument("--video-codec", default="mp4v", help="OpenCV fourcc for video output.")
    parser.add_argument("--image-format", choices=("png", "jpg", "jpeg"), default="png", help="Image sequence format.")
    parser.add_argument("--connect", action="store_true", help="Connect to Piper when GUI opens.")
    parser.add_argument("--no-can-judge", action="store_true", help="Use judge_flag=False for non-official CAN.")
    parser.add_argument("--allow-missing-camera", action="store_true", help="Allow recording with missing camera frames.")
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--standby-on-exit", action="store_true")
    parser.add_argument("--x-min", type=float, default=None, help="Optional min X workspace limit in mm.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional max X workspace limit in mm.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional min Y workspace limit in mm.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional max Y workspace limit in mm.")
    parser.add_argument("--list-cameras", action="store_true", help="Probe OpenCV camera indexes, then exit.")
    parser.add_argument("--list-camera-max-index", type=int, default=16)
    return parser


def main() -> int:
    """运行相机探测，或者启动 Qt 采集界面。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be > 0")
    if args.command_hz <= 0:
        parser.error("--command-hz must be > 0")
    if args.reset_duration <= 0:
        parser.error("--reset-duration must be > 0")
    if args.reset_hz <= 0:
        parser.error("--reset-hz must be > 0")
    if args.replay_reset_duration <= 0:
        parser.error("--replay-reset-duration must be > 0")
    if args.replay_hz <= 0:
        parser.error("--replay-hz must be > 0")
    if args.disable_delay < 0:
        parser.error("--disable-delay must be >= 0")
    if args.save_formats is None:
        args.save_formats = ["hdf5"]
    if len(args.video_codec) != 4:
        parser.error("--video-codec must be exactly 4 characters, e.g. mp4v")
    if args.list_cameras:
        list_cameras(args.list_camera_max_index)
        return 0
    app = QApplication(sys.argv)
    window = DataCollectorWindow(args)
    window.show()
    return app.exec_()
