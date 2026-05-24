"""Piper joint-action 数据集格式的 episode 写入器。

每个 HDF5 文件对应一个 episode。所有数据集的第一维都是时间步 T。
主训练 action 保存为下一步实际关节角，同时保留 XY 命令字段用于审计。
也可以额外保存视频或图片序列及逐帧时间戳，方便人工检查和通用视觉工具读取。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

from .camera import FramePacket
from .constants import JOINT_NAMES, KEY_NAMES, POSE_NAMES, SCHEMA_VERSION
from .robot import RobotFeedback, pose_to_sdk_units


def json_ready_config(args: argparse.Namespace) -> dict:
    """把 argparse 参数转换成可以写入 JSON 元数据的普通类型。"""

    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def compact_json(value: object) -> str:
    """把结构化元数据压成稳定的 JSON 字符串，便于写入 HDF5 属性。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_save_formats(args: argparse.Namespace) -> tuple[str, ...]:
    """解析保存格式列表，默认只保存 HDF5。"""

    values = getattr(args, "save_formats", None) or ["hdf5"]
    formats = []
    for value in values:
        text = str(value).strip().lower()
        if text and text not in formats:
            formats.append(text)
    return tuple(formats)


class EpisodeWriter:
    """单个采集 episode 的追加式多格式写入器。"""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        image_shape: tuple[int, int, int],
        start_pose: np.ndarray,
    ) -> None:
        """创建新的 episode 输出，并初始化所需格式。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.hdf5_path = path
        self.sidecar_dir = path.with_suffix("")
        self.save_formats = normalize_save_formats(args)
        unknown_formats = sorted(set(self.save_formats) - {"hdf5", "video", "images"})
        if unknown_formats:
            raise ValueError(f"Unsupported save format(s): {unknown_formats}")
        if not self.save_formats:
            raise ValueError("At least one save format is required.")

        self.hdf5_enabled = "hdf5" in self.save_formats
        self.video_enabled = "video" in self.save_formats
        self.images_enabled = "images" in self.save_formats
        self.path = self.hdf5_path if self.hdf5_enabled else self.sidecar_dir
        self.output_paths: list[Path] = []
        self.f: h5py.File | None = None
        self.count = 0
        self.prev_eef_pose: np.ndarray | None = None
        self.prev_qpos: np.ndarray | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.frame_csv_file = None
        self.frame_csv_writer: csv.DictWriter | None = None
        self.metadata: dict[str, object] | None = None
        self.video_paths: dict[str, str] = {}
        self.image_dirs: dict[str, Path] = {}
        self.image_extension = str(getattr(args, "image_format", "png")).lower().lstrip(".")

        if self.hdf5_enabled:
            self.f = h5py.File(path, "w")
            self.output_paths.append(self.hdf5_path)
            self._write_hdf5_attrs(args, start_pose)
            self._create_common_datasets(image_shape, args.image_compression)
        if self.video_enabled or self.images_enabled:
            self._init_sidecar_outputs(args, image_shape, start_pose)

    def _write_hdf5_attrs(self, args: argparse.Namespace, start_pose: np.ndarray) -> None:
        """写入 HDF5 文件级元数据。"""

        if self.f is None:
            return
        self.f.attrs["schema_version"] = SCHEMA_VERSION
        self.f.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        self.f.attrs["control_hz"] = float(args.hz)
        self.f.attrs["command_hz"] = float(getattr(args, "command_hz", args.hz))
        self.f.attrs["save_formats_json"] = json.dumps(self.save_formats)
        self.f.attrs["dt_seconds"] = float(1.0 / args.hz)
        self.f.attrs["command_dt_seconds"] = float(1.0 / getattr(args, "command_hz", args.hz))
        self.f.attrs["alignment"] = (
            "obs[t] is sampled at dataset rate; action[t] is the next measured joint "
            "position observations/qpos[t+1]; actions/command_delta_xy_mm[t] is the "
            "accumulated teleop XY command since the previous dataset sample."
        )
        self.f.attrs["action_units"] = (
            "radians; action[t] and actual_action[t] are next-step absolute joint positions"
        )
        self.f.attrs["eef_pose_units"] = "x,y,z in millimeters; rx,ry,rz in degrees"
        self.f.attrs["qpos_units"] = "radians"
        self.f.attrs["image_encoding"] = "RGB uint8"
        self.f.attrs["joint_names_json"] = json.dumps(JOINT_NAMES)
        self.f.attrs["pose_names_json"] = json.dumps(POSE_NAMES)
        self.f.attrs["key_names_json"] = json.dumps(KEY_NAMES)
        self.f.attrs["start_pose_mm_deg_json"] = json.dumps(start_pose.astype(float).tolist())
        self.f.attrs["config_json"] = compact_json(json_ready_config(args))

        initial_pose_payload = getattr(args, "initial_pose_payload", None)
        if initial_pose_payload is not None:
            self.f.attrs["initial_pose_json"] = compact_json(initial_pose_payload)
            self.f.attrs["work_plane_json"] = compact_json(initial_pose_payload.get("work_plane", {}))
            self.f.attrs["reset_json"] = compact_json(initial_pose_payload.get("reset", {}))
            self.f.attrs["initial_pose_source"] = str(getattr(args, "initial_pose_json", ""))

    def _init_sidecar_outputs(
        self,
        args: argparse.Namespace,
        image_shape: tuple[int, int, int],
        start_pose: np.ndarray,
    ) -> None:
        """初始化视频/图片序列输出目录及逐帧时间戳 CSV。"""

        self.sidecar_dir.mkdir(parents=True, exist_ok=True)
        self.output_paths.append(self.sidecar_dir)
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "save_formats": list(self.save_formats),
            "control_hz": float(args.hz),
            "command_hz": float(getattr(args, "command_hz", args.hz)),
            "video_fps": float(args.hz),
            "image_shape_hwc": list(image_shape),
            "image_encoding": "RGB uint8 source; videos/images are written with OpenCV BGR conversion",
            "hdf5_path": str(self.hdf5_path) if self.hdf5_enabled else None,
            "start_pose_mm_deg": start_pose.astype(float).tolist(),
            "joint_names": list(JOINT_NAMES),
            "pose_names": list(POSE_NAMES),
            "key_names": list(KEY_NAMES),
            "config": json_ready_config(args),
            "num_steps": None,
        }

        self.frame_csv_file = (self.sidecar_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        self.frame_csv_writer = csv.DictWriter(self.frame_csv_file, fieldnames=self._frame_csv_fields())
        self.frame_csv_writer.writeheader()

        if self.video_enabled:
            videos_dir = self.sidecar_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            codec = str(getattr(args, "video_codec", "mp4v"))
            fourcc = cv2.VideoWriter_fourcc(*codec)
            height, width, _ = image_shape
            for camera_name in ("wrist", "global"):
                video_path = videos_dir / f"{camera_name}.mp4"
                writer = cv2.VideoWriter(str(video_path), fourcc, float(args.hz), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {video_path}")
                self.video_writers[camera_name] = writer
                self.video_paths[camera_name] = str(video_path)
            self.metadata["video_codec"] = codec
            self.metadata["video_paths"] = dict(self.video_paths)

        if self.images_enabled:
            images_dir = self.sidecar_dir / "images"
            for camera_name in ("wrist", "global"):
                camera_dir = images_dir / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                self.image_dirs[camera_name] = camera_dir
            self.metadata["image_format"] = self.image_extension
            self.metadata["image_dirs"] = {name: str(path) for name, path in self.image_dirs.items()}

        self._write_sidecar_metadata()

    def _frame_csv_fields(self) -> list[str]:
        """返回视频/图片 sidecar 的逐帧索引字段。"""

        return [
            "step",
            "timestamp_ns",
            "robot_timestamp_ns",
            "command_sent_timestamp_ns",
            "camera_wrist_timestamp_ns",
            "camera_global_timestamp_ns",
            "camera_wrist_age_s",
            "camera_global_age_s",
            "valid_wrist_camera",
            "valid_global_camera",
            "valid_robot_feedback",
            "valid_command_sent",
            "wrist_frame_index",
            "global_frame_index",
            "wrist_image_path",
            "global_image_path",
            "command_count",
            "xy_direction_x",
            "xy_direction_y",
            *[f"qpos_{idx + 1}_rad" for idx in range(6)],
            *[f"eef_{name}_mm_deg" for name in POSE_NAMES],
        ]

    def _create_common_datasets(self, image_shape: tuple[int, int, int], image_compression: str) -> None:
        """创建第一维为统一时间步、可增长的数据集。"""

        compression = None if image_compression == "none" else image_compression
        image_kwargs = {"chunks": (1, *image_shape), "compression": compression}
        if compression == "gzip":
            image_kwargs["compression_opts"] = 4

        self._datasets["observations/images/wrist"] = self._create_resizable(
            "observations/images/wrist", image_shape, np.uint8, **image_kwargs
        )
        self._datasets["observations/images/global"] = self._create_resizable(
            "observations/images/global", image_shape, np.uint8, **image_kwargs
        )

        # 非图像数据比较小，可以按“每个时间步一行”的方式存储。
        # NaN 用来标记有意缺失的字段，例如最后一行没有 actual_action。
        for name, tail, dtype, fillvalue in [
            ("observations/eef_pose", (6,), np.float32, np.nan),
            ("observations/qpos", (6,), np.float32, np.nan),
            ("observations/state", (12,), np.float32, np.nan),
            ("action", (6,), np.float32, np.nan),
            ("actual_action", (6,), np.float32, np.nan),
            ("actions/command_delta_xy_mm", (2,), np.float32, np.nan),
            ("actions/command_eef_pose_mm_deg", (6,), np.float32, np.nan),
            ("actions/command_sdk_units", (6,), np.int32, 0),
            ("actions/command_count", (), np.int32, 0),
            ("actions/actual_delta_xy_mm", (2,), np.float32, np.nan),
            ("actions/actual_eef_pose_next_mm_deg", (6,), np.float32, np.nan),
            ("actions/actual_joint_delta_rad", (6,), np.float32, np.nan),
            ("actions/actual_qpos_next_rad", (6,), np.float32, np.nan),
            ("time/step", (), np.int64, 0),
            ("time/timestamp_ns", (), np.int64, 0),
            ("time/robot_timestamp_ns", (), np.int64, 0),
            ("time/command_sent_timestamp_ns", (), np.int64, 0),
            ("time/camera_wrist_timestamp_ns", (), np.int64, 0),
            ("time/camera_global_timestamp_ns", (), np.int64, 0),
            ("time/camera_wrist_age_s", (), np.float32, np.nan),
            ("time/camera_global_age_s", (), np.float32, np.nan),
            ("valid/wrist_camera", (), np.bool_, False),
            ("valid/global_camera", (), np.bool_, False),
            ("valid/robot_feedback", (), np.bool_, False),
            ("valid/action", (), np.bool_, False),
            ("valid/actual_action", (), np.bool_, False),
            ("valid/command_sent", (), np.bool_, False),
            ("control/key_state", (4,), np.bool_, False),
            ("control/xy_direction", (2,), np.float32, 0.0),
        ]:
            self._datasets[name] = self._create_resizable(name, tail, dtype, fillvalue=fillvalue)

    def _create_resizable(self, name: str, tail_shape: tuple[int, ...], dtype, **kwargs) -> h5py.Dataset:
        """创建一个第一维会随着 append 增长的数据集。"""

        shape = (0, *tail_shape)
        maxshape = (None, *tail_shape)
        chunks = kwargs.pop("chunks", (1, *tail_shape) if tail_shape else (1024,))
        return self.f.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            **kwargs,
        )

    def append(
        self,
        *,
        timestamp_ns: int,
        wrist: FramePacket | None,
        global_cam: FramePacket | None,
        feedback: RobotFeedback,
        command_delta_xy: np.ndarray,
        command_pose: np.ndarray,
        command_sent_timestamp_ns: int,
        command_sent: bool,
        command_count: int,
        key_state: np.ndarray,
        xy_direction: np.ndarray,
        zero_image: np.ndarray,
    ) -> None:
        """追加一个已经按时间对齐的时间步。

        action[t] 使用 t+1 的关节反馈作为监督标签。
        因此每次 append 开始时，会用当前反馈回填上一行的 action、actual_action 和实际增量。
        """

        if self.prev_eef_pose is not None and self.prev_qpos is not None and feedback.valid:
            actual_delta = feedback.eef_pose_mm_deg[:2] - self.prev_eef_pose[:2]
            actual_joint_delta = feedback.qpos_rad - self.prev_qpos
            prev_index = self.count - 1
            if self.hdf5_enabled:
                self._datasets["action"][prev_index] = feedback.qpos_rad
                self._datasets["actual_action"][prev_index] = feedback.qpos_rad
                self._datasets["actions/actual_delta_xy_mm"][prev_index] = actual_delta
                self._datasets["actions/actual_eef_pose_next_mm_deg"][prev_index] = feedback.eef_pose_mm_deg
                self._datasets["actions/actual_joint_delta_rad"][prev_index] = actual_joint_delta
                self._datasets["actions/actual_qpos_next_rad"][prev_index] = feedback.qpos_rad
                self._datasets["valid/action"][prev_index] = True
                self._datasets["valid/actual_action"][prev_index] = True

        index = self.count
        if self.hdf5_enabled:
            self._resize_all(index + 1)

        wrist_valid = wrist is not None
        global_valid = global_cam is not None
        wrist_image = wrist.image_rgb if wrist_valid else zero_image
        global_image = global_cam.image_rgb if global_valid else zero_image
        wrist_ts = int(wrist.timestamp_ns) if wrist_valid else 0
        global_ts = int(global_cam.timestamp_ns) if global_valid else 0

        if self.hdf5_enabled:
            self._append_hdf5_row(
                index=index,
                timestamp_ns=timestamp_ns,
                wrist_image=wrist_image,
                global_image=global_image,
                wrist_valid=wrist_valid,
                global_valid=global_valid,
                wrist_ts=wrist_ts,
                global_ts=global_ts,
                feedback=feedback,
                command_delta_xy=command_delta_xy,
                command_pose=command_pose,
                command_sent_timestamp_ns=command_sent_timestamp_ns,
                command_sent=command_sent,
                command_count=command_count,
                key_state=key_state,
                xy_direction=xy_direction,
            )
        if self.video_enabled or self.images_enabled:
            self._append_sidecar_row(
                index=index,
                timestamp_ns=timestamp_ns,
                wrist=wrist,
                global_cam=global_cam,
                wrist_image=wrist_image,
                global_image=global_image,
                wrist_valid=wrist_valid,
                global_valid=global_valid,
                wrist_ts=wrist_ts,
                global_ts=global_ts,
                feedback=feedback,
                command_sent_timestamp_ns=command_sent_timestamp_ns,
                command_sent=command_sent,
                command_count=command_count,
                xy_direction=xy_direction,
            )

        if feedback.valid:
            self.prev_eef_pose = feedback.eef_pose_mm_deg.copy()
            self.prev_qpos = feedback.qpos_rad.copy()
        self.count += 1

    def _append_hdf5_row(
        self,
        *,
        index: int,
        timestamp_ns: int,
        wrist_image: np.ndarray,
        global_image: np.ndarray,
        wrist_valid: bool,
        global_valid: bool,
        wrist_ts: int,
        global_ts: int,
        feedback: RobotFeedback,
        command_delta_xy: np.ndarray,
        command_pose: np.ndarray,
        command_sent_timestamp_ns: int,
        command_sent: bool,
        command_count: int,
        key_state: np.ndarray,
        xy_direction: np.ndarray,
    ) -> None:
        """向 HDF5 写入当前时间步。"""

        self._datasets["observations/images/wrist"][index] = wrist_image
        self._datasets["observations/images/global"][index] = global_image
        self._datasets["observations/eef_pose"][index] = feedback.eef_pose_mm_deg
        self._datasets["observations/qpos"][index] = feedback.qpos_rad
        self._datasets["observations/state"][index] = np.concatenate(
            [feedback.eef_pose_mm_deg, feedback.qpos_rad]
        ).astype(np.float32)
        self._datasets["action"][index] = np.full(6, np.nan, dtype=np.float32)
        self._datasets["actual_action"][index] = np.full(6, np.nan, dtype=np.float32)
        self._datasets["actions/command_delta_xy_mm"][index] = command_delta_xy
        self._datasets["actions/command_eef_pose_mm_deg"][index] = command_pose
        self._datasets["actions/command_sdk_units"][index] = pose_to_sdk_units(command_pose)
        self._datasets["actions/command_count"][index] = int(command_count)
        self._datasets["actions/actual_delta_xy_mm"][index] = np.asarray([np.nan, np.nan], dtype=np.float32)
        self._datasets["actions/actual_eef_pose_next_mm_deg"][index] = np.full(6, np.nan, dtype=np.float32)
        self._datasets["actions/actual_joint_delta_rad"][index] = np.full(6, np.nan, dtype=np.float32)
        self._datasets["actions/actual_qpos_next_rad"][index] = np.full(6, np.nan, dtype=np.float32)
        self._datasets["time/step"][index] = index
        self._datasets["time/timestamp_ns"][index] = int(timestamp_ns)
        self._datasets["time/robot_timestamp_ns"][index] = int(feedback.timestamp_ns)
        self._datasets["time/command_sent_timestamp_ns"][index] = int(command_sent_timestamp_ns)
        self._datasets["time/camera_wrist_timestamp_ns"][index] = wrist_ts
        self._datasets["time/camera_global_timestamp_ns"][index] = global_ts
        self._datasets["time/camera_wrist_age_s"][index] = (
            (timestamp_ns - wrist_ts) / 1e9 if wrist_valid else np.nan
        )
        self._datasets["time/camera_global_age_s"][index] = (
            (timestamp_ns - global_ts) / 1e9 if global_valid else np.nan
        )
        self._datasets["valid/wrist_camera"][index] = wrist_valid
        self._datasets["valid/global_camera"][index] = global_valid
        self._datasets["valid/robot_feedback"][index] = feedback.valid
        self._datasets["valid/action"][index] = False
        self._datasets["valid/actual_action"][index] = False
        self._datasets["valid/command_sent"][index] = bool(command_sent)
        self._datasets["control/key_state"][index] = key_state
        self._datasets["control/xy_direction"][index] = xy_direction

    def _append_sidecar_row(
        self,
        *,
        index: int,
        timestamp_ns: int,
        wrist: FramePacket | None,
        global_cam: FramePacket | None,
        wrist_image: np.ndarray,
        global_image: np.ndarray,
        wrist_valid: bool,
        global_valid: bool,
        wrist_ts: int,
        global_ts: int,
        feedback: RobotFeedback,
        command_sent_timestamp_ns: int,
        command_sent: bool,
        command_count: int,
        xy_direction: np.ndarray,
    ) -> None:
        """写入视频/图片帧和对应 CSV 时间索引。"""

        wrist_image_path = ""
        global_image_path = ""
        if self.video_enabled:
            self._write_video_frame("wrist", wrist_image)
            self._write_video_frame("global", global_image)
        if self.images_enabled:
            wrist_image_path = self._write_image_frame("wrist", index, wrist_image)
            global_image_path = self._write_image_frame("global", index, global_image)
        if self.frame_csv_writer is not None:
            row = {
                "step": index,
                "timestamp_ns": int(timestamp_ns),
                "robot_timestamp_ns": int(feedback.timestamp_ns),
                "command_sent_timestamp_ns": int(command_sent_timestamp_ns),
                "camera_wrist_timestamp_ns": wrist_ts,
                "camera_global_timestamp_ns": global_ts,
                "camera_wrist_age_s": (timestamp_ns - wrist_ts) / 1e9 if wrist_valid else np.nan,
                "camera_global_age_s": (timestamp_ns - global_ts) / 1e9 if global_valid else np.nan,
                "valid_wrist_camera": bool(wrist_valid),
                "valid_global_camera": bool(global_valid),
                "valid_robot_feedback": bool(feedback.valid),
                "valid_command_sent": bool(command_sent),
                "wrist_frame_index": int(wrist.frame_index) if wrist is not None else -1,
                "global_frame_index": int(global_cam.frame_index) if global_cam is not None else -1,
                "wrist_image_path": wrist_image_path,
                "global_image_path": global_image_path,
                "command_count": int(command_count),
                "xy_direction_x": float(xy_direction[0]),
                "xy_direction_y": float(xy_direction[1]),
            }
            for idx, value in enumerate(feedback.qpos_rad):
                row[f"qpos_{idx + 1}_rad"] = float(value)
            for name, value in zip(POSE_NAMES, feedback.eef_pose_mm_deg):
                row[f"eef_{name}_mm_deg"] = float(value)
            self.frame_csv_writer.writerow(row)

    def _write_video_frame(self, camera_name: str, image_rgb: np.ndarray) -> None:
        """把 RGB 帧转换为 BGR 后写入 OpenCV 视频。"""

        writer = self.video_writers.get(camera_name)
        if writer is None:
            return
        writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    def _write_image_frame(self, camera_name: str, index: int, image_rgb: np.ndarray) -> str:
        """写入单张图片，并返回相对 sidecar 目录的路径。"""

        directory = self.image_dirs[camera_name]
        image_path = directory / f"{index:06d}.{self.image_extension}"
        ok = cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Cannot write image: {image_path}")
        return str(image_path.relative_to(self.sidecar_dir))

    def _resize_all(self, size: int) -> None:
        """把所有数据集扩展到包含 size 行。"""

        for dataset in self._datasets.values():
            dataset.resize((size, *dataset.shape[1:]))

    def close(self) -> None:
        """刷新元数据并关闭所有输出。"""

        if self.hdf5_enabled and self.f is not None:
            self.f.attrs["num_steps"] = int(self.count)
            self.f.flush()
            self.f.close()
            self.f = None
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        if self.frame_csv_file is not None:
            self.frame_csv_file.flush()
            self.frame_csv_file.close()
            self.frame_csv_file = None
            self.frame_csv_writer = None
        if self.metadata is not None:
            self.metadata["num_steps"] = int(self.count)
            self._write_sidecar_metadata()

    def _write_sidecar_metadata(self) -> None:
        """把视频/图片 sidecar 的元数据写成 JSON。"""

        if self.metadata is None:
            return
        path = self.sidecar_dir / "metadata.json"
        path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
