"""OpenCV 相机采集基础模块。

本模块负责相机探测、后台取帧，以及把 OpenCV 帧转换成 Qt 预览和
HDF5 数据集统一使用的 RGB 格式。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage


@dataclass
class FramePacket:
    """一帧相机图像，以及用于时间对齐的单调时钟时间戳。"""

    image_rgb: np.ndarray
    timestamp_ns: int
    frame_index: int


def parse_camera_id(value: str) -> int | str:
    """同时支持 OpenCV 数字编号和稳定的 /dev 设备路径。"""

    try:
        return int(value)
    except ValueError:
        return value


def open_video_capture(camera_id: int | str):
    """用尽量可靠的后端打开相机。

    在这台机器上，有些 RealSense 节点用 OpenCV 默认后端更稳，
    有些节点用 V4L2 更稳。这里两个都试一下，调用方不用关心细节。
    """

    attempts = [None, cv2.CAP_V4L2]
    last_cap = None
    for backend in attempts:
        cap = cv2.VideoCapture(camera_id) if backend is None else cv2.VideoCapture(camera_id, backend)
        last_cap = cap
        if cap.isOpened():
            return cap
        cap.release()
    return last_cap


def frame_to_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """把 OpenCV 读到的帧统一成固定尺寸、连续内存的 RGB uint8 图像。"""

    if frame.ndim == 2:
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if width > 0 and height > 0 and (rgb.shape[1] != width or rgb.shape[0] != height):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def rgb_to_qimage(image_rgb: np.ndarray) -> QImage:
    """把 RGB numpy 数组转换成 Qt 自己持有内存的 QImage，用于实时预览。"""

    height, width, channels = image_rgb.shape
    return QImage(
        image_rgb.data,
        width,
        height,
        channels * width,
        QImage.Format_RGB888,
    ).copy()


def list_cameras(max_index: int) -> None:
    """打印哪些 OpenCV 数字相机编号能打开并成功读到帧。"""

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


class CameraCapture(QObject):
    """在后台守护线程里持续采集最新图像。

    数采主循环按自己的固定频率取“最新一帧”。这样相机读取抖动不会阻塞
    机械臂控制，同时每帧仍然带时间戳，后续可以审查图像的新鲜度。
    """

    frame_ready = pyqtSignal(str, QImage)
    status = pyqtSignal(str)

    def __init__(
        self,
        name: str,
        camera_id: int | str,
        width: int,
        height: int,
        fps: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._running = False
        self._cap = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """启动相机采集循环。"""

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """请求采集循环停止，并尽快释放相机设备。"""

        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def join(self, timeout: float = 1.0) -> None:
        """短暂等待后台采集线程结束。"""

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def latest(self) -> FramePacket | None:
        """返回最新帧；如果还没有读到第一帧，则返回 None。"""

        with self._lock:
            return self._latest

    def _run(self) -> None:
        """相机工作循环：发送预览图像，并持续更新 _latest。"""

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

        self.status.emit(f"{self.name}: opened camera {self.camera_id}")
        frame_index = 0
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status.emit(f"{self.name}: read failed")
                time.sleep(0.05)
                continue

            timestamp_ns = time.monotonic_ns()
            image_rgb = frame_to_rgb(frame, self.width, self.height)
            packet = FramePacket(image_rgb=image_rgb, timestamp_ns=timestamp_ns, frame_index=frame_index)
            with self._lock:
                self._latest = packet
            self.frame_ready.emit(self.name, rgb_to_qimage(image_rgb))
            frame_index += 1

        cap.release()
        self._cap = None
        self.status.emit(f"{self.name}: released")
