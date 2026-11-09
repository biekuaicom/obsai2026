"""视频输入源：USB 相机、视频文件、合成显微镜演示。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pcb_ar.core.synthetic import demoTrackPose, synthetic_view


@dataclass
class FrameData:
    image: np.ndarray
    timestamp: float
    frame_index: int
    truth_matrix: np.ndarray | None = None
    source_label: str = ""


class VideoSource(Protocol):
    def read(self) -> FrameData | None: ...
    def release(self) -> None: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def description(self) -> str: ...


class CameraSource:
    def __init__(
        self,
        index: int = 0,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        backend: str = "auto",
    ) -> None:
        self.index = int(index)
        self.cap: cv2.VideoCapture | None = None
        backends: list[int] = []
        if backend == "dshow":
            backends = [cv2.CAP_DSHOW]
        elif backend == "msmf":
            backends = [cv2.CAP_MSMF]
        else:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for api in backends:
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(f"无法打开摄像头 {self.index}，请检查连接或是否被其他程序占用")
        if width > 0 and height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame_index = 0
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._description = f"摄像头 #{self.index}  {actual_w}x{actual_h}"
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def is_open(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    @property
    def description(self) -> str:
        return self._description

    def read(self) -> FrameData | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self._frame_index += 1
        return FrameData(frame, time.time(), self._frame_index, None, self._description)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileSource:
    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件：{self.path}")
        self.loop = loop
        self._frame_index = 0
        self._description = f"视频文件 {self.path.name}"
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._next_deadline = time.perf_counter()

    @property
    def is_open(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    @property
    def description(self) -> str:
        return self._description

    def read(self) -> FrameData | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        if self._fps > 1.0:
            self._next_deadline += 1.0 / self._fps
            delay = self._next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(min(delay, 0.1))
            else:
                self._next_deadline = time.perf_counter()
        self._frame_index += 1
        return FrameData(frame, time.time(), self._frame_index, None, self._description)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SyntheticMicroscopeSource:
    """以合成板图为基础生成连续移动的显微镜视野。"""

    def __init__(
        self,
        board: np.ndarray,
        size: tuple[int, int] = (1600, 1000),
        fps: float = 15.0,
        seed: int = 7,
    ) -> None:
        self.board = board
        self.size = (int(size[0]), int(size[1]))
        self.fps = max(1.0, float(fps))
        self.seed = int(seed)
        self._frame_index = 0
        self._start = time.perf_counter()
        self._paused_at: float | None = None
        self._paused_total = 0.0
        self._next_deadline = self._start
        self._description = "模拟显微镜（合成演示板）"
        self.last_truth: np.ndarray | None = None

    @property
    def is_open(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return self._description

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.perf_counter()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.perf_counter() - self._paused_at
            self._paused_at = None
            self._next_deadline = time.perf_counter()

    def read(self) -> FrameData | None:
        self._next_deadline += 1.0 / self.fps
        delay = self._next_deadline - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, 0.2))
        else:
            self._next_deadline = time.perf_counter()
        now = time.perf_counter()
        if self._paused_at is not None:
            t = self._paused_at - self._start - self._paused_total
        else:
            t = now - self._start - self._paused_total
        center, angle, scale = demoTrackPose(t)
        # 实时演示不做逐帧高斯噪声（4.8M 随机数会让 1600x1000 生成耗时增加约 50 倍）。
        # 真实相机本身的噪声已足够；如需要纹理，可在离线素材中预生成。
        frame, truth = synthetic_view(
            self.board,
            center,
            scale,
            angle,
            self.size,
            seed=0,
        )
        self._frame_index += 1
        self.last_truth = truth
        return FrameData(frame, time.time(), self._frame_index, truth, self._description)

    def release(self) -> None:
        return None


def open_camera(index: int = 0, width: int = 0, height: int = 0) -> CameraSource:
    return CameraSource(index=index, width=width, height=height)
