"""实时采集线程：采集 -> 降采样 -> 跟踪 -> 发回 UI。"""
from __future__ import annotations

import time
import traceback
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from pcb_ar.core.tracker import GlobalLocalTracker, TrackResult
from pcb_ar.core.video import FrameData, VideoSource


class LiveWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    fatal_error = pyqtSignal(str)
    tracker_error = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    def __init__(
        self,
        source: VideoSource,
        tracker: GlobalLocalTracker | None = None,
        max_width: int = 1600,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.tracker = tracker
        self.max_width = max(640, int(max_width))
        self._running = True
        self._paused = False
        self._paused_source = False
        self._frame_times: deque[float] = deque(maxlen=20)
        self._last_frame: FrameData | None = None

    @property
    def source_description(self) -> str:
        try:
            return self.source.description
        except Exception:  # noqa: BLE001
            return "视频源"

    def is_paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()

    def run(self) -> None:  # noqa: C901
        failures = 0
        try:
            while self._running and not self.isInterruptionRequested():
                if self._paused:
                    if not self._paused_source and hasattr(self.source, "pause"):
                        try:
                            self.source.pause()  # type: ignore[attr-defined]
                            self._paused_source = True
                        except Exception:  # noqa: BLE001
                            pass
                    self.msleep(40)
                    continue
                if self._paused_source and hasattr(self.source, "resume"):
                    try:
                        self.source.resume()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    self._paused_source = False
                try:
                    data = self.source.read()
                except Exception as exc:  # noqa: BLE001
                    self.fatal_error.emit(f"读取视频源失败：{exc}\n{traceback.format_exc(limit=2)}")
                    break
                if data is None:
                    failures += 1
                    if failures >= 12:
                        self.fatal_error.emit(
                            "视频源连续多帧读取失败，可能已断开或被占用。"
                        )
                        break
                    self.msleep(50)
                    continue
                failures = 0
                frame = data.image
                original_h, original_w = frame.shape[:2]
                scale = 1.0
                if original_w > self.max_width:
                    scale = self.max_width / float(original_w)
                    frame = cv2.resize(
                        frame,
                        (self.max_width, max(1, int(round(original_h * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                    truth = data.truth_matrix
                    if truth is not None:
                        truth = np.array(
                            [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                            dtype=np.float64,
                        ) @ _as_h(truth)
                else:
                    truth = data.truth_matrix
                track: TrackResult | None = None
                if self.tracker is not None:
                    try:
                        track = self.tracker.track(frame)
                    except Exception as exc:  # noqa: BLE001
                        self.tracker_error.emit(f"跟踪线程异常，已继续显示原始画面：{exc}")
                        track = None
                display = FrameData(
                    image=frame,
                    timestamp=data.timestamp,
                    frame_index=data.frame_index,
                    truth_matrix=truth,
                    source_label=data.source_label,
                )
                self._last_frame = display
                now = time.monotonic()
                self._frame_times.append(now)
                self.frame_ready.emit(display, track)
                if len(self._frame_times) >= 2 and (
                    len(self._frame_times) % 5 == 0 or len(self._frame_times) == 20
                ):
                    span = self._frame_times[-1] - self._frame_times[0]
                    if span > 0:
                        self.fps_updated.emit((len(self._frame_times) - 1) / span)
        except Exception as exc:  # noqa: BLE001
            self.fatal_error.emit(f"采集线程异常：{exc}\n{traceback.format_exc(limit=3)}")
        finally:
            try:
                self.source.release()
            except Exception:  # noqa: BLE001
                pass


def _as_h(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = m
        return h
    return m
