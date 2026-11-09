"""兼容中文/Unicode 路径的图像读写。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageIOError(RuntimeError):
    pass


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise ImageIOError(f"图像不存在：{path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, flags)
    except (OSError, ValueError) as exc:
        raise ImageIOError(f"读取图像失败：{path} ({exc})") from exc
    if image is None:
        raise ImageIOError(f"无法解码图像：{path}")
    return image


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """兼容旧调用的别名。"""
    return read_image(path, flags)


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    path = Path(path)
    if image is None or not hasattr(image, "shape"):
        raise ImageIOError("待保存图像为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    try:
        ok, buffer = cv2.imencode(suffix, image, params)
    except cv2.error as exc:
        raise ImageIOError(f"编码图像失败：{path} ({exc})") from exc
    if not ok:
        raise ImageIOError(f"编码图像失败：{path}")
    try:
        buffer.tofile(str(path))
    except OSError as exc:
        raise ImageIOError(f"写入图像失败：{path} ({exc})") from exc
    return path


def image_size(path: str | Path) -> tuple[int, int]:
    image = read_image(path)
    h, w = image.shape[:2]
    return int(w), int(h)
