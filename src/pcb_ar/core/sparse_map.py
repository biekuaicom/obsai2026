"""稀疏视觉地图：关键帧、ORB 特征、板坐标位姿与覆盖多边形。

设计目标
--------
* 默认不再保存/匹配整张高分辨率拼图；
* 每个关键帧只保存 1200px 压缩图、ORB 特征、板坐标位姿和覆盖区域；
* 定位时检索少量关键帧，恢复 当前视野 -> 板坐标 的相似变换；
* 全图模式仍可生成一张低分辨率总览，作为人工查看/全图增强的后备。
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pcb_ar.core.geo import (
    apply_affine,
    as_homography,
    invert_affine,
    is_valid_similarity,
)
from pcb_ar.core.imageio import read_image, write_image

SPARSE_MANIFEST = "sparse_map.json"
SPARSE_FEATURES = "sparse_features.npz"
KEYFRAME_DIR = "keyframes"


class SparseMapError(RuntimeError):
    pass


@dataclass
class FeatureSet:
    image: np.ndarray
    gray: np.ndarray
    scale: float  # work width / original width
    keypoints: np.ndarray  # Nx2 in work coordinates
    descriptors: np.ndarray  # Nx32 uint8
    global_descriptor: np.ndarray  # 1024 uint8 bits


@dataclass
class SparseKeyframe:
    index: int
    shape: tuple[int, int]  # stored keyframe (h,w)
    pose: np.ndarray  # 3x3: keyframe pixels -> board/overview pixels
    coverage: np.ndarray  # 4x2 board coordinates
    keypoints: np.ndarray  # Nx2
    descriptors: np.ndarray  # Nx32
    global_descriptor: np.ndarray  # 1024
    neighbors: list[int] = field(default_factory=list)
    image_file: str = ""
    jpeg: bytes | None = None

    @property
    def point_count(self) -> int:
        return int(self.keypoints.shape[0])


def extract_features(
    image: np.ndarray,
    *,
    max_width: int = 1200,
    nfeatures: int = 2600,
    clahe: bool = True,
) -> FeatureSet:
    """提取 ORB 特征，坐标与返回的 work 图像一致。"""
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        patchSize=31,
        fastThreshold=12,
    )
    kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((0, 32), dtype=np.uint8)
        points = np.zeros((0, 2), dtype=np.float32)
    else:
        points = np.float32([[kp.pt[0], kp.pt[1]] for kp in kps]).reshape(-1, 2)
    return FeatureSet(
        image=work,
        gray=gray,
        scale=scale,
        keypoints=points,
        descriptors=des,
        global_descriptor=global_descriptor(work),
    )


def global_descriptor(image: np.ndarray, size: int = 32) -> np.ndarray:
    """低分辨率边缘布局描述子，用于粗检索；最终匹配仍用 ORB。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 45, 130)
    edges = cv2.resize(edges, (size, size), interpolation=cv2.INTER_AREA)
    return (edges > 16).astype(np.uint8).reshape(-1)


@dataclass
class SparseMap:
    keyframes: list[SparseKeyframe]
    board_width: int
    board_height: int
    version: int = 1
    manifest_path: Path | None = None

    @property
    def count(self) -> int:
        return len(self.keyframes)

    @property
    def total_points(self) -> int:
        return int(sum(k.point_count for k in self.keyframes))

    def sorted_keyframes(self) -> list[SparseKeyframe]:
        return sorted(self.keyframes, key=lambda k: k.index)

    def coverage_polygons(self) -> list[np.ndarray]:
        return [k.coverage for k in self.sorted_keyframes()]

    def global_descriptors(self) -> np.ndarray:
        if not self.keyframes:
            return np.zeros((0, 1024), dtype=np.uint8)
        return np.stack([k.global_descriptor for k in self.sorted_keyframes()], axis=0)

    def save(self, assets_dir: str | Path) -> Path:
        """保存到工程 assets 目录，返回 manifest 路径。"""
        assets = Path(assets_dir)
        assets.mkdir(parents=True, exist_ok=True)
        kf_dir = assets / KEYFRAME_DIR
        kf_dir.mkdir(parents=True, exist_ok=True)
        if not self.keyframes:
            raise SparseMapError("稀疏地图没有关键帧")
        keypoints = []
        descriptors = []
        offsets = [0]
        global_desc = []
        entries = []
        for idx, kf in enumerate(self.sorted_keyframes()):
            if kf.descriptors.size and kf.keypoints.size:
                keypoints.append(np.asarray(kf.keypoints, dtype=np.float32))
                descriptors.append(np.asarray(kf.descriptors, dtype=np.uint8))
                offsets.append(offsets[-1] + int(kf.keypoints.shape[0]))
            else:
                offsets.append(offsets[-1])
            global_desc.append(np.asarray(kf.global_descriptor, dtype=np.uint8))
            image_name = f"kf_{idx:04d}.jpg"
            image_path = kf_dir / image_name
            if kf.jpeg is not None:
                image_path.write_bytes(kf.jpeg)
            elif kf.image_file and Path(kf.image_file).exists():
                shutil.copy2(kf.image_file, image_path)
            elif image_path.exists():
                pass
            else:
                raise SparseMapError(f"关键帧 {idx} 缺少图像数据")
            entries.append(
                {
                    "index": idx,
                    "shape": [int(kf.shape[0]), int(kf.shape[1])],
                    "pose": np.asarray(kf.pose, dtype=float).reshape(3, 3).tolist(),
                    "coverage": np.asarray(kf.coverage, dtype=float).reshape(-1, 2).tolist(),
                    "neighbors": [int(n) for n in kf.neighbors],
                    "image": f"{KEYFRAME_DIR}/{image_name}",
                }
            )
        kp_all = np.vstack(keypoints) if keypoints else np.zeros((0, 2), np.float32)
        des_all = np.vstack(descriptors) if descriptors else np.zeros((0, 32), np.uint8)
        np.savez_compressed(
            assets / SPARSE_FEATURES,
            keypoints=kp_all,
            descriptors=des_all,
            offsets=np.asarray(offsets, dtype=np.int32),
            global_descriptors=np.stack(global_desc, axis=0),
        )
        manifest = {
            "version": self.version,
            "board_width": int(self.board_width),
            "board_height": int(self.board_height),
            "features": SPARSE_FEATURES,
            "keyframes": entries,
        }
        path = assets / SPARSE_MANIFEST
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        self.manifest_path = path
        return path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "SparseMap":
        path = Path(manifest_path)
        if not path.exists():
            raise SparseMapError(f"稀疏地图不存在：{path}")
        assets = path.parent
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            data = np.load(assets / str(manifest.get("features", SPARSE_FEATURES)))
        except (OSError, ValueError, KeyError) as exc:
            raise SparseMapError(f"读取稀疏地图失败：{path}\n{exc}") from exc
        kp_all = data["keypoints"]
        des_all = data["descriptors"]
        offsets = data["offsets"]
        globals_all = data["global_descriptors"]
        if len(offsets) != len(manifest.get("keyframes", [])) + 1:
            raise SparseMapError("稀疏地图特征索引与关键帧数量不一致")
        keyframes: list[SparseKeyframe] = []
        for idx, entry in enumerate(manifest.get("keyframes", [])):
            start, end = int(offsets[idx]), int(offsets[idx + 1])
            image_rel = str(entry.get("image", ""))
            image_path = assets / image_rel
            if not image_path.exists():
                raise SparseMapError(f"关键帧图像缺失：{image_path}")
            keyframes.append(
                SparseKeyframe(
                    index=int(entry.get("index", idx)),
                    shape=(int(entry["shape"][0]), int(entry["shape"][1])),
                    pose=np.asarray(entry["pose"], dtype=np.float64).reshape(3, 3),
                    coverage=np.asarray(entry.get("coverage", []), dtype=np.float32).reshape(-1, 2),
                    keypoints=np.asarray(kp_all[start:end], dtype=np.float32),
                    descriptors=np.asarray(des_all[start:end], dtype=np.uint8),
                    global_descriptor=np.asarray(globals_all[idx], dtype=np.uint8),
                    neighbors=[int(n) for n in entry.get("neighbors", [])],
                    image_file=str(image_path),
                )
            )
        return cls(
            keyframes=keyframes,
            board_width=int(manifest.get("board_width", 0)),
            board_height=int(manifest.get("board_height", 0)),
            version=int(manifest.get("version", 1)),
            manifest_path=path,
        )

    def load_keyframe_image(self, keyframe: SparseKeyframe) -> np.ndarray:
        path = Path(keyframe.image_file)
        if not path.exists():
            raise SparseMapError(f"关键帧图像不存在：{path}")
        return read_image(path)

    def save_overview(self, assets_dir: str | Path, image: np.ndarray, name: str = "overview.png") -> str:
        assets = Path(assets_dir)
        assets.mkdir(parents=True, exist_ok=True)
        write_image(assets / name, image)
        return name


def ensure_sparse_path(project_path: Path, sparse_manifest: str | Path) -> str:
    """把 manifest 路径转换为相对工程 JSON 的路径。"""
    try:
        return Path(os.path.relpath(Path(sparse_manifest), project_path.parent)).as_posix()
    except ValueError:
        return str(sparse_manifest)


def point_to_board(pose: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    out = apply_affine(as_homography(pose)[:2, :], np.float32([[point[0], point[1]]]))[0]
    return float(out[0]), float(out[1])


def board_to_live(matrix_live_to_board: np.ndarray) -> np.ndarray:
    return invert_affine(matrix_live_to_board)
