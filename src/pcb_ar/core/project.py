"""工程文件（JSON + 全局图）的创建、读取与原子保存。"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pcb_ar.core.imageio import read_image, write_image
from pcb_ar.core.models import Project

PROJECT_SUFFIX = ".pcbproj.json"


class ProjectError(RuntimeError):
    pass


def _ensure_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    return path.with_name(path.name + PROJECT_SUFFIX)


def _project_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(PROJECT_SUFFIX):
        return name[: -len(PROJECT_SUFFIX)]
    return path.stem


class ProjectStore:
    """一个工程 = 一个 JSON 文件 + 同目录下的 assets 全局图。"""

    def __init__(self, path: Path, project: Project, assets_dir: Path | None = None) -> None:
        self.path = Path(path)
        self.project = project
        self.assets_dir = assets_dir or self.path.parent / f"{_project_stem(self.path)}_assets"
        self._dirty = False

    @property
    def image_path(self) -> Path:
        image = Path(self.project.global_image)
        if image.is_absolute():
            return image
        return (self.path.parent / image).resolve()

    @property
    def sparse_map_path(self) -> Path | None:
        if not self.project.sparse_map:
            return None
        path = Path(self.project.sparse_map)
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty

    def load_image(self) -> np.ndarray:
        try:
            return read_image(self.image_path)
        except Exception as exc:  # noqa: BLE001 - 转换为用户可读错误
            raise ProjectError(f"无法加载工程全局图：{self.image_path}\n{exc}") from exc

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str,
        image: np.ndarray,
        mm_per_px: float = 0.0,
        *,
        demo: bool = False,
    ) -> "ProjectStore":
        path = _ensure_suffix(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = _project_stem(path)
        assets = path.parent / f"{stem}_assets"
        assets.mkdir(parents=True, exist_ok=True)
        image_path = assets / "global_board.png"
        write_image(image_path, image)
        height, width = image.shape[:2]
        project = Project(
            name=name or stem,
            global_image=f"{assets.name}/global_board.png",
            width=int(width),
            height=int(height),
            mm_per_px=float(mm_per_px or 0.0),
            demo=demo,
        )
        store = cls(path, project, assets)
        store.save()
        return store

    @classmethod
    def load(cls, path: str | Path) -> "ProjectStore":
        path = Path(path)
        if not path.exists():
            raise ProjectError(f"工程文件不存在：{path}")
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProjectError(f"读取工程失败：{path}\n{exc}") from exc
        if not isinstance(raw, dict):
            raise ProjectError("工程文件格式错误：根节点必须是对象")
        try:
            project = Project.from_dict(raw)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProjectError(f"工程数据损坏：{exc}") from exc
        store = cls(path, project)
        if not store.image_path.exists():
            raise ProjectError(f"工程引用的全局图不存在：{store.image_path}")
        try:
            image = read_image(store.image_path)
            height, width = image.shape[:2]
        except Exception as exc:  # noqa: BLE001
            raise ProjectError(f"全局图无法读取：{store.image_path}\n{exc}") from exc
        if not project.width or not project.height:
            project.width, project.height = width, height
        elif (project.width, project.height) != (width, height):
            raise ProjectError(
                "工程记录尺寸与全局图不一致："
                f"记录 {project.width}x{project.height}，实际 {width}x{height}"
            )
        return store

    def save(self) -> None:
        if not self.image_path.exists():
            raise ProjectError(f"保存失败，全局图不存在：{self.image_path}")
        self.project.touch()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
            except OSError:
                pass
        payload = json.dumps(self.project.to_dict(), ensure_ascii=False, indent=2)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            raise ProjectError(f"写入工程失败：{self.path}\n{exc}") from exc
        self._dirty = False

    def save_as(self, new_path: str | Path) -> "ProjectStore":
        old_assets = self.assets_dir
        old_sparse = self.sparse_map_path
        new_path = _ensure_suffix(Path(new_path))
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_stem = _project_stem(new_path)
        new_assets = new_path.parent / f"{new_stem}_assets"
        if new_assets.resolve() != old_assets.resolve():
            new_assets.mkdir(parents=True, exist_ok=True)
            # 稀疏地图含 sparse_map.json、特征 npz 和 keyframes/，
            # 必须整体复制，不能只复制总览图。
            if old_assets.exists():
                for item in old_assets.iterdir():
                    target_item = new_assets / item.name
                    if item.is_dir():
                        shutil.copytree(item, target_item, dirs_exist_ok=True)
                    elif item.resolve() != target_item.resolve():
                        shutil.copy2(item, target_item)
            target = new_assets / "global_board.png"
            if self.image_path.resolve() != target.resolve():
                shutil.copy2(self.image_path, target)
        self.project.global_image = f"{new_assets.name}/global_board.png"
        if old_sparse is not None and old_sparse.exists() and old_sparse.parent.resolve() == old_assets.resolve():
            self.project.sparse_map = f"{new_assets.name}/{old_sparse.name}"
        self.path = new_path
        self.assets_dir = new_assets
        self.save()
        return self

    def set_global_image(self, image: np.ndarray, name: str = "global_board.png") -> None:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        target = self.assets_dir / name
        write_image(target, image)
        self.project.global_image = f"{self.assets_dir.name}/{name}"
        self.project.width, self.project.height = int(image.shape[1]), int(image.shape[0])
        self.project.touch()
        self.mark_dirty()

    @property
    def display_name(self) -> str:
        return self.project.name or _project_stem(self.path)
