from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pcb_ar.core.geo import BBox
from pcb_ar.core.imageio import read_image
from pcb_ar.core.models import Component, Param, Project, new_component
from pcb_ar.core.project import ProjectError, ProjectStore


def _sample_image() -> np.ndarray:
    image = np.full((240, 360, 3), (35, 70, 30), dtype=np.uint8)
    cv2.rectangle(image, (50, 60), (150, 120), (205, 210, 215), -1)
    cv2.putText(image, "PCB TEST", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 235, 230), 2)
    return image


def test_component_core_slots_are_limited():
    comp = Component(
        "R1",
        BBox(1, 2, 3, 4),
        core=[Param(str(i), str(i)) for i in range(6)],
    )
    assert len(comp.core) == 3
    assert comp.core[0].name == "0"
    comp.set_core_values(["10k", "0603", "note", "ignored"])
    assert comp.core_values() == ["10k", "0603", "note"]


def test_project_deduplicates_refdes():
    project = Project(
        components=[
            new_component("R1", BBox(0, 0, 10, 10)),
            new_component("R1", BBox(20, 0, 10, 10)),
            new_component("", BBox(40, 0, 10, 10)),
        ]
    )
    refs = [c.refdes for c in project.components]
    assert len(refs) == len(set(refs))
    assert project.next_refdes("R") == "R2"


def test_project_store_roundtrip_unicode(tmp_path: Path):
    project_dir = tmp_path / "中文目录"
    project_dir.mkdir()
    path = project_dir / "测试工程.pcbproj.json"
    image = _sample_image()
    store = ProjectStore.create(path, "测试工程", image, mm_per_px=0.01)
    comp = new_component("U1", BBox(50, 60, 100, 60), "芯片", "MCU", "LQFP", "测试")
    store.project.add_component(comp)
    store.project.fiducials = []
    store.project.notes = "中文路径与中文内容"
    store.save()

    loaded = ProjectStore.load(path)
    assert loaded.project.name == "测试工程"
    assert loaded.project.count == 1
    assert loaded.project.components[0].refdes == "U1"
    assert loaded.project.components[0].core[0].value == "MCU"
    assert loaded.project.mm_per_px == pytest.approx(0.01)
    reloaded_image = read_image(loaded.image_path)
    assert reloaded_image.shape == image.shape

    loaded.project.notes = "第二次保存"
    loaded.save()
    assert path.with_name(path.name + ".bak").exists()


def test_project_store_save_as(tmp_path: Path):
    path = tmp_path / "a.pcbproj.json"
    store = ProjectStore.create(path, "A", _sample_image())
    new_path = tmp_path / "sub" / "b.pcbproj.json"
    store.save_as(new_path)
    assert new_path.exists()
    assert (new_path.parent / "b_assets" / "global_board.png").exists()
    loaded = ProjectStore.load(new_path)
    assert loaded.project.name == "A"


def test_project_store_missing_image(tmp_path: Path):
    path = tmp_path / "bad.pcbproj.json"
    store = ProjectStore.create(path, "Bad", _sample_image())
    store.image_path.unlink()
    with pytest.raises(ProjectError):
        ProjectStore.load(path)
