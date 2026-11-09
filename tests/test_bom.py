from __future__ import annotations

from pathlib import Path

from pcb_ar.core.bom import apply_bom, export_bom, load_bom, parse_bom_text
from pcb_ar.core.geo import BBox
from pcb_ar.core.models import Project, new_component


def _project() -> Project:
    project = Project(name="BOM 测试")
    project.add_component(new_component("R1", BBox(0, 0, 10, 10), "电阻", "旧值", "0603"))
    project.add_component(new_component("U1", BBox(20, 0, 30, 30), "芯片"))
    return project


def test_parse_and_apply_chinese_bom(tmp_path: Path):
    text = (
        "位号,参数,封装,类别,备注,供应商\n"
        "R1,10k,0603,电阻,精度 1%,立创\n"
        "U1,STM32F407,LQFP100,芯片,主控,ST\n"
        "X9,不存在的,0603,电阻,,\n"
    )
    rows = parse_bom_text(text)
    assert len(rows) == 3
    project = _project()
    report = apply_bom(project, rows)
    assert report.updated == 2
    assert report.unmatched == ["X9"]
    r1 = project.find_by_refdes("R1")
    assert r1 is not None
    assert r1.core[0].value == "10k"
    assert r1.core[1].value == "0603"
    assert r1.extended["供应商"] == "立创"
    assert r1.notes == "精度 1%"


def test_bom_export_roundtrip(tmp_path: Path):
    project = _project()
    path = tmp_path / "bom.csv"
    export_bom(project, path)
    rows = load_bom(path)
    refs = {row.refdes for row in rows}
    assert refs == {"R1", "U1"}
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_english_headers_and_semicolon():
    rows = parse_bom_text("RefDes;Value;Package;Supplier\nR1;22R;0402;A\n")
    assert rows[0].refdes == "R1"
    assert rows[0].value == "22R"
    assert rows[0].extra["Supplier"] == "A"
