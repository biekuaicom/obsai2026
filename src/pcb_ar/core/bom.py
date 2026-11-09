"""BOM / 元件清单 CSV 导入导出。"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcb_ar.core.models import STATUS_LABELS, Component, Project


class BomError(RuntimeError):
    pass


HEADER_ALIASES = {
    "refdes": {"refdes", "ref", "reference", "designator", "位号", "元件号", "器件位号", "标号"},
    "value": {"value", "comment", "参数", "值", "元件值", "规格"},
    "package": {"package", "footprint", "封装", "封装形式"},
    "category": {"category", "type", "类别", "类型", "元件类型"},
    "notes": {"notes", "note", "description", "备注", "说明"},
}


@dataclass
class BomRow:
    refdes: str
    value: str = ""
    package: str = ""
    category: str = ""
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class BomApplyReport:
    updated: int = 0
    created: int = 0
    unmatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"更新 {self.updated} 个元件"]
        if self.created:
            parts.append(f"新建 {self.created} 个")
        if self.unmatched:
            parts.append(f"未匹配 {len(self.unmatched)} 行")
        if self.missing:
            parts.append(f"BOM 缺少 {len(self.missing)} 个已标注元件")
        if self.duplicates:
            parts.append(f"重复位号 {len(self.duplicates)} 个")
        return "，".join(parts)


def _normalize_header(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "").replace("_", "")


def _canonical_header(header: str) -> str:
    normalized = _normalize_header(header)
    for canonical, aliases in HEADER_ALIASES.items():
        if normalized in {_normalize_header(a) for a in aliases}:
            return canonical
    return header.strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(line for line in text.splitlines()[:10] if line.strip())
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        counts = {delim: sample.count(delim) for delim in (",", ";", "\t")}
        return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def parse_bom_text(text: str) -> list[BomRow]:
    delimiter = _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        raise BomError("BOM 文件为空")
    header = [_canonical_header(cell) for cell in rows[0]]
    if "refdes" not in header:
        raise BomError("BOM 缺少位号列（支持：位号 / RefDes / Designator / Reference）")
    result: list[BomRow] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not any(str(cell).strip() for cell in row):
            continue
        record = {header[i]: str(row[i]).strip() for i in range(min(len(header), len(row)))}
        refdes = record.pop("refdes", "").strip()
        if not refdes:
            continue
        result.append(
            BomRow(
                refdes=refdes,
                value=record.pop("value", ""),
                package=record.pop("package", ""),
                category=record.pop("category", ""),
                notes=record.pop("notes", ""),
                extra={k: v for k, v in record.items() if k and v},
            )
        )
    if not result:
        raise BomError("BOM 中没有有效数据行")
    return result


def load_bom(path: str | Path) -> list[BomRow]:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BomError(f"读取 BOM 失败：{path}\n{exc}") from exc
    return parse_bom_text(_decode(data))


def apply_bom(project: Project, rows: list[BomRow], match_case: bool = False) -> BomApplyReport:
    report = BomApplyReport()
    by_ref: dict[str, Component] = {}
    for comp in project.components:
        key = comp.refdes.strip() if match_case else comp.refdes.strip().lower()
        by_ref[key] = comp
    seen: set[str] = set()
    for row in rows:
        key = row.refdes.strip() if match_case else row.refdes.strip().lower()
        if key in seen:
            report.duplicates.append(row.refdes)
            continue
        seen.add(key)
        comp = by_ref.get(key)
        if comp is None:
            report.unmatched.append(row.refdes)
            continue
        if row.value:
            comp.core[0].value = row.value
        if row.package:
            comp.core[1].value = row.package
        if row.category:
            comp.category = row.category
        if row.notes:
            comp.notes = row.notes
        for name, value in row.extra.items():
            comp.extended[name] = value
        comp.touch()
        report.updated += 1
    bom_keys = {row.refdes.strip() if match_case else row.refdes.strip().lower() for row in rows}
    for comp in project.components:
        key = comp.refdes.strip() if match_case else comp.refdes.strip().lower()
        if key not in bom_keys:
            report.missing.append(comp.refdes)
    if report.updated:
        project.touch()
    return report


def _ordered_extra_keys(components: list[Component]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for comp in components:
        for key in comp.extended:
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def bom_csv_text(project: Project) -> str:
    extra_keys = _ordered_extra_keys(project.components)
    core_names = []
    for comp in project.components:
        for param in comp.core:
            if param.name and param.name not in core_names and param.name not in ("参数", "封装"):
                core_names.append(param.name)
    header = ["位号", "类别", "参数", "封装", "备注", "状态", *extra_keys, *core_names]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for comp in project.components:
        row = [
            comp.refdes,
            comp.category,
            comp.core[0].value if len(comp.core) > 0 else "",
            comp.core[1].value if len(comp.core) > 1 else "",
            comp.notes,
            STATUS_LABELS.get(comp.status, comp.status),
        ]
        row.extend(comp.extended.get(key, "") for key in extra_keys)
        for name in core_names:
            value = ""
            for param in comp.core:
                if param.name == name:
                    value = param.value
                    break
            row.append(value)
        writer.writerow(row)
    return buffer.getvalue()


def export_bom(project: Project, path: str | Path) -> Path:
    path = Path(path)
    try:
        path.write_text(bom_csv_text(project), encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise BomError(f"写入 BOM 失败：{path}\n{exc}") from exc
    return path
