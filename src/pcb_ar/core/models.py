"""工程数据模型：元件、参数、基准点与工程。"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from pcb_ar.core.geo import BBox

SCHEMA_VERSION = 1
CORE_SLOT_COUNT = 3

STATUS_PENDING = "pending"
STATUS_VERIFIED = "verified"
STATUS_ISSUE = "issue"
STATUSES = (STATUS_PENDING, STATUS_VERIFIED, STATUS_ISSUE)
STATUS_LABELS = {
    STATUS_PENDING: "待核对",
    STATUS_VERIFIED: "已核对",
    STATUS_ISSUE: "异常",
}
STATUS_COLORS = {
    STATUS_PENDING: "#F5A524",
    STATUS_VERIFIED: "#2DD4A7",
    STATUS_ISSUE: "#FF5C7A",
}
DEFAULT_CORE_NAMES = ("参数", "封装", "备注")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in STATUSES:
        return text
    reverse = {label: key for key, label in STATUS_LABELS.items()}
    return reverse.get(str(value or "").strip(), STATUS_PENDING)


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


@dataclass
class Param:
    name: str
    value: str = ""

    def __post_init__(self) -> None:
        self.name = _clean_text(self.name)
        self.value = _clean_text(self.value)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Any) -> "Param":
        if isinstance(raw, Param):
            return cls(raw.name, raw.value)
        if isinstance(raw, dict):
            return cls(_clean_text(raw.get("name")), _clean_text(raw.get("value")))
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return cls(_clean_text(raw[0]), _clean_text(raw[1]))
        return cls("", _clean_text(raw))


@dataclass
class Component:
    refdes: str
    bbox: BBox
    category: str = "未分类"
    core: list[Param] = field(default_factory=list)
    extended: dict[str, str] = field(default_factory=dict)
    status: str = STATUS_PENDING
    notes: str = ""
    source: str = "manual"
    angle: float = 0.0
    id: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.refdes = _clean_text(self.refdes, "未命名")
        self.category = _clean_text(self.category, "未分类") or "未分类"
        self.bbox = BBox.from_sequence(self.bbox)
        self.core = [Param.from_dict(p) for p in list(self.core)[:CORE_SLOT_COUNT]]
        while len(self.core) < CORE_SLOT_COUNT:
            idx = len(self.core)
            self.core.append(Param(DEFAULT_CORE_NAMES[idx], ""))
        self.extended = {
            _clean_text(k): _clean_text(v) for k, v in dict(self.extended or {}).items() if _clean_text(k)
        }
        self.status = normalize_status(self.status)
        self.notes = _clean_text(self.notes)
        self.source = _clean_text(self.source, "manual") or "manual"
        self.id = self.id or f"cmp-{uuid.uuid4().hex[:10]}"
        self.updated_at = self.updated_at or now_iso()

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def status_color(self) -> str:
        return STATUS_COLORS[self.status]

    @property
    def primary_value(self) -> str:
        for param in self.core:
            if param.value:
                return param.value
        return self.category

    def inline_lines(self, limit: int = CORE_SLOT_COUNT) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        for index, param in enumerate(self.core[:limit]):
            if param.value or param.name:
                name = param.name or f"参数{index + 1}"
                lines.append((name, param.value))
        return lines

    def core_values(self) -> list[str]:
        return [param.value for param in self.core[:CORE_SLOT_COUNT]]

    def set_core_values(self, values: Iterable[str]) -> None:
        values_list = list(values)[:CORE_SLOT_COUNT]
        for index in range(CORE_SLOT_COUNT):
            if index < len(values_list):
                self.core[index].value = _clean_text(values_list[index])
        self.touch()

    def set_status(self, status: str) -> None:
        self.status = normalize_status(status)
        self.touch()

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "refdes": self.refdes,
            "category": self.category,
            "bbox": self.bbox.as_list(),
            "angle": round(float(self.angle), 3),
            "core": [p.to_dict() for p in self.core],
            "extended": dict(self.extended),
            "status": self.status,
            "notes": self.notes,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Component":
        raw = dict(raw or {})
        bbox_raw = raw.get("bbox")
        if bbox_raw is None:
            bbox_raw = raw.get("base_pixel_box")
        if bbox_raw is None:
            bbox_raw = [0, 0, 1, 1]
        core_raw = raw.get("core", raw.get("core_params", []))
        if isinstance(core_raw, dict):
            names = list(DEFAULT_CORE_NAMES)
            core = [
                Param(name, _clean_text(value))
                for name, value in zip(names, list(core_raw.values())[:CORE_SLOT_COUNT])
            ]
        else:
            core = [Param.from_dict(item) for item in list(core_raw or [])[:CORE_SLOT_COUNT]]
        extended = raw.get("extended", raw.get("extended_params", {}))
        return cls(
            refdes=_clean_text(raw.get("refdes") or raw.get("name") or "未命名", "未命名"),
            bbox=BBox.from_sequence(bbox_raw),
            category=_clean_text(raw.get("category"), "未分类"),
            core=core,
            extended={str(k): _clean_text(v) for k, v in dict(extended or {}).items()},
            status=normalize_status(raw.get("status", STATUS_PENDING)),
            notes=_clean_text(raw.get("notes") or raw.get("sop_note")),
            source=_clean_text(raw.get("source"), "manual"),
            angle=float(raw.get("angle", 0.0) or 0.0),
            id=_clean_text(raw.get("id")),
            updated_at=_clean_text(raw.get("updated_at")),
        )


@dataclass
class Fiducial:
    name: str
    x: float
    y: float
    kind: str = "fiducial"
    radius: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "kind": self.kind,
            "radius": round(float(self.radius), 3),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Fiducial":
        return cls(
            name=_clean_text(raw.get("name"), "FID"),
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            kind=_clean_text(raw.get("kind"), "fiducial"),
            radius=float(raw.get("radius", 12.0)),
        )


@dataclass
class Project:
    name: str = "未命名工程"
    global_image: str = "assets/global_board.png"
    width: int = 0
    height: int = 0
    mm_per_px: float = 0.0
    components: list[Component] = field(default_factory=list)
    fiducials: list[Fiducial] = field(default_factory=list)
    notes: str = ""
    demo: bool = False
    sparse_map: str = ""
    structural_map: str = ""
    map_mode: str = "full"
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _clean_text(self.name, "未命名工程") or "未命名工程"
        self.components = [
            c if isinstance(c, Component) else Component.from_dict(c) for c in self.components
        ]
        self.fiducials = [
            f if isinstance(f, Fiducial) else Fiducial.from_dict(f) for f in self.fiducials
        ]
        self.width = int(self.width)
        self.height = int(self.height)
        self.mm_per_px = float(self.mm_per_px or 0.0)
        self.sparse_map = _clean_text(self.sparse_map)
        self.structural_map = _clean_text(self.structural_map)
        self.map_mode = _clean_text(self.map_mode, "full") or "full"
        if self.map_mode not in ("full", "sparse", "hybrid"):
            self.map_mode = "full"
        self._dedupe()

    def _dedupe(self) -> None:
        seen_ids: set[str] = set()
        seen_refs: set[str] = set()
        for comp in self.components:
            if not comp.id or comp.id in seen_ids:
                comp.id = f"cmp-{uuid.uuid4().hex[:10]}"
            seen_ids.add(comp.id)
            if not comp.refdes:
                comp.refdes = "C1"
            base = comp.refdes.strip()
            candidate = base
            suffix = 2
            while candidate.lower() in seen_refs:
                candidate = f"{base}_{suffix}"
                suffix += 1
            comp.refdes = candidate
            seen_refs.add(candidate.lower())

    @property
    def image_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def count(self) -> int:
        return len(self.components)

    def find_by_id(self, component_id: str) -> Component | None:
        return next((c for c in self.components if c.id == component_id), None)

    def find_by_refdes(self, refdes: str) -> Component | None:
        target = refdes.strip().lower()
        return next((c for c in self.components if c.refdes.strip().lower() == target), None)

    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for comp in self.components:
            counts[comp.status] = counts.get(comp.status, 0) + 1
        return counts

    def add_component(self, component: Component) -> Component:
        if not component.refdes or component.refdes == "未命名":
            component.refdes = self.next_refdes(_prefix_for_category(component.category))
        if any(c.refdes.lower() == component.refdes.lower() for c in self.components):
            component.refdes = self.unique_refdes(component.refdes)
        if any(c.id == component.id for c in self.components):
            component.id = f"cmp-{uuid.uuid4().hex[:10]}"
        self.components.append(component)
        self.touch()
        return component

    def remove_component(self, component_id: str) -> bool:
        before = len(self.components)
        self.components = [c for c in self.components if c.id != component_id]
        if len(self.components) != before:
            self.touch()
            return True
        return False

    def unique_refdes(self, preferred: str) -> str:
        base = _clean_text(preferred, "C1") or "C1"
        if not any(c.refdes.lower() == base.lower() for c in self.components):
            return base
        match = re.match(r"^(.*?)(\d+)$", base)
        prefix, number = (match.group(1), int(match.group(2))) if match else (base, 1)
        while True:
            candidate = f"{prefix}{number}"
            if not any(c.refdes.lower() == candidate.lower() for c in self.components):
                return candidate
            number += 1

    def next_refdes(self, prefix: str = "C") -> str:
        prefix = _clean_text(prefix, "C") or "C"
        used: set[int] = set()
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
        for comp in self.components:
            match = pattern.match(comp.refdes)
            if match:
                used.add(int(match.group(1)))
        number = 1
        while number in used:
            number += 1
        return f"{prefix}{number}"

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "global_image": self.global_image,
            "width": self.width,
            "height": self.height,
            "mm_per_px": self.mm_per_px,
            "demo": self.demo,
            "sparse_map": self.sparse_map,
            "structural_map": self.structural_map,
            "map_mode": self.map_mode,
            "components": [c.to_dict() for c in self.components],
            "fiducials": [f.to_dict() for f in self.fiducials],
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Project":
        raw = dict(raw or {})
        schema = int(raw.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
        if schema > SCHEMA_VERSION:
            raise ValueError(f"工程版本 {schema} 高于当前程序支持的 {SCHEMA_VERSION}")
        return cls(
            name=_clean_text(raw.get("name"), "未命名工程"),
            global_image=_clean_text(raw.get("global_image"), "assets/global_board.png"),
            width=int(raw.get("width", 0) or 0),
            height=int(raw.get("height", 0) or 0),
            mm_per_px=float(raw.get("mm_per_px", 0.0) or 0.0),
            components=[Component.from_dict(c) for c in raw.get("components", [])],
            fiducials=[Fiducial.from_dict(f) for f in raw.get("fiducials", [])],
            notes=_clean_text(raw.get("notes")),
            demo=bool(raw.get("demo", False)),
            sparse_map=_clean_text(raw.get("sparse_map")),
            structural_map=_clean_text(raw.get("structural_map")),
            map_mode=_clean_text(raw.get("map_mode"), "full"),
            schema_version=schema,
            created_at=_clean_text(raw.get("created_at"), now_iso()),
            updated_at=_clean_text(raw.get("updated_at"), now_iso()),
            extra=dict(raw.get("extra", {}) or {}),
        )


def _prefix_for_category(category: str) -> str:
    text = category.lower()
    if "电阻" in text or "resistor" in text or text == "r":
        return "R"
    if "电容" in text or "capacitor" in text or text == "c":
        return "C"
    if "芯片" in text or "ic" in text or "mcu" in text:
        return "U"
    if "连接" in text or "connector" in text or "接口" in text:
        return "J"
    if "二极管" in text or "diode" in text:
        return "D"
    if "电感" in text or "inductor" in text:
        return "L"
    if "晶振" in text or "crystal" in text:
        return "Y"
    return "C"


def new_component(
    refdes: str,
    bbox: BBox,
    category: str = "未分类",
    value: str = "",
    package: str = "",
    note: str = "",
    source: str = "manual",
) -> Component:
    comp = Component(
        refdes=refdes,
        bbox=bbox,
        category=category,
        core=[
            Param("参数", value),
            Param("封装", package),
            Param("备注", note),
        ],
        source=source,
    )
    return comp
