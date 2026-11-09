"""轻量应用配置（保存在 LocalAppData，不使用注册表）。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from pcb_ar.paths import config_path


@dataclass
class AppConfig:
    last_project: str = ""
    max_display_width: int = 1600
    global_match_width: int = 1800
    tracker_global_features: int = 5000
    tracker_live_features: int = 1600
    tracker_min_matches: int = 18
    tracker_min_inliers: int = 12
    tracker_ratio: float = 0.76
    tracker_ransac_px: float = 4.0
    tracker_smoothing: float = 0.65
    tracker_max_lost_frames: int = 8
    mm_per_px: float = 0.0
    labels_visible: bool = True
    expanded_labels: bool = False
    window_geometry: str = ""
    window_state: str = ""
    autoplay_demo: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or config_path()
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    def save(self, path: Path | None = None) -> None:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def update(self, **values: Any) -> None:
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)
