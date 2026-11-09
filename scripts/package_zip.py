"""用标准 forward-slash 路径压缩 onedir 目录，兼容 Windows/Linux 解压工具。"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: package_zip.py <source_dir> <output.zip>")
        return 2
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    if not source.is_dir():
        print(f"source dir not found: {source}")
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(source).as_posix())
    print(f"zip written: {target} ({len(files)} files, {target.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
