"""Build a sanitized release ZIP from the validated Qwen demo workspace."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.package_hygiene import scan_entries, scan_tree
from scripts.prepare_demo_workspace import prepare


def scan_release(destination: Path) -> dict:
    destination = Path(destination).resolve()
    with zipfile.ZipFile(destination) as archive:
        result = scan_entries(
            (PurePosixPath(name), archive.read(name))
            for name in archive.namelist()
            if not name.endswith("/")
        )
    if result["failures"]:
        raise ValueError(f"Release ZIP hygiene scan failed: {result['failures']}")
    result["release_zip"] = str(destination)
    scan_path = destination.with_suffix(".secret-scan.json")
    scan_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["scan_result"] = str(scan_path)
    return result


def build_release(destination: Path) -> dict:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Release destination already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix="simdata-release-") as temporary:
        workspace = Path(temporary) / "simdata-loop-agent"
        demo_result = prepare(workspace)
        scan_tree(workspace)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(workspace.rglob("*")):
                if source.is_file():
                    archive.write(source, source.relative_to(workspace).as_posix())
    result = scan_release(destination)
    result.update({"release_zip": str(destination), "demo_validation": demo_result["qwen_validation"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sanitized SimProbe release ZIP")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scan-existing", action="store_true")
    args = parser.parse_args()
    result = scan_release(args.destination) if args.scan_existing else build_release(args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
