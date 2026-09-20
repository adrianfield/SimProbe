"""Deterministic secret and packaging hygiene checks for demo/release artifacts."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Iterable


FORBIDDEN_DIRECTORY_NAMES = {".venv", "__pycache__"}
FORBIDDEN_FILE_NAMES = {".env"}
KEY_ASSIGNMENT = re.compile(
    r"(?mi)^[^\S\r\n]*DASHSCOPE_API_KEY[^\S\r\n]*=[^\S\r\n]*([^\r\n]*)$"
)


def _path_violations(relative: PurePosixPath) -> list[str]:
    violations: list[str] = []
    if any(part in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts):
        violations.append("forbidden runtime/cache directory")
    if relative.name in FORBIDDEN_FILE_NAMES:
        violations.append("forbidden environment file")
    if relative.suffix.lower() == ".pyc":
        violations.append("forbidden bytecode file")
    if relative.name.startswith("test_") and relative.suffix == ".py":
        violations.append("forbidden test source")
    if relative.name.startswith("acceptance_") and relative.suffix == ".py":
        violations.append("forbidden acceptance source")
    return violations


def _content_violations(relative: PurePosixPath, content: bytes) -> list[str]:
    if b"\x00" in content:
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return [
        "non-empty DASHSCOPE_API_KEY assignment"
        for match in KEY_ASSIGNMENT.finditer(text)
        if match.group(1).strip().strip('"\'')
    ]


def scan_entries(entries: Iterable[tuple[PurePosixPath, bytes]]) -> dict:
    failures: list[dict[str, str]] = []
    scanned = 0
    for relative, content in entries:
        scanned += 1
        for reason in [*_path_violations(relative), *_content_violations(relative, content)]:
            failures.append({"path": relative.as_posix(), "reason": reason})
    return {"status": "PASSED" if not failures else "FAILED", "scanned_files": scanned, "failures": failures}


def scan_tree(root: Path) -> dict:
    root = Path(root).resolve()
    entries = (
        (PurePosixPath(path.relative_to(root).as_posix()), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    )
    result = scan_entries(entries)
    if result["failures"]:
        raise ValueError(f"Packaging hygiene scan failed: {result['failures']}")
    return result
