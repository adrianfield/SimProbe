"""Filesystem persistence for traceable SimData Loop runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .models import SeedScenario


PLATFORM_VERSION = "v0.4.5.2-final"
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "output" / "runs"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class RunStore:
    """Owns one immutable run directory and its incrementally updated assets."""

    SUBDIRECTORIES = (
        "seed",
        "coarse",
        "simulation/trajectories",
        "boundary",
        "fine",
        "report",
    )

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.trace_path = self.run_dir / "run_trace.json"

    @classmethod
    def create(
        cls,
        seed: SeedScenario,
        source_version: str,
        selection_reason: str,
        runs_root: Path = DEFAULT_RUNS_ROOT,
    ) -> "RunStore":
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        date_token = datetime.now().astimezone().strftime("%Y%m%d")
        run_dir = None
        for sequence in range(1, 1000):
            candidate = runs_root / f"CL-{date_token}-{sequence:03d}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                run_dir = candidate
                break
            except FileExistsError:
                continue
        if run_dir is None:
            raise RuntimeError("Could not allocate a unique run_id for today")

        store = cls(run_dir)
        for relative in cls.SUBDIRECTORIES:
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        now = _now_iso()
        manifest = {
            "schema_version": 1,
            "platform_version": PLATFORM_VERSION,
            "run_id": run_dir.name,
            "created_at": now,
            "updated_at": now,
            "source_version": source_version,
            "seed_case_id": seed.case_id,
            "scenario_type": seed.scenario_type,
            "status": "CREATED",
            "selection_reason": selection_reason,
            "backend": {
                "simulation": "LocalKinematicSimulator",
                "evaluation": "EvaluationEngine",
                "mode": "local_surrogate",
            },
            "assets": {},
        }
        store.write_json("manifest.json", manifest)
        return store

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def write_dataframe(self, relative_path: str, frame: pd.DataFrame) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def update_manifest(self, status: str, **fields: Any) -> dict:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(fields)
        manifest["status"] = status
        manifest["updated_at"] = _now_iso()
        manifest["assets"] = self.asset_counts()
        self.write_json("manifest.json", manifest)
        return manifest

    def asset_counts(self) -> dict[str, int]:
        files = [path for path in self.run_dir.rglob("*") if path.is_file()]
        directories = [path for path in self.run_dir.rglob("*") if path.is_dir()]

        def count_cases(relative_path: str) -> int:
            path = self.run_dir / relative_path
            if not path.exists():
                return 0
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return len(payload) if isinstance(payload, list) else 0
            except (json.JSONDecodeError, OSError):
                return 0

        return {
            "files": len(files),
            "directories": len(directories),
            "scenario_cases": count_cases("coarse/cases.json")
            + count_cases("fine/cases.json"),
            "trajectories": len(
                list((self.run_dir / "simulation" / "trajectories").glob("*.csv"))
            ),
        }


def list_runs(runs_root: Path = DEFAULT_RUNS_ROOT) -> list[dict]:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    manifests = []
    for path in sorted(runs_root.glob("CL-*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["run_dir"] = str(path.parent.resolve())
            manifests.append(manifest)
        except (json.JSONDecodeError, OSError):
            continue
    return manifests


def load_run(run_id: str, runs_root: Path = DEFAULT_RUNS_ROOT) -> dict:
    run_dir = Path(runs_root) / run_id
    if not run_dir.is_dir() or run_dir.parent.resolve() != Path(runs_root).resolve():
        raise FileNotFoundError(f"Unknown run_id: {run_id}")

    def read_json(relative: str, default):
        path = run_dir / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def read_csv(relative: str) -> pd.DataFrame:
        path = run_dir / relative
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "run_dir": run_dir,
        "manifest": read_json("manifest.json", {}),
        "trace": read_json("run_trace.json", {"events": []}),
        "seed": read_json("seed/seed_case.json", {}),
        "plan": read_json("coarse/generalization_plan.json", {}),
        "coarse_cases": read_csv("coarse/cases.csv"),
        "validation": read_csv("coarse/validation_results.csv"),
        "coarse_results": read_csv("simulation/coarse_results.csv"),
        "boundary": read_json("boundary/boundary_region.json", {}),
        "fine_cases": read_csv("fine/cases.csv"),
        "fine_results": read_csv("fine/fine_results.csv"),
    }
