"""Create a clean SimProbe demo workspace from explicit allowlists."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.package_hygiene import scan_tree
from core.batch import build_batch_report, build_user_delivery_zip
from core.report_parser import classify_daily_cases, load_daily_report
from core.run_store import load_run


DEMO_RUNS = ("CL-20260915-031", "CL-20260915-032")
DEMO_BATCHES = ("BATCH-20260915-016",)
TOP_FILES = (
    "app.py", "README.md", "DEMO_WORKSPACE.md", "requirements.txt", ".env.example", ".gitignore",
)


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", "test_*.py", ".env",
    ))


def validate_qwen_artifacts() -> dict:
    validated_runs = {}
    required_stages = {
        "AI Issue Interpretation", "Coarse Exploration — Attempt 1",
        "Metric Sensitivity Analysis", "Boundary Detection", "Fine Generalization",
    }
    required_context = (
        "case_id", "diff", "severity", "version_prev", "version_curr",
        "prev_result", "curr_result",
    )
    for run_id in DEMO_RUNS:
        run_root = ROOT / "output" / "runs" / run_id
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        trace = json.loads((run_root / "run_trace.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"Demo run is not completed: {run_id}")
        if trace.get("llm_provider") != "qwen" or trace.get("llm_model") != "qwen3.8-flash":
            raise ValueError(f"Demo run is not the required Qwen model: {run_id}")
        context = trace.get("source_task_context") or {}
        if any(str(context.get(key, "")).strip().upper() in {"", "UNKNOWN", "N/A", "NONE", "NULL"}
               for key in required_context):
            raise ValueError(f"Demo run has incomplete source task context: {run_id}")
        if context["case_id"] != manifest.get("seed_case_id"):
            raise ValueError(f"Demo source task does not match its seed case: {run_id}")
        facts = trace.get("local_transition_boundary_facts") or []
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"Demo run has no local transition boundary facts: {run_id}")
        completed_stages = {
            event.get("step") for event in trace.get("events", [])
            if event.get("status") == "COMPLETED"
        }
        if not required_stages.issubset(completed_stages):
            raise ValueError(f"Demo run does not have the current five-stage evidence: {run_id}")
        for artifact in ("seed/seed_case.json", "coarse/attempt_1/cases.csv",
                         "fine/cases.csv", "boundary/boundary_region.json"):
            if not (run_root / artifact).is_file():
                raise ValueError(f"Demo run is missing a required artifact: {run_id}/{artifact}")
        validated_runs[run_id] = {
            "provider": "qwen", "model": "qwen3.8-flash",
            "status": manifest["status"], "case_id": context["case_id"],
            "diff": context["diff"], "severity": context["severity"],
            "boundary_fact_count": len(facts), "completed_stages": sorted(completed_stages),
        }
    for batch_id in DEMO_BATCHES:
        metadata = json.loads((ROOT / "output" / "batches" / batch_id / "batch.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "COMPLETED" or tuple(metadata.get("run_ids", [])) != DEMO_RUNS:
            raise ValueError(f"Demo batch does not reference the validated Qwen runs: {batch_id}")
    return {
        "provider": "qwen", "model": "qwen3.8-flash",
        "runs": list(DEMO_RUNS), "batches": list(DEMO_BATCHES),
        "run_validation": validated_runs,
    }


def rebuild_demo_report(destination: Path, batch_id: str) -> None:
    """Rebuild only the copied user report; frozen source Batch artifacts stay byte-identical."""
    batch_root = destination / "output" / "batches" / batch_id
    metadata = json.loads((batch_root / "batch.json").read_text(encoding="utf-8"))
    source_daily = batch_root / "source_daily.xlsx"
    daily = load_daily_report(source_daily)
    candidates, ineligible, non_candidates = classify_daily_cases(daily)
    outcomes = []
    for item in metadata["run_outcomes"]:
        run = load_run(item["run_id"], destination / "output" / "runs")
        outcomes.append({
            **item,
            "result": {
                "summary": run["manifest"].get("summary", {}),
                "local_transition_boundary_facts": run["trace"].get("local_transition_boundary_facts", []),
                "ai_issue_interpretation": run["trace"].get("ai_issue_interpretation", {}),
            },
        })
    report = build_batch_report(
        daily, candidates, ineligible, metadata["selected_case_ids"], outcomes,
        metadata["source_daily_name"], batch_id, ineligible, non_candidates,
    )
    report_path = batch_root / "report" / "closed_loop_analysis.xlsx"
    report_path.write_bytes(report)
    delivery = batch_root / metadata["delivery_zip"]
    build_user_delivery_zip(batch_root, batch_id, delivery)


def prepare(destination: Path) -> dict:
    qwen_validation = validate_qwen_artifacts()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Demo destination already exists: {destination}")
    destination.mkdir(parents=True)
    for name in TOP_FILES:
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    copy_tree(ROOT / "core", destination / "core")
    copy_tree(ROOT / ".streamlit", destination / ".streamlit")
    (destination / "data").mkdir()
    for name in ("sample_daily.xlsx", "seed_scenarios.json"):
        shutil.copy2(ROOT / "data" / name, destination / "data" / name)
    (destination / "scripts").mkdir()
    shutil.copy2(Path(__file__), destination / "scripts" / Path(__file__).name)
    shutil.copy2(ROOT / "scripts" / "build_release_package.py", destination / "scripts" / "build_release_package.py")
    for group, names in (("runs", DEMO_RUNS), ("batches", DEMO_BATCHES)):
        target_root = destination / "output" / group
        target_root.mkdir(parents=True)
        for name in names:
            source = ROOT / "output" / group / name
            if not source.is_dir():
                raise FileNotFoundError(f"Allowlisted demo artifact is missing: {source}")
            copy_tree(source, target_root / name)
    for batch_id in DEMO_BATCHES:
        rebuild_demo_report(destination, batch_id)
    manifest = {
        "platform_version": "v0.4.5.2-final",
        "artifact_policy": "explicit_allowlist",
        "qwen_validation": qwen_validation,
    }
    (destination / "DEMO_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    scan = scan_tree(destination)
    return {"destination": str(destination), "qwen_validation": qwen_validation, "secret_scan": scan}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean SimProbe demo workspace")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
