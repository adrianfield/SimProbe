"""Persist and recover user task context without changing candidate decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .report_parser import CHINESE_HEADER_MAP


TASK_FIELDS = (
    "case_id", "diff", "severity", "version_prev", "version_curr",
    "prev_result", "curr_result", "seed_case_ref",
)
DEFAULT_BATCHES_ROOT = Path(__file__).resolve().parents[1] / "output" / "batches"
MISSING_TEXT = {"", "UNKNOWN", "N/A", "NA", "NONE", "NULL", "NAN"}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    rendered = str(value).strip()
    return rendered if rendered.upper() not in MISSING_TEXT else None


def source_task_context(trigger_row: dict[str, Any], case_id: str) -> dict[str, str | None]:
    """Reuse Daily's canonical columns; this is trace context, not a new IR schema."""
    return {
        field: _clean(case_id if field == "case_id" else trigger_row.get(field))
        for field in TASK_FIELDS
    }


def _fill_missing(context: dict, source: dict[str, Any] | None) -> None:
    if not isinstance(source, dict):
        return
    canonical = {CHINESE_HEADER_MAP.get(key, key): value for key, value in source.items()}
    for field in TASK_FIELDS:
        if not context.get(field):
            recovered = _clean(canonical.get(field))
            if recovered:
                context[field] = recovered


def _workbook_row(path: Path, case_id: str, sheets: tuple[str, ...]) -> dict | None:
    if not path.is_file() or not case_id:
        return None
    try:
        with pd.ExcelFile(path) as workbook:
            for sheet_name in sheets:
                if sheet_name not in workbook.sheet_names:
                    continue
                frame = workbook.parse(sheet_name)
                frame = frame.rename(columns={
                    name: CHINESE_HEADER_MAP[name]
                    for name in frame.columns if name in CHINESE_HEADER_MAP
                })
                if "case_id" not in frame:
                    continue
                matching = frame[frame["case_id"].astype(str).str.strip() == case_id]
                if not matching.empty:
                    return matching.iloc[0].to_dict()
    except (OSError, ValueError, KeyError, ImportError):
        pass
    return None


def _matching_batch(run_id: str, batches_root: Path) -> tuple[Path, dict] | None:
    if not run_id or not batches_root.is_dir():
        return None
    for path in sorted(batches_root.glob("BATCH-*/batch.json"), reverse=True):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        outcomes = metadata.get("run_outcomes") or []
        if run_id in (metadata.get("run_ids") or []) or any(
            item.get("run_id") == run_id for item in outcomes if isinstance(item, dict)
        ):
            return path.parent, metadata
    return None


def recover_run_task_context(run: dict, batches_root: Path = DEFAULT_BATCHES_ROOT) -> dict:
    """Recover old Runs by source precedence; never manufacture unavailable facts."""
    manifest = run.get("manifest") or {}
    trace = run.get("trace") or {}
    seed = run.get("seed") or {}
    case_id = _clean(manifest.get("seed_case_id")) or _clean(seed.get("case_id"))
    run_id = _clean(manifest.get("run_id"))
    context: dict[str, str] = {}
    if case_id:
        context["case_id"] = case_id
    scenario_type = _clean(manifest.get("scenario_type")) or _clean(seed.get("scenario_type"))
    if scenario_type:
        context["scenario_type"] = scenario_type

    # 1. New source context, or equivalent context already embedded in an old trace.
    _fill_missing(context, trace.get("source_task_context"))
    for event in trace.get("events") or []:
        data = event.get("data") or {}
        if _clean(data.get("case_id")) == case_id:
            _fill_missing(context, data)

    # 2. The owning Batch's persisted candidate/task context and candidate sheet.
    batch = _matching_batch(run_id, Path(batches_root))
    if batch:
        batch_dir, metadata = batch
        for outcome in metadata.get("run_outcomes") or []:
            if isinstance(outcome, dict) and outcome.get("run_id") == run_id:
                for key in ("source_task_context", "task_context", "candidate_context"):
                    _fill_missing(context, outcome.get(key))
        for key in ("source_task_contexts", "task_contexts", "candidate_contexts"):
            contexts = metadata.get(key) or {}
            if isinstance(contexts, dict):
                _fill_missing(context, contexts.get(run_id) or contexts.get(case_id))
        _fill_missing(context, _workbook_row(
            batch_dir / "report" / "closed_loop_analysis.xlsx", case_id or "",
            ("候选任务清单", "Candidate Tasks"),
        ))

    # 3. Enriched Daily first, then the original Daily copied into the Batch.
    run_dir = Path(run.get("run_dir")) if run.get("run_dir") else None
    if run_dir:
        _fill_missing(context, _workbook_row(
            run_dir / "report" / "daily_enriched.xlsx", case_id or "",
            ("Daily回归明细", "Daily Regression", "Technical Detail"),
        ))
    if batch:
        batch_dir, _ = batch
        _fill_missing(context, _workbook_row(
            batch_dir / "source_daily.xlsx", case_id or "",
            ("Daily回归明细", "Daily Regression"),
        ))
        _fill_missing(context, _workbook_row(
            batch_dir / "report" / "closed_loop_analysis.xlsx", case_id or "",
            ("Daily回归明细", "Daily Regression", "Technical Detail"),
        ))
    return context
