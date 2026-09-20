"""Daily-level batch orchestration and the v0.4.5 user delivery report."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .pipeline import run_closed_loop
from .presentation import (
    format_user_datetime,
    interaction_summary_for_user,
    localize_user_prose,
)
from .report_parser import (
    DIFF_DISPLAY, ELIGIBILITY_DISPLAY, EXECUTION_DISPLAY,
    SEVERITY_DISPLAY, canonical_metric_snapshot, classify_daily_cases,
)
from .scenario_registry import parameter_metadata


PLATFORM_VERSION = "v0.4.5.2-final"
DEFAULT_BATCHES_ROOT = Path(__file__).resolve().parents[1] / "output" / "batches"
USER_SHEETS = ("探索分析总览", "Daily回归明细", "候选任务清单", "局部边界摘要")
TECHNICAL_SHEETS = ("Technical Detail", "Local Boundary Technical")
SCENARIO_DISPLAY = {"highway_merge": "高速汇入", "lead_brake": "前车制动"}

COLORS = {
    "navy": "#18212B", "blue": "#2F5D7C", "page": "#F4F6F8",
    "pass": "#3E7C59", "risk": "#C58A2A", "fail": "#B84A4A",
    "text": "#1F2933", "muted": "#66717D", "border": "#D9DEE5",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class BatchStore:
    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self.metadata_path = self.batch_dir / "batch.json"

    @classmethod
    def create(
        cls, source_daily_name: str, source_daily_bytes: bytes,
        candidate_case_ids: list[str], selected_case_ids: list[str],
        batches_root: Path = DEFAULT_BATCHES_ROOT,
        ineligible_case_ids: list[str] | None = None,
        non_candidate_case_ids: list[str] | None = None,
    ) -> "BatchStore":
        batches_root = Path(batches_root)
        batches_root.mkdir(parents=True, exist_ok=True)
        token = datetime.now().astimezone().strftime("%Y%m%d")
        batch_dir = None
        for sequence in range(1, 1000):
            candidate = batches_root / f"BATCH-{token}-{sequence:03d}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                batch_dir = candidate
                break
            except FileExistsError:
                continue
        if batch_dir is None:
            raise RuntimeError("Could not allocate a unique batch_id for today")
        (batch_dir / "report").mkdir()
        # This is deliberately a byte-identical copy of the upload.
        (batch_dir / "source_daily.xlsx").write_bytes(source_daily_bytes)
        store = cls(batch_dir)
        now = _now_iso()
        ineligible = ineligible_case_ids or []
        non_candidates = non_candidate_case_ids or []
        store.write_metadata({
            "schema_version": 2, "platform_version": PLATFORM_VERSION,
            "batch_id": batch_dir.name, "source_daily_name": source_daily_name,
            "source_daily_sha256": hashlib.sha256(source_daily_bytes).hexdigest(),
            "created_at": now, "updated_at": now,
            "candidate_groups": {
                "candidate_case_ids": candidate_case_ids,
                "ineligible_case_ids": ineligible,
                "non_candidate_case_ids": non_candidates,
            },
            "candidate_case_ids": candidate_case_ids,
            "ineligible_case_ids": ineligible,
            "non_candidate_case_ids": non_candidates,
            "selected_case_ids": selected_case_ids, "run_ids": [], "run_outcomes": [],
            "status": "READY",
        })
        return store

    @property
    def batch_id(self) -> str:
        return self.batch_dir.name

    def read_metadata(self) -> dict:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def write_metadata(self, payload: dict) -> None:
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    def update(self, status: str, **fields: Any) -> dict:
        metadata = self.read_metadata()
        metadata.update(fields)
        metadata["status"] = status
        metadata["updated_at"] = _now_iso()
        self.write_metadata(metadata)
        return metadata


def list_batches(batches_root: Path = DEFAULT_BATCHES_ROOT) -> list[dict]:
    root = Path(batches_root)
    if not root.exists():
        return []
    records = []
    for path in sorted(root.glob("BATCH-*/batch.json"), reverse=True):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["batch_dir"] = str(path.parent.resolve())
            records.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def load_batch(batch_id: str, batches_root: Path = DEFAULT_BATCHES_ROOT) -> dict:
    root = Path(batches_root).resolve()
    batch_dir = (Path(batches_root) / batch_id).resolve()
    if batch_dir.parent != root or not batch_dir.is_dir():
        raise FileNotFoundError(f"Unknown batch_id: {batch_id}")
    metadata = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    metadata["batch_dir"] = str(batch_dir)
    metadata["report_path"] = str(batch_dir / "report" / "closed_loop_analysis.xlsx")
    metadata["delivery_zip_path"] = str(batch_dir / f"{batch_id}_user_delivery.zip")
    return metadata


def report_download_name(batch_id: str) -> str:
    return f"SimProbe_场景探索与边界分析报告_{batch_id}.xlsx"


def delivery_download_name(batch_id: str) -> str:
    return f"SimProbe_探索结果包_{batch_id}.zip"


def _parameter_name(key: str) -> str:
    return parameter_metadata(key)["display_name_zh"]


def _execution_display(status: str, outcome: dict | None = None) -> str:
    result = (outcome or {}).get("result", {}) or {}
    warning_code = result.get("warning_code") or (outcome or {}).get("warning_code")
    if status == "COMPLETED_WITH_WARNINGS" and warning_code == "AI_FEEDBACK_UNAVAILABLE":
        return "已完成"
    return EXECUTION_DISPLAY[status]


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


def _interval(interval: dict | None, key: str) -> str:
    if not interval:
        return "未形成可靠精化区间"
    unit = parameter_metadata(key)["unit"]
    suffix = f" {unit}" if unit else ""
    return f"{_number(interval.get('min'))} – {_number(interval.get('max'))}{suffix}"


def _fixed_conditions(values: dict[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        metadata = parameter_metadata(key)
        unit = f" {metadata['unit']}" if metadata["unit"] else ""
        parts.append(f"{metadata['display_name_zh']} = {_number(value)}{unit}")
    return "；".join(parts)


def _distribution_values(result: dict, stage: str) -> tuple[int, int, int, int]:
    summary = result.get("summary", {})
    count = int(summary.get(f"{stage}_generated", 0) or 0)
    values = summary.get(f"{stage}_results", {}) or {}
    return count, int(values.get("PASS", 0)), int(values.get("HIGH_RISK", 0)), int(values.get("FAIL", 0))


def _distribution_short(result: dict, stage: str) -> str:
    count, passed, risk, failed = _distribution_values(result, stage)
    return f"{count}\nPASS {passed} / HIGH_RISK {risk} / FAIL {failed}"


def _boundary_conclusion(result: dict) -> str:
    facts = result.get("local_transition_boundary_facts", []) or []
    if not facts:
        return "本次未观察到可靠的局部状态转换区域。"
    names = list(dict.fromkeys(_parameter_name(item["changed_parameter"]) for item in facts))
    return (
        f"本次发现 {len(facts)} 个局部状态转换区域，涉及{'、'.join(names)}。"
        "每条结果仅适用于对应固定条件和已采样区间，具体范围见《局部边界摘要》。"
    )


def _outcome_by_case(outcomes: list[dict]) -> dict[str, dict]:
    return {item["case_id"]: item for item in outcomes}


def _execution_status(case_id: str, candidate_ids: set[str], selected_ids: set[str], outcomes: dict[str, dict]) -> str:
    if case_id in outcomes:
        return outcomes[case_id].get("status", "FAILED")
    if case_id in candidate_ids:
        return "NOT_SELECTED" if case_id not in selected_ids else "FAILED"
    return "NOT_APPLICABLE"


def _processing_status(case_id: str, candidate_ids: set[str], selected_ids: set[str], outcomes: dict[str, dict], excluded: pd.DataFrame | None = None) -> str:
    """Legacy user-copy helper retained for v0.4.4 readers."""
    return EXECUTION_DISPLAY[_execution_status(case_id, candidate_ids, selected_ids, outcomes)]


def _frames_for_report(
    daily_df: pd.DataFrame, candidates: pd.DataFrame, ineligible: pd.DataFrame,
    non_candidates: pd.DataFrame, selected_case_ids: list[str], outcomes: list[dict], batch_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_ids = set(candidates.get("case_id", pd.Series(dtype=str)).astype(str))
    selected_ids = set(selected_case_ids)
    outcome_map = _outcome_by_case(outcomes)
    groups = {}
    for code, frame in (("CANDIDATE", candidates), ("INELIGIBLE_NO_SEED", ineligible), ("NOT_CANDIDATE", non_candidates)):
        for _, row in frame.iterrows():
            groups[str(row["case_id"])] = (code, row.to_dict())

    compact_rows, daily_rows, candidate_rows, technical_rows = [], [], [], []
    for _, source in daily_df.iterrows():
        row = source.to_dict()
        case_id = str(row["case_id"])
        eligibility, classified = groups.get(case_id, ("NOT_CANDIDATE", row))
        outcome = outcome_map.get(case_id, {})
        execution = _execution_status(case_id, candidate_ids, selected_ids, outcome_map)
        run_id = outcome.get("run_id", "")
        scenario_type = outcome.get("scenario_type", row.get("scenario_type", ""))
        seed_id = classified.get("seed_case_id", "") if eligibility != "NOT_CANDIDATE" else ""
        reason = classified.get("candidate_reason") or classified.get("exclusion_reason", "")
        recommendation = classified.get("recommendation", "")
        execution_display = "待补充种子场景" if eligibility == "INELIGIBLE_NO_SEED" else _execution_display(execution, outcome)
        daily_rows.append({
            "用例 ID": case_id, "上一版本": row.get("version_prev", ""), "当前版本": row.get("version_curr", ""),
            "上一版本结果": row.get("prev_result", ""), "当前评测结果": row.get("curr_result", ""),
            "结果变化": DIFF_DISPLAY.get(row.get("diff", ""), row.get("diff", "")),
            "严重度": SEVERITY_DISPLAY.get(row.get("severity", ""), row.get("severity", "")),
            "问题描述": localize_user_prose(row.get("issue_description", "")),
            "关键指标": row.get("key_metrics_zh", ""), "候选状态": ELIGIBILITY_DISPLAY[eligibility],
            "候选原因": localize_user_prose(reason), "处理建议": recommendation,
            "执行状态": execution_display,
        })
        technical_rows.append({
            "case_id": case_id, "version_prev": row.get("version_prev", ""), "version_curr": row.get("version_curr", ""),
            "prev_result": row.get("prev_result", ""), "curr_result": row.get("curr_result", ""),
            "diff": row.get("diff", ""), "severity": row.get("severity", ""),
            "issue_description": row.get("issue_description", ""),
            "input_evaluation_reason": row.get("input_evaluation_reason", ""),
            "evaluation_reason": row.get("evaluation_reason", ""),
            "evaluation_reason_source": row.get("evaluation_reason_source", "本地评测规则"),
            "local_review_result": row.get("local_review_result", ""),
            "local_review_description": row.get("local_review_description", ""),
            "evaluation_review_status": row.get("evaluation_review_status", "UNKNOWN"),
            "metric_snapshot": row.get("metric_snapshot_canonical") or canonical_metric_snapshot(row.get("metric_snapshot")),
            "candidate_eligibility": eligibility,
            "candidate_reason_code": classified.get("candidate_reason_code", "CURRENT_RESULT_NOT_ABNORMAL"),
            "candidate_reason": reason, "recommendation": recommendation,
            "execution_status": execution, "seed_case_ref": row.get("seed_case_ref", ""),
            "seed_case_id": seed_id,
            "batch_id": batch_id, "run_id": run_id,
            "closed_loop_status": execution,
        })
        if eligibility == "CANDIDATE":
            candidate_rows.append({
                "用例 ID": case_id, "当前结果": row.get("curr_result", ""),
                "结果变化": DIFF_DISPLAY.get(row.get("diff", ""), row.get("diff", "")),
                "严重度": SEVERITY_DISPLAY.get(row.get("severity", ""), row.get("severity", "")),
                "候选原因": localize_user_prose(reason), "处理建议": recommendation,
                "是否默认选择": "是" if classified.get("default_selected") else "否",
                "是否实际执行": "是" if case_id in selected_ids else "否",
                "执行状态": _execution_display(execution, outcome), "运行 ID": run_id,
            })
        if outcome:
            result = outcome.get("result", {})
            facts = result.get("local_transition_boundary_facts", []) or []
            compact_rows.append({
                "用例 ID": case_id, "当前结果": row.get("curr_result", ""),
                "结果变化": DIFF_DISPLAY.get(row.get("diff", ""), row.get("diff", "")),
                "严重度": SEVERITY_DISPLAY.get(row.get("severity", ""), row.get("severity", "")),
                "场景类型": SCENARIO_DISPLAY.get(scenario_type, scenario_type),
                "执行状态": _execution_display(execution, outcome),
                "粗粒度探索": _distribution_short(result, "coarse") if result else "—",
                "定向细化": _distribution_short(result, "fine") if result else "—",
                "局部边界": f"{len(facts)} 个" if result else "—", "运行 ID": run_id,
            })

    boundary_rows, boundary_technical = [], []
    for outcome in outcomes:
        for fact in outcome.get("result", {}).get("local_transition_boundary_facts", []) or []:
            key = fact["changed_parameter"]
            boundary_rows.append({
                "用例 ID": outcome["case_id"], "边界参数": _parameter_name(key),
                "固定条件": _fixed_conditions(fact["fixed_parameters"]),
                "粗粒度转换区间": _interval(fact.get("coarse_interval"), key),
                "精化区间": _interval(fact.get("refined_interval"), key),
                "状态变化": " → ".join(fact.get("ordered_transition_states", [])),
                "宽度缩减": f"{fact['width_reduction']:.1%}" if fact.get("width_reduction") is not None else "—",
                "运行 ID": outcome.get("run_id", ""),
            })
            boundary_technical.append({
                "case_id": outcome["case_id"], "changed_parameter": key,
                "fixed_parameters": json.dumps(fact["fixed_parameters"], ensure_ascii=False, sort_keys=True),
                "coarse_interval": json.dumps(fact.get("coarse_interval"), ensure_ascii=False, sort_keys=True),
                "refined_interval": json.dumps(fact.get("refined_interval"), ensure_ascii=False, sort_keys=True),
                "ordered_transition_states": json.dumps(fact.get("ordered_transition_states", []), ensure_ascii=False),
                "width_reduction": fact.get("width_reduction"),
                "backend": "LocalKinematicSimulator + EvaluationEngine", "run_id": outcome.get("run_id", ""),
            })
    return (
        pd.DataFrame(compact_rows, columns=["用例 ID", "当前结果", "结果变化", "严重度", "场景类型", "执行状态", "粗粒度探索", "定向细化", "局部边界", "运行 ID"]),
        pd.DataFrame(daily_rows), pd.DataFrame(candidate_rows),
        pd.DataFrame(boundary_rows, columns=["用例 ID", "边界参数", "固定条件", "粗粒度转换区间", "精化区间", "状态变化", "宽度缩减", "运行 ID"]),
        pd.DataFrame(technical_rows), pd.DataFrame(boundary_technical, columns=["case_id", "changed_parameter", "fixed_parameters", "coarse_interval", "refined_interval", "ordered_transition_states", "width_reduction", "backend", "run_id"]),
    )


def _style_table(writer, frame: pd.DataFrame, sheet_name: str, startrow: int = 0, user_sheet: bool = True) -> None:
    workbook, sheet = writer.book, writer.sheets[sheet_name]
    header = workbook.add_format({
        "bold": True,
        "font_color": "#24313C" if user_sheet else "#FFFFFF",
        "bg_color": "#E7EDF2" if user_sheet else COLORS["navy"],
        "border": 1 if user_sheet else 0,
        "border_color": "#C7D0D9",
        "align": "center", "valign": "vcenter", "text_wrap": True,
    })
    body = workbook.add_format({"font_color": COLORS["text"], "border": 1, "border_color": COLORS["border"], "valign": "top", "text_wrap": True})
    nowrap = workbook.add_format({"font_color": COLORS["text"], "border": 1, "border_color": COLORS["border"], "valign": "top", "text_wrap": False})
    for col, name in enumerate(frame.columns):
        sheet.write(startrow, col, name, header)
        width = min(34, max(12, len(str(name)) * 2 + 4))
        if name in {"问题描述", "评测说明", "本地规则复核", "候选 / 排除原因", "候选原因", "固定条件"}:
            width = 30
        if name in {"用例 ID", "运行 ID", "case_id", "run_id", "batch_id"}:
            width = max(width, 20)
            column_format = nowrap
        else:
            column_format = body
        sheet.set_column(col, col, width, column_format)
    for row_index in range(len(frame)):
        sheet.set_row(startrow + 1 + row_index, 42)
    if len(frame.columns):
        sheet.autofilter(startrow, 0, startrow + len(frame), len(frame.columns) - 1)
    sheet.freeze_panes(startrow + 1, 1 if sheet_name != "探索分析总览" else 0)
    if user_sheet:
        sheet.hide_gridlines(2)
        sheet.set_zoom(90)
    state_fmt = {
        "PASS": workbook.add_format({"font_color": COLORS["pass"], "bg_color": "#ECFDF3", "bold": True}),
        "HIGH_RISK": workbook.add_format({"font_color": "#92400E", "bg_color": "#FFF7E6", "bold": True}),
        "FAIL": workbook.add_format({"font_color": COLORS["fail"], "bg_color": "#FEF2F2", "bold": True}),
    }
    for col, name in enumerate(frame.columns):
        if name in {"当前结果", "当前评测结果", "上一版本结果"}:
            for value, fmt in state_fmt.items():
                sheet.conditional_format(startrow + 1, col, startrow + len(frame), col, {"type": "text", "criteria": "containing", "value": value, "format": fmt})
        if name == "严重度":
            severity_formats = {
                "高": workbook.add_format({"font_color": "#991B1B", "bg_color": "#FEF2F2"}),
                "中": workbook.add_format({"font_color": "#92400E", "bg_color": "#FFF7E6"}),
                "低": workbook.add_format({"font_color": "#475569", "bg_color": "#F1F5F9"}),
            }
            for value, fmt in severity_formats.items():
                sheet.conditional_format(startrow + 1, col, startrow + len(frame), col, {"type": "text", "criteria": "containing", "value": value, "format": fmt})
        if sheet_name == "局部边界摘要" and name in {"边界参数", "精化区间", "状态变化"}:
            sheet.set_column(col, col, 22, workbook.add_format({"font_color": COLORS["blue"], "bg_color": "#EDF1F4", "border": 1, "border_color": COLORS["border"], "text_wrap": True, "valign": "top"}))


def build_batch_report(
    daily_df: pd.DataFrame, candidates: pd.DataFrame, excluded: pd.DataFrame,
    selected_case_ids: list[str], outcomes: list[dict], source_daily_name: str, batch_id: str,
    ineligible: pd.DataFrame | None = None, non_candidates: pd.DataFrame | None = None,
) -> bytes:
    if ineligible is None or non_candidates is None:
        _, ineligible, non_candidates = classify_daily_cases(daily_df)
    compact, daily_user, candidate_user, boundary_user, technical, local_technical = _frames_for_report(
        daily_df, candidates, ineligible, non_candidates, selected_case_ids, outcomes, batch_id,
    )
    success_count = sum(item.get("status") == "COMPLETED" for item in outcomes)
    warning_count = sum(item.get("status") == "COMPLETED_WITH_WARNINGS" for item in outcomes)
    failure_count = sum(item.get("status") == "FAILED" for item in outcomes)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        compact.to_excel(writer, sheet_name="探索分析总览", index=False, startrow=10)
        daily_user.to_excel(writer, sheet_name="Daily回归明细", index=False)
        candidate_user.to_excel(writer, sheet_name="候选任务清单", index=False)
        boundary_user.to_excel(writer, sheet_name="局部边界摘要", index=False, startrow=3)
        technical.to_excel(writer, sheet_name="Technical Detail", index=False)
        local_technical.to_excel(writer, sheet_name="Local Boundary Technical", index=False)
        workbook = writer.book
        summary = writer.sheets["探索分析总览"]
        title_fmt = workbook.add_format({
            "bold": True, "font_size": 22, "font_color": "#1F2933", "bg_color": "#FFFFFF",
            "bottom": 1, "bottom_color": COLORS["blue"], "align": "left", "valign": "vcenter",
        })
        subtitle_fmt = workbook.add_format({"font_size": 11, "font_color": COLORS["muted"], "bg_color": "#FFFFFF"})
        meta_fmt = workbook.add_format({"font_size": 9, "font_color": COLORS["muted"]})
        kpi_label = workbook.add_format({"bold": True, "font_color": COLORS["muted"], "bg_color": "#EEF3FA", "align": "center", "border": 1, "border_color": COLORS["border"]})
        kpi_value = workbook.add_format({"bold": True, "font_size": 18, "font_color": COLORS["navy"], "bg_color": "#FFFFFF", "align": "center", "border": 1, "border_color": COLORS["border"]})
        summary.merge_range("A1:L1", "场景探索与局部边界分析报告", title_fmt)
        summary.merge_range("A2:L2", "SimProbe · 仿真探索与边界分析", subtitle_fmt)
        summary.merge_range("A3:L3", f"批次 ID：{batch_id}    来源报告：{source_daily_name}    生成时间：{format_user_datetime(_now_iso())}", meta_fmt)
        summary.merge_range("A4:L4", "仿真环境：本地运动学仿真", meta_fmt)
        labels = ["总用例数", "候选任务", "本次执行", "已完成", "有警告", "失败"]
        values = [len(daily_df), len(candidates), len(selected_case_ids), success_count, warning_count, failure_count]
        for index, (label, value) in enumerate(zip(labels, values)):
            first = index * 2
            summary.merge_range(5, first, 5, first + 1, label, kpi_label)
            summary.merge_range(6, first, 7, first + 1, value, kpi_value)
        summary.write(9, 0, "本次探索结果", workbook.add_format({"bold": True, "font_size": 13, "font_color": COLORS["navy"]}))
        _style_table(writer, compact, "探索分析总览", 10)
        _style_table(writer, daily_user, "Daily回归明细")
        _style_table(writer, candidate_user, "候选任务清单")
        _style_table(writer, boundary_user, "局部边界摘要", 3)
        _style_table(writer, technical, "Technical Detail", user_sheet=False)
        _style_table(writer, local_technical, "Local Boundary Technical", user_sheet=False)
        writer.sheets["Technical Detail"].hide()
        writer.sheets["Local Boundary Technical"].hide()
        notice = workbook.add_format({"font_color": COLORS["blue"], "bg_color": "#EDF1F4", "border": 1, "border_color": COLORS["border"], "text_wrap": True, "valign": "vcenter"})
        boundary_sheet = writer.sheets["局部边界摘要"]
        # Hidden compatibility header lets old machine readers discover the schema;
        # the visible product table starts below the single notice box.
        for column, name in enumerate(boundary_user.columns):
            boundary_sheet.write(0, column, name)
        boundary_sheet.set_row(0, None, None, {"hidden": True})
        end_column = max(0, len(boundary_user.columns) - 1)
        boundary_sheet.merge_range(1, 0, 2, end_column, "本表展示局部状态转换边界。每条结果仅适用于对应固定条件，不代表其他场景上下文下的全局规律。", notice)
        # One restrained horizontal stacked chart; source data is kept in hidden columns.
        chart_rows = []
        for outcome in outcomes:
            result = outcome.get("result", {})
            for stage, label in (("coarse", "粗粒度探索"), ("fine", "定向细化")):
                _, passed, risk, failed = _distribution_values(result, stage)
                if passed + risk + failed > 0:
                    chart_rows.append((f"{SCENARIO_DISPLAY.get(outcome.get('scenario_type', ''), outcome.get('scenario_type', ''))} · {label}", passed, risk, failed))
        for col, value in enumerate(("探索评测分布", "PASS", "HIGH_RISK", "FAIL"), 23):
            summary.write(0, col, value)
        for row_idx, values_row in enumerate(chart_rows, 1):
            for col_idx, value in enumerate(values_row, 23):
                summary.write(row_idx, col_idx, value)
        if chart_rows:
            chart = workbook.add_chart({"type": "bar", "subtype": "stacked"})
            for offset, (name, color) in enumerate((("PASS", COLORS["pass"]), ("HIGH_RISK", COLORS["risk"]), ("FAIL", COLORS["fail"])), 24):
                chart.add_series({"name": name, "categories": ["探索分析总览", 1, 23, len(chart_rows), 23], "values": ["探索分析总览", 1, offset, len(chart_rows), offset], "fill": {"color": color}, "border": {"none": True}})
            chart.show_hidden_data()
            chart.set_title({"name": "探索评测分布"})
            chart.set_legend({"position": "bottom"})
            chart.set_x_axis({"major_gridlines": {"visible": False}})
            chart.set_style(10)
            chart.set_size({"width": 500, "height": 260})
            summary.insert_chart("L11", chart)
        summary.set_column(23, 26, None, None, {"hidden": True})
        # Readable conclusion blocks below the compact table.
        conclusion_row = 12 + len(compact) + 2
        section_fmt = workbook.add_format({"bold": True, "font_size": 13, "font_color": COLORS["navy"]})
        case_fmt = workbook.add_format({"bold": True, "font_color": COLORS["blue"], "bg_color": "#EDF1F4"})
        conclusion_fmt = workbook.add_format({"font_color": COLORS["text"], "bg_color": "#FFFFFF", "border": 1, "border_color": COLORS["border"], "text_wrap": True, "valign": "top"})
        summary.write(conclusion_row, 0, "关键结论", section_fmt)
        conclusion_row += 1
        outcome_map = _outcome_by_case(outcomes)
        for _, candidate in candidates.iterrows():
            outcome = outcome_map.get(str(candidate["case_id"]))
            if not outcome:
                continue
            result = outcome.get("result", {})
            interpretation = result.get("ai_issue_interpretation", {}) or {}
            summary.merge_range(conclusion_row, 0, conclusion_row, 9, f"结论 · {candidate['case_id']}", case_fmt)
            conclusion_row += 1
            lines = [
                f"评测说明：{localize_user_prose(candidate.get('evaluation_reason', ''))}",
                f"候选原因：{localize_user_prose(candidate.get('candidate_reason', ''))}",
                f"场景理解：{interaction_summary_for_user(interpretation.get('interaction_summary', '未执行或未形成场景理解。'))}",
                f"局部边界结论：{_boundary_conclusion(result)}",
            ]
            summary.merge_range(conclusion_row, 0, conclusion_row + 3, 9, "\n".join(lines), conclusion_fmt)
            summary.set_row(conclusion_row, 25)
            conclusion_row += 5
    return buffer.getvalue()


def build_user_delivery_zip(batch_root: Path, batch_id: str, zip_path: Path | None = None) -> Path:
    """Build the sole user package format from persisted Batch artifacts."""
    batch_root = Path(batch_root)
    zip_path = Path(zip_path) if zip_path is not None else batch_root / f"{batch_id}_user_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(batch_root / "source_daily.xlsx", "原始回归报告.xlsx")
        archive.write(
            batch_root / "report" / "closed_loop_analysis.xlsx",
            report_download_name(batch_id),
        )
    return zip_path


def _delivery_zip(store: BatchStore) -> Path:
    return build_user_delivery_zip(store.batch_dir, store.batch_id)


def execute_batch(
    daily_df: pd.DataFrame, selected_case_ids: list[str], seeds: dict,
    source_daily_name: str, source_daily_bytes: bytes, runs_root: Path,
    batches_root: Path = DEFAULT_BATCHES_ROOT,
    run_executor: Callable[..., dict] = run_closed_loop,
    planner_factory: Callable[[], Any] | None = None,
) -> dict:
    candidates, ineligible, non_candidates = classify_daily_cases(daily_df, seeds)
    available = set(candidates.get("case_id", pd.Series(dtype=str)).astype(str))
    selected = list(dict.fromkeys(selected_case_ids))
    if not selected or not set(selected).issubset(available):
        raise ValueError("At least one eligible candidate must be selected")
    store = BatchStore.create(
        source_daily_name, source_daily_bytes, candidates["case_id"].tolist(), selected,
        batches_root, ineligible.get("case_id", pd.Series(dtype=str)).astype(str).tolist(),
        non_candidates.get("case_id", pd.Series(dtype=str)).astype(str).tolist(),
    )
    store.update("RUNNING")
    outcomes = []
    for case_id in selected:
        row = candidates[candidates["case_id"] == case_id].iloc[0]
        seed_case_id = str(row.get("seed_case_id") or case_id)
        source_seed = seeds[seed_case_id]
        # A Daily case may explicitly reference an existing seed asset with a different
        # case id. Clone only trace identity; scenario type and parameters remain exact.
        seed = source_seed if seed_case_id == case_id else source_seed.model_copy(update={"case_id": case_id})
        try:
            kwargs = {"runs_root": runs_root}
            if planner_factory is not None:
                kwargs["planner"] = planner_factory()
            result = run_executor(daily_df, seed, row.to_dict(), **kwargs)
            outcome_status = result.get("status", "COMPLETED")
            outcomes.append({
                "case_id": case_id, "seed_case_ref": row.get("seed_case_ref"),
                "seed_case_id": seed_case_id, "scenario_type": seed.scenario_type,
                "run_id": result["run_id"], "status": outcome_status, "result": result,
            })
        except Exception as exc:
            outcomes.append({
                "case_id": case_id, "seed_case_ref": row.get("seed_case_ref"),
                "seed_case_id": seed_case_id, "scenario_type": seed.scenario_type,
                "run_id": getattr(exc, "run_id", ""), "status": "FAILED",
                "error_type": type(exc).__name__, "error_summary": str(exc), "result": {},
            })
    completed = sum(item["status"] in {"COMPLETED", "COMPLETED_WITH_WARNINGS"} for item in outcomes)
    warnings = sum(item["status"] == "COMPLETED_WITH_WARNINGS" for item in outcomes)
    failed = sum(item["status"] == "FAILED" for item in outcomes)
    status = (
        "COMPLETED_WITH_WARNINGS" if failed == 0 and warnings
        else "COMPLETED" if failed == 0
        else "PARTIAL_FAILED" if completed else "FAILED"
    )
    report = build_batch_report(
        daily_df, candidates, pd.concat([ineligible, non_candidates], ignore_index=True), selected,
        outcomes, source_daily_name, store.batch_id, ineligible, non_candidates,
    )
    report_path = store.batch_dir / "report" / "closed_loop_analysis.xlsx"
    report_path.write_bytes(report)
    serializable_outcomes = [{key: value for key, value in item.items() if key != "result"} for item in outcomes]
    metadata = store.update(
        status, run_ids=[item["run_id"] for item in outcomes if item.get("run_id")],
        run_outcomes=serializable_outcomes, report_path="report/closed_loop_analysis.xlsx",
        delivery_zip=f"{store.batch_id}_user_delivery.zip",
    )
    zip_path = _delivery_zip(store)
    return {
        "batch_id": store.batch_id, "batch_dir": str(store.batch_dir.resolve()),
        "status": status, "outcomes": outcomes, "metadata": metadata,
        "report_path": str(report_path.resolve()), "report_bytes": report,
        "delivery_zip_path": str(zip_path.resolve()), "delivery_zip_bytes": zip_path.read_bytes(),
    }
