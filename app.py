from __future__ import annotations

import html
import io
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from core.boundary import assess_refinement
from core.batch import (
    DEFAULT_BATCHES_ROOT,
    delivery_download_name,
    execute_batch,
    load_batch,
    report_download_name,
)
from core.presentation import (
    format_user_datetime,
    has_persisted_scenario_assets,
    interaction_summary_for_user,
    local_boundary_count,
    persisted_scenario_asset_count,
    run_has_results,
    status_class,
    status_display,
)
from core.report_enricher import (
    build_local_boundary_facts,
    build_local_boundary_records,
)
from core.risk_map import build_risk_map_slice, nearest_observed_value
from core.report_parser import (
    DIFF_DISPLAY,
    SEVERITY_DISPLAY,
    candidate_filter_rule_text,
    classify_daily_cases,
    load_daily_report,
)
from core.run_store import list_runs, load_run
from core.run_context import recover_run_task_context
from core.scenario_registry import (
    SCENARIO_REGISTRY,
    actor_metadata,
    parameter_metadata,
    registry_summary,
)
from core.scenario_store import load_seed_scenarios


ROOT = Path(__file__).resolve().parent
SAMPLE_REPORT = ROOT / "data" / "sample_daily.xlsx"
RUNS_ROOT = ROOT / "output" / "runs"
BATCHES_ROOT = Path(os.getenv("SIMDATA_BATCHES_ROOT", str(DEFAULT_BATCHES_ROOT)))
RUNS_ROOT = Path(os.getenv("SIMDATA_RUNS_ROOT", str(RUNS_ROOT)))

st.set_page_config(
    page_title="SimProbe",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#1F2933; --muted:#66717D; --line:#D9DEE5; --blue:#2F5D7C; }
    .stApp { background:#F4F6F8; color:var(--ink); }
    [data-testid="stSidebar"] { background:#18212B; border-right:1px solid #26313C; }
    [data-testid="stSidebar"] * { color:#B9C1C9; }
    [data-testid="stSidebar"] .sidebar-brand { margin-top:.35rem; margin-bottom:1.8rem; }
    [data-testid="stSidebar"] .sidebar-brand-title {
        color:#E6EDF3; font-size:1.15rem; font-weight:680; line-height:1.35;
    }
    [data-testid="stSidebar"] .sidebar-brand-subtitle {
        color:#8B98A5; font-size:.8rem; font-weight:400; line-height:1.45; margin-top:.16rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding:.55rem .72rem; border-radius:.18rem; margin:.1rem 0; border-left:3px solid transparent;
    }
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-baseweb="radio"],
    [data-testid="stSidebar"] label[data-testid="stRadioOption"] > span:has(input[type="radio"]) + div > div:first-child {
        display:none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:#242F3A; border-left-color:#2F5D7C;
    }
    [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"] p::before { color:#fff; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) svg { color:#fff !important; fill:#fff !important; }
    .block-container { padding-top:1.4rem; max-width:1500px; }
    .platform-head { display:flex; align-items:flex-start; justify-content:space-between;
        margin-bottom:1rem; }
    .platform-head h1 { color:var(--ink); font-size:1.72rem; margin:0 0 .2rem 0; }
    .subtle { color:var(--muted); font-size:.88rem; }
    .environment-note { color:var(--muted); font-size:.78rem; white-space:nowrap; padding-top:.35rem; }
    .flow { display:grid; grid-template-columns:repeat(5,1fr); gap:.35rem; margin:.55rem 0 .85rem; }
    .flow-step { background:white; border:1px solid var(--line); border-radius:.2rem;
        padding:.52rem .35rem; font-size:.72rem; color:#66717D; text-align:center; }
    .flow-step.done { border-color:#8AA391; color:#315F45; background:#EDF4EF; }
    .flow-step.warning { border-color:#D6B66F; color:#8B641F; background:#FAF3E7; }
    .flow-step.failed { border-color:#D49A9A; color:#A23F3F; background:#F9ECEC; }
    .flow-step.skipped,.flow-step.pending { border-color:#D9DEE5; color:#66717D; background:#F4F6F8; }
    .flow-step strong { display:block; font-size:.74rem; color:inherit; margin-bottom:.12rem; }
    .meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.45rem; }
    .meta-item { background:#fff; border:1px solid var(--line); padding:.7rem;
        border-radius:.2rem; min-height:62px; }
    .meta-key { color:var(--muted); font-size:.7rem; text-transform:uppercase; }
    .meta-value { color:var(--ink); font-size:.88rem; font-weight:650; margin-top:.25rem; }
    .batch-card,.candidate-card { background:#fff; border:1px solid var(--line);
        border-radius:.2rem; padding:.8rem 1rem; margin:.55rem 0; }
    .candidate-card { padding:.75rem 1rem; margin:.35rem 0 .8rem; }
    .badge { display:inline-block; border-radius:.15rem; padding:.16rem .42rem; font-size:.72rem; font-weight:700; }
    .badge-pass,.status-completed { color:#3E7C59; background:#EDF4EF; }
    .badge-risk,.status-warning { color:#9A6B20; background:#FAF3E7; }
    .badge-fail,.status-failed { color:#B84A4A; background:#F9ECEC; }
    .status-running { color:#2F5D7C; background:#EDF1F4; }
    .status-unknown { color:#66717D; background:#EEF0F2; }
    .overview-strip { display:grid; grid-template-columns:repeat(6,1fr); border:1px solid var(--line); background:#fff; margin:.5rem 0 1rem; }
    .overview-cell { padding:.62rem .75rem; border-right:1px solid var(--line); }
    .overview-cell:last-child { border-right:0; }
    .overview-label { color:var(--muted); font-size:.72rem; }
    .overview-value { color:var(--ink); font-size:1.05rem; font-weight:700; margin-top:.12rem; }
    .run-head { display:flex; align-items:center; gap:1rem; border-bottom:1px solid var(--line);
        padding:.15rem 0 .55rem; margin:.2rem 0 .45rem; }
    .run-head-id { font-size:1.18rem; font-weight:760; color:var(--ink); }
    .run-head-type { color:#475569; font-size:.9rem; }
    .batch-title { display:flex; justify-content:space-between; align-items:center; font-size:1.08rem; font-weight:750; color:var(--ink); }
    div[data-testid="stButton"] > button[kind="primary"], div[data-testid="stDownloadButton"] > button[kind="primary"] {
        background:#2F5D7C !important; border-color:#2F5D7C !important; color:#fff !important;
        border-radius:.2rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] > button[kind="primary"]:hover { background:#264D68 !important; }
    #MainMenu, footer,
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { display:none !important; }
    header[data-testid="stHeader"] { background:transparent !important; }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display:flex !important; visibility:visible !important; opacity:1 !important;
        pointer-events:auto !important; z-index:1000000 !important;
        position:fixed !important; top:.5rem !important; left:.5rem !important;
        width:2.5rem !important; height:2.5rem !important;
        min-width:2.5rem !important; min-height:2.5rem !important;
        align-items:center !important; justify-content:center !important;
        background:#fff !important; border:1px solid #cbd5e1 !important;
        border-radius:.2rem !important;
    }
    @media (max-width:900px) {
        .flow,.meta-grid { grid-template-columns:repeat(2,1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)




def render_metadata(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        '<div class="meta-item">'
        f'<div class="meta-key">{html.escape(key)}</div>'
        f'<div class="meta-value">{html.escape(str(value))}</div>'
        "</div>"
        for key, value in items
    )
    st.markdown(f'<div class="meta-grid">{cells}</div>', unsafe_allow_html=True)


def format_number(value, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def display_parameter(parameter: str, include_key: bool = True) -> str:
    metadata = parameter_metadata(parameter)
    name = metadata["display_name_zh"]
    return f"{name}（{parameter}）" if include_key and name != parameter else name


def display_actor(actor_id: str) -> str:
    return actor_metadata(actor_id)["display_name_zh"]


SCENARIO_DISPLAY = {"highway_merge": "高速汇入", "lead_brake": "前车制动"}


def _scenario_name(value: object) -> str:
    rendered = str(value or "").strip()
    if rendered.upper() in {"", "UNKNOWN", "NONE", "NULL", "N/A"}:
        return "未记录"
    return SCENARIO_DISPLAY.get(rendered, rendered)


def _status_badge(status: object, warning_code: object = None) -> str:
    css = "status-completed" if status == "COMPLETED_WITH_WARNINGS" and warning_code == "AI_FEEDBACK_UNAVAILABLE" else status_class(status)
    return (
        f'<span class="badge {css}">'
        f'{html.escape(status_display(status, warning_code))}</span>'
    )


AI_FEEDBACK_WARNING_TEXT = (
    "该历史运行的核心仿真、评测及局部边界分析已完成；"
    "旧版生成式摘要未产出，不影响边界结果。"
)


def warning_code_for(run: dict) -> str:
    manifest = run.get("manifest", {}) or {}
    trace = run.get("trace", {}) or {}
    return str(manifest.get("warning_code") or trace.get("warning_code") or "")


def run_status_display(run: dict) -> str:
    manifest = run.get("manifest", {}) or {}
    trace = run.get("trace", {}) or {}
    status = manifest.get("status", trace.get("final_status", "UNKNOWN"))
    return status_display(status, warning_code_for(run))


def run_warning_text(run: dict) -> str:
    if warning_code_for(run) == "AI_FEEDBACK_UNAVAILABLE":
        return AI_FEEDBACK_WARNING_TEXT
    manifest = run.get("manifest", {}) or {}
    trace = run.get("trace", {}) or {}
    return manifest.get("warning_message") or trace.get("warning_message") or "核心结果可用，但部分非关键表达未生成。"


def scenario_asset_runs(runs: list[dict]) -> list[dict]:
    return [run for run in runs if has_persisted_scenario_assets(run)]


def scenario_asset_total(runs: list[dict]) -> int:
    return sum(persisted_scenario_asset_count(run) for run in runs)


def seed_asset_rows(seeds: dict) -> list[dict]:
    rows = []
    for seed in seeds.values():
        parts = []
        for key, value in seed.parameters.items():
            metadata = parameter_metadata(key, seed.scenario_type)
            suffix = f" {metadata['unit']}" if metadata["unit"] else ""
            parts.append(f"{metadata['display_name_zh']} {format_number(value)}{suffix}")
        road = seed.road.get("road_ref", "") if isinstance(seed.road, dict) else seed.road
        rows.append({
            "种子场景 ID": seed.case_id,
            "场景类型": _scenario_name(seed.scenario_type),
            "参数摘要": "；".join(parts),
            "道路": road or "—",
        })
    return rows




def _parameter_summary(record: dict, scenario_type: str, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    parts = []
    specs = SCENARIO_REGISTRY.get(scenario_type, {}).get("parameters", {})
    for key in specs:
        if key in excluded or key not in record or pd.isna(record[key]):
            continue
        metadata = parameter_metadata(key, scenario_type)
        suffix = f" {metadata['unit']}" if metadata["unit"] else ""
        parts.append(f"{metadata['display_name_zh']} {format_number(record[key])}{suffix}")
    return "；".join(parts) or "—"


def _result_lookup(results: pd.DataFrame) -> dict[str, str]:
    if results.empty or "generated_case_id" not in results or "result" not in results:
        return {}
    return dict(zip(results["generated_case_id"], results["result"]))


def coarse_asset_frame(run: dict) -> pd.DataFrame:
    cases = run.get("coarse_cases", pd.DataFrame())
    if cases.empty:
        return pd.DataFrame()
    run_id = run.get("manifest", {}).get("run_id", "")
    results = _result_lookup(run.get("coarse_results", pd.DataFrame()))
    rows = []
    for record in cases.to_dict("records"):
        scenario_type = str(record.get("scenario_type", ""))
        case_id = record.get("generated_case_id", "")
        rows.append({
            "场景实例 ID": case_id,
            "种子场景 ID": record.get("seed_case_id", "—"),
            "场景类型": _scenario_name(scenario_type),
            "参数摘要": _parameter_summary(record, scenario_type),
            "评测结果": results.get(case_id, "—"),
            "来源运行": run_id,
        })
    return pd.DataFrame(rows)


def _fixed_condition_summary(values: dict, scenario_type: str) -> str:
    parts = []
    for key, value in values.items():
        metadata = parameter_metadata(key, scenario_type)
        suffix = f" {metadata['unit']}" if metadata["unit"] else ""
        parts.append(f"{metadata['display_name_zh']} = {format_number(value)}{suffix}")
    return "；".join(parts) or "—"


def _fine_anchor(record: dict, boundary: dict) -> tuple[str, dict]:
    scenario_type = str(record.get("scenario_type", ""))
    candidates = boundary.get("selected_transition_pairs", []) or boundary.get("transition_pairs", [])
    for pair in candidates:
        fixed = pair.get("fixed_parameters", {}) or {}
        if all(key in record and abs(float(record[key]) - float(value)) < 1e-9 for key, value in fixed.items()):
            return str(pair.get("changed_parameter", "")), fixed
    specs = SCENARIO_REGISTRY.get(scenario_type, {}).get("parameters", {})
    return next((key for key in specs if key in record), ""), {}


def fine_asset_frame(run: dict) -> pd.DataFrame:
    cases = run.get("fine_cases", pd.DataFrame())
    if cases.empty:
        return pd.DataFrame()
    run_id = run.get("manifest", {}).get("run_id", "")
    results = _result_lookup(run.get("fine_results", pd.DataFrame()))
    boundary = run.get("boundary", {}) or {}
    rows = []
    for record in cases.to_dict("records"):
        scenario_type = str(record.get("scenario_type", ""))
        parameter, fixed = _fine_anchor(record, boundary)
        metadata = parameter_metadata(parameter, scenario_type) if parameter else {"display_name_zh": "—", "unit": ""}
        suffix = f" {metadata['unit']}" if metadata["unit"] else ""
        case_id = record.get("generated_case_id", "")
        rows.append({
            "细化场景 ID": case_id,
            "种子场景": record.get("seed_case_id", "—"),
            "边界参数": metadata["display_name_zh"],
            "参数值": f"{format_number(record.get(parameter, '—'))}{suffix}" if parameter else "—",
            "固定条件摘要": _fixed_condition_summary(fixed, scenario_type),
            "评测结果": results.get(case_id, "—"),
            "来源运行": run_id,
        })
    return pd.DataFrame(rows)


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def run_duration_text(manifest: dict, trace: dict) -> str:
    started = _parse_timestamp(trace.get("started_at") or manifest.get("created_at"))
    finished = _parse_timestamp(
        trace.get("finished_at") or manifest.get("completed_at") or manifest.get("updated_at")
    )
    if not started or not finished:
        return "—"
    seconds = max(0, int((finished - started).total_seconds()))
    return f"{seconds // 60}分{seconds % 60}秒" if seconds >= 60 else f"{seconds}秒"


def run_scene_count(run: dict) -> int:
    return len(run.get("coarse_cases", pd.DataFrame())) + len(run.get("fine_cases", pd.DataFrame()))


def result_distribution_frame(run: dict) -> pd.DataFrame:
    rows = []
    for stage, key in (("粗粒度探索", "coarse_results"), ("定向细化", "fine_results")):
        frame = run.get(key, pd.DataFrame())
        counts = frame["result"].value_counts().to_dict() if not frame.empty and "result" in frame else {}
        for state in ("PASS", "HIGH_RISK", "FAIL"):
            rows.append({"探索阶段": stage, "评测结果": state, "场景数": int(counts.get(state, 0))})
    return pd.DataFrame(rows)




def distribution_text(distribution: dict) -> str:
    return " / ".join(
        f"{label} {int(distribution.get(label, 0))}"
        for label in ("PASS", "HIGH_RISK", "FAIL")
    )


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="platform-head"><div><h1>{html.escape(title)}</h1>
        <div class="subtle">{html.escape(subtitle)}</div>
        </div><div class="environment-note">仿真环境：本地运动学仿真</div></div>
        """,
        unsafe_allow_html=True,
    )


def load_selected_run(run_id: str):
    try:
        return load_run(run_id, RUNS_ROOT)
    except FileNotFoundError:
        st.warning(f"未找到运行记录：{run_id}")
        return None


def _run_task_context(run: dict) -> dict:
    return recover_run_task_context(run, BATCHES_ROOT)


def _task_metadata_items(task: dict) -> list[tuple[str, str]]:
    items = []
    if task.get("case_id"):
        items.append(("用例 ID", task["case_id"]))
    if task.get("diff"):
        items.append(("结果变化", DIFF_DISPLAY.get(task["diff"], task["diff"])))
    if task.get("severity"):
        items.append(("严重度", SEVERITY_DISPLAY.get(task["severity"], task["severity"])))
    if task.get("scenario_type"):
        items.append(("场景类型", _scenario_name(task["scenario_type"])))
    return items




PIPELINE_STAGE_DEFINITIONS = (
    ("场景理解", ("AI Issue Interpretation",)),
    ("粗粒度探索", ("Coarse Exploration — Attempt 1", "Coarse Exploration — Attempt 2")),
    ("敏感度分析", ("Metric Sensitivity Analysis",)),
    ("自适应重规划", ("AI Adaptive Replan",)),
    ("局部边界", ("Boundary Detection", "Fine Generalization")),
)


def pipeline_stage_states(run: dict) -> list[dict]:
    """Build the five-stage UI projection of persisted trace stage states."""
    trace = run.get("trace", {}) or {}
    events = trace.get("events", []) or []
    indexed: dict[str, list[str]] = {}
    for event in events:
        indexed.setdefault(str(event.get("step", "")), []).append(str(event.get("status", "")))
    rows = []
    for label, steps in PIPELINE_STAGE_DEFINITIONS:
        statuses = [status for step in steps for status in indexed.get(step, [])]
        if "FAILED" in statuses:
            state, css = "失败", "failed"
        elif "NO_BOUNDARY" in statuses:
            state, css = "已完成", "done"
        elif any(status == "COMPLETED" for status in statuses):
            state, css = "已完成", "done"
        elif statuses and all(status in {"SKIPPED", "NO_BOUNDARY"} for status in statuses):
            state, css = "未触发", "skipped"
        elif label == "自适应重规划" and not trace.get("ai_adaptive_replan"):
            state, css = "未触发", "skipped"
        else:
            state, css = "未执行", "pending"
        rows.append({"stage": label, "state": state, "class": css})
    return rows


def render_pipeline_status(run: dict) -> None:
    cells = "".join(
        f'<div class="flow-step {item["class"]}"><strong>{html.escape(item["stage"])}</strong>'
        f'{html.escape(item["state"])}</div>'
        for item in pipeline_stage_states(run)
    )
    st.markdown(f'<div class="flow">{cells}</div>', unsafe_allow_html=True)


def _render_run_task(run: dict) -> None:
    task = _run_task_context(run)
    st.markdown("#### 本次任务")
    render_metadata(_task_metadata_items(task))


def _render_interpretation(run: dict) -> None:
    trace = run.get("trace", {}) or {}
    task = _run_task_context(run)
    interpretation = trace.get("ai_issue_interpretation") or {}
    st.markdown("#### 场景理解摘要")
    if not interpretation:
        st.info("该历史运行记录未保存结构化场景理解字段。")
        return
    scenario_type = interpretation.get("scenario_type", task["scenario_type"])
    canonical_actors = list(registry_summary(scenario_type).get("actors", {})) if scenario_type in SCENARIO_REGISTRY else []
    saved_actors = interpretation.get("actors", [])
    actors = saved_actors if set(saved_actors) == set(canonical_actors) else canonical_actors
    st.write(f"**关键参与者：** {'、'.join(display_actor(actor) for actor in actors) or '未记录'}")
    st.write("**交互关系：** " + interaction_summary_for_user(
        interpretation.get("interaction_summary", "未记录"), task
    ))
    factors = "、".join(display_parameter(item, include_key=False) for item in interpretation.get("candidate_factors", []))
    st.write(f"**候选探索因素：** {factors or '未记录'}")


def _render_distribution_summary(run: dict) -> None:
    distribution = result_distribution_frame(run)
    pivot = distribution.pivot(index="探索阶段", columns="评测结果", values="场景数").reset_index()
    st.markdown("#### 探索分布摘要")
    st.dataframe(pivot[["探索阶段", "PASS", "HIGH_RISK", "FAIL"]], hide_index=True, width="stretch")
    adaptive = run.get("trace", {}).get("ai_adaptive_replan")
    st.caption("自适应重规划：已触发" if adaptive else "自适应重规划：未触发；首轮已覆盖多个评测状态。")


def _render_overview_boundary_discoveries(run: dict) -> None:
    conclusions = build_boundary_conclusions(run)
    st.markdown("#### 本次边界发现")
    if not conclusions:
        st.caption("本次未观察到可靠的局部状态转换区域。")
        return
    st.write(f"发现 {len(conclusions)} 个局部状态转换边界")
    for item in conclusions[:3]:
        with st.container(border=True):
            st.write(f"**{item['parameter_name']}**　{item['refined_interval']}　{item['ordered_transition_states']}")
            st.caption(f"固定：{item['fixed_conditions']}")
    st.caption("详细固定条件和证据请查看《局部边界》页签。")


def _parameter_range_rows(ranges: dict, scenario_type: str) -> list[dict]:
    rows = []
    for key, values in (ranges or {}).items():
        metadata = parameter_metadata(key, scenario_type)
        rows.append({
            "参数": metadata["display_name_zh"],
            "取值": " / ".join(format_number(value) for value in values),
            "单位": metadata["unit"] or "—",
        })
    return rows


DIRECTION_DISPLAY = {
    "increase_for_safer": "增大时趋向较低风险",
    "decrease_for_safer": "减小时趋向较低风险",
    "no_reliable_direction": "未形成可靠方向",
}


def sensitivity_frame(run: dict) -> pd.DataFrame:
    trace = run.get("trace") or {}
    values = (trace.get("metric_sensitivity") or {}).get("parameters") or []
    rows = []
    for item in values:
        parameter = item.get("parameter", "")
        direction = item.get("direction")
        if parameter == "lead_decel_mps2" and direction == "increase_for_safer":
            direction_text = "制动减弱（数值增大）时趋向较低风险"
        elif parameter == "lead_decel_mps2" and direction == "decrease_for_safer":
            direction_text = "制动增强（数值减小）时趋向较低风险"
        else:
            direction_text = DIRECTION_DISPLAY.get(direction, direction or "—")
        support = int(item.get("support_count", 0))
        consistent = int(item.get("consistent_support_count", support))
        rows.append({
            "参数": display_parameter(parameter, include_key=False),
            "方向": direction_text,
            "方向一致证据": f"{consistent} / {support} 组",
        })
    return pd.DataFrame(rows)


def _render_parameter_space(run: dict) -> None:
    trace = run.get("trace", {}) or {}
    seed = run.get("seed", {}) or {}
    scenario_type = seed.get("scenario_type") or run.get("manifest", {}).get("scenario_type", "")
    st.markdown("#### 种子场景参数")
    seed_rows = []
    for key, value in (seed.get("parameters", {}) or {}).items():
        metadata = parameter_metadata(key, scenario_type)
        seed_rows.append({"参数": metadata["display_name_zh"], "取值": format_number(value), "单位": metadata["unit"] or "—"})
    st.dataframe(pd.DataFrame(seed_rows), hide_index=True, width="stretch")
    attempt_1_trace = trace.get("coarse_attempt_1") or {}
    attempt_1 = (attempt_1_trace.get("effective_plan") or {}).get("parameter_ranges") or trace.get("parameter_space", {})
    st.markdown("#### 第 1 轮参数空间")
    st.dataframe(pd.DataFrame(_parameter_range_rows(attempt_1, scenario_type)), hide_index=True, width="stretch")
    effective_replan = trace.get("effective_replan") or {}
    effective = (effective_replan.get("expanded_plan") or {}).get("parameter_ranges")
    if effective:
        st.markdown("#### 第 2 轮生效参数空间")
        st.dataframe(pd.DataFrame(_parameter_range_rows(effective, scenario_type)), hide_index=True, width="stretch")
    else:
        st.caption("未触发第 2 轮自适应重规划。")
    st.markdown("#### 指标敏感度摘要")
    sensitivity = sensitivity_frame(run)
    if sensitivity.empty:
        st.caption("当前运行没有可展示的指标敏感度记录。")
    else:
        st.dataframe(sensitivity, hide_index=True, width="stretch")
        st.caption(
            "“方向一致证据”表示：在可比较的相邻样本对中，有多少组的局部变化方向与总体趋势一致。"
            "该数量用于说明趋势的一致性，不代表场景数、参数重要性、置信度或成功率。"
        )


def _boundary_context(run: dict) -> tuple[list[dict], list[dict]]:
    task = _run_task_context(run)
    boundary = run.get("boundary", {}) or {}
    refinement = assess_refinement(boundary, run.get("fine_cases", pd.DataFrame()), run.get("fine_results", pd.DataFrame()))
    facts = build_local_boundary_facts(task["case_id"], boundary, refinement, run.get("fine_cases", pd.DataFrame()), run.get("fine_results", pd.DataFrame()))
    records = build_local_boundary_records(task["case_id"], boundary, refinement, run.get("fine_cases", pd.DataFrame()), run.get("fine_results", pd.DataFrame()))
    return facts, records


def build_boundary_conclusions(run: dict) -> list[dict]:
    """One deterministic user conclusion per selected local boundary."""
    _, records = _boundary_context(run)
    return records


def render_boundary_cards(run: dict) -> None:
    conclusions = build_boundary_conclusions(run)
    if not conclusions:
        st.caption("本次未观察到可靠的局部状态转换区域。")
        return
    for item in conclusions:
        with st.container(border=True):
            st.markdown(f"##### {item['parameter_name']}")
            st.write(f"**固定条件：** {item['fixed_conditions']}")
            st.write(f"**局部状态转换区：** {item['refined_interval']}")
            st.write(f"**状态变化：** {item['ordered_transition_states']}")
            st.caption(
                f"粗粒度转换区间：{item['coarse_interval']}　｜　精化后：{item['refined_interval']}　｜　"
                f"区间收窄：{item['width_reduction']}"
            )
            st.write(
                f"在上述固定条件及已采样区间内，{item['parameter_name']}约在 "
                f"{item['refined_interval']} 观察到局部状态转换。"
            )


def _render_boundaries(run: dict) -> None:
    records = build_boundary_conclusions(run)
    st.info("以下为当前固定条件下的局部状态转换边界，不代表其他场景上下文中的全局规律。")
    if records:
        st.markdown("#### 边界发现")
        render_boundary_cards(run)
        with st.expander("边界明细", expanded=False):
            st.table(pd.DataFrame([{
                "边界参数": item["parameter_name"],
                "固定条件": item["fixed_conditions"],
                "粗粒度转换区间": item["coarse_interval"],
                "精化区间": item["refined_interval"],
                "状态变化": item["ordered_transition_states"],
                "宽度缩减": item["width_reduction"],
            } for item in records]))
        st.caption("固定条件来自真实状态转换样本组合，不保证等于种子场景参数。")
    elif run.get("boundary", {}).get("status") == "NO_TRANSITION_DETECTED":
        st.warning("两轮粗粒度探索均未观察到局部状态转换。")
    else:
        st.warning("当前未形成可靠的精化边界。")




def render_run_detail(run: dict, key_prefix: str = "run_detail") -> None:
    manifest = run.get("manifest", {}) or {}
    trace = run.get("trace", {}) or {}
    status = manifest.get("status", trace.get("final_status", "UNKNOWN"))
    run_id = manifest.get("run_id") or trace.get("run_id") or "—"
    task = _run_task_context(run)
    st.markdown(
        f'<div class="run-head"><span class="run-head-id">{html.escape(run_id)}</span>'
        f'<span class="run-head-type">{html.escape(_scenario_name(task["scenario_type"]))}</span>'
        f'{_status_badge(status, warning_code_for(run))}</div>', unsafe_allow_html=True,
    )
    render_metadata([
        ("开始时间", format_user_datetime(trace.get("started_at") or manifest.get("created_at"))),
        ("总耗时", run_duration_text(manifest, trace)),
        ("场景资产数", str(run_scene_count(run))),
    ])
    if status == "FAILED":
        st.error("本次运行最终失败；已成功阶段和已落盘数据仍可查看。")
    elif status == "COMPLETED_WITH_WARNINGS" and warning_code_for(run) == "AI_FEEDBACK_UNAVAILABLE":
        st.caption("历史摘要未生成")
        st.caption(run_warning_text(run))
    elif status == "COMPLETED_WITH_WARNINGS":
        st.warning(run_warning_text(run))
    st.markdown("#### 执行链路")
    render_pipeline_status(run)
    overview_tab, parameter_tab, boundary_tab = st.tabs(["运行概览", "参数空间", "局部边界"])
    with overview_tab:
        _render_run_task(run)
        _render_interpretation(run)
        _render_overview_boundary_discoveries(run)
        _render_distribution_summary(run)
    with parameter_tab:
        _render_parameter_space(run)
    with boundary_tab:
        _render_boundaries(run)


def render_risk_map(run: dict, key_prefix: str = "risk_map") -> None:
    manifest = run["manifest"]
    coarse_results = run["coarse_results"]
    scenario_type = manifest.get("scenario_type")
    metadata = SCENARIO_REGISTRY.get(scenario_type, {}).get("risk_map")
    if not metadata or coarse_results.empty:
        st.caption("当前运行没有可展示的真实粗粒度风险切片。")
        return
    coarse_cases = run["coarse_cases"]
    seed_parameters = run.get("seed", {}).get("parameters", {})
    fixed_values = {}
    selector_columns = st.columns(len(metadata["fixed_parameters"]))
    for column, parameter in zip(selector_columns, metadata["fixed_parameters"]):
        values = sorted({float(value) for value in coarse_cases[parameter]})
        default = nearest_observed_value(values, seed_parameters.get(parameter, values[0]))
        fixed_values[parameter] = column.selectbox(
            f"固定 {display_parameter(parameter, include_key=False)}", values, index=values.index(default),
            format_func=lambda value, current=parameter: (
                f"{format_number(value)} {parameter_metadata(current, scenario_type)['unit']}".strip()
            ),
            key=f"{key_prefix}_risk_slice_{run['manifest'].get('run_id', 'unknown')}_{parameter}",
        )
    risk_slice = build_risk_map_slice(
        coarse_cases, coarse_results, seed_parameters, fixed_values,
        x_parameter=metadata["x_parameter"],
        y_parameter=metadata["y_parameter"],
        fixed_parameters=metadata["fixed_parameters"],
    )
    x_parameter, y_parameter = metadata["x_parameter"], metadata["y_parameter"]
    x_meta = parameter_metadata(x_parameter, scenario_type)
    y_meta = parameter_metadata(y_parameter, scenario_type)
    x_label = f"{x_meta['display_name_zh']}（{x_meta['unit']}）"
    y_label = f"{y_meta['display_name_zh']}（{y_meta['unit']}）"
    risk_map = risk_slice["cells"].rename(columns={
        "generated_case_id": "场景实例 ID",
        x_parameter: x_label,
        y_parameter: y_label,
        "result": "评测结果",
    }).copy()
    risk_map["评测结果"] = risk_map["评测结果"].replace({"N/A": "未采样"})
    x_values = sorted({float(value) for value in risk_map[x_label]})
    y_values = sorted({float(value) for value in risk_map[y_label]}, reverse=True)
    heatmap = (
        alt.Chart(risk_map)
        .mark_rect(stroke="white", strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X(
                f"{x_label}:O", title=x_label, sort=x_values,
                axis=alt.Axis(labelAngle=0, labelPadding=6),
            ),
            y=alt.Y(f"{y_label}:O", title=y_label, sort=y_values),
            color=alt.Color(
                "评测结果:N",
                title="状态",
                scale=alt.Scale(
                    domain=["PASS", "HIGH_RISK", "FAIL", "未采样"],
                    range=["#3E7C59", "#C58A2A", "#B84A4A", "#D9DEE5"],
                ),
            ),
            tooltip=["场景实例 ID", x_label, y_label, "评测结果"],
        )
        .properties(
            height=310,
            title=metadata["title_zh"] + " · 固定：" + "，".join(
                f"{parameter_metadata(parameter, scenario_type)['display_name_zh']} "
                f"{format_number(fixed_values[parameter])} "
                f"{parameter_metadata(parameter, scenario_type)['unit']}"
                for parameter in metadata["fixed_parameters"]
            ),
        )
    )
    chart_col, count_col = st.columns([3, 1])
    chart_col.altair_chart(heatmap, width="stretch")
    count_col.markdown("**全部粗粒度分布**")
    count_col.caption(distribution_text(risk_slice["overall_distribution"]))
    count_col.markdown("**当前切片分布**")
    count_col.caption(distribution_text(risk_slice["slice_distribution"]))
    count_col.caption(f"切片已采样场景：{risk_slice['real_case_count']}；灰色表示未采样区域。")


def navigate_to_run(run_id: str) -> None:
    """Switch to the single authoritative Run detail page without touching task state."""
    st.session_state["pending_run_id"] = run_id
    st.session_state["active_page"] = "运行记录"


def render_dashboard(runs: list[dict]) -> None:
    render_page_header("总览", "基于现有运行记录汇总仿真探索状态。")
    run_total = len(runs)
    status_counts = Counter(run.get("status", "UNKNOWN") for run in runs)
    values = [
        ("运行总数", run_total),
        ("已完成", status_counts.get("COMPLETED", 0)),
        ("运行中", status_counts.get("RUNNING", 0)),
        ("有警告", status_counts.get("COMPLETED_WITH_WARNINGS", 0)),
        ("失败", status_counts.get("FAILED", 0)),
        ("场景资产数", scenario_asset_total(runs)),
    ]
    cells = "".join(
        f'<div class="overview-cell"><div class="overview-label">{label}</div>'
        f'<div class="overview-value">{value}</div></div>' for label, value in values
    )
    st.markdown(f'<div class="overview-strip">{cells}</div>', unsafe_allow_html=True)
    st.markdown("### 最近运行记录")
    recent = []
    recent_runs = []
    for manifest in runs[:6]:
        loaded = None
        try:
            loaded = load_run(manifest["run_id"], RUNS_ROOT)
        except FileNotFoundError:
            pass
        summary = manifest.get("summary", {}) or {}
        trace = (loaded or {}).get("trace", {})
        recent.append({
            "运行 ID": manifest.get("run_id", ""),
            "场景类型": _scenario_name(manifest.get("scenario_type", "")),
            "状态": run_status_display(loaded) if loaded else status_display(manifest.get("status"), manifest.get("warning_code")),
            "场景资产数": run_scene_count(loaded) if loaded else summary.get("coarse_generated", 0) + summary.get("fine_generated", 0),
            "开始时间": format_user_datetime(trace.get("started_at", manifest.get("created_at"))),
            "耗时": run_duration_text(manifest, trace),
        })
        if loaded and run_has_results(loaded):
            recent_runs.append(loaded)
    st.dataframe(pd.DataFrame(recent), width="stretch", hide_index=True)
    if recent:
        selected = st.selectbox("选择最近运行", [item["运行 ID"] for item in recent], key="overview_run_id")
        st.button("查看运行详情", key="overview_view_run", on_click=navigate_to_run, args=(selected,))
    st.markdown("### 近期边界发现")
    discoveries = []
    for manifest in runs:
        try:
            loaded = load_run(manifest["run_id"], RUNS_ROOT)
        except FileNotFoundError:
            continue
        if local_boundary_count(loaded) > 0:
            discoveries.append(loaded)
        if len(discoveries) == 4:
            break
    if discoveries:
        for loaded in discoveries:
            item_manifest = loaded.get("manifest", {})
            boundary_names = list(dict.fromkeys(
                item["parameter_name"] for item in build_boundary_conclusions(loaded)
            ))
            run_id = item_manifest.get("run_id", "")
            with st.container(border=True):
                st.write(f"**{run_id}**　{_scenario_name(item_manifest.get('scenario_type'))} · {item_manifest.get('case_id', loaded.get('seed', {}).get('case_id', ''))}")
                st.write(f"发现 {local_boundary_count(loaded)} 个局部状态转换边界")
                st.caption("主要边界参数：" + "、".join(boundary_names))
                st.button("查看运行详情", key=f"recent_boundary_{run_id}", on_click=navigate_to_run, args=(run_id,))
    else:
        st.caption("近期暂无形成局部状态转换边界的运行记录。")


def render_scenario_assets(runs: list[dict], seeds: dict) -> None:
    render_page_header("场景资产", "查看种子场景及探索过程中生成的粗粒度与细化场景资产。")
    seed_tab, coarse_tab, fine_tab = st.tabs(["种子场景", "粗粒度场景", "细化场景"])
    loaded_assets = []
    for manifest in scenario_asset_runs(runs):
        loaded = load_selected_run(manifest["run_id"])
        if loaded is not None:
            loaded_assets.append((manifest, loaded))
    with seed_tab:
        st.dataframe(pd.DataFrame(seed_asset_rows(seeds)), width="stretch", hide_index=True)
        st.caption("种子场景是独立资产，无需选择运行记录。")
    with coarse_tab:
        coarse_runs = [(manifest, loaded) for manifest, loaded in loaded_assets if not loaded["coarse_cases"].empty]
        if not coarse_runs:
            st.info("暂无有效的粗粒度场景资产。")
        else:
            labels = {manifest["run_id"]: f"{manifest['run_id']} · {run_status_display(loaded)}" + ("但存在可用资产" if manifest.get("status") == "FAILED" else "") for manifest, loaded in coarse_runs}
            run_id = st.selectbox("选择运行记录", list(labels), format_func=lambda value: labels[value], key="coarse_asset_run_id")
            selected = next(loaded for manifest, loaded in coarse_runs if manifest["run_id"] == run_id)
            st.dataframe(coarse_asset_frame(selected), width="stretch", hide_index=True)
    with fine_tab:
        fine_runs = [(manifest, loaded) for manifest, loaded in loaded_assets if not loaded["fine_cases"].empty]
        if not fine_runs:
            st.info("暂无有效的细化场景资产。")
        else:
            labels = {manifest["run_id"]: f"{manifest['run_id']} · {run_status_display(loaded)}" + ("但存在可用资产" if manifest.get("status") == "FAILED" else "") for manifest, loaded in fine_runs}
            run_id = st.selectbox("选择运行记录", list(labels), format_func=lambda value: labels[value], key="fine_asset_run_id")
            selected = next(loaded for manifest, loaded in fine_runs if manifest["run_id"] == run_id)
            st.dataframe(fine_asset_frame(selected), width="stretch", hide_index=True)


def render_run_records(runs: list[dict]) -> None:
    render_page_header("运行记录", "查看运行列表及单个运行的执行与边界分析结果。")
    rows = []
    loaded_by_id = {}
    for manifest in runs:
        trace = {}
        try:
            loaded_by_id[manifest["run_id"]] = load_run(manifest["run_id"], RUNS_ROOT)
            trace = loaded_by_id[manifest["run_id"]]["trace"]
        except FileNotFoundError:
            pass
        summary = manifest.get("summary", {}) or {}
        rows.append({
            "运行 ID": manifest.get("run_id") or "—",
            "场景类型": _scenario_name(manifest.get("scenario_type")),
            "状态": run_status_display(loaded_by_id[manifest["run_id"]]) if manifest.get("run_id") in loaded_by_id else status_display(manifest.get("status", "UNKNOWN"), manifest.get("warning_code")),
            "开始时间": format_user_datetime(trace.get("started_at", manifest.get("created_at"))),
            "耗时": run_duration_text(manifest, trace),
            "粗粒度场景": summary.get("coarse_generated", len(loaded_by_id.get(manifest.get("run_id"), {}).get("coarse_cases", []))),
            "细化场景": summary.get("fine_generated", len(loaded_by_id.get(manifest.get("run_id"), {}).get("fine_cases", []))),
            "局部边界": local_boundary_count(loaded_by_id[manifest["run_id"]]) if manifest.get("run_id") in loaded_by_id else 0,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=225)
    if not rows:
        st.info("暂无运行记录。")
        return
    pending = st.session_state.pop("pending_run_id", None)
    ids = [row["运行 ID"] for row in rows]
    if pending in ids:
        st.session_state["record_run_id"] = pending
    elif st.session_state.get("record_run_id") not in ids:
        st.session_state["record_run_id"] = ids[0]
    run_id = st.selectbox("选择运行记录", ids, key="record_run_id")
    selected = loaded_by_id.get(run_id) or load_selected_run(run_id)
    if selected:
        render_run_detail(selected, key_prefix=f"record_{run_id}")


def render_evaluation(runs: list[dict]) -> None:
    render_page_header("评测分析", "以局部边界结论为核心，查看风险切片、敏感度证据与探索样本。")
    result_ids = []
    for manifest in runs:
        try:
            if run_has_results(load_run(manifest["run_id"], RUNS_ROOT)):
                result_ids.append(manifest["run_id"])
        except FileNotFoundError:
            continue
    if not result_ids:
        st.info("暂无可展示的运行结果。")
        return
    run_id = st.selectbox("选择运行记录", result_ids, key="evaluation_run_id")
    selected = load_selected_run(run_id)
    if selected is None:
        return
    boundary_tab, risk_tab, sensitivity_tab, sample_tab = st.tabs(["边界结论", "风险分布", "参数敏感度", "样本分布"])
    with boundary_tab:
        st.caption(
            "以下结论来自当前本地运动学仿真、对应固定条件和已采样区间，"
            "属于局部状态转换边界，不代表真实自动驾驶系统的全局能力边界。"
        )
        render_boundary_cards(selected)
    with sample_tab:
        st.markdown("#### 主动探索样本状态分布")
        st.caption("主动探索样本数")
        distribution = result_distribution_frame(selected)
        chart = alt.Chart(distribution).mark_bar(size=34).encode(
            x=alt.X("探索阶段:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("场景数:Q", title=None, axis=alt.Axis(labels=True, ticks=True, labelPadding=4)),
            xOffset="评测结果:N",
            color=alt.Color("评测结果:N", scale=alt.Scale(domain=["PASS", "HIGH_RISK", "FAIL"], range=["#3E7C59", "#C58A2A", "#B84A4A"]), title="评测结果"),
            tooltip=["探索阶段", "评测结果", "场景数"],
        ).properties(height=330, padding={"left": 4, "right": 4, "top": 4, "bottom": 4})
        st.altair_chart(chart, width="stretch")
        st.caption("该图展示当前主动构造探索样本的评测状态分布，不代表自然场景分布下的通过率或失败率。")
    with risk_tab:
        st.info("风险分布是当前固定条件切片下的局部投影，不代表参数空间的全局风险地图。")
        render_risk_map(selected, key_prefix=f"evaluation_{run_id}")
    with sensitivity_tab:
        sensitivity = sensitivity_frame(selected)
        if sensitivity.empty:
            st.info("当前运行没有可展示的参数敏感度数据。")
        else:
            st.dataframe(sensitivity, hide_index=True, width="stretch")
            st.caption(
                "“方向一致证据”表示：在可比较的相邻样本对中，有多少组的局部变化方向与总体趋势一致。"
                "该数量用于说明趋势的一致性，不代表场景数、参数重要性、置信度或成功率。"
            )
            with st.expander("证据说明", expanded=False):
                st.write(
                    "程序先固定其他全部注册参数，把只有当前参数不同的场景组成可比较样本组，"
                    "并按当前参数值从小到大排序。随后比较组内相邻样本，计算两个样本之间的“安全裕度”变化。"
                )
                st.write(
                    "安全裕度综合考虑最小碰撞时间（TTC）和最小间距相对 FAIL 判定阈值的余量，"
                    "并按 FAIL 与 HIGH_RISK 两级阈值之间的范围换算到同一尺度，取其中更保守的一项；"
                    "发生碰撞时额外扣减。"
                )
                st.write(
                    "每组相邻样本的安全裕度变化除以参数变化量，得到该组的局部变化率。"
                    "总体趋势取所有有效局部变化率的中位数方向。"
                    "例如 35 / 48 组表示共找到 48 组有效可比较样本对，其中 35 组的局部变化方向与总体趋势一致。"
                )


seeds = load_seed_scenarios()
runs = list_runs(RUNS_ROOT)
NAVIGATION_LABELS = {
    "总览": "▦  总览",
    "场景资产": "▤  场景资产",
    "运行记录": "☷  运行记录",
    "评测分析": "⌁  评测分析",
    "场景探索": "↻  场景探索",
}
st.session_state.setdefault("active_page", "总览")

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">SimProbe</div>
            <div class="sidebar-brand-subtitle">仿真探索与边界分析</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "工作区",
        [
            "总览",
            "场景资产",
            "运行记录",
            "评测分析",
            "场景探索",
        ],
        format_func=lambda value: NAVIGATION_LABELS[value],
        key="active_page",
        label_visibility="collapsed",
    )

if page == "总览":
    render_dashboard(runs)
    st.stop()
if page == "场景资产":
    render_scenario_assets(runs, seeds)
    st.stop()
if page == "运行记录":
    render_run_records(runs)
    st.stop()
if page == "评测分析":
    render_evaluation(runs)
    st.stop()
render_page_header(
    "场景探索",
    "从版本回归问题中筛选候选任务，执行参数探索与局部边界分析。",
)
st.caption(
    "智能规划：Qwen3.8-Flash · 用于场景理解、因素规划和条件触发的自适应重规划；"
    "仿真、评测、敏感度与边界判定由确定性引擎完成。"
)

st.markdown("### Daily 回归报告")
st.caption("支持 XLSX · 支持中英文表头")
uploaded = st.file_uploader("Daily 报告", type=["xlsx"], key="daily_upload", label_visibility="collapsed")
template_col, state_col = st.columns([1, 3])
with template_col:
    st.download_button("下载示例模板", data=SAMPLE_REPORT.read_bytes(), file_name="SimProbe_示例回归报告.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if uploaded is not None:
    st.session_state["daily_upload_bytes"] = uploaded.getvalue()
    st.session_state["daily_upload_name"] = getattr(uploaded, "name", "uploaded.xlsx")
    daily_source = uploaded
elif st.session_state.get("daily_upload_bytes"):
    daily_source = io.BytesIO(st.session_state["daily_upload_bytes"])
    daily_source.name = st.session_state.get("daily_upload_name", "uploaded.xlsx")
else:
    daily_source = None

daily_df = None
candidates, ineligible_cases, non_candidate_cases = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
selected_case_ids: list[str] = []
if daily_source is None:
    state_col.caption("尚未上传报告")
else:
    try:
        daily_df = load_daily_report(daily_source)
        candidates, ineligible_cases, non_candidate_cases = classify_daily_cases(daily_df, seeds)
        anomaly_count = int(daily_df["curr_result"].isin(["FAIL", "HIGH_RISK"]).sum())
        state_col.success(f"文件：{getattr(daily_source, 'name', 'uploaded.xlsx')} · 总用例数：{len(daily_df)} · 异常数：{anomaly_count} · 可执行候选数：{len(candidates)} · 不可执行异常数：{len(ineligible_cases)}")
        if daily_df.attrs.get("daily_format") == "legacy_v0.3.2":
            st.warning(daily_df.attrs.get("compatibility_notice"))
    except Exception as exc:
        st.error(f"Daily 报告加载失败：{exc}")

st.markdown("### 候选任务")
if daily_df is not None:
    candidate_values = [
        ("总用例数", len(daily_df)),
        ("异常数", int(daily_df["curr_result"].isin(["FAIL", "HIGH_RISK"]).sum())),
        ("可执行候选数", len(candidates)),
        ("不可执行异常数", len(ineligible_cases)),
    ]
    candidate_cells = "".join(f'<div class="overview-cell"><div class="overview-label">{label}</div><div class="overview-value">{value}</div></div>' for label, value in candidate_values)
    st.markdown(f'<div class="overview-strip" style="grid-template-columns:repeat(4,1fr)">{candidate_cells}</div>', unsafe_allow_html=True)
    st.caption("候选资格由确定性规则判断，是否执行由用户决定。")
    for _, candidate in candidates.iterrows():
        case_id, state = candidate["case_id"], candidate["curr_result"]
        badge_class = "badge-fail" if state == "FAIL" else "badge-risk"
        seed_case_id = str(candidate.get("seed_case_id", ""))
        seed_mapping = (
            f" · 关联种子场景：{html.escape(seed_case_id)}"
            if seed_case_id and seed_case_id != case_id else ""
        )
        st.markdown(f'<div class="candidate-card"><span class="badge {badge_class}">{html.escape(state)}</span> <strong>{html.escape(case_id)}</strong><br><span class="subtle">{html.escape(DIFF_DISPLAY.get(candidate["diff"], candidate["diff"]))} · {html.escape(SEVERITY_DISPLAY.get(candidate["severity"], candidate["severity"]))} · 建议：{html.escape(candidate["recommendation"])}{seed_mapping}</span></div>', unsafe_allow_html=True)
        checked = st.checkbox(case_id, value=bool(candidate["default_selected"]), key=f"candidate_selected_{case_id}")
        if checked:
            selected_case_ids.append(case_id)
        st.caption(f"原因：{candidate['candidate_reason']}")
    if candidates.empty:
            st.info("当前没有具备有效种子场景的异常候选任务。")

    with st.expander(f"不可执行异常 ({len(ineligible_cases)})", expanded=False):
        if ineligible_cases.empty:
            st.caption("当前没有不可执行异常。")
        else:
            display = ineligible_cases.rename(columns={"case_id": "用例 ID", "curr_result": "当前评测结果", "severity": "严重度", "exclusion_reason": "原因"}).copy()
            display["严重度"] = display["严重度"].map(lambda value: SEVERITY_DISPLAY.get(value, value))
            st.dataframe(display[["用例 ID", "当前评测结果", "严重度", "原因"]], hide_index=True, width="stretch")
    with st.expander(f"未进入候选 ({len(non_candidate_cases)})", expanded=False):
        if non_candidate_cases.empty:
            st.caption("当前没有未进入候选的用例。")
        else:
            display = non_candidate_cases.rename(columns={"case_id": "用例 ID", "curr_result": "当前评测结果", "diff": "结果变化", "exclusion_reason": "原因"}).copy()
            display["结果变化"] = display["结果变化"].map(lambda value: DIFF_DISPLAY.get(value, value))
            st.dataframe(display[["用例 ID", "当前评测结果", "结果变化", "原因"]], hide_index=True, width="stretch")
    with st.expander("查看候选判定规则", expanded=False):
        st.write(candidate_filter_rule_text())
        st.caption("候选资格由确定性规则判断，不由大模型决定。")

execute_disabled = daily_df is None or not selected_case_ids
execute_requested = st.button(f"执行选中的 {len(selected_case_ids)} 个探索任务", type="primary", disabled=execute_disabled, key="execute_closed_loop")
if execute_requested:
    with st.spinner("正在依次执行选中的独立探索任务并汇总中文报告..."):
        try:
            source_bytes = st.session_state.get("daily_upload_bytes") or daily_source.getvalue()
            result = execute_batch(daily_df=daily_df, selected_case_ids=selected_case_ids, seeds=seeds, source_daily_name=getattr(daily_source, "name", "uploaded.xlsx"), source_daily_bytes=source_bytes, runs_root=RUNS_ROOT, batches_root=BATCHES_ROOT)
            st.session_state["current_batch_id"] = result["batch_id"]
            st.rerun()
        except Exception as exc:
            st.error(f"批量探索执行失败：{type(exc).__name__}: {exc}")

current_batch_id = st.session_state.get("current_batch_id")
if current_batch_id:
    try:
        current_batch = load_batch(current_batch_id, BATCHES_ROOT)
        outcomes = current_batch.get("run_outcomes", [])
        completed = sum(item.get("status") == "COMPLETED" for item in outcomes)
        warned = sum(item.get("status") == "COMPLETED_WITH_WARNINGS" for item in outcomes)
        failed = sum(item.get("status") == "FAILED" for item in outcomes)
        status_zh = status_display(current_batch.get("status"))
        status_badge = status_class(current_batch.get("status"))
        batch_created = _parse_timestamp(current_batch.get("created_at"))
        batch_updated = _parse_timestamp(current_batch.get("updated_at"))
        batch_duration = "—" if not batch_created or not batch_updated else run_duration_text({"created_at": current_batch.get("created_at"), "updated_at": current_batch.get("updated_at")}, {})
        st.markdown(f'<div class="batch-card"><div class="batch-title">批次执行结果 <span class="badge {status_badge}">{html.escape(status_zh)}</span></div><div style="display:flex;gap:2.4rem;margin-top:.55rem"><div>批次 ID<br><strong>{html.escape(current_batch_id)}</strong></div><div>执行数<br><strong>{len(current_batch.get("selected_case_ids", []))}</strong></div><div>完成<br><strong>{completed}</strong></div><div>警告<br><strong>{warned}</strong></div><div>失败<br><strong>{failed}</strong></div><div>耗时<br><strong>{batch_duration}</strong></div></div></div>', unsafe_allow_html=True)
        report_path = Path(current_batch["report_path"])
        if report_path.is_file():
            st.download_button("下载分析报告", data=report_path.read_bytes(), file_name=report_download_name(current_batch_id), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        zip_path = Path(current_batch["delivery_zip_path"])
        if zip_path.is_file():
            st.download_button("下载结果包", data=zip_path.read_bytes(), file_name=delivery_download_name(current_batch_id), mime="application/zip")
            st.caption("结果包包含原始回归报告和中文分析报告；完整运行记录已单独保存。")
        batch_rows = []
        for outcome in outcomes:
            run_id = outcome.get("run_id", "未创建")
            scenario_name = {"highway_merge": "高速汇入", "lead_brake": "前车制动"}.get(outcome.get("scenario_type", ""), outcome.get("scenario_type", ""))
            loaded_run = {}
            try:
                loaded_run = load_run(run_id, RUNS_ROOT) if run_id else {}
                run_summary = loaded_run.get("manifest", {}).get("summary", {}) or {}
            except FileNotFoundError:
                run_summary = {}
            boundary_count = local_boundary_count(loaded_run) if loaded_run else 0
            candidate_row = candidates[candidates["case_id"] == outcome.get("case_id")] if not candidates.empty else pd.DataFrame()
            warning_code = (outcome.get("result", {}) or {}).get("warning_code") or outcome.get("warning_code")
            batch_rows.append({"运行 ID": run_id or "—", "用例 ID": outcome.get("case_id") or "—", "场景类型": scenario_name or "未记录", "状态": status_display(outcome.get("status"), warning_code), "场景资产数": run_summary.get("coarse_generated", 0) + run_summary.get("fine_generated", 0), "局部边界": boundary_count})
        st.dataframe(pd.DataFrame(batch_rows), hide_index=True, width="stretch")
        for row in batch_rows:
            if row["运行 ID"] and row["运行 ID"] != "未创建":
                st.button(f"查看详情 · {row['运行 ID']}", key=f"batch_view_{current_batch_id}_{row['运行 ID']}", on_click=navigate_to_run, args=(row["运行 ID"],))
    except FileNotFoundError:
        st.warning(f"未找到批次记录：{current_batch_id}")
