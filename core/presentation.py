"""User-facing run status and persisted-asset helpers.

These helpers deliberately inspect files instead of inferring data availability
from a run's terminal status.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .report_parser import DIFF_DISPLAY, SEVERITY_DISPLAY
from .scenario_registry import SCENARIO_REGISTRY, actor_metadata, parameter_metadata


RUN_STATUS_DISPLAY = {
    "READY": "待执行",
    "RUNNING": "运行中",
    "COMPLETED": "已完成",
    "COMPLETED_WITH_WARNINGS": "已完成（有警告）",
    "FAILED": "失败",
    "PARTIAL_FAILED": "部分失败",
    "UNKNOWN": "状态未知",
}

RUN_STATUS_CLASS = {
    "READY": "status-running",
    "RUNNING": "status-running",
    "COMPLETED": "status-completed",
    "COMPLETED_WITH_WARNINGS": "status-warning",
    "FAILED": "status-failed",
    "PARTIAL_FAILED": "status-warning",
    "UNKNOWN": "status-unknown",
}


def status_display(status: Any, warning_code: Any = None) -> str:
    value = str(status or "UNKNOWN")
    if value == "COMPLETED_WITH_WARNINGS" and warning_code == "AI_FEEDBACK_UNAVAILABLE":
        return "已完成"
    return RUN_STATUS_DISPLAY.get(value, "状态未知")


def status_class(status: Any) -> str:
    return RUN_STATUS_CLASS.get(str(status or "UNKNOWN"), "status-unknown")


def localize_user_prose(value: Any) -> str:
    """Localize generated prose without mutating canonical technical artifacts."""
    text = str(value if value is not None else "")
    parameter_keys = {
        key for scenario in SCENARIO_REGISTRY.values()
        for key in scenario["parameters"]
    }
    for key in sorted(parameter_keys, key=len, reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])",
            parameter_metadata(key)["display_name_zh"],
            text,
        )
    actor_ids = {
        actor_id for scenario in SCENARIO_REGISTRY.values()
        for actor_id in scenario["actors"]
    }
    for actor_id in sorted(actor_ids, key=len, reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(actor_id)}(?![A-Za-z0-9_])",
            actor_metadata(actor_id)["display_name_zh"],
            text,
        )
    for token, display in sorted(DIFF_DISPLAY.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            display,
            text,
            flags=re.IGNORECASE,
        )
    for token, display in SEVERITY_DISPLAY.items():
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            display,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"collision\s*=\s*false", "未发生碰撞", text, flags=re.IGNORECASE)
    text = re.sub(r"collision\s*=\s*true", "发生碰撞", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![A-Za-z0-9_])min_ttc_s(?![A-Za-z0-9_])", "最小碰撞时间（TTC）", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![A-Za-z0-9_])min_gap_m(?![A-Za-z0-9_])", "最小间距", text, flags=re.IGNORECASE)
    text = text.replace(
        "LocalKinematicSimulator + EvaluationEngine",
        "本地运动学仿真与确定性评测",
    )
    text = re.sub(r"Local\s+EvaluationEngine", "本地评测规则", text, flags=re.IGNORECASE)
    text = re.sub(r"ground\s+truth", "内部参考标签", text, flags=re.IGNORECASE)
    text = re.sub(r"safety\s+margin", "安全裕度", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmedian\b", "中位数", text, flags=re.IGNORECASE)
    text = re.sub(r"本地\s+surrogate(?:\s+backend)?", "本地运动学仿真", text, flags=re.IGNORECASE)
    text = re.sub(r"local\s+surrogate(?:\s+backend)?", "本地运动学仿真", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsurrogate\b", "运动学仿真", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSeed\b", "种子场景", text)
    text = re.sub(r"\bN/A\b", "未采样", text, flags=re.IGNORECASE)
    replacements = {
        "fixed_parameters": "固定条件",
        "Scenario Assets": "场景资产",
        "Transition Anchor": "对应固定条件",
        "Anchor": "对应固定条件",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"本地\s+运动学仿真", "本地运动学仿真", text)
    text = re.sub(r"([\u4e00-\u9fff]{1,24})（\1）", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?==)", "", text)
    text = re.sub(r"[ \t]+([，。；：！？、）])", r"\1", text)
    text = re.sub(r"（[ \t]+", "（", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


_SYSTEM_CAUSAL_TERMS = (
    "跟车逻辑", "跟车策略", "纵向控制逻辑", "间距控制逻辑", "控制策略",
    "速度匹配策略", "相对速度匹配策略", "控制响应", "规划响应", "算法响应", "系统响应", "响应及时性",
)
_INTERACTION_UNSAFE_TERMS = (
    "建议", "推荐", "需进一步", "需要进一步", "需要扩展", "建议扩展", "建议探索",
    "需排查", "重点排查", "验证方案", "高保真", "安全裕度", "敏感", *_SYSTEM_CAUSAL_TERMS,
)
_TASK_STATUS_PHRASES = tuple(dict.fromkeys(DIFF_DISPLAY.values()))


def _split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[。！？；])\s*|\n+", text) if item.strip()]


def _has_standalone_severity(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])(?:HIGH|MEDIUM|LOW)(?![A-Za-z0-9_])", text, re.I))


def _contains_registered_key(text: str) -> bool:
    keys = {
        key for scenario in SCENARIO_REGISTRY.values() for key in scenario["parameters"]
    } | {
        actor for scenario in SCENARIO_REGISTRY.values() for actor in scenario["actors"]
    }
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", text) for key in keys)


def _interaction_fallback(context: dict[str, Any] | None) -> str:
    context = context or {}
    metric_snapshot = context.get("metric_snapshot") or (
        context.get("source_task_context") or {}
    ).get("metric_snapshot")
    if metric_snapshot:
        return "当前记录的直接观测指标为：" + localize_user_prose(metric_snapshot).strip("。") + "。"
    return "当前记录未包含可独立展示的交互事实。"


def interaction_summary_for_user(value: Any, fallback_context: dict[str, Any] | None = None) -> str:
    """Project an issue summary into factual user prose without rewriting its evidence."""
    original = str(value if value is not None else "").strip()
    if not original:
        return ""
    localized = localize_user_prose(original)
    safe = []
    for sentence in _split_sentences(localized):
        if any(term in sentence for term in _INTERACTION_UNSAFE_TERMS):
            continue
        if any(status in sentence for status in _TASK_STATUS_PHRASES):
            continue
        if _has_standalone_severity(sentence):
            continue
        if re.search(r"V\d+\.\d+(?:\.\d+)?", sentence, re.I):
            continue
        if any(marker in sentence for marker in (
            "仅基于当前本地运动学仿真", "仅限于当前本地运动学仿真",
            "产品限制", "真实自动驾驶系统", "真实 ADS", "不涉及对控制",
        )):
            continue
        if _contains_registered_key(sentence):
            continue
        safe.append(sentence)
    result = "".join(safe).strip()
    return result or _interaction_fallback(fallback_context)


def format_user_datetime(value: Any) -> str:
    """Format persisted ISO timestamps for user-facing views only."""
    if value in (None, ""):
        return "—"
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def local_boundary_count(run: dict) -> int:
    """Count final local boundaries, never all candidate transition pairs."""
    trace = run.get("trace", {}) or {}
    if "local_transition_boundary_facts" in trace:
        return len(trace.get("local_transition_boundary_facts") or [])
    boundary = run.get("boundary", {}) or {}
    return len(boundary.get("selected_transition_pairs") or [])


def _valid_case_list(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict) and item.get("case_id")]


def persisted_scenario_asset_count(run_or_manifest: dict) -> int:
    run_dir = Path(run_or_manifest.get("run_dir", ""))
    if not run_dir.is_dir():
        return 0
    return sum(
        len(_valid_case_list(run_dir / relative))
        for relative in ("coarse/cases.json", "fine/cases.json")
    )


def has_persisted_scenario_assets(run_or_manifest: dict) -> bool:
    return persisted_scenario_asset_count(run_or_manifest) > 0


def run_has_results(run: dict) -> bool:
    return any(
        not run.get(name).empty
        for name in ("coarse_cases", "coarse_results", "fine_cases", "fine_results")
        if hasattr(run.get(name), "empty")
    ) or bool(run.get("boundary"))
