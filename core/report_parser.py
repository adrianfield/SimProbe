from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from .evaluator import EVALUATION_THRESHOLDS


COMMON_REQUIRED = {
    "case_id", "version_prev", "version_curr", "prev_result", "curr_result",
    "severity",
}
FORMAL_REQUIRED = COMMON_REQUIRED | {"issue_description", "metric_snapshot"}
LEGACY_FIELDS = {"scenario_name", "scenario_type", "issue_type"}

CHINESE_HEADER_MAP = {
    "用例ID": "case_id", "用例 ID": "case_id", "上一版本": "version_prev", "当前版本": "version_curr",
    "上一版本结果": "prev_result", "当前评测结果": "curr_result", "结果变化": "diff",
    "严重度": "severity", "问题描述": "issue_description", "评测原因": "evaluation_reason",
    "评测说明": "evaluation_reason", "关键指标": "metric_snapshot", "日志摘要": "log_summary",
    "备注": "comment", "种子场景ID": "seed_case_ref", "种子场景 ID": "seed_case_ref",
    "关联种子场景": "seed_case_ref", "Seed Scenario": "seed_case_ref",
    "Seed ID": "seed_case_ref",
}
USER_DAILY_HEADERS = {
    "case_id": "用例 ID", "version_prev": "上一版本", "version_curr": "当前版本",
    "prev_result": "上一版本结果", "curr_result": "当前评测结果", "diff": "结果变化",
    "severity": "严重度", "issue_description": "问题描述", "evaluation_reason": "评测原因",
    "metric_snapshot": "关键指标", "log_summary": "日志摘要", "comment": "备注",
    "seed_case_ref": "关联种子场景",
}
DIFF_DISPLAY = {
    "NEW_FAIL": "新增失败", "REPEATED_FAIL": "持续失败",
    "NEW_HIGH_RISK": "新增高风险", "REPEATED_HIGH_RISK": "持续高风险",
    "ESCALATED_TO_FAIL": "风险升级为失败", "RISK_REDUCED": "风险程度下降",
    "RECOVERED": "已恢复", "FIXED": "已恢复", "UNCHANGED": "无变化",
}
SEVERITY_DISPLAY = {"HIGH": "高", "MEDIUM": "中", "LOW": "低"}
ELIGIBILITY_DISPLAY = {
    "CANDIDATE": "可执行候选", "INELIGIBLE_NO_SEED": "不可执行异常",
    "NOT_CANDIDATE": "未进入候选",
}
EXECUTION_DISPLAY = {
    "RUNNING": "运行中", "COMPLETED": "已完成",
    "COMPLETED_WITH_WARNINGS": "已完成（有警告）", "FAILED": "执行失败",
    "NOT_SELECTED": "用户未选择",
    "NOT_APPLICABLE": "不适用",
}
REVIEW_DISPLAY = {"CONSISTENT": "一致", "MISMATCH": "规则可能不同", "UNKNOWN": "无法复核"}
_UNPARSEABLE_EVALUATION_REASON = "关键指标不足或无法解析，无法按当前本地评测规则重建具体判定原因。"
_DIFF_CANONICAL = {value: key for key, value in DIFF_DISPLAY.items()}
_DIFF_CANONICAL["已修复"] = "RECOVERED"
_SEVERITY_CANONICAL = {value: key for key, value in SEVERITY_DISPLAY.items()}
_VALID_RESULTS = {"PASS", "HIGH_RISK", "FAIL"}
_VALID_SEVERITIES = {"HIGH", "MEDIUM", "LOW"}

# Compatibility-only table retained for old callers. v0.4.5 classification does not use it.
CANDIDATE_PRIORITY = {
    ("NEW_FAIL", "HIGH"): 10, ("NEW_FAIL", "MEDIUM"): 20,
    ("REPEATED_FAIL", "HIGH"): 30, ("REPEATED_FAIL", "MEDIUM"): 40,
}
_SEVERITY_SORT = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_STATE_SORT = {"FAIL": 0, "HIGH_RISK": 1}
_RECOMMENDATION = {"HIGH": "优先处理", "MEDIUM": "建议处理", "LOW": "可选处理"}


def derive_diff(prev: str, curr: str) -> str:
    pair = (str(prev).upper(), str(curr).upper())
    return {
        ("PASS", "FAIL"): "NEW_FAIL", ("FAIL", "FAIL"): "REPEATED_FAIL",
        ("PASS", "HIGH_RISK"): "NEW_HIGH_RISK", ("HIGH_RISK", "HIGH_RISK"): "REPEATED_HIGH_RISK",
        ("HIGH_RISK", "FAIL"): "ESCALATED_TO_FAIL", ("FAIL", "HIGH_RISK"): "RISK_REDUCED",
        ("FAIL", "PASS"): "RECOVERED", ("HIGH_RISK", "PASS"): "RECOVERED",
        ("PASS", "PASS"): "UNCHANGED",
    }.get(pair, "UNCHANGED")


def _normalise_headers(frame: pd.DataFrame) -> pd.DataFrame:
    sources: dict[str, list[str]] = {}
    for column in frame.columns:
        target = CHINESE_HEADER_MAP.get(column, column)
        sources.setdefault(target, []).append(str(column))
    collisions = {target: names for target, names in sources.items() if len(names) > 1}
    if collisions:
        labels = "、".join(USER_DAILY_HEADERS.get(target, target) for target in collisions)
        raise ValueError(f"检测到重复字段：请不要同时保留中文列和对应英文列：{labels}。")
    return frame.rename(columns={name: CHINESE_HEADER_MAP[name] for name in frame.columns if name in CHINESE_HEADER_MAP})


def is_blank_business_value(value) -> bool:
    """Treat spreadsheet nulls and placeholder null strings as missing business values."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _business_text(value) -> str:
    return "" if is_blank_business_value(value) else str(value).strip()


def _canonical_value(value, reverse_map: dict[str, str]) -> str:
    rendered = _business_text(value)
    return reverse_map.get(rendered, rendered.upper())


def _normalise_business_values(df: pd.DataFrame) -> None:
    for field in (
        "case_id", "version_prev", "version_curr", "prev_result", "curr_result",
        "severity", "issue_description", "metric_snapshot",
    ):
        if field in df:
            df[field] = df[field].map(_business_text)
    df["prev_result"] = df["prev_result"].str.upper()
    df["curr_result"] = df["curr_result"].str.upper()
    df["severity"] = [_canonical_value(value, _SEVERITY_CANONICAL) for value in df["severity"]]


def _validate_daily_rows(df: pd.DataFrame, require_metric_snapshot: bool) -> None:
    errors: list[str] = []
    seen_case_rows: dict[str, int] = {}
    required_text = (
        "case_id", "version_prev", "version_curr", "issue_description",
    ) + (("metric_snapshot",) if require_metric_snapshot else ())
    for position, (_, row) in enumerate(df.iterrows(), start=2):
        for field in required_text:
            if field in df and is_blank_business_value(row.get(field)):
                errors.append(f"第 {position} 行：“{USER_DAILY_HEADERS[field]}”为空。")

        case_id = _business_text(row.get("case_id"))
        if case_id:
            if case_id in seen_case_rows:
                errors.append(
                    f"第 {position} 行：用例 ID “{case_id}”与前面记录重复。"
                )
            else:
                seen_case_rows[case_id] = position

        for field in ("prev_result", "curr_result"):
            value = _business_text(row.get(field)).upper()
            label = USER_DAILY_HEADERS[field]
            if not value:
                errors.append(f"第 {position} 行：“{label}”为空。")
            elif value not in _VALID_RESULTS:
                errors.append(
                    f"第 {position} 行：“{label}”为“{value}”，允许值为 PASS、HIGH_RISK、FAIL。"
                )

        severity = _business_text(row.get("severity")).upper()
        if not severity:
            errors.append(f"第 {position} 行：“严重度”为空。")
        elif severity not in _VALID_SEVERITIES:
            errors.append(
                f"第 {position} 行：“严重度”为“{row.get('severity')}”，允许值为高、中、低。"
            )

    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"报告中发现 {len(errors)} 个输入问题：\n{details}")


def _read_daily_sheet(path_or_buffer) -> pd.DataFrame:
    with pd.ExcelFile(path_or_buffer) as workbook:
        for name in ("Daily回归明细", "Daily Regression"):
            if name in workbook.sheet_names:
                return workbook.parse(name)
        return workbook.parse(workbook.sheet_names[0])


def parse_metric_snapshot(value) -> dict[str, float | bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    metrics: dict[str, float | bool] = {}
    patterns = {
        "min_ttc_s": r"(?:min_ttc_s\s*=|最小碰撞时间\s*)(-?\d+(?:\.\d+)?)",
        "min_gap_m": r"(?:min_gap_m\s*=|最小间距\s*)(-?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    collision = re.search(r"collision\s*=\s*(true|false)", text, flags=re.IGNORECASE)
    if collision:
        metrics["collision"] = collision.group(1).lower() == "true"
    elif "未发生碰撞" in text:
        metrics["collision"] = False
    elif "发生碰撞" in text or "检测到碰撞" in text:
        metrics["collision"] = True
    return metrics


def canonical_metric_snapshot(value) -> str:
    metrics = parse_metric_snapshot(value)
    parts = []
    if "min_ttc_s" in metrics:
        parts.append(f"min_ttc_s={metrics['min_ttc_s']:.2f}")
    if "min_gap_m" in metrics:
        parts.append(f"min_gap_m={metrics['min_gap_m']:.2f}")
    if "collision" in metrics:
        parts.append(f"collision={str(metrics['collision']).lower()}")
    return "; ".join(parts)


def metric_snapshot_zh(value) -> str:
    metrics = parse_metric_snapshot(value)
    parts = []
    if "min_ttc_s" in metrics:
        parts.append(f"最小碰撞时间 {metrics['min_ttc_s']:.2f} s")
    if "min_gap_m" in metrics:
        parts.append(f"最小间距 {metrics['min_gap_m']:.2f} m")
    if "collision" in metrics:
        parts.append("发生碰撞" if metrics["collision"] else "未发生碰撞")
    return "；".join(parts) if parts else "原始报告未提供可解析的关键指标。"


def reconstruct_local_result(value) -> str | None:
    metrics = parse_metric_snapshot(value)
    if not {"min_gap_m", "collision"}.issubset(metrics):
        return None
    ttc, gap = metrics.get("min_ttc_s"), float(metrics["min_gap_m"])
    if metrics["collision"] or (ttc is not None and ttc < EVALUATION_THRESHOLDS["fail_ttc_s"]) or gap < EVALUATION_THRESHOLDS["fail_gap_m"]:
        return "FAIL"
    if (ttc is not None and ttc < EVALUATION_THRESHOLDS["high_risk_ttc_s"]) or gap < EVALUATION_THRESHOLDS["high_risk_gap_m"]:
        return "HIGH_RISK"
    return "PASS"


def local_evaluation_review(row: dict | pd.Series) -> dict[str, str]:
    input_result = str(row.get("curr_result", "")).upper()
    local_result = reconstruct_local_result(row.get("metric_snapshot"))
    if local_result is None:
        return {"local_result": "", "status": "UNKNOWN", "description": _UNPARSEABLE_EVALUATION_REASON}
    metrics = metric_snapshot_zh(row.get("metric_snapshot"))
    if local_result == input_result:
        return {"local_result": local_result, "status": "CONSISTENT", "description": f"本地规则复核：按当前本地评测规则，根据{metrics}重建为 {local_result}，与输入报告一致。"}
    return {
        "local_result": local_result, "status": "MISMATCH",
        "description": (
            f"输入报告结果为 {input_result}；按当前本地评测规则，根据{metrics}，"
            f"复核结果为 {local_result}。两者判定不一致，可能存在评测口径差异。"
        ),
    }


def deterministic_evaluation_reason(row: dict | pd.Series) -> str:
    """Compatibility helper: explain only what the local thresholds establish."""
    review = local_evaluation_review(row)
    if review["status"] == "UNKNOWN":
        return _UNPARSEABLE_EVALUATION_REASON
    if review["status"] == "MISMATCH":
        return review["description"]

    result = str(row.get("curr_result", "")).upper()
    metrics = parse_metric_snapshot(row.get("metric_snapshot"))
    collision, ttc, gap = metrics.get("collision"), metrics.get("min_ttc_s"), metrics.get("min_gap_m")
    reasons: list[str] = []
    if result == "FAIL":
        if collision is True:
            reasons.append("检测到碰撞，触发 FAIL 判定。")
        if ttc is not None and ttc < EVALUATION_THRESHOLDS["fail_ttc_s"]:
            reasons.append(f"最小碰撞时间 {ttc:.2f} s，低于 FAIL 阈值 {EVALUATION_THRESHOLDS['fail_ttc_s']:.2f} s。")
        if gap is not None and gap < EVALUATION_THRESHOLDS["fail_gap_m"]:
            reasons.append(f"最小间距 {gap:.2f} m，低于 FAIL 阈值 {EVALUATION_THRESHOLDS['fail_gap_m']:.2f} m。")
    elif result == "HIGH_RISK":
        if ttc is not None and EVALUATION_THRESHOLDS["fail_ttc_s"] <= ttc < EVALUATION_THRESHOLDS["high_risk_ttc_s"]:
            reasons.append(f"最小碰撞时间 {ttc:.2f} s，低于 HIGH_RISK 阈值 {EVALUATION_THRESHOLDS['high_risk_ttc_s']:.2f} s。")
        if gap is not None and EVALUATION_THRESHOLDS["fail_gap_m"] <= gap < EVALUATION_THRESHOLDS["high_risk_gap_m"]:
            reasons.append(f"最小间距 {gap:.2f} m，低于 HIGH_RISK 阈值 {EVALUATION_THRESHOLDS['high_risk_gap_m']:.2f} m。")
    elif result == "PASS" and collision is False and ttc is not None and gap is not None and ttc >= EVALUATION_THRESHOLDS["high_risk_ttc_s"] and gap >= EVALUATION_THRESHOLDS["high_risk_gap_m"]:
        return "碰撞、最小间距和最小碰撞时间均未触发当前风险判定条件。"
    return "".join(reasons) if reasons else review["description"]


def _attach_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = df.get("evaluation_reason", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    reviews = [local_evaluation_review(row) for _, row in df.iterrows()]
    deterministic = [deterministic_evaluation_reason(row) for _, row in df.iterrows()]
    df["input_evaluation_reason"] = original
    df["evaluation_reason_source"] = "本地评测规则"
    df["local_review_result"] = [item["local_result"] for item in reviews]
    df["local_review_description"] = [item["description"] for item in reviews]
    df["evaluation_review_status"] = [item["status"] for item in reviews]
    df["evaluation_reason"] = deterministic
    df["key_metrics_zh"] = df["metric_snapshot"].map(metric_snapshot_zh)
    df["metric_snapshot_canonical"] = df["metric_snapshot"].map(canonical_metric_snapshot)
    return df


def load_daily_report(path_or_buffer) -> pd.DataFrame:
    raw = _read_daily_sheet(path_or_buffer)
    had_chinese_headers = any(name in CHINESE_HEADER_MAP for name in raw.columns)
    had_seed_mapping = any(
        CHINESE_HEADER_MAP.get(name, name) == "seed_case_ref" for name in raw.columns
    )
    df = _normalise_headers(raw)
    missing_common = COMMON_REQUIRED - set(df.columns)
    if missing_common:
        labels = "、".join(USER_DAILY_HEADERS.get(name, name) for name in sorted(missing_common))
        raise ValueError(f"报告缺少必要字段：{labels}。")
    is_formal = FORMAL_REQUIRED.issubset(df.columns)
    is_legacy = LEGACY_FIELDS.issubset(df.columns)
    if not is_formal and not is_legacy:
        missing_business = FORMAL_REQUIRED - set(df.columns)
        labels = "、".join(USER_DAILY_HEADERS.get(name, name) for name in sorted(missing_business))
        raise ValueError(f"无法识别该回归报告格式，请使用当前示例模板或受支持的旧版格式。缺少业务字段：{labels}。")
    if is_legacy:
        if "issue_description" not in df:
            df["issue_description"] = df["scenario_name"].map(
                lambda value: f"旧版 Daily 兼容记录：{_business_text(value)}"
            )
        if "metric_snapshot" not in df:
            df["metric_snapshot"] = ""
    _normalise_business_values(df)
    _validate_daily_rows(df, require_metric_snapshot=is_formal)
    df["diff"] = [derive_diff(a, b) for a, b in zip(df.prev_result, df.curr_result)]
    if had_seed_mapping:
        df["seed_case_ref"] = df["seed_case_ref"].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    else:
        df["seed_case_ref"] = df["case_id"].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    if "comment" not in df:
        df["comment"] = ""
    else:
        df["comment"] = df["comment"].fillna("").astype(str)
    if "evaluation_reason" not in df:
        df["evaluation_reason"] = ""
    if "log_summary" not in df:
        df["log_summary"] = ""
    else:
        df["log_summary"] = df["log_summary"].fillna("").astype(str)
    if is_formal:
        if "scenario_type" not in df and "ground_truth_scenario_type" in df:
            df["scenario_type"] = df["ground_truth_scenario_type"]
        if "issue_type" not in df and "ground_truth_issue_type" in df:
            df["issue_type"] = df["ground_truth_issue_type"]
        _attach_review_columns(df)
        df.attrs["daily_format"] = "v0.4_chinese" if had_chinese_headers else "v0.4"
        df.attrs["compatibility_notice"] = "当前 Daily 格式"
        df.attrs["seed_mapping_source"] = "explicit" if had_seed_mapping else "case_id_default"
        return df
    if is_legacy:
        _attach_review_columns(df)
        df.attrs["daily_format"] = "legacy_v0.3.2"
        df.attrs["compatibility_notice"] = "检测到旧版 Daily 格式，已按兼容规则读取；内部参考标签不会用于智能规划。"
        df.attrs["seed_mapping_source"] = "explicit" if had_seed_mapping else "case_id_default"
        return df
    raise AssertionError("Daily format classification reached an unreachable branch")


def seed_case_id_from_ref(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text.rsplit("#", 1)[-1].strip() if text else ""


def _candidate_reason(diff: str, severity: str) -> str:
    return f"当前版本{DIFF_DISPLAY.get(diff, diff)}，且严重度为{SEVERITY_DISPLAY.get(severity, severity)}。"


def candidate_rule_descriptions() -> list[str]:
    """Legacy four-row rendering retained for compatibility tests."""
    return [f"{DIFF_DISPLAY[d]} + {SEVERITY_DISPLAY[s]}严重度" for (d, s), _ in sorted(CANDIDATE_PRIORITY.items(), key=lambda item: item[1])]


def candidate_filter_rule_text() -> str:
    return "当前结果为 FAIL 或 HIGH_RISK，且存在有效种子场景 → 可执行候选。严重度只影响处理建议与默认选择，不决定候选资格。"


def classify_daily_cases(df: pd.DataFrame, available_seed_ids: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_ids = set(available_seed_ids) if available_seed_ids is not None else None
    candidates, ineligible, non_candidates = [], [], []
    for _, source in df.iterrows():
        row = source.to_dict()
        row["diff"] = derive_diff(row.get("prev_result"), row.get("curr_result"))
        row["severity"] = str(row.get("severity", "")).upper()
        row["curr_result"] = str(row.get("curr_result", "")).upper()
        row["seed_case_id"] = seed_case_id_from_ref(row.get("seed_case_ref"))
        seed_available = bool(row["seed_case_id"]) and (seed_ids is None or row["seed_case_id"] in seed_ids)
        abnormal = row["curr_result"] in {"FAIL", "HIGH_RISK"}
        if abnormal and seed_available:
            severity = row["severity"] if row["severity"] in _RECOMMENDATION else "LOW"
            row.update({
                "candidate_eligibility": "CANDIDATE", "candidate_reason_code": "ABNORMAL_WITH_VALID_SEED",
                "candidate_reason": _candidate_reason(row["diff"], severity),
                "recommendation": _RECOMMENDATION[severity],
                # Legacy field retained without changing old v0.4.4 tests.
                "handling_level": ("可选处理" if row["diff"] == "REPEATED_FAIL" and severity == "MEDIUM" else _RECOMMENDATION[severity]),
                "default_selected": severity == "HIGH", "seed_available": True,
                "internal_priority": _SEVERITY_SORT[severity] * 10 + _STATE_SORT[row["curr_result"]],
                "deterministic_priority": _SEVERITY_SORT[severity] * 10 + _STATE_SORT[row["curr_result"]],
                "execution_status": "NOT_SELECTED", "eligibility_status": "可执行",
            })
            candidates.append(row)
        elif abnormal:
            row.update({
                "candidate_eligibility": "INELIGIBLE_NO_SEED", "candidate_reason_code": "NO_VALID_SEED",
                "exclusion_reason": "当前为异常状态，但缺少有效种子场景，暂时无法进入场景探索。",
                "execution_status": "NOT_APPLICABLE", "seed_available": False,
            })
            ineligible.append(row)
        else:
            reason = "当前评测结果为 PASS" if row["diff"] != "RECOVERED" else "当前评测结果已恢复为 PASS"
            row.update({
                "candidate_eligibility": "NOT_CANDIDATE", "candidate_reason_code": "CURRENT_RESULT_NOT_ABNORMAL",
                "exclusion_reason": reason, "execution_status": "NOT_APPLICABLE", "seed_available": seed_available,
            })
            non_candidates.append(row)
    candidate_frame = pd.DataFrame(candidates)
    if not candidate_frame.empty:
        candidate_frame = candidate_frame.sort_values(["default_selected", "internal_priority", "case_id"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    return candidate_frame, pd.DataFrame(ineligible), pd.DataFrame(non_candidates)


def analyze_daily_candidates(df: pd.DataFrame, available_seed_ids: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility API. New product surfaces should call classify_daily_cases."""
    candidates, ineligible, non_candidates = classify_daily_cases(df, available_seed_ids)
    excluded_rows = []
    for _, row in ineligible.iterrows():
        item = row.to_dict()
        item["exclusion_reason"] = "无有效种子场景"
        excluded_rows.append(item)
    for _, row in non_candidates.iterrows():
        item = row.to_dict()
        reasons = [str(item["exclusion_reason"]).replace("当前评测结果为 PASS", "当前结果为 PASS")]
        if item.get("severity") == "LOW":
            reasons.append("严重度为 LOW")
        item["exclusion_reason"] = "；".join(reasons)
        excluded_rows.append(item)
    return candidates, pd.DataFrame(excluded_rows)


def find_closed_loop_triggers(df: pd.DataFrame) -> pd.DataFrame:
    candidates, _, _ = classify_daily_cases(df)
    return candidates


def issue_interpreter_input(row: dict) -> dict:
    """Allowlist prevents ground-truth and evaluation-reason leakage into AI interpretation."""
    issue_description = row.get("issue_description")
    if issue_description is None or (isinstance(issue_description, float) and pd.isna(issue_description)):
        issue_description = ""
    issue_description = str(issue_description).strip()
    if not issue_description:
        raise ValueError("Issue Interpreter requires a non-empty issue_description")
    payload = {"issue_description": issue_description}
    for key in ("metric_snapshot", "log_summary"):
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        rendered = str(value).strip()
        if rendered:
            payload[key] = rendered
    return payload
