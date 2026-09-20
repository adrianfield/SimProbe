from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from .presentation import interaction_summary_for_user, localize_user_prose
from .scenario_registry import parameter_metadata
from .report_parser import (
    DIFF_DISPLAY,
    SEVERITY_DISPLAY,
    analyze_daily_candidates,
)


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


def _unit(parameter: str) -> str:
    return parameter_metadata(parameter)["unit"]


def _parameter_label(parameter: str, include_key: bool = True) -> str:
    metadata = parameter_metadata(parameter)
    if include_key and metadata["display_name_zh"] != parameter:
        return f"{metadata['display_name_zh']}（{parameter}）"
    return metadata["display_name_zh"]


def _fixed_conditions_text(fixed_parameters: dict[str, Any]) -> str:
    conditions = []
    for parameter, value in fixed_parameters.items():
        metadata = parameter_metadata(parameter)
        suffix = f" {metadata['unit']}" if metadata["unit"] else ""
        conditions.append(
            f"{metadata['display_name_zh']} = {_number(value)}{suffix}"
        )
    return "；".join(conditions)


def _interval_text(interval: dict | None, parameter: str) -> str:
    if not interval:
        return "当前未形成可靠的精化边界"
    return (
        f"{_number(interval.get('min'))} ～ {_number(interval.get('max'))}"
        f" {_unit(parameter)}".rstrip()
    )


def build_local_boundary_facts(
    seed_case_id: str,
    boundary: dict,
    refined_boundary: dict | None = None,
    fine_cases_df: pd.DataFrame | None = None,
    fine_results_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic facts ordered by increasing changed-parameter value."""
    selected = boundary.get("selected_transition_pairs") or []
    refined_by_index = {
        int(item.get("pair_index", index)): item
        for index, item in enumerate((refined_boundary or {}).get("pairs", []), start=1)
    }
    merged_fine = pd.DataFrame()
    if (
        fine_cases_df is not None
        and fine_results_df is not None
        and not fine_cases_df.empty
        and not fine_results_df.empty
        and "generated_case_id" in fine_cases_df
        and {"generated_case_id", "result"}.issubset(fine_results_df.columns)
    ):
        merged_fine = fine_cases_df.merge(
            fine_results_df[["generated_case_id", "result"]],
            on="generated_case_id",
            how="inner",
            validate="one_to_one",
        )

    facts: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selected, start=1):
        parameter = pair.get("changed_parameter", "")
        fixed = dict(pair.get("fixed_parameters") or {})
        coarse_interval = dict(pair.get("interval") or {})
        refined = refined_by_index.get(pair_index, {})
        points = [
            {"value": float(coarse_interval["min"]), "result": pair.get("result_a")},
            {"value": float(coarse_interval["max"]), "result": pair.get("result_b")},
        ] if {"min", "max"}.issubset(coarse_interval) else []

        if not merged_fine.empty and parameter in merged_fine:
            mask = pd.Series(True, index=merged_fine.index)
            for fixed_parameter, fixed_value in fixed.items():
                if fixed_parameter not in merged_fine:
                    mask &= False
                else:
                    mask &= (
                        merged_fine[fixed_parameter].astype(float) - float(fixed_value)
                    ).abs() < 1e-8
            if coarse_interval:
                mask &= merged_fine[parameter].astype(float).between(
                    float(coarse_interval["min"]),
                    float(coarse_interval["max"]),
                    inclusive="neither",
                )
            points.extend(
                {"value": float(row[parameter]), "result": row["result"]}
                for _, row in merged_fine.loc[mask].iterrows()
            )

        ordered_states: list[str] = []
        for point in sorted(points, key=lambda item: item["value"]):
            state = point.get("result")
            if state and (not ordered_states or ordered_states[-1] != state):
                ordered_states.append(state)
        facts.append({
            "changed_parameter": parameter,
            "fixed_parameters": fixed,
            "coarse_interval": coarse_interval or None,
            "refined_interval": refined.get("refined_interval"),
            "ordered_transition_states": ordered_states,
            "width_reduction": refined.get("width_reduction_ratio"),
        })
    return facts


def build_local_boundary_records(
    seed_case_id: str,
    boundary: dict,
    refined_boundary: dict | None = None,
    fine_cases_df: pd.DataFrame | None = None,
    fine_results_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build one human-readable row per selected transition anchor."""
    records: list[dict[str, Any]] = []
    facts = build_local_boundary_facts(
        seed_case_id, boundary, refined_boundary, fine_cases_df, fine_results_df
    )
    for fact in facts:
        parameter = fact["changed_parameter"]
        fixed = fact["fixed_parameters"]
        refined_interval = fact["refined_interval"]
        ordered_states = fact["ordered_transition_states"]
        fixed_text = "; ".join(
            f"{name}={_number(value)}" for name, value in fixed.items()
        )
        condition_text = _fixed_conditions_text(fixed)
        parameter_name = _parameter_label(parameter, include_key=False)
        interval_for_summary = refined_interval or fact["coarse_interval"]
        summary = (
            f"在 {condition_text} 固定的条件下，{parameter_name}在约 "
            f"{_interval_text(interval_for_summary, parameter)} 区间内观察到 "
            f"{' → '.join(ordered_states)} 的局部状态转换。"
            f"该结果仅适用于上述固定条件，不代表其他场景上下文下的全局规律。"
        )
        reduction = fact["width_reduction"]
        records.append({
            "case_id": seed_case_id,
            "parameter_name": parameter_name,
            "parameter_key": parameter,
            "fixed_conditions": condition_text,
            "changed_parameter": parameter,
            "fixed_parameters": fixed_text,
            "coarse_interval": _interval_text(fact["coarse_interval"], parameter),
            "refined_interval": _interval_text(refined_interval, parameter),
            "ordered_transition_states": " → ".join(ordered_states),
            "width_reduction": f"{reduction:.1%}" if reduction is not None else "—",
            "backend": "LocalKinematicSimulator + EvaluationEngine local surrogate",
            "boundary_summary": summary,
        })
    return records


def build_boundary_summary_text(facts: list[dict[str, Any]]) -> str:
    """Summarize selected deterministic facts without merging local anchors."""
    if not facts:
        return "本次未观察到可靠的局部状态转换区域。"
    names = list(dict.fromkeys(
        _parameter_label(item["changed_parameter"], include_key=False)
        for item in facts
    ))
    return (
        f"本次发现 {len(facts)} 个局部状态转换区域，涉及{'、'.join(names)}。"
        "每条结果仅适用于对应固定条件及已采样区间；具体区间见《局部边界摘要》。"
    )


def build_enriched_report(
    original_df: pd.DataFrame,
    seed_case_id: str,
    coarse_count: int,
    fine_count: int,
    boundary: dict,
    final_summary: str,
    issue_interpretation: dict | None = None,
    factor_plan: dict | None = None,
    adaptive_replan: dict | None = None,
    refined_boundary: dict | None = None,
    local_boundary_facts: list[dict[str, Any]] | None = None,
    run_id: str = "",
    source_daily_name: str = "Daily Report",
) -> bytes:
    out = original_df.copy()
    text_cols = [
        "closed_loop_status", "ai_scenario_interpretation", "candidate_factors",
        "adaptive_replan_summary", "boundary_parameter", "boundary_parameter_key",
        "boundary_context",
        "boundary_summary", "scenario_assets_generated", "boundary_region",
        "data_loop_conclusion",
    ]
    for column in text_cols:
        if column not in out:
            out[column] = pd.Series([None] * len(out), dtype="object")
        else:
            out[column] = out[column].astype("object")
    for column in ("coarse_scenario_count", "fine_scenario_count"):
        if column not in out:
            out[column] = pd.Series([pd.NA] * len(out), dtype="Int64")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    interpretation = issue_interpretation or {}
    factors = factor_plan or {}
    replan = adaptive_replan or {}
    selected = "、".join(
        _parameter_label(parameter, include_key=False)
        for parameter in replan.get("selected_factors", [])
    )
    if local_boundary_facts is None:
        local_boundaries = build_local_boundary_records(
            seed_case_id, boundary, refined_boundary
        )
    else:
        local_boundaries = []
        for fact in local_boundary_facts:
            synthetic_boundary = {"selected_transition_pairs": [{
                "changed_parameter": fact["changed_parameter"],
                "fixed_parameters": fact["fixed_parameters"],
                "interval": fact["coarse_interval"],
                "result_a": (fact.get("ordered_transition_states") or [None])[0],
                "result_b": (fact.get("ordered_transition_states") or [None])[-1],
            }]}
            synthetic_refined = {"pairs": [{
                "pair_index": 1,
                "refined_interval": fact.get("refined_interval"),
                "width_reduction_ratio": fact.get("width_reduction"),
            }]}
            record = build_local_boundary_records(
                seed_case_id, synthetic_boundary, synthetic_refined
            )[0]
            record["ordered_transition_states"] = " → ".join(
                fact.get("ordered_transition_states") or []
            )
            condition_text = _fixed_conditions_text(fact.get("fixed_parameters", {}))
            parameter_name = _parameter_label(fact["changed_parameter"], include_key=False)
            record["parameter_name"] = parameter_name
            record["parameter_key"] = fact["changed_parameter"]
            record["fixed_conditions"] = condition_text
            interval_for_summary = fact.get("refined_interval") or fact.get("coarse_interval")
            record["boundary_summary"] = (
                f"在 {condition_text} 固定的条件下，{parameter_name}在约 "
                f"{_interval_text(interval_for_summary, fact['changed_parameter'])} 区间内观察到 "
                f"{' → '.join(fact.get('ordered_transition_states') or [])} 的局部状态转换。"
                "该结果仅适用于上述固定条件，不代表其他场景上下文下的全局规律。"
            )
            local_boundaries.append(record)
    mask = out["case_id"] == seed_case_id
    out.loc[mask, "closed_loop_status"] = "ENRICHED"
    out.loc[mask, "coarse_scenario_count"] = int(coarse_count)
    out.loc[mask, "fine_scenario_count"] = int(fine_count)
    out.loc[mask, "ai_scenario_interpretation"] = interaction_summary_for_user(
        interpretation.get("interaction_summary", "未形成场景理解摘要。")
    )
    out.loc[mask, "candidate_factors"] = "、".join(
        _parameter_label(item.get("name", "")) for item in factors.get("factors", [])
    )
    out.loc[mask, "adaptive_replan_summary"] = (
        f"首轮结果状态单一，第二轮优先探索 {selected}；"
        f"{localize_user_prose(replan.get('reason', ''))}"
        if replan else "首轮探索已覆盖多个评测状态，无需触发二次自适应重规划。"
    )
    out.loc[mask, "boundary_parameter"] = "\n".join(
        item["parameter_name"] for item in local_boundaries
    )
    out.loc[mask, "boundary_parameter_key"] = "\n".join(
        item["parameter_key"] for item in local_boundaries
    )
    out.loc[mask, "boundary_context"] = "\n".join(
        item["fixed_conditions"] for item in local_boundaries
    )
    out.loc[mask, "boundary_summary"] = (
        "\n".join(item["boundary_summary"] for item in local_boundaries)
        if local_boundaries else "当前未形成可靠的精化边界。"
    )
    out.loc[mask, "scenario_assets_generated"] = (
        f"已生成 {coarse_count} 个粗粒度与 {fine_count} 个定向细化场景资产。"
    )
    out.loc[mask, "boundary_region"] = json.dumps(boundary, ensure_ascii=False)
    out.loc[mask, "data_loop_conclusion"] = localize_user_prose(final_summary)

    summary_columns = [
        "参数名称", "parameter_key", "固定条件", "coarse_interval",
        "refined_interval", "状态变化", "宽度缩减", "boundary_summary", "backend",
    ]
    boundary_frame = pd.DataFrame([
        {
            "参数名称": item["parameter_name"],
            "parameter_key": item["parameter_key"],
            "固定条件": item["fixed_conditions"],
            "coarse_interval": item["coarse_interval"],
            "refined_interval": item["refined_interval"],
            "状态变化": item["ordered_transition_states"],
            "宽度缩减": item["width_reduction"],
            "boundary_summary": item["boundary_summary"],
            "backend": item["backend"],
        }
        for item in local_boundaries
    ], columns=summary_columns)
    technical_facts = local_boundary_facts or build_local_boundary_facts(
        seed_case_id, boundary, refined_boundary
    )
    technical_columns = [
        "changed_parameter", "fixed_parameters", "coarse_interval",
        "refined_interval", "ordered_transition_states", "width_reduction",
    ]
    technical_frame = pd.DataFrame([
        {
            "changed_parameter": fact["changed_parameter"],
            "fixed_parameters": json.dumps(fact["fixed_parameters"], ensure_ascii=False),
            "coarse_interval": json.dumps(fact["coarse_interval"], ensure_ascii=False),
            "refined_interval": json.dumps(fact["refined_interval"], ensure_ascii=False),
            "ordered_transition_states": json.dumps(fact["ordered_transition_states"], ensure_ascii=False),
            "width_reduction": fact["width_reduction"],
        }
        for fact in technical_facts
    ], columns=technical_columns)
    candidates, _ = analyze_daily_candidates(out)
    candidate_rows = []
    for _, item in candidates.iterrows():
        candidate_rows.append({
            "用例 ID": item["case_id"],
            "结果变化": DIFF_DISPLAY.get(item["diff"], item["diff"]),
            "严重度": SEVERITY_DISPLAY.get(item["severity"], item["severity"]),
            "候选原因": item["candidate_reason"],
            "处理等级": item["handling_level"],
            "是否默认选择": "是" if item["default_selected"] else "否",
            "是否实际执行": "是" if item["case_id"] == seed_case_id else "否",
            "执行状态": "已完成" if item["case_id"] == seed_case_id else "本次未执行",
            "运行 ID": run_id if item["case_id"] == seed_case_id else "",
        })
    candidate_frame = pd.DataFrame(candidate_rows)

    user_rows = []
    for _, item in out.iterrows():
        is_seed = item.get("case_id") == seed_case_id
        user_rows.append({
            "用例 ID": item.get("case_id", ""),
            "上一版本": item.get("version_prev", ""),
            "当前版本": item.get("version_curr", ""),
            "上一版本结果": item.get("prev_result", ""),
            "当前评测结果": item.get("curr_result", ""),
            "结果变化": DIFF_DISPLAY.get(str(item.get("diff", "")), item.get("diff", "")),
            "严重度": SEVERITY_DISPLAY.get(str(item.get("severity", "")), item.get("severity", "")),
            "问题描述": localize_user_prose(item.get("issue_description", "")),
            "评测说明": localize_user_prose(item.get("evaluation_reason", "")),
            "关键指标": item.get("key_metrics_zh", item.get("metric_snapshot", "")),
            "分析状态": (
                "已完成" if is_seed and local_boundaries
                else "未发现局部边界" if is_seed else "本次未执行"
            ),
        })
    user_daily = pd.DataFrame(user_rows)

    local_user = boundary_frame.rename(columns={
        "参数名称": "边界参数",
        "coarse_interval": "粗粒度转换区间", "refined_interval": "精化区间",
        "boundary_summary": "局部边界说明", "backend": "仿真环境",
    }).copy()
    local_user = local_user[[
        "边界参数", "固定条件", "粗粒度转换区间", "精化区间",
        "状态变化", "宽度缩减", "局部边界说明", "仿真环境",
    ]]
    local_user["仿真环境"] = "本地运动学仿真"
    local_user.insert(0, "用例 ID", seed_case_id)
    local_user["运行 ID"] = run_id
    local_user["使用说明"] = "仅适用于对应固定条件下的局部状态转换，不代表其他场景上下文下的全局规律。"

    technical_frame.insert(0, "case_id", seed_case_id)
    technical_frame["backend"] = "LocalKinematicSimulator + EvaluationEngine (local surrogate)"
    technical_frame["run_id"] = run_id
    technical_detail = out.drop(columns=[
        column for column in out.columns
        if column.startswith("ground_truth_") or column == "boundary_region"
    ]).copy()
    technical_detail["run_id"] = run_id

    overview_meta = pd.DataFrame({
        "项目": ["来源报告", "总用例数", "候选任务数", "本次选择执行数", "成功完成数", "执行失败数", "仿真环境"],
        "内容": [source_daily_name, len(out), len(candidates), 1, 1, 0,
                 "本地运动学仿真"],
    })
    selected_candidate = candidates[candidates["case_id"] == seed_case_id]
    selected_reason = selected_candidate.iloc[0]["candidate_reason"] if not selected_candidate.empty else ""
    overview = pd.DataFrame([{
        "用例 ID": seed_case_id,
        "当前评测结果": out.loc[mask, "curr_result"].iloc[0] if mask.any() else "",
        "评测说明": out.loc[mask, "evaluation_reason"].iloc[0] if mask.any() and "evaluation_reason" in out else "",
        "候选原因": selected_reason,
        "分析状态": "已完成" if local_boundaries else "未发现局部边界",
        "场景理解": interaction_summary_for_user(interpretation.get("interaction_summary", "")),
        "探索结果": f"粗粒度探索 {coarse_count} 个场景；定向细化 {fine_count} 个场景。",
        "局部边界结果": f"发现 {len(local_boundaries)} 个局部状态转换边界。" if local_boundaries else "未发现局部状态转换边界。",
        "局部边界结论": localize_user_prose(final_summary),
        "运行 ID": run_id,
    }])

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        overview_meta.to_excel(writer, sheet_name="探索分析总览", index=False)
        overview.to_excel(writer, sheet_name="探索分析总览", index=False, startrow=9)
        user_daily.to_excel(writer, sheet_name="Daily回归明细", index=False)
        candidate_frame.to_excel(writer, sheet_name="候选任务清单", index=False)
        local_user.to_excel(writer, sheet_name="局部边界摘要", index=False)
        technical_detail.to_excel(writer, sheet_name="Technical Detail", index=False)
        technical_frame.to_excel(writer, sheet_name="Local Boundary Technical", index=False)
        writer.sheets["Technical Detail"].hide()
        writer.sheets["Local Boundary Technical"].hide()
        user_sheets = {"探索分析总览": (0, 9), "Daily回归明细": (0,),
                       "候选任务清单": (0,), "局部边界摘要": (0,)}
        user_header = writer.book.add_format({
            "bold": True, "font_color": "#24313C", "bg_color": "#E7EDF2",
            "border": 1, "border_color": "#C7D0D9",
            "align": "center", "valign": "vcenter", "text_wrap": True,
        })
        technical_header = writer.book.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#18212B",
            "align": "center", "valign": "vcenter",
        })
        frames = {"探索分析总览": (overview_meta, overview),
                  "Daily回归明细": (user_daily,), "候选任务清单": (candidate_frame,),
                  "局部边界摘要": (local_user,), "Technical Detail": (technical_detail,),
                  "Local Boundary Technical": (technical_frame,)}
        for sheet_name, sheet_frames in frames.items():
            worksheet = writer.sheets[sheet_name]
            rows = user_sheets.get(sheet_name, (0,))
            style = user_header if sheet_name in user_sheets else technical_header
            for row_index, frame in zip(rows, sheet_frames):
                for column_index, name in enumerate(frame.columns):
                    worksheet.write(row_index, column_index, name, style)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, 0, 20)
            worksheet.set_column(1, 20, 24)
    return buffer.getvalue()
