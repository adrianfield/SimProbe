"""End-to-end v0.4 orchestration with explicit AI/deterministic boundaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .boundary import assess_refinement, mine_boundary
from .evaluator import EvaluationEngine
from .generalizer import coarse_generalize, fine_generalize
from .llm_planner import (
    MockPlanner,
    QwenPlanner,
    get_planner,
    validate_factor_plan,
    validate_generalization_plan,
    validate_issue_interpretation,
    validate_replan,
)
from .models import GeneratedCase, SeedScenario
from .plan_guardrail import apply_plan_guardrail, expand_plan_for_retry
from .report_enricher import build_enriched_report, build_local_boundary_facts, build_boundary_summary_text
from .report_parser import find_closed_loop_triggers, issue_interpreter_input, load_daily_report
from .run_context import source_task_context
from .run_store import DEFAULT_RUNS_ROOT, RunStore
from .scenario_registry import build_parameter_space, registry_summary
from .scenario_store import load_seed_scenarios
from .sensitivity import MetricSensitivityAnalyzer, safety_margin_score
from .simulator import LocalKinematicSimulator
from .validator import validate_case


PLATFORM_VERSION = "v0.4.5.2-final"
MAX_COARSE_ATTEMPTS = 2


def enforce_directional_replan_policy(
    adaptive_replan: dict[str, Any], sensitivity: dict[str, Any],
    observed_state: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically choose execution direction from measured sensitivity."""
    execution = dict(adaptive_replan)
    reliable = {
        item["parameter"]
        for item in sensitivity.get("parameters", [])
        if item.get("direction") != "no_reliable_direction"
    }
    selected_reliable = sorted(set(execution.get("selected_factors", [])) & reliable)
    execution["strategy"] = (
        "symmetric_expansion"
        if observed_state == "HIGH_RISK"
        else "follow_metric_gradient" if selected_reliable else "symmetric_expansion"
    )
    return execution, selected_reliable


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _case_rows(cases: list[GeneratedCase]) -> list[dict]:
    return [{
        "generated_case_id": case.generated_case_id,
        "seed_case_id": case.seed_case_id,
        "scenario_type": case.scenario_type,
        "round_name": case.round_name,
        **case.parameters,
    } for case in cases]


def _result_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["result"].value_counts() if not frame.empty and "result" in frame else {}
    return {label: int(counts.get(label, 0)) for label in ("PASS", "HIGH_RISK", "FAIL")}


def _llm_schema_trace(planner) -> list[dict[str, Any]]:
    return json.loads(json.dumps(getattr(planner, "schema_validation_trace", []), ensure_ascii=False, default=str))


def _trace_event(trace, step, status, summary, data=None) -> None:
    trace["events"].append({
        "timestamp": _now_iso(), "step": step, "status": status,
        "summary": summary, "data": data or {},
    })


def _single_observed_state(distribution: dict[str, int]) -> Optional[str]:
    observed = [state for state, count in distribution.items() if count > 0]
    return observed[0] if len(observed) == 1 else None


def _run_coarse_attempt(seed, plan, store, simulator, evaluator, attempt_number, limit=128):
    cases, sampling = coarse_generalize(
        seed, parameter_ranges=plan.parameter_ranges, limit=limit, return_metadata=True
    )
    cases_df = pd.DataFrame(_case_rows(cases))
    validation_rows, valid_cases = [], []
    for case in cases:
        validation = validate_case(case)
        validation_rows.append({"generated_case_id": case.generated_case_id, "valid": validation.valid, "reason": validation.reason})
        if validation.valid:
            valid_cases.append(case)
    validation_df = pd.DataFrame(validation_rows)
    result_rows = []
    for case in valid_cases:
        trajectory = simulator.run(case)
        store.write_dataframe(f"simulation/trajectories/{case.generated_case_id}.csv", trajectory)
        result_rows.append(evaluator.evaluate(case, trajectory).model_dump())
    results_df = pd.DataFrame(result_rows)

    # Standard paths always represent the effective/final attempt; archives make
    # both attempts independently auditable.
    ir_cases = [case.to_scenario_ir() for case in cases]
    store.write_json("coarse/cases.json", ir_cases)
    store.write_dataframe("coarse/cases.csv", cases_df)
    store.write_dataframe("coarse/validation_results.csv", validation_df)
    store.write_dataframe("simulation/coarse_results.csv", results_df)
    prefix = f"coarse/attempt_{attempt_number}"
    store.write_json(f"{prefix}/cases.json", ir_cases)
    store.write_dataframe(f"{prefix}/cases.csv", cases_df)
    store.write_dataframe(f"{prefix}/validation_results.csv", validation_df)
    store.write_dataframe(f"{prefix}/results.csv", results_df)
    return {
        "plan": plan, "cases": cases, "cases_df": cases_df,
        "validation_df": validation_df, "valid_cases": valid_cases,
        "rejected": len(cases) - len(valid_cases), "results_df": results_df,
        "result_distribution": _result_counts(results_df), "sampling": sampling,
    }


def _closest_boundary_samples(cases_df, results_df, limit=5):
    if cases_df.empty or results_df.empty:
        return []
    merged = cases_df.merge(results_df, on="generated_case_id", how="inner", validate="one_to_one")
    merged["safety_margin"] = [
        safety_margin_score(row.min_ttc_s, row.min_gap_m, row.collision)
        for row in merged.itertuples()
    ]
    return merged.assign(_abs=merged["safety_margin"].abs()).sort_values(
        ["_abs", "generated_case_id"]
    ).drop(columns="_abs").head(limit).to_dict("records")


def _legacy_custom_numeric_plan(planner, seed, context):
    """Narrow compatibility bridge for v0.3 custom planner subclasses."""
    if planner.__class__ not in {MockPlanner, QwenPlanner} and "plan" in planner.__class__.__dict__:
        return validate_generalization_plan(planner.plan(seed, issue_context=context), seed)
    return None


def run_closed_loop(
    daily_df: pd.DataFrame,
    seed: SeedScenario,
    trigger_row: dict[str, Any],
    runs_root: Path = DEFAULT_RUNS_ROOT,
    planner=None,
) -> dict:
    planner = planner or get_planner()
    candidates = find_closed_loop_triggers(daily_df)
    selected = candidates[candidates["case_id"] == seed.case_id]
    if selected.empty:
        raise RuntimeError(f"{seed.case_id} does not satisfy deterministic candidate rules")
    priority = int(selected.iloc[0]["deterministic_priority"])
    source_version = str(trigger_row.get("version_curr", "UNKNOWN"))
    candidate_reason = trigger_row.get("candidate_reason") or (
        f"{trigger_row.get('diff')} + {trigger_row.get('severity')}"
    )
    selection_reason = f"确定性候选规则命中：{candidate_reason} priority={priority}。"
    store = RunStore.create(seed, source_version, selection_reason, runs_root)
    store.update_manifest("CREATED", platform_version=PLATFORM_VERSION)
    trace = {
        "schema_version": "2.0", "run_id": store.run_id, "started_at": _now_iso(),
        "backend": "LocalKinematicSimulator + EvaluationEngine (local surrogate)",
        "llm_provider": planner.provider_name, "llm_model": planner.model_name,
        "source_task_context": source_task_context(trigger_row, seed.case_id),
        "candidate_filter": {"method": "deterministic_priority_table", "selected_case_id": seed.case_id, "priority": priority, "reason": selection_reason},
        "issue_prioritization": {"source": "deterministic_candidate_filter", "decisions": [{"case_id": seed.case_id, "priority": priority, "should_expand": True, "reason": selection_reason}]},
        "ai_issue_interpretation": None, "ai_factor_plan": None,
        "parameter_space": None, "coarse_attempt_1": None,
        "metric_sensitivity": None, "ai_adaptive_replan": None,
        "effective_replan": None, "coarse_attempt_2": None,
        "ai_feedback": None, "llm_schema_trace": [], "events": [],
    }
    _trace_event(trace, "Candidate Filter", "COMPLETED", selection_reason, trace["candidate_filter"])
    store.write_json("seed/seed_case.json", {
        "schema_version": seed.schema_version, "case_id": seed.case_id,
        "scenario_type": seed.scenario_type, "description": seed.description,
        "actors": seed.actors, "road": seed.road, "parameters": seed.parameters,
        "constraints": seed.constraints, "provenance": seed.provenance,
    })
    store.write_json("run_trace.json", trace)
    store.update_manifest("RUNNING")

    try:
        ai_input = issue_interpreter_input(trigger_row)
        interpretation = validate_issue_interpretation(planner.interpret(ai_input))
        if interpretation.scenario_type != seed.scenario_type:
            raise RuntimeError(f"AI interpreted {interpretation.scenario_type}, but seed {seed.case_id} is {seed.scenario_type}")
        trace["ai_issue_interpretation"] = interpretation.model_dump()
        _trace_event(trace, "AI Issue Interpretation", "COMPLETED", interpretation.interaction_summary, {"input_fields": sorted(ai_input), "output": interpretation.model_dump(), "decision_owner": "AI"})

        registry = registry_summary(seed.scenario_type, seed.parameters)
        factor_plan = validate_factor_plan(planner.plan_factors(interpretation, registry), seed.scenario_type)
        trace["ai_factor_plan"] = factor_plan.model_dump()
        _trace_event(trace, "AI Scenario Factor Plan", "COMPLETED", factor_plan.hypothesis, {"factor_plan": factor_plan.model_dump(), "numeric_ranges_supplied_by_ai": False, "decision_owner": "AI"})

        legacy_plan = _legacy_custom_numeric_plan(planner, seed, trace["candidate_filter"])
        if legacy_plan is not None:
            plan, adjustments = apply_plan_guardrail(legacy_plan, seed)
            coarse_limit = 128
            trace["legacy_numeric_plan_compatibility"] = True
        else:
            plan = build_parameter_space(seed, factor_plan)
            plan, adjustments = apply_plan_guardrail(plan, seed)
            coarse_limit = 128
        trace["parameter_space"] = plan.model_dump()
        trace["ai_generalization_plan"] = plan.model_dump()  # historical reader alias
        trace["ai_original_plan"] = (legacy_plan or plan).model_dump()
        trace["effective_plan"] = plan.model_dump()
        trace["guardrail_adjustments"] = adjustments
        store.write_json("coarse/generalization_plan.json", plan.model_dump())
        _trace_event(trace, "Parameter Space Generation", "COMPLETED", "Scenario Registry、Seed 与 AI factor priority 已生成确定性数值空间。", {"plan": plan.model_dump(), "registry": registry, "decision_owner": "DETERMINISTIC_PYTHON"})

        simulator, evaluator = LocalKinematicSimulator(), EvaluationEngine()
        attempt = _run_coarse_attempt(seed, plan, store, simulator, evaluator, 1, limit=coarse_limit)
        trace["coarse_attempt_1"] = {
            **attempt["sampling"], "effective_plan": plan.model_dump(),
            "generated": len(attempt["cases"]), "valid": len(attempt["valid_cases"]),
            "rejected": attempt["rejected"], "result_distribution": attempt["result_distribution"],
        }
        _trace_event(trace, "Coarse Exploration — Attempt 1", "COMPLETED", "本地 surrogate 仿真与确定性评测完成。", trace["coarse_attempt_1"])

        analyzer = MetricSensitivityAnalyzer()
        sensitivity = analyzer.analyze(attempt["cases_df"], attempt["results_df"], seed.scenario_type)
        trace["metric_sensitivity"] = sensitivity
        _trace_event(trace, "Metric Sensitivity Analysis", "COMPLETED", "已基于 TTC/gap 阈值 margin 与轴对齐邻接样本计算局部敏感度。", {"parameters": sensitivity["parameters"], "decision_owner": "DETERMINISTIC_PYTHON"})

        adaptive = None
        uniform_state = _single_observed_state(attempt["result_distribution"])
        if uniform_state is not None:
            replan_context = {
                "scenario_type": seed.scenario_type,
                "issue_interpretation": interpretation.model_dump(),
                "factor_plan": factor_plan.model_dump(),
                "result_distribution": attempt["result_distribution"],
                "metric_sensitivity": sensitivity,
                "closest_to_boundary_samples": _closest_boundary_samples(attempt["cases_df"], attempt["results_df"]),
                "current_parameter_space": plan.parameter_ranges,
                "scenario_registry": registry,
            }
            adaptive = validate_replan(
                planner.replan(replan_context), seed.scenario_type, sensitivity,
                uniform_state,
            )
            adaptive_trace = adaptive.model_dump()
            if uniform_state == "HIGH_RISK":
                trace["ai_adaptive_replan_raw"] = adaptive_trace
                adaptive_trace = {
                    **adaptive_trace,
                    "strategy": "symmetric_expansion",
                    "reason": "ALL_HIGH_RISK requires bidirectional boundary search.",
                }
            trace["ai_adaptive_replan"] = adaptive_trace
            _trace_event(trace, "AI Adaptive Replan", "COMPLETED", adaptive_trace["reason"], {"replan": adaptive_trace, "numeric_points_supplied_by_ai": False, "decision_owner": "AI_ASSISTED_WITH_DETERMINISTIC_POLICY"})
            execution_replan, selected_reliable = enforce_directional_replan_policy(
                adaptive.model_dump(), sensitivity, uniform_state
            )
            expanded_plan, expansion = expand_plan_for_retry(
                plan, seed, uniform_state, sensitivity, execution_replan
            )
            trace["effective_replan"] = {
                "observed_state": uniform_state,
                "expanded_plan": expanded_plan.model_dump(),
                "adjustments": expansion,
                "maximum_attempts": MAX_COARSE_ATTEMPTS,
                "execution_strategy": execution_replan["strategy"],
                "selected_reliable_factors": selected_reliable,
                "strategy_reason": (
                    "ALL_HIGH_RISK requires bidirectional boundary search"
                    if uniform_state == "HIGH_RISK" else
                    "Measured local direction is applied when reliable; otherwise symmetric expansion"
                ),
            }
            trace["why_replan"] = f"Attempt 1 produced only {uniform_state}; metric evidence drives the single allowed retry."
            _trace_event(trace, "Deterministic Expansion", "COMPLETED", "Python 已将 AI factor selection 与 metric direction 转换为合法数值。", {**trace["effective_replan"], "decision_owner": "DETERMINISTIC_PYTHON"})
            store.write_json("coarse/generalization_plan.json", expanded_plan.model_dump())
            attempt = _run_coarse_attempt(seed, expanded_plan, store, simulator, evaluator, 2)
            trace["coarse_attempt_2"] = {
                **attempt["sampling"], "expanded_plan": expanded_plan.model_dump(),
                "generated": len(attempt["cases"]), "valid": len(attempt["valid_cases"]),
                "rejected": attempt["rejected"], "result_distribution": attempt["result_distribution"],
                "final_if_no_transition": "NO_TRANSITION_DETECTED",
            }
            _trace_event(trace, "Coarse Exploration — Attempt 2", "COMPLETED", "第二轮完成；不会执行第三轮 coarse attempt。", trace["coarse_attempt_2"])

        cases, cases_df = attempt["cases"], attempt["cases_df"]
        valid_cases, results_df = attempt["valid_cases"], attempt["results_df"]
        validation_df, counts = attempt["validation_df"], attempt["result_distribution"]
        valid_ids = {case.generated_case_id for case in valid_cases}
        valid_cases_df = cases_df[cases_df["generated_case_id"].isin(valid_ids)]
        boundary = mine_boundary(results_df, valid_cases_df, seed.parameters, seed.scenario_type)
        store.write_json("boundary/boundary_region.json", boundary)
        _trace_event(trace, "Boundary Detection", "COMPLETED" if boundary["status"] == "DETECTED" else "NO_BOUNDARY", boundary["reason"], {"status": boundary["status"], "transition_pair_count": boundary.get("transition_pair_count", 0), "selected_anchors": boundary.get("selected_transition_pairs", []), "decision_owner": "DETERMINISTIC_PYTHON"})

        fine_cases = fine_generalize(seed, boundary)
        fine_cases_df = pd.DataFrame(_case_rows(fine_cases))
        store.write_json("fine/cases.json", [case.to_scenario_ir() for case in fine_cases])
        store.write_dataframe("fine/cases.csv", fine_cases_df)
        valid_fine = [case for case in fine_cases if validate_case(case).valid]
        fine_rows = []
        for case in valid_fine:
            fine_rows.append(evaluator.evaluate(case, simulator.run(case)).model_dump())
        fine_results_df = pd.DataFrame(fine_rows)
        store.write_dataframe("fine/fine_results.csv", fine_results_df)
        fine_counts = _result_counts(fine_results_df)
        refined = assess_refinement(boundary, fine_cases_df, fine_results_df)
        local_boundary_facts = build_local_boundary_facts(
            seed.case_id,
            boundary,
            refined,
            fine_cases_df,
            fine_results_df,
        )
        trace["local_transition_boundary_facts"] = local_boundary_facts
        _trace_event(trace, "Fine Generalization", "COMPLETED" if fine_cases else "SKIPPED", f"生成 {len(fine_cases)} 个确定性插值样本。", {"generated": len(fine_cases), "result_distribution": fine_counts, "decision_owner": "DETERMINISTIC_PYTHON"})

        # The formal conclusion is a deterministic projection of selected local
        # boundary facts. Historical ai_feedback remains readable, but new Runs
        # never request a generative final summary.
        final_summary = build_boundary_summary_text(local_boundary_facts)
        trace["deterministic_summary"] = {
            "source": "DETERMINISTIC_PYTHON",
            "boundary_summary": final_summary,
        }
        warning_code = None
        warning_message = None
        trace["llm_schema_trace"] = _llm_schema_trace(planner)
        enriched = build_enriched_report(
            daily_df,
            seed.case_id,
            len(results_df),
            len(fine_results_df),
            boundary,
            final_summary,
            interpretation.model_dump(),
            factor_plan.model_dump(),
            trace["ai_adaptive_replan"] if adaptive else None,
            refined,
            local_boundary_facts,
            store.run_id,
            str(trigger_row.get("source_daily_name", "Daily Report")),
        )
        store.write_bytes("report/daily_enriched.xlsx", enriched)
        final_status = "COMPLETED"
        trace["completed_at"], trace["final_status"] = _now_iso(), final_status
        store.write_json("run_trace.json", trace)
        summary = {
            "coarse_generated": len(cases), "coarse_valid": len(valid_cases),
            "coarse_rejected": attempt["rejected"], "coarse_results": counts,
            "candidate_space_size": attempt["sampling"]["candidate_space_size"],
            "sampling_strategy": attempt["sampling"]["sampling_strategy"],
            "transition_pairs": boundary.get("transition_pair_count", 0),
            "fine_generated": len(fine_cases), "fine_valid": len(valid_fine),
            "fine_rejected": len(fine_cases) - len(valid_fine), "fine_results": fine_counts,
        }
        manifest_fields = {
            "completed_at": trace["completed_at"], "summary": summary,
            "boundary_status": boundary["status"],
        }
        if warning_code:
            manifest_fields.update({"warning_code": warning_code, "warning_message": warning_message})
        store.update_manifest(final_status, **manifest_fields)
        return {
            "run_id": store.run_id, "run_dir": str(store.run_dir.resolve()),
            "case_id": seed.case_id, "seed_case_ref": trigger_row.get("seed_case_ref"),
            "scenario_type": seed.scenario_type,
            "llm_provider": planner.provider_name, "llm_model": planner.model_name,
            "llm_schema_trace": trace["llm_schema_trace"], "candidate_filter": trace["candidate_filter"],
            "issue_prioritization": trace["issue_prioritization"],
            "ai_issue_interpretation": interpretation.model_dump(), "ai_factor_plan": factor_plan.model_dump(),
            "ai_generalization_plan": plan.model_dump(), "effective_plan": plan.model_dump(),
            "guardrail_adjustments": adjustments, "coarse_attempt_1": trace["coarse_attempt_1"],
            "metric_sensitivity": sensitivity, "ai_adaptive_replan": trace["ai_adaptive_replan"],
            "effective_replan": trace["effective_replan"], "coarse_attempt_2": trace["coarse_attempt_2"],
            "why_replan": trace.get("why_replan"), "summary": summary, "boundary": boundary,
            "refined_boundary": refined, "local_transition_boundary_facts": local_boundary_facts,
            "status": final_status,
            "warning_code": warning_code, "warning_message": warning_message,
            "deterministic_summary": trace.get("deterministic_summary"),
            "ai_feedback": None,
            "final_summary": final_summary,
        }
    except Exception as exc:
        trace["llm_schema_trace"] = _llm_schema_trace(planner)
        _trace_event(trace, "Pipeline", "FAILED", f"{type(exc).__name__}: {exc}")
        trace["final_status"], trace["completed_at"] = "FAILED", _now_iso()
        store.write_json("run_trace.json", trace)
        store.update_manifest("FAILED", error=f"{type(exc).__name__}: {exc}")
        setattr(exc, "run_id", store.run_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimData Loop v0.4")
    parser.add_argument("--case-id", default="CASE_MERGE_001")
    parser.add_argument("--report", type=Path, default=Path("data/sample_daily.xlsx"))
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    args = parser.parse_args()
    daily = load_daily_report(args.report)
    trigger = find_closed_loop_triggers(daily)
    selected = trigger[trigger["case_id"] == args.case_id]
    if selected.empty:
        raise SystemExit(f"No deterministic closed-loop trigger found for {args.case_id}")
    seeds = load_seed_scenarios()
    if args.case_id not in seeds:
        raise SystemExit(f"No seed scenario found for {args.case_id}")
    print(json.dumps(run_closed_loop(daily, seeds[args.case_id], selected.iloc[0].to_dict(), args.runs_root), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
