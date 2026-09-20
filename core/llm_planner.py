"""Strict structured AI semantics for the v0.4 data loop."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Iterable

from dotenv import load_dotenv
from pydantic import ValidationError

from .models import (
    AIDataFeedback,
    AdaptiveReplan,
    GeneralizationPlan,
    IssueCandidate,
    IssueInterpretation,
    IssuePrioritization,
    IssuePriorityDecision,
    ScenarioFactor,
    ScenarioFactorPlan,
    SeedScenario,
)
from .scenario_registry import (
    actor_metadata,
    build_parameter_space,
    parameter_metadata,
    registered_actors,
    registered_parameters,
)


load_dotenv()


class AIOutputValidationError(ValueError):
    """Structured AI output violated a semantic or whitelist contract."""


AI_TRUTH_BOUNDARY = (
    "Only use the current input evidence and local kinematic observations. Do not infer "
    "a real ADS root cause, controller/planner defect, real-vehicle safety conclusion, "
    "confirmed physical mechanism, or a global rule from local evidence. Product-level "
    "limitations are rendered elsewhere: do not repeat them in every generated field and "
    "do not append generic higher-fidelity-simulation recommendations."
)

FORBIDDEN_AI_CLAIMS = (
    "安全边界", "能力边界", "全局安全阈值", "真实车辆安全能力",
    "已确认控制器缺陷", "已确认规划缺陷", "已确认真实 ads 根因",
    "算法根因", "根因在于", "真实 ads 根因", "规划模块", "控制模块", "决策模块",
    "控制器能力", "纵向控制响应不足",
    "决策阈值", "算法不鲁棒", "控制能力不足", "规划缺陷", "控制缺陷",
    "真实车辆安全问题", "真实物理机制已确认", "planning module defect",
    "control defect", "controller capability", "decision logic is not robust",
    "root cause is", "caused by the algorithm", "algorithm causes",
    "最重要参数", "最重要的参数", "首要主导变量", "显著高于其他参数",
    "dominant factor",
)

FORBIDDEN_FORMULA_REASONING = (
    "ttc 的分母", "ttc的分母", "ttc 的分子", "ttc的分子", "分母项", "分子项",
    "共同构成 ttc", "共同构成ttc", "物理公式", "公式", "ttc =", "ttc=",
)

INTERNAL_DIFF_TOKENS = (
    "NEW_FAIL", "REPEATED_FAIL", "NEW_HIGH_RISK", "REPEATED_HIGH_RISK",
    "ESCALATED_TO_FAIL", "RISK_REDUCED", "RECOVERED", "FIXED", "UNCHANGED",
)

SYSTEM_CAUSAL_CLAIMS = (
    "跟车逻辑", "跟车策略", "纵向控制逻辑", "间距控制逻辑", "控制策略",
    "速度匹配策略", "相对速度匹配策略", "控制响应", "规划响应", "算法响应", "响应及时性",
    "响应滞后", "响应延迟", "响应迟滞", "制动响应滞后", "控制滞后", "跟车响应滞后",
)

INTERACTION_ACTION_TERMS = (
    "建议", "推荐", "需进一步", "需要进一步", "需要扩展", "建议扩展", "建议探索",
    "需排查", "重点排查", "验证方案", "高保真",
)

HYPOTHESIS_ACTION_TERMS = (
    "建议", "推荐", "需要扩展", "需关注", "需排查", "应探索", "重点探索",
    "需探索", "需要探索", "进一步验证", "用于验证", "以验证", "进一步探索",
    "继续探索", "高保真",
)

PRE_EXPLORATION_CLAIMS = (
    "高敏感度", "明显敏感", "显著敏感", "主导因素", "主导影响", "首要敏感因素",
    "关键变量", "高度相关", "直接导致", "影响显著高于", "影响弱于其他参数",
    "影响强于其他参数", "直接决定", "共同决定", "驱动因素", "影响弱于", "影响强于",
    "安全裕度", "敏感区域", "关键运动学条件", "关键风险边界", "风险边界", "参数扫描",
    "需通过", "需要通过", "主导", "数据证明", "数据显示", "数据表明", "观测表明",
)


def _all_narrative_canonical_keys() -> tuple[str, ...]:
    parameters = {
        key for scenario in ("highway_merge", "lead_brake")
        for key in registered_parameters(scenario)
    }
    actors = {
        actor for scenario in ("highway_merge", "lead_brake")
        for actor in registered_actors(scenario)
    }
    return tuple(sorted(parameters | actors, key=len, reverse=True))


def _token_hits(text: str, tokens: Iterable[str]) -> list[str]:
    return sorted({
        token for token in tokens
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text, re.I)
    })


def _semantic_hits(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in terms if term.lower() in lowered})


def _validate_common_narrative(text: str, stage: str) -> None:
    hits = _token_hits(text, INTERNAL_DIFF_TOKENS)
    hits += _token_hits(text, ("HIGH", "MEDIUM", "LOW"))
    hits += _token_hits(text, _all_narrative_canonical_keys())
    hits += _semantic_hits(text, SYSTEM_CAUSAL_CLAIMS)
    if re.search(r"(?<![A-Za-z0-9_])V\d+\.\d+(?:\.\d+)?(?![A-Za-z0-9_])", text, re.I):
        hits.append("version metadata")
    if hits:
        raise AIOutputValidationError(
            f"{stage} contains internal metadata, canonical identifiers, or system attribution: "
            f"{', '.join(sorted(set(hits)))}"
        )


def is_within_ai_truth_boundary(text: str) -> bool:
    narrative = text.lower()
    return not any(term in narrative for term in FORBIDDEN_AI_CLAIMS)


def _validate_truth_boundary(text: str, stage: str) -> None:
    narrative = text.lower()
    violations = sorted({term for term in FORBIDDEN_AI_CLAIMS if term in narrative})
    if violations:
        raise AIOutputValidationError(
            f"{stage} must stay within the current local surrogate truth boundary; "
            f"forbidden content: {', '.join(violations)}"
        )


def _normalise_candidates(candidates: Iterable[dict | IssueCandidate]) -> list[IssueCandidate]:
    return [candidate if isinstance(candidate, IssueCandidate) else IssueCandidate.model_validate(candidate) for candidate in candidates]


def validate_prioritization(prioritization, candidates) -> IssuePrioritization:
    candidate_models = _normalise_candidates(candidates)
    candidate_ids = {candidate.case_id for candidate in candidate_models}
    decision_ids = [decision.case_id for decision in prioritization.decisions]
    unknown = sorted(set(decision_ids) - candidate_ids)
    missing = sorted(candidate_ids - set(decision_ids))
    duplicates = sorted({case_id for case_id in decision_ids if decision_ids.count(case_id) > 1})
    if unknown:
        raise AIOutputValidationError(f"AI prioritization returned unknown case_id values: {unknown}")
    if missing:
        raise AIOutputValidationError(f"AI prioritization omitted candidate case_id values: {missing}")
    if duplicates:
        raise AIOutputValidationError(f"AI prioritization duplicated case_id values: {duplicates}")
    prioritization.decisions.sort(key=lambda item: (-item.priority, item.case_id))
    return prioritization


def validate_generalization_plan(plan: GeneralizationPlan, seed: SeedScenario) -> GeneralizationPlan:
    if plan.scenario_type != seed.scenario_type:
        raise AIOutputValidationError("AI plan scenario_type does not match seed scenario_type")
    registered = set(registered_parameters(seed.scenario_type))
    supplied = set(plan.parameter_ranges)
    if supplied - registered:
        raise AIOutputValidationError(f"AI plan used unregistered parameters for {seed.scenario_type}: {sorted(supplied - registered)}")
    if registered - supplied:
        raise AIOutputValidationError(f"AI plan omitted required parameters for {seed.scenario_type}: {sorted(registered - supplied)}")
    return plan


def validate_issue_input(payload: dict[str, Any]) -> dict[str, Any]:
    prohibited = {"ground_truth_scenario_type", "ground_truth_issue_type", "scenario_type", "issue_type"}
    leaked = sorted(prohibited & set(payload))
    if leaked:
        raise AIOutputValidationError(f"Issue Interpreter input contains prohibited ground-truth fields: {leaked}")
    allowed = {"issue_description", "log_summary", "metric_snapshot"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise AIOutputValidationError(f"Issue Interpreter input contains unsupported fields: {unknown}")
    issue_description = payload.get("issue_description")
    if not isinstance(issue_description, str) or not issue_description.strip():
        raise AIOutputValidationError("Issue Interpreter input requires a non-empty issue_description")
    for key in ("metric_snapshot", "log_summary"):
        if key in payload and (not isinstance(payload[key], str) or not payload[key].strip()):
            raise AIOutputValidationError(f"Issue Interpreter optional field must be omitted when empty: {key}")
    return payload


def validate_issue_interpretation(value: IssueInterpretation) -> IssueInterpretation:
    registered = set(registered_parameters(value.scenario_type))
    unknown = sorted(set(value.candidate_factors) - registered)
    if unknown:
        raise AIOutputValidationError(f"Issue interpretation used unregistered factors: {unknown}")
    if len(value.candidate_factors) != len(set(value.candidate_factors)):
        raise AIOutputValidationError("Issue interpretation duplicated candidate factors")
    canonical_actors = registered_actors(value.scenario_type)
    if len(value.actors) != len(set(value.actors)):
        raise AIOutputValidationError("Issue interpretation duplicated actors")
    if set(value.actors) != set(canonical_actors):
        raise AIOutputValidationError(
            f"Issue interpretation actors must exactly match the canonical actor whitelist for "
            f"{value.scenario_type}: {canonical_actors}"
        )
    summary = value.interaction_summary
    hypothesis = value.risk_hypothesis
    _validate_common_narrative(summary, "Issue interpretation interaction_summary")
    _validate_common_narrative(hypothesis, "Issue interpretation risk_hypothesis")
    summary_actions = _semantic_hits(summary, INTERACTION_ACTION_TERMS)
    if summary_actions:
        raise AIOutputValidationError(
            "Issue interpretation interaction_summary must contain observable facts only; "
            f"forbidden action content: {', '.join(summary_actions)}"
        )
    hypothesis_actions = _semantic_hits(hypothesis, HYPOTHESIS_ACTION_TERMS)
    if hypothesis_actions:
        raise AIOutputValidationError(
            "Issue interpretation risk_hypothesis must state a conditional observable relation, "
            f"not an action: {', '.join(hypothesis_actions)}"
        )
    unsupported_hypothesis = _semantic_hits(
        hypothesis,
        (
            "安全裕度", "safety margin", "敏感度", "敏感性", "主导因素", "主导变量",
            "安全阈值", "风险阈值", "临界值", "阈值下限", "阈值上限",
        ),
    )
    if unsupported_hypothesis:
        raise AIOutputValidationError(
            "Issue interpretation risk_hypothesis cannot claim post-exploration analysis: "
            f"{', '.join(unsupported_hypothesis)}"
        )
    narrative = f"{summary} {hypothesis}".lower()
    _validate_truth_boundary(narrative, "Issue interpretation")
    return value


def validate_factor_plan(value: ScenarioFactorPlan, scenario_type: str) -> ScenarioFactorPlan:
    if value.scenario_type != scenario_type:
        raise AIOutputValidationError("Factor plan scenario_type does not match interpretation")
    registered = set(registered_parameters(scenario_type))
    names = [factor.name for factor in value.factors]
    unknown = sorted(set(names) - registered)
    if unknown:
        raise AIOutputValidationError(f"Factor plan used unregistered factors: {unknown}")
    if len(names) != len(set(names)):
        raise AIOutputValidationError("Factor plan duplicated factors")
    narrative = " ".join([value.hypothesis, *(factor.reason for factor in value.factors)]).lower()
    _validate_common_narrative(narrative, "Factor plan narrative")
    _validate_truth_boundary(narrative, "Factor plan")
    if any(term in narrative for term in FORBIDDEN_FORMULA_REASONING):
        raise AIOutputValidationError(
            "Factor plan must provide qualitative relevance without deriving a formula"
        )
    measured_claims = _semantic_hits(narrative, PRE_EXPLORATION_CLAIMS)
    measured_patterns = (
        r"数据(?:显示|表明).{0,24}(?:显著|主导)",
        r"观测表明.{0,24}主导",
        r"主要受.{0,24}主导",
        r"(?:可能|能够|可以)?.{0,16}主导",
    )
    if measured_claims or any(re.search(pattern, narrative) for pattern in measured_patterns):
        raise AIOutputValidationError(
            "Factor plan priority is semantic exploration priority, not measured sensitivity, "
            f"dominance, correlation, or causal contribution: {', '.join(measured_claims) or 'measured claim'}"
        )
    return value


def validate_replan(
    value: AdaptiveReplan, scenario_type: str, sensitivity: dict,
    observed_state: str | None = None,
) -> AdaptiveReplan:
    registered = set(registered_parameters(scenario_type))
    unknown = sorted(set(value.selected_factors) - registered)
    if unknown:
        raise AIOutputValidationError(f"Adaptive replan used unregistered factors: {unknown}")
    allowed_refs = {item["evidence_id"] for item in sensitivity.get("evidence", [])}
    bad_refs = sorted(set(value.evidence_refs) - allowed_refs)
    if bad_refs:
        raise AIOutputValidationError(f"Adaptive replan invented evidence references: {bad_refs}")
    evidence_owner = {
        item["evidence_id"]: item.get("parameter")
        for item in sensitivity.get("evidence", [])
    }
    reliable = {
        item["parameter"] for item in sensitivity.get("parameters", [])
        if item["direction"] != "no_reliable_direction"
    }
    selected_reliable = set(value.selected_factors) & reliable
    if observed_state == "HIGH_RISK" and value.strategy != "symmetric_expansion":
        raise AIOutputValidationError(
            "ALL_HIGH_RISK requires symmetric_expansion for bidirectional boundary search"
        )
    if observed_state != "HIGH_RISK" and selected_reliable and value.strategy == "symmetric_expansion":
        raise AIOutputValidationError(
            "Adaptive replan strategy conflicts with reliable metric sensitivity; "
            "selected reliable factors require follow_metric_gradient: "
            f"{sorted(selected_reliable)}"
        )
    if value.strategy == "follow_metric_gradient":
        if not selected_reliable or not value.evidence_refs:
            raise AIOutputValidationError(
                "Gradient replan must include a selected reliable factor and cite evidence"
            )
    elif not selected_reliable and value.strategy != "symmetric_expansion":
        raise AIOutputValidationError(
            "Only selected factors without a reliable direction may use symmetric expansion"
        )
    missing_own_evidence = sorted(
        parameter for parameter in selected_reliable
        if not any(evidence_owner.get(ref) == parameter for ref in value.evidence_refs)
    )
    if missing_own_evidence:
        raise AIOutputValidationError(
            "Each selected reliable factor must cite evidence belonging to itself: "
            f"{missing_own_evidence}"
        )
    _validate_truth_boundary(value.reason, "Adaptive replan")
    return value


def validate_feedback(feedback: AIDataFeedback) -> AIDataFeedback:
    combined = " ".join([
        feedback.finding,
        feedback.risk_pattern,
        feedback.boundary_summary,
        feedback.data_recommendation,
        feedback.limitations,
    ]).lower()
    _validate_truth_boundary(combined, "AI feedback")
    if "local surrogate" not in feedback.limitations.lower():
        raise AIOutputValidationError("AI feedback limitations must identify the local surrogate backend")
    return feedback


GLOBAL_BOUNDARY_RULE_TERMS = (
    " or ", " if ", "或", "只要", "小于", "大于", "低于", "高于",
    "超过", "不超过", "以上", "以下", "以内", "都会", "均会", "一定会",
    "<", ">", "≤", "≥",
)


def validate_grounded_boundary_feedback(
    feedback: AIDataFeedback,
    local_boundary_facts: list[dict[str, Any]] | None,
) -> AIDataFeedback:
    """Reject global rules and numeric boundary claims not present in Python facts."""
    feedback = validate_feedback(feedback)
    facts = local_boundary_facts or []
    boundary_text = feedback.boundary_summary
    lowered = f" {boundary_text.lower()} "
    if any(term in lowered for term in GLOBAL_BOUNDARY_RULE_TERMS):
        raise AIOutputValidationError(
            "AI feedback must not merge local anchors into a global if/or threshold rule"
        )
    if not facts:
        if re.search(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", boundary_text):
            raise AIOutputValidationError(
                "AI feedback cannot state numeric boundaries without deterministic facts"
            )
        return feedback

    allowed_numbers = {float(value) for value in range(len(facts) + 1)}
    for fact in facts:
        for value in fact.get("fixed_parameters", {}).values():
            allowed_numbers.add(float(value))
        for interval_name in ("coarse_interval", "refined_interval"):
            interval = fact.get(interval_name) or {}
            for value in interval.values():
                allowed_numbers.add(float(value))
        reduction = fact.get("width_reduction")
        if reduction is not None:
            allowed_numbers.update({float(reduction), float(reduction) * 100.0})
    stated_numbers = {
        float(token)
        for token in re.findall(
            r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", boundary_text
        )
    }
    if any(
        not any(abs(number - allowed) < 1e-8 for allowed in allowed_numbers)
        for number in stated_numbers
    ):
        raise AIOutputValidationError(
            "AI feedback introduced a numeric boundary not present in deterministic facts"
        )

    for fact in facts:
        parameter = fact.get("changed_parameter", "")
        if parameter not in boundary_text:
            continue
        fact_numbers = []
        for interval_name in ("coarse_interval", "refined_interval"):
            fact_numbers.extend((fact.get(interval_name) or {}).values())
        mentions_interval = any(
            re.search(rf"(?<![\d.]){re.escape(format(float(value), 'g'))}(?![\d.])", boundary_text)
            for value in fact_numbers
        )
        if mentions_interval:
            for name, value in fact.get("fixed_parameters", {}).items():
                if name not in boundary_text or not re.search(
                    rf"(?<![\d.]){re.escape(format(float(value), 'g'))}(?![\d.])",
                    boundary_text,
                ):
                    raise AIOutputValidationError(
                        "Every numeric local boundary must include its fixed_parameters context"
                    )
    return feedback


class MockPlanner:
    provider_name = "mock"
    model_name = "deterministic-structured-mock"

    def interpret(self, payload: dict[str, Any]) -> IssueInterpretation:
        payload = validate_issue_input(payload)
        text = " ".join(str(value) for value in payload.values()).lower()
        merge = any(token in text for token in ("merge", "汇入", "并入", "匝道", "后车"))
        if merge:
            value = IssueInterpretation(
                scenario_type="highway_merge",
                actors=["ego_vehicle", "rear_vehicle_on_target_lane"],
                interaction_summary="高速汇入过程中，目标车道后车持续接近自车，纵向间距随之缩小。",
                candidate_factors=["rear_speed_kph", "rear_distance_m", "ego_speed_kph", "merge_time_s"],
                risk_hypothesis="目标车道后车的接近速度、后向间距与汇入时机组合，可能对应更紧迫的局部交互状态。",
            )
        else:
            value = IssueInterpretation(
                scenario_type="lead_brake",
                actors=["ego_vehicle", "lead_vehicle"],
                interaction_summary="前车制动后，自车与前车之间的纵向间距快速缩小。",
                candidate_factors=["ego_speed_kph", "lead_speed_kph", "initial_gap_m", "lead_decel_mps2"],
                risk_hypothesis="前车制动后的相对运动与间距变化，可能共同对应更紧迫的局部交互状态。",
            )
        return validate_issue_interpretation(value)

    def plan_factors(self, interpretation: IssueInterpretation, registry: dict) -> ScenarioFactorPlan:
        factors = [
            ScenarioFactor(
                name=name,
                priority="high" if index < 2 else "medium",
                reason=(
                    f"{parameter_metadata(name, interpretation.scenario_type)['display_name_zh']}"
                    "可能改变场景中的相对运动或间距变化，因此值得作为探索因素。"
                ),
            )
            for index, name in enumerate(interpretation.candidate_factors)
        ]
        return validate_factor_plan(ScenarioFactorPlan(
            scenario_type=interpretation.scenario_type,
            factors=factors,
            hypothesis=interpretation.risk_hypothesis,
        ), interpretation.scenario_type)

    def replan(self, context: dict[str, Any]) -> AdaptiveReplan:
        sensitivity = context["metric_sensitivity"]
        distribution = context.get("result_distribution", {})
        all_high_risk = (
            int(distribution.get("HIGH_RISK", 0)) > 0
            and int(distribution.get("PASS", 0)) == 0
            and int(distribution.get("FAIL", 0)) == 0
        )
        reliable = [item for item in sensitivity.get("parameters", []) if item["direction"] != "no_reliable_direction" and item["support_count"] > 0]
        factor_names = [item["name"] for item in context["factor_plan"]["factors"]]
        if reliable and not all_high_risk:
            selected = [item["parameter"] for item in reliable[:2]]
            refs = []
            for item in reliable[:2]:
                refs.extend(item["evidence_refs"][:2])
            value = AdaptiveReplan(
                selected_factors=selected,
                strategy="follow_metric_gradient",
                reason="依据局部 TTC/gap safety margin 斜率，选择具有稳定方向性证据的参数轴。",
                evidence_refs=refs,
            )
        else:
            value = AdaptiveReplan(
                selected_factors=factor_names[:2],
                strategy="symmetric_expansion",
                reason=(
                    "ALL_HIGH_RISK requires bidirectional boundary search."
                    if all_high_risk else "当前指标没有形成可靠单边梯度，采用确定性对称扩展。"
                ),
                evidence_refs=(
                    [ref for item in reliable[:2] for ref in item["evidence_refs"][:2]]
                    if all_high_risk else []
                ),
            )
        return validate_replan(
            value, context["scenario_type"], sensitivity,
            "HIGH_RISK" if all_high_risk else None,
        )

    def feedback(self, actual_run_data: dict[str, Any]) -> AIDataFeedback:
        coarse = actual_run_data["coarse_summary"]
        fine = actual_run_data["fine_summary"]
        facts = actual_run_data.get("local_transition_boundaries")
        transitions = actual_run_data.get("transition_pairs", [])
        refined = actual_run_data.get("refined_boundary", {})
        cc, fc = coarse["result_counts"], fine["result_counts"]
        local_count = len(facts) if facts is not None else len(refined.get("pairs", []))
        narrowed = (
            sum(item.get("width_reduction") is not None for item in facts)
            if facts is not None
            else sum(bool(pair.get("boundary_width_shrunk")) for pair in refined.get("pairs", []))
        )
        if facts is None:
            boundary_summary = (
                f"系统评估了 {local_count} 个确定性 Anchor，其中 {narrowed} 个得到更窄的观测转换区间"
                f"（{narrowed} produced a narrower interval）。"
            )
        elif local_count:
            boundary_summary = (
                f"当前 deterministic Python facts 包含 {local_count} 条带 fixed_parameters 的局部状态转换记录，"
                f"其中 {narrowed} 条形成了更窄的观测转换区间。"
            )
        else:
            boundary_summary = "当前 deterministic Python facts 未形成可总结的局部状态转换记录。"
        legacy_transition_note = (
            f"，粗粒度检测到 {len(transitions)} 个状态转换对"
            f"（{len(transitions)} detected coarse transition pairs）"
            if facts is None else ""
        )
        feedback = AIDataFeedback(
            finding=f"当前 local surrogate 数据中共评测 {coarse['evaluated']} 个粗粒度和 {fine['evaluated']} 个细化场景（{coarse['evaluated']} coarse and {fine['evaluated']} fine）；粗粒度结果为 PASS {cc['PASS']}、HIGH_RISK {cc['HIGH_RISK']}、FAIL {cc['FAIL']}。",
            risk_pattern=f"参数变化与 TTC/gap 风险趋势存在关联；细化结果为 PASS={fc['PASS']}、HIGH_RISK={fc['HIGH_RISK']}、FAIL={fc['FAIL']}{legacy_transition_note}。",
            boundary_summary=boundary_summary,
            data_recommendation="建议将边界附近样本沉淀为 Scenario Assets，并在后续更高保真仿真后端继续验证。",
            limitations="所有结论仅来自 LocalKinematicSimulator + EvaluationEngine local surrogate backend，不代表真实车辆、真实 ADS 算法或商业仿真平台结果。",
        )
        return (
            validate_grounded_boundary_feedback(feedback, facts)
            if facts is not None else validate_feedback(feedback)
        )

    # v0.3 API compatibility only.
    def prioritize(self, candidates) -> IssuePrioritization:
        models = _normalise_candidates(candidates)
        decisions = [IssuePriorityDecision(case_id=item.case_id, priority=80, should_expand=True, reason="兼容旧接口的确定性候选结果。") for item in models]
        return validate_prioritization(IssuePrioritization(decisions=decisions), models)

    def plan(self, seed: SeedScenario, issue_context=None) -> GeneralizationPlan:
        return validate_generalization_plan(build_parameter_space(seed), seed)


class QwenPlanner:
    provider_name = "qwen"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Qwen provider is unavailable: DASHSCOPE_API_KEY is not set. Set LLM_PROVIDER=mock for deterministic local execution.")
        self.model_name = os.getenv("QWEN_MODEL") or "qwen3.8-flash"
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=30.0, max_retries=0)
        self.schema_validation_trace: list[dict[str, Any]] = []

    def _request(self, output_model, schema_name: str, system_prompt: str, payload: dict[str, Any], response_schema=None, reasoning_effort: str = "none") -> str | None:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            timeout=30.0,
            extra_body={"reasoning_effort": reasoning_effort, "preserve_thinking": False},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": response_schema or output_model.model_json_schema()}},
        )
        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _validate_response(raw_output, output_model, semantic_validator):
        if not raw_output:
            raise AIOutputValidationError("provider returned an empty or malformed structured response")
        return semantic_validator(output_model.model_validate_json(raw_output))

    @staticmethod
    def _validation_error(exc: Exception) -> str:
        text = json.dumps(exc.errors(), ensure_ascii=False, default=str) if isinstance(exc, ValidationError) else f"{type(exc).__name__}: {exc}"
        api_key = os.getenv("DASHSCOPE_API_KEY")
        return text.replace(api_key, "<redacted>") if api_key else text

    @staticmethod
    def _safe_output(raw_output) -> str:
        text = (raw_output or "")[:8000]
        api_key = os.getenv("DASHSCOPE_API_KEY")
        return text.replace(api_key, "<redacted>") if api_key else text

    @staticmethod
    def _repair_schema(output_model, exact_allowed_fields=None) -> dict[str, Any]:
        schema = deepcopy(output_model.model_json_schema())
        allowed_parameters = (exact_allowed_fields or {}).get("parameter_ranges")
        if allowed_parameters:
            ranges_schema = schema["properties"]["parameter_ranges"]
            value_schema = ranges_schema.get("additionalProperties", {"type": "array", "items": {"type": "number"}})
            schema["properties"]["parameter_ranges"] = {"type": "object", "title": ranges_schema.get("title", "Parameter Ranges"), "properties": {name: deepcopy(value_schema) for name in allowed_parameters}, "required": list(allowed_parameters), "additionalProperties": False}
        allowed_factors = (exact_allowed_fields or {}).get("allowed_factors")
        if allowed_factors:
            enum_schema = {"type": "string", "enum": list(allowed_factors)}
            if "candidate_factors" in schema.get("properties", {}):
                schema["properties"]["candidate_factors"]["items"] = deepcopy(enum_schema)
            if "selected_factors" in schema.get("properties", {}):
                schema["properties"]["selected_factors"]["items"] = deepcopy(enum_schema)
            if "ScenarioFactor" in schema.get("$defs", {}):
                schema["$defs"]["ScenarioFactor"]["properties"]["name"] = deepcopy(enum_schema)
        allowed_actors = (exact_allowed_fields or {}).get("allowed_actors")
        if allowed_actors and "actors" in schema.get("properties", {}):
            schema["properties"]["actors"]["items"] = {
                "type": "string", "enum": list(allowed_actors)
            }
        allowed_evidence = (exact_allowed_fields or {}).get("allowed_evidence_refs")
        if allowed_evidence and "evidence_refs" in schema.get("properties", {}):
            schema["properties"]["evidence_refs"]["items"] = {
                "type": "string", "enum": list(allowed_evidence)
            }
        return schema

    def _structured_call(self, output_model, schema_name, system_prompt, payload, semantic_validator=None, exact_allowed_fields=None, reasoning_effort="none"):
        semantic_validator = semantic_validator or (lambda value: value)
        call_trace = {"schema_name": schema_name, "reasoning_effort": reasoning_effort, "schema_validation_attempt_1": {"status": "PENDING", "validation_errors": []}, "schema_repair_triggered": False, "schema_validation_attempt_2": None, "schema_retry_count": 0, "repair_result": None}
        self.schema_validation_trace.append(call_trace)
        try:
            strict_schema = self._repair_schema(output_model, exact_allowed_fields)
            raw_output = self._request(output_model, schema_name, system_prompt, payload, response_schema=strict_schema, reasoning_effort=reasoning_effort)
        except Exception as exc:
            call_trace["schema_validation_attempt_1"] = {"status": "NOT_RUN_NETWORK_ERROR", "validation_errors": []}
            call_trace["network_error"] = self._validation_error(exc)
            raise
        try:
            result = self._validate_response(raw_output, output_model, semantic_validator)
            call_trace["schema_validation_attempt_1"] = {"status": "SUCCESS", "validation_errors": []}
            return result
        except (ValidationError, AIOutputValidationError) as first_error:
            first_error_text = self._validation_error(first_error)
            call_trace["schema_validation_attempt_1"] = {"status": "FAILED", "validation_errors": [first_error_text]}
            call_trace["schema_repair_triggered"] = True
            call_trace["schema_retry_count"] = 1
            call_trace["schema_validation_attempt_2"] = {"status": "PENDING", "validation_errors": []}
        allowed_fields = exact_allowed_fields or {"top_level": list(output_model.model_fields)}
        repair_schema = self._repair_schema(output_model, allowed_fields)
        repair_prompt = "The previous output failed Schema or business semantic validation. Retry exactly once using only explicitly allowed fields. Correct every violation named in the validation error. If forbidden content is listed, omit each literal term entirely, including negated or disclaimer forms; express limitations positively using only current Local surrogate observations and local state-transition wording. Do not rename, fuzzy-match, add, or omit fields. Return only valid JSON. " + f"Exact allowed fields: {json.dumps(allowed_fields, ensure_ascii=False)}. Previous validation error: {first_error_text}"
        repair_payload = {"original_task": {"system_prompt": system_prompt, "input": payload}, "original_output": self._safe_output(raw_output), "validation_error": first_error_text, "exact_allowed_fields": allowed_fields, "json_schema": repair_schema}
        try:
            repaired = self._request(output_model, schema_name, repair_prompt, repair_payload, response_schema=repair_schema, reasoning_effort="none")
        except Exception as exc:
            call_trace["schema_validation_attempt_2"] = {"status": "NOT_RUN_NETWORK_ERROR", "validation_errors": []}
            call_trace["repair_result"] = "FAILED"
            call_trace["network_error"] = self._validation_error(exc)
            raise
        try:
            result = self._validate_response(repaired, output_model, semantic_validator)
            call_trace["schema_validation_attempt_2"] = {"status": "SUCCESS", "validation_errors": []}
            call_trace["repair_result"] = "SUCCESS"
            return result
        except (ValidationError, AIOutputValidationError) as second_error:
            error_text = self._validation_error(second_error)
            call_trace["schema_validation_attempt_2"] = {"status": "FAILED", "validation_errors": [error_text]}
            call_trace["repair_result"] = "FAILED"
            raise AIOutputValidationError(f"Qwen structured output failed during {schema_name} after one schema repair retry: {error_text}") from second_error

    def interpret(self, payload):
        payload = validate_issue_input(payload)
        all_factors = sorted({name for scenario in ("highway_merge", "lead_brake") for name in registered_parameters(scenario)})
        all_actors = sorted({name for scenario in ("highway_merge", "lead_brake") for name in registered_actors(scenario)})
        registry = {
            scenario: {
                "parameters": registered_parameters(scenario),
                "actors": registered_actors(scenario),
                "parameter_display_names": {
                    key: parameter_metadata(key, scenario)["display_name_zh"]
                    for key in registered_parameters(scenario)
                },
                "actor_display_names": {
                    actor: actor_metadata(actor, scenario)["display_name_zh"]
                    for actor in registered_actors(scenario)
                },
            }
            for scenario in ("highway_merge", "lead_brake")
        }
        prompt = (
            "理解回归问题语义并输出中文业务文本。只识别场景、参与者、输入证据中的交互事实、"
            "注册的候选探索因素，以及待探索验证的条件性场景风险假设。scenario_type、"
            "candidate_factors 与 actors 必须逐字来自对应场景注册表；actors 必须完整且只能使用"
            "当前 scenario_type 的 canonical actor whitelist。自然语言字段必须使用注册表提供的"
            "中文显示名，不得出现 canonical actor/parameter key。interaction_summary 只写事实，"
            "不得包含建议、行动方案、diff/severity/version、算法或控制根因；risk_hypothesis 只写"
            "可观察变量或交互特征之间的条件性关系，不得写探索、验证等行动措辞，不得声称"
            "敏感度、主导性、安全裕度或输入未提供的阈值。"
            "不得决定 PASS/FAIL；只可复述输入中明确存在的指标数值，不得创造数值。 "
            + AI_TRUTH_BOUNDARY
        )
        return self._structured_call(IssueInterpretation, "ai_issue_interpretation", prompt, {"issue": payload, "scenario_registry": registry}, semantic_validator=validate_issue_interpretation, exact_allowed_fields={"top_level": list(IssueInterpretation.model_fields), "allowed_factors": all_factors, "allowed_actors": all_actors}, reasoning_effort="none")

    def plan_factors(self, interpretation, registry):
        prompt = (
            "选择语义上值得探索的因素、探索优先级与中文原因。priority 的 high/medium/low 只表示"
            "探索顺序，不表示参数重要性、已测敏感度或因果贡献。此阶段尚未执行参数扫描和 "
            "MetricSensitivity；不得声称数据已证明敏感性、相关性、主导性、因素比较或控制根因。"
            "允许使用‘可能影响’和‘值得优先探索’。结构化 factor.name 必须保持 canonical key，"
            "但 reason 与 hypothesis 必须使用注册表中文显示名且不得出现 canonical key。"
            "不得使用‘直接决定’、‘共同决定’、‘驱动因素’、‘主导’、‘影响弱于/强于’、"
            "‘安全裕度’、‘敏感区域’、‘关键风险边界’或‘参数扫描验证’等越权措辞。"
            "每条 reason 只按‘[中文因素名]可能影响当前场景的相对运动或间距变化，因此值得"
            "（优先）探索’这一语义写作；不要解释 TTC、最小间距或任何确定性机制。hypothesis "
            "只表达‘多个候选因素组合可能对应更紧迫的局部交互状态’。"
            "只输出因素名称，不得输出数值、min/max 或 parameter_ranges。"
            "Do not derive physical equations. Do not explain TTC formula. "
            "Only provide qualitative factor relevance as semantic exploration priority. "
            + AI_TRUTH_BOUNDARY
        )
        return self._structured_call(ScenarioFactorPlan, "ai_scenario_factor_plan", prompt, {"issue_interpretation": interpretation.model_dump(), "scenario_registry": registry}, semantic_validator=lambda value: validate_factor_plan(value, interpretation.scenario_type), exact_allowed_fields={"top_level": list(ScenarioFactorPlan.model_fields), "allowed_factors": registered_parameters(interpretation.scenario_type)}, reasoning_effort="none")

    def replan(self, context):
        evidence_refs = [item["evidence_id"] for item in context["metric_sensitivity"].get("evidence", [])]
        distribution = context.get("result_distribution", {})
        all_high_risk = (
            int(distribution.get("HIGH_RISK", 0)) > 0
            and int(distribution.get("PASS", 0)) == 0
            and int(distribution.get("FAIL", 0)) == 0
        )
        prompt = (
            "仅依据给定 metric sensitivity evidence 选择下一轮优先因素并用中文解释。"
            "不得输出数值点或自行推断方向。除 ALL_HIGH_RISK 外，若任一 selected factor "
            "已有可靠 direction，strategy 必须为 follow_metric_gradient 并引用真实 evidence_refs；"
            "只有 selected factors 均无可靠 direction 时才允许 symmetric_expansion。"
            "不同参数量纲不同，不得根据 raw strength 数值大小判断哪个因素更重要，"
            "不得声称最重要参数、首要主导变量、显著高于其他参数或 dominant factor。"
            + (
                "当前首轮结果为 ALL_HIGH_RISK；strategy 必须为 symmetric_expansion，"
                "reason 必须说明 bidirectional boundary search，以同时寻找 PASS 与 FAIL。"
                if all_high_risk else ""
            )
            + "用户可见文本只能使用‘局部状态转换边界’，不得使用‘安全边界’或‘能力边界’。 "
            + AI_TRUTH_BOUNDARY
        )
        return self._structured_call(AdaptiveReplan, "ai_adaptive_replan", prompt, context, semantic_validator=lambda value: validate_replan(value, context["scenario_type"], context["metric_sensitivity"], "HIGH_RISK" if all_high_risk else None), exact_allowed_fields={"top_level": list(AdaptiveReplan.model_fields), "allowed_factors": registered_parameters(context["scenario_type"]), "allowed_evidence_refs": evidence_refs}, reasoning_effort="low")

    def feedback(self, actual_run_data):
        prompt = (
            "仅总结给定的真实 surrogate 数据，value 使用中文。limitations 必须声明 "
            "LocalKinematicSimulator + EvaluationEngine local surrogate backend。 "
            "local_transition_boundaries 是 deterministic Python 生成的逐 Anchor 结构化事实。"
            "不得把不同 Anchor 合并成 if/or 全局规则，不得外推未采样阈值。"
            "任何具体数值边界都必须逐条复述对应 fixed_parameters；没有事实时不得生成数值边界。 "
            + AI_TRUTH_BOUNDARY
        )
        facts = actual_run_data.get("local_transition_boundaries", [])
        return self._structured_call(AIDataFeedback, "ai_data_feedback", prompt, actual_run_data, semantic_validator=lambda value: validate_grounded_boundary_feedback(value, facts), exact_allowed_fields={"top_level": list(AIDataFeedback.model_fields)}, reasoning_effort="none")

    # v0.3 compatibility calls retained for historical tests and integrations.
    def prioritize(self, candidates):
        models = _normalise_candidates(candidates)
        return self._structured_call(IssuePrioritization, "issue_prioritization", "Rank only supplied candidates.", {"candidates": [item.model_dump() for item in models]}, semantic_validator=lambda value: validate_prioritization(value, models), exact_allowed_fields={"top_level": ["decisions"], "decision_fields": list(IssuePriorityDecision.model_fields), "allowed_case_ids": [item.case_id for item in models]})

    def plan(self, seed, issue_context=None):
        registered = registered_parameters(seed.scenario_type)
        return self._structured_call(GeneralizationPlan, "ai_generalization_plan", "Compatibility endpoint: return all exact registered numeric ranges.", {"seed_scenario": seed.model_dump(), "issue_context": issue_context or {}, "registered_parameter_names": registered}, semantic_validator=lambda value: validate_generalization_plan(value, seed), exact_allowed_fields={"top_level": list(GeneralizationPlan.model_fields), "parameter_ranges": registered})


def get_planner(provider: str | None = None):
    selected = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if selected == "mock":
        return MockPlanner()
    if selected == "qwen":
        return QwenPlanner()
    raise RuntimeError(f"Unsupported LLM_PROVIDER={selected!r}; expected 'mock' or 'qwen'.")
