from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScenarioType = Literal["lead_brake", "highway_merge"]
FactorPriority = Literal["high", "medium", "low"]


class SeedScenario(BaseModel):
    """Seed model with defaults that migrate flat v0.3 seed JSON."""

    case_id: str
    scenario_type: ScenarioType
    description: str = ""
    parameters: Dict[str, float]
    schema_version: str = "1.0"
    actors: Dict[str, dict[str, Any]] = Field(default_factory=dict)
    road: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class GeneratedCase(BaseModel):
    generated_case_id: str
    seed_case_id: str
    scenario_type: ScenarioType
    round_name: Literal["coarse", "fine"]
    parameters: Dict[str, float]
    schema_version: str = "1.0"
    actors: Dict[str, dict[str, Any]] = Field(default_factory=dict)
    road: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    def to_scenario_ir(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.generated_case_id,
            "generated_case_id": self.generated_case_id,
            "seed_case_id": self.seed_case_id,
            "scenario_type": self.scenario_type,
            "round_name": self.round_name,
            "actors": self.actors,
            "road": self.road,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "provenance": self.provenance,
        }


class ValidationResult(BaseModel):
    valid: bool
    reason: str = "VALID"


class EvalResult(BaseModel):
    generated_case_id: str
    seed_case_id: str
    scenario_type: ScenarioType
    round_name: str
    result: Literal["PASS", "HIGH_RISK", "FAIL"]
    collision: bool
    min_gap_m: float
    min_ttc_s: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class IssueInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_type: ScenarioType
    actors: list[str] = Field(min_length=1)
    interaction_summary: str = Field(
        min_length=1,
        description=(
            "中文事实摘要；只描述输入中可观察的场景、参与者、动作、间距、TTC 与碰撞事实；"
            "不得包含建议、根因、内部枚举、版本/严重度或 canonical key。"
        ),
    )
    candidate_factors: list[str] = Field(min_length=1)
    risk_hypothesis: str = Field(
        min_length=1,
        description=(
            "中文、条件性且待探索验证的场景可观察关系；不得包含行动建议、控制/规划根因、"
            "diff/severity、已测敏感度结论或 canonical key。"
        ),
    )


class ScenarioFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    priority: FactorPriority
    reason: str = Field(
        min_length=1,
        description=(
            "用中文说明该因素为什么在场景语义上可能影响相对运动或间距、因而值得探索。"
            "不得使用决定、驱动、主导、关键、敏感、相关、比较、安全裕度、风险边界或参数扫描等措辞。"
        ),
    )


class ScenarioFactorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_type: ScenarioType
    factors: list[ScenarioFactor] = Field(min_length=1)
    hypothesis: str = Field(
        min_length=1,
        description=(
            "仅用中文表达多个候选因素组合可能对应某种局部交互状态。"
            "不得冒充参数扫描、指标敏感度、主导性、重要性、因果贡献或行动方案。"
        ),
    )


class AdaptiveReplan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_factors: list[str] = Field(min_length=1)
    strategy: Literal["follow_metric_gradient", "symmetric_expansion"]
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class AIDataFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding: str = Field(min_length=1)
    risk_pattern: str = Field(min_length=1)
    boundary_summary: str = Field(min_length=1)
    data_recommendation: str = Field(min_length=1)
    limitations: str = Field(min_length=1)


# Compatibility contracts for v0.3 callers. The v0.4 pipeline never asks AI
# to produce GeneralizationPlan numeric values or simple candidate ranking.
class GeneralizationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_type: ScenarioType
    parameter_ranges: Dict[str, list[float]]
    rationale: str

    @field_validator("parameter_ranges")
    @classmethod
    def validate_ranges(cls, ranges: Dict[str, list[float]]):
        if not ranges:
            raise ValueError("parameter_ranges cannot be empty")
        for parameter, values in ranges.items():
            if not values:
                raise ValueError(f"parameter range cannot be empty: {parameter}")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"parameter range contains a non-finite value: {parameter}")
        return ranges


class EffectiveReplanTrace(BaseModel):
    """Explicit persisted schema for deterministic Attempt 2 execution."""

    model_config = ConfigDict(extra="forbid")
    observed_state: Literal["PASS", "HIGH_RISK", "FAIL"]
    expanded_plan: GeneralizationPlan
    adjustments: list[dict[str, Any]] = Field(default_factory=list)
    maximum_attempts: int = Field(ge=1)
    execution_strategy: Optional[
        Literal["follow_metric_gradient", "symmetric_expansion"]
    ] = None
    selected_reliable_factors: list[str] = Field(default_factory=list)
    strategy_reason: Optional[str] = None


class IssueCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    scenario_type: ScenarioType
    diff: Literal["NEW_FAIL", "REPEATED_FAIL"]
    severity: str
    issue_type: str = "unknown"


class IssuePriorityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    priority: int = Field(ge=0, le=100)
    should_expand: bool
    reason: str = Field(min_length=1)


class IssuePrioritization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[IssuePriorityDecision] = Field(min_length=1)
