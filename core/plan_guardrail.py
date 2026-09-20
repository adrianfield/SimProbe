"""Deterministic coverage checks and one-shot coarse-plan expansion."""

from __future__ import annotations

from copy import deepcopy

from .generalizer import DEFAULT_COARSE
from .llm_planner import validate_generalization_plan
from .models import GeneralizationPlan, SeedScenario
from .validator import LIMITS


MIN_EXPLORATION_SPAN = {
    "highway_merge": {
        "ego_speed_kph": 20.0,
        "rear_speed_kph": 20.0,
        "rear_distance_m": 20.0,
        "merge_time_s": 1.5,
    },
    "lead_brake": {
        "ego_speed_kph": 30.0,
        "lead_speed_kph": 30.0,
        "initial_gap_m": 25.0,
        "lead_decel_mps2": 3.0,
    },
}

def _clamp(parameter: str, value: float) -> float:
    lower, upper = LIMITS[parameter]
    return min(float(upper), max(float(lower), float(value)))


def apply_plan_guardrail(
    plan: GeneralizationPlan,
    seed: SeedScenario,
) -> tuple[GeneralizationPlan, list[dict]]:
    """Ensure each registered axis has deterministic exploratory coverage."""
    validate_generalization_plan(plan, seed)
    effective_ranges = deepcopy(plan.parameter_ranges)
    adjustments = []

    for parameter in DEFAULT_COARSE[seed.scenario_type]:
        original = sorted({float(value) for value in plan.parameter_ranges[parameter]})
        seed_value = float(seed.parameters[parameter])
        minimum_span = MIN_EXPLORATION_SPAN[seed.scenario_type][parameter]
        minimum_side_coverage = minimum_span * 0.25
        failures = []
        if len(original) < 3:
            failures.append("fewer_than_3_values")
        if not any(value < seed_value for value in original):
            failures.append("no_lower_seed_coverage")
        elif seed_value - min(original) < minimum_side_coverage:
            failures.append("insufficient_lower_seed_coverage")
        if not any(value > seed_value for value in original):
            failures.append("no_upper_seed_coverage")
        elif max(original) - seed_value < minimum_side_coverage:
            failures.append("insufficient_upper_seed_coverage")
        current_span = max(original) - min(original)
        if current_span < minimum_span:
            failures.append("span_below_minimum")
        near_radius = minimum_span * 0.30
        if all(abs(value - seed_value) <= near_radius for value in original):
            failures.append("values_concentrated_near_seed")

        if not failures:
            effective_ranges[parameter] = original
            continue

        lower = min(
            original[0], _clamp(parameter, seed_value - minimum_span / 2.0)
        )
        upper = max(
            original[-1], _clamp(parameter, seed_value + minimum_span / 2.0)
        )
        effective = sorted({lower, seed_value, upper})
        if len(effective) < 3:
            baseline = [float(value) for value in DEFAULT_COARSE[seed.scenario_type][parameter]]
            effective = sorted({*effective, *baseline})
        effective_ranges[parameter] = effective
        adjustments.append({
            "parameter": parameter,
            "checks_failed": failures,
            "minimum_span": minimum_span,
            "minimum_side_coverage": minimum_side_coverage,
            "original_values": original,
            "effective_values": effective,
        })

    effective_plan = GeneralizationPlan(
        scenario_type=plan.scenario_type,
        parameter_ranges=effective_ranges,
        rationale=plan.rationale,
    )
    return validate_generalization_plan(effective_plan, seed), adjustments


def expand_plan_for_retry(
    current_plan: GeneralizationPlan,
    seed: SeedScenario,
    observed_state: str,
    sensitivity: dict | None = None,
    adaptive_replan: dict | None = None,
) -> tuple[GeneralizationPlan, list[dict]]:
    """Turn AI factor choice plus measured metric direction into numeric levels."""
    if observed_state not in {"PASS", "HIGH_RISK", "FAIL"}:
        raise ValueError(f"Cannot expand from unsupported state: {observed_state}")

    sensitivity = sensitivity or {"parameters": []}
    by_parameter = {
        item["parameter"]: item for item in sensitivity.get("parameters", [])
    }
    selected = set((adaptive_replan or {}).get("selected_factors", current_plan.parameter_ranges))
    strategy = (adaptive_replan or {}).get("strategy", "symmetric_expansion")
    expanded_ranges = {}
    adjustments = []
    for parameter, values in current_plan.parameter_ranges.items():
        ordered = sorted({float(value) for value in values})
        minimum_span = MIN_EXPLORATION_SPAN[seed.scenario_type][parameter]
        span = max(max(ordered) - min(ordered), minimum_span)
        step = span * 0.75
        metric = by_parameter.get(parameter, {})
        safer_direction = metric.get("direction", "no_reliable_direction")
        evidence_backed = (
            parameter in selected
            and strategy == "follow_metric_gradient"
            and safer_direction != "no_reliable_direction"
        )
        if evidence_backed and observed_state in {"PASS", "FAIL"}:
            increase_for_safer = safer_direction == "increase_for_safer"
            increase = increase_for_safer if observed_state == "FAIL" else not increase_for_safer
            target = max(ordered) + step if increase else min(ordered) - step
            expanded = sorted({*ordered, _clamp(parameter, target)})
            effective_direction = "increase" if increase else "decrease"
        elif parameter in selected:
            expanded = sorted({
                *ordered,
                _clamp(parameter, min(ordered) - step),
                _clamp(parameter, max(ordered) + step),
            })
            effective_direction = "symmetric"
        else:
            expanded = ordered
            effective_direction = "unchanged"
        expanded_ranges[parameter] = expanded
        adjustments.append({
            "parameter": parameter,
            "observed_state": observed_state,
            "metric_direction_for_safer": safer_direction,
            "metric_strength": metric.get("strength", 0.0),
            "evidence_refs": metric.get("evidence_refs", []),
            "effective_direction": effective_direction,
            "previous_values": ordered,
            "expanded_values": expanded,
        })

    expanded_plan = GeneralizationPlan(
        scenario_type=current_plan.scenario_type,
        parameter_ranges=expanded_ranges,
        rationale=(
            f"基于 metric sensitivity 与 AI factor selection 对统一 {observed_state} "
            "结果执行一次确定性数值扩展。"
        ),
    )
    return validate_generalization_plan(expanded_plan, seed), adjustments
