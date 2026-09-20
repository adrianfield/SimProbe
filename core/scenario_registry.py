"""Authoritative deterministic parameter registry and Scenario IR metadata."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

from .models import GeneralizationPlan, ScenarioFactorPlan, SeedScenario


@dataclass(frozen=True)
class ParameterSpec:
    canonical_key: str
    display_name_zh: str
    unit: str
    minimum: float
    maximum: float
    coarse_half_span: float
    step: float


SCENARIO_REGISTRY: dict[str, dict[str, Any]] = {
    "highway_merge": {
        "actors": {
            "ego_vehicle": {
                "canonical_actor_id": "ego_vehicle",
                "display_name_zh": "自车",
                "type": "vehicle",
                "role": "ego",
            },
            "rear_vehicle_on_target_lane": {
                "canonical_actor_id": "rear_vehicle_on_target_lane",
                "display_name_zh": "目标车道后车",
                "type": "vehicle",
                "role": "target_lane_rear",
            },
        },
        "road_ref": "highway_merge_01",
        "risk_map": {
            "x_parameter": "rear_speed_kph",
            "y_parameter": "rear_distance_m",
            "fixed_parameters": ["ego_speed_kph", "merge_time_s"],
            "title_zh": "高速汇入风险切片",
        },
        "parameters": {
            "ego_speed_kph": ParameterSpec("ego_speed_kph", "自车速度", "km/h", 0.0, 180.0, 20.0, 5.0),
            "rear_speed_kph": ParameterSpec("rear_speed_kph", "目标车道后车速度", "km/h", 0.0, 200.0, 30.0, 5.0),
            "rear_distance_m": ParameterSpec("rear_distance_m", "后向间距", "m", 3.0, 150.0, 20.0, 5.0),
            "merge_time_s": ParameterSpec("merge_time_s", "汇入时长", "s", 1.0, 8.0, 1.0, 0.5),
        },
        "assumptions": [
            "自车与目标车道后车保持恒速纵向运动。",
            "达到 merge_time_s 的 70% 后开始计入车道重叠风险。",
            "不模拟感知、规划、控制器或高保真车辆动力学。",
        ],
    },
    "lead_brake": {
        "actors": {
            "ego_vehicle": {
                "canonical_actor_id": "ego_vehicle",
                "display_name_zh": "自车",
                "type": "vehicle",
                "role": "ego",
            },
            "lead_vehicle": {
                "canonical_actor_id": "lead_vehicle",
                "display_name_zh": "前车",
                "type": "vehicle",
                "role": "lead",
            },
        },
        "road_ref": "highway_follow_01",
        "risk_map": {
            "x_parameter": "initial_gap_m",
            "y_parameter": "lead_decel_mps2",
            "fixed_parameters": ["ego_speed_kph", "lead_speed_kph"],
            "title_zh": "前车制动风险切片",
        },
        "parameters": {
            "ego_speed_kph": ParameterSpec("ego_speed_kph", "自车速度", "km/h", 0.0, 180.0, 30.0, 5.0),
            "lead_speed_kph": ParameterSpec("lead_speed_kph", "前车速度", "km/h", 0.0, 180.0, 25.0, 5.0),
            "initial_gap_m": ParameterSpec("initial_gap_m", "初始纵向间距", "m", 3.0, 150.0, 20.0, 5.0),
            "lead_decel_mps2": ParameterSpec("lead_decel_mps2", "前车制动加速度", "m/s²", -10.0, 0.0, 2.0, 0.5),
        },
        "assumptions": [
            "前车在固定时刻制动，自车按固定反应时间与减速度响应。",
            "不模拟感知、规划、控制器或高保真车辆动力学。",
        ],
    },
}


def registered_parameters(scenario_type: str) -> list[str]:
    return sorted(SCENARIO_REGISTRY[scenario_type]["parameters"])


def registered_actors(scenario_type: str) -> list[str]:
    return list(SCENARIO_REGISTRY[scenario_type]["actors"])


def parameter_metadata(canonical_key: str, scenario_type: str | None = None) -> dict[str, str]:
    scenarios = (
        [SCENARIO_REGISTRY[scenario_type]]
        if scenario_type else SCENARIO_REGISTRY.values()
    )
    for scenario in scenarios:
        spec = scenario["parameters"].get(canonical_key)
        if spec is not None:
            return {
                "canonical_key": spec.canonical_key,
                "display_name_zh": spec.display_name_zh,
                "unit": spec.unit,
            }
    return {
        "canonical_key": canonical_key,
        "display_name_zh": canonical_key,
        "unit": "",
    }


def actor_metadata(canonical_actor_id: str, scenario_type: str | None = None) -> dict[str, str]:
    scenarios = (
        [SCENARIO_REGISTRY[scenario_type]]
        if scenario_type else SCENARIO_REGISTRY.values()
    )
    for scenario in scenarios:
        metadata = scenario["actors"].get(canonical_actor_id)
        if metadata is not None:
            return deepcopy(metadata)
    return {
        "canonical_actor_id": canonical_actor_id,
        "display_name_zh": canonical_actor_id,
        "type": "unknown",
        "role": "unknown",
    }


def parameter_limits() -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for scenario in SCENARIO_REGISTRY.values():
        for name, spec in scenario["parameters"].items():
            limits[name] = (spec.minimum, spec.maximum)
    return limits


def registry_summary(
    scenario_type: str,
    seed_parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    entry = SCENARIO_REGISTRY[scenario_type]
    return {
        "scenario_type": scenario_type,
        "actors": deepcopy(entry["actors"]),
        "road_ref": entry["road_ref"],
        "parameters": {
            name: {
                "canonical_key": spec.canonical_key,
                "display_name_zh": spec.display_name_zh,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
                "coarse_half_span": spec.coarse_half_span,
                "step": spec.step,
                "unit": spec.unit,
                "seed_value": (
                    float(seed_parameters[name])
                    if seed_parameters and name in seed_parameters
                    else None
                ),
            }
            for name, spec in sorted(entry["parameters"].items())
        },
        "assumptions": entry["assumptions"],
    }


def _quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 9)


def build_parameter_space(
    seed: SeedScenario,
    factor_plan: ScenarioFactorPlan | None = None,
) -> GeneralizationPlan:
    """Create numeric levels from registry policy, the seed and factor priority."""
    entry = SCENARIO_REGISTRY[seed.scenario_type]
    priorities = {
        factor.name: factor.priority for factor in (factor_plan.factors if factor_plan else [])
    }
    ranges: dict[str, list[float]] = {}
    notes = []
    for name, spec in sorted(entry["parameters"].items()):
        seed_value = float(seed.parameters[name])
        priority = priorities.get(name, "medium")
        scale = {"high": 1.0, "medium": 0.75, "low": 0.5}[priority]
        half_span = max(spec.step, spec.coarse_half_span * scale)
        lower = max(spec.minimum, _quantize(seed_value - half_span, spec.step))
        upper = min(spec.maximum, _quantize(seed_value + half_span, spec.step))
        values = sorted({float(lower), seed_value, float(upper)})
        ranges[name] = values
        notes.append(f"{name}={priority}:{values}")
    return GeneralizationPlan(
        scenario_type=seed.scenario_type,
        parameter_ranges=ranges,
        rationale="Scenario Registry 根据 Seed 与 AI factor priority 生成：" + "；".join(notes),
    )


def scenario_ir_fields(seed: SeedScenario, source: str = "daily_regression") -> dict[str, Any]:
    entry = SCENARIO_REGISTRY[seed.scenario_type]
    return {
        "schema_version": "1.0",
        "actors": deepcopy(entry["actors"]),
        "road": {"road_ref": entry["road_ref"]},
        "constraints": {
            name: {"min": spec.minimum, "max": spec.maximum, "unit": spec.unit}
            for name, spec in sorted(entry["parameters"].items())
        },
        "provenance": {"source": source, "source_case_id": seed.case_id},
    }
