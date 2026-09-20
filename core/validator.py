from __future__ import annotations

from .models import GeneratedCase, ValidationResult
from .scenario_registry import parameter_limits


LIMITS = parameter_limits()


def validate_case(case: GeneratedCase) -> ValidationResult:
    p = case.parameters
    for key, (lo, hi) in LIMITS.items():
        if key in p and not (lo <= float(p[key]) <= hi):
            return ValidationResult(valid=False, reason=f"OUT_OF_RANGE:{key}")

    if case.scenario_type == "lead_brake":
        required = {"ego_speed_kph", "lead_speed_kph", "initial_gap_m", "lead_decel_mps2"}
        if not required.issubset(p):
            return ValidationResult(valid=False, reason="MISSING_PARAMETER")
        if p["ego_speed_kph"] + 15 < p["lead_speed_kph"]:
            return ValidationResult(valid=False, reason="LOGIC_CONFLICT")

    if case.scenario_type == "highway_merge":
        required = {"ego_speed_kph", "rear_speed_kph", "rear_distance_m", "merge_time_s"}
        if not required.issubset(p):
            return ValidationResult(valid=False, reason="MISSING_PARAMETER")
        if p["rear_speed_kph"] + 10 < p["ego_speed_kph"]:
            return ValidationResult(valid=False, reason="LOW_VALUE_FOR_APPROACH_RISK")

    return ValidationResult(valid=True)
