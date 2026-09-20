from __future__ import annotations
import math
from .models import EvalResult, GeneratedCase

EVALUATION_THRESHOLDS = {
    "fail_ttc_s": 1.5,
    "high_risk_ttc_s": 3.0,
    "fail_gap_m": 4.0,
    "high_risk_gap_m": 10.0,
}

class EvaluationEngine:
    def evaluate(self, case: GeneratedCase, trajectory) -> EvalResult:
        active = trajectory[trajectory["overlap_active"] == True].copy()
        if active.empty:
            active = trajectory.copy()

        min_gap = float(active["gap_m"].min())
        finite_ttc = active["ttc_s"].replace([math.inf], float("nan")).dropna()
        min_ttc = float(finite_ttc.min()) if not finite_ttc.empty else None
        collision = min_gap <= 0

        if collision:
            result = "FAIL"
        elif (
            min_ttc is not None and min_ttc < EVALUATION_THRESHOLDS["fail_ttc_s"]
        ) or min_gap < EVALUATION_THRESHOLDS["fail_gap_m"]:
            result = "FAIL"
        elif (
            min_ttc is not None
            and min_ttc < EVALUATION_THRESHOLDS["high_risk_ttc_s"]
        ) or min_gap < EVALUATION_THRESHOLDS["high_risk_gap_m"]:
            result = "HIGH_RISK"
        else:
            result = "PASS"

        return EvalResult(
            generated_case_id=case.generated_case_id,
            seed_case_id=case.seed_case_id,
            scenario_type=case.scenario_type,
            round_name=case.round_name,
            result=result,
            collision=collision,
            min_gap_m=round(min_gap, 3),
            min_ttc_s=round(min_ttc, 3) if min_ttc is not None else None,
            metrics={"trajectory_frames": float(len(trajectory))},
        )
