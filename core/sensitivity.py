"""Deterministic local metric-sensitivity analysis for coarse results."""

from __future__ import annotations

import math
from statistics import median

import pandas as pd

from .evaluator import EVALUATION_THRESHOLDS
from .scenario_registry import registered_parameters


def safety_margin_score(min_ttc_s, min_gap_m, collision: bool) -> float:
    """Continuous margin aligned to EvaluationEngine's FAIL thresholds.

    Zero is the deterministic FAIL boundary for each metric and larger is safer.
    The minimum margin mirrors the evaluator's OR rule. Collision is explicit.
    """
    gap_scale = (
        EVALUATION_THRESHOLDS["high_risk_gap_m"]
        - EVALUATION_THRESHOLDS["fail_gap_m"]
    )
    gap_margin = (float(min_gap_m) - EVALUATION_THRESHOLDS["fail_gap_m"]) / gap_scale
    if min_ttc_s is None or (isinstance(min_ttc_s, float) and math.isnan(min_ttc_s)):
        return gap_margin - (2.0 if bool(collision) else 0.0)
    ttc_scale = (
        EVALUATION_THRESHOLDS["high_risk_ttc_s"]
        - EVALUATION_THRESHOLDS["fail_ttc_s"]
    )
    ttc_margin = (float(min_ttc_s) - EVALUATION_THRESHOLDS["fail_ttc_s"]) / ttc_scale
    base_margin = min(ttc_margin, gap_margin)
    return base_margin - (2.0 if bool(collision) else 0.0)


class MetricSensitivityAnalyzer:
    def __init__(self, minimum_strength: float = 1e-6):
        self.minimum_strength = minimum_strength

    def analyze(
        self,
        cases_df: pd.DataFrame,
        results_df: pd.DataFrame,
        scenario_type: str,
    ) -> dict:
        required = {"generated_case_id", "result", "min_gap_m", "min_ttc_s", "collision"}
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"Missing sensitivity result columns: {sorted(missing)}")
        parameters = [p for p in registered_parameters(scenario_type) if p in cases_df.columns]
        merged = cases_df.merge(
            results_df[list(required)],
            on="generated_case_id",
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            return {"method": "axis_aligned_local_metric_slopes", "parameters": [], "evidence": []}
        merged["safety_margin"] = [
            safety_margin_score(row.min_ttc_s, row.min_gap_m, row.collision)
            for row in merged.itertuples()
        ]
        summaries = []
        all_evidence = []
        for parameter in parameters:
            fixed = [name for name in parameters if name != parameter]
            slopes = []
            evidence = []
            grouped = merged.groupby(fixed, dropna=False, sort=True)
            for _, group in grouped:
                ordered = group.sort_values([parameter, "generated_case_id"], kind="stable")
                records = ordered.to_dict("records")
                for left, right in zip(records, records[1:]):
                    delta = float(right[parameter]) - float(left[parameter])
                    if delta == 0:
                        continue
                    margin_delta = float(right["safety_margin"]) - float(left["safety_margin"])
                    slope = margin_delta / delta
                    evidence_id = f"{parameter}:{left['generated_case_id']}->{right['generated_case_id']}"
                    record = {
                        "evidence_id": evidence_id,
                        "parameter": parameter,
                        "case_a": left["generated_case_id"],
                        "case_b": right["generated_case_id"],
                        "parameter_delta": round(delta, 9),
                        "safety_margin_a": round(float(left["safety_margin"]), 9),
                        "safety_margin_b": round(float(right["safety_margin"]), 9),
                        "safety_margin_delta": round(margin_delta, 9),
                        "local_slope": round(slope, 12),
                    }
                    slopes.append(slope)
                    evidence.append(record)
                    all_evidence.append(record)
            aggregate = median(slopes) if slopes else 0.0
            nonzero = [s for s in slopes if abs(s) >= self.minimum_strength]
            same_sign = (
                bool(nonzero)
                and sum(1 for s in nonzero if s * aggregate > 0) / len(nonzero) >= 0.6
            )
            reliable = abs(aggregate) >= self.minimum_strength and same_sign
            direction = (
                "increase_for_safer"
                if reliable and aggregate > 0
                else "decrease_for_safer"
                if reliable
                else "no_reliable_direction"
            )
            summaries.append({
                "parameter": parameter,
                "direction": direction,
                "strength": round(abs(aggregate), 12),
                "signed_median_slope": round(aggregate, 12),
                "support_count": len(slopes),
                "consistent_support_count": sum(1 for s in nonzero if s * aggregate > 0),
                "evidence_refs": [item["evidence_id"] for item in evidence],
            })
        summaries.sort(key=lambda item: (-item["strength"], item["parameter"]))
        return {
            "method": "axis_aligned_local_metric_slopes",
            "score_definition": "min(normalized TTC FAIL margin, normalized gap FAIL margin); collision subtracts 2",
            "thresholds": EVALUATION_THRESHOLDS,
            "parameters": summaries,
            "evidence": all_evidence,
        }
