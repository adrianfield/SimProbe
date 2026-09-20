"""Deterministic transition-zone mining for coarse scenario results."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .evaluator import EVALUATION_THRESHOLDS


FOCUS_PARAMETERS = {
    "highway_merge": [
        "rear_speed_kph",
        "rear_distance_m",
        "ego_speed_kph",
        "merge_time_s",
    ],
    "lead_brake": [
        "ego_speed_kph",
        "lead_speed_kph",
        "initial_gap_m",
        "lead_decel_mps2",
    ],
}
RESULT_LEVEL = {"PASS": 0, "HIGH_RISK": 1, "FAIL": 2}


class BoundaryResult(dict):
    """Keep the v0.2 pipeline callable while persisting the renamed score only."""

    def __getitem__(self, key):
        if key == "confidence" and key not in self:
            key = "boundary_evidence_score"
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "confidence" and key not in self:
            key = "boundary_evidence_score"
        return super().get(key, default)


def _transition_priority(result_a: str, result_b: str) -> int:
    states = frozenset((result_a, result_b))
    if states == frozenset(("PASS", "FAIL")):
        return 3
    if states == frozenset(("PASS", "HIGH_RISK")):
        return 2
    if states == frozenset(("HIGH_RISK", "FAIL")):
        return 1
    return 0


def _local_interval(values: list[float], value: float) -> tuple[float, float]:
    """Return the grid cell around value, bounded by adjacent midpoints."""
    if len(values) == 1:
        delta = max(abs(value) * 0.05, 0.5)
        return value - delta, value + delta

    index = min(range(len(values)), key=lambda i: abs(values[i] - value))
    lower = value if index == 0 else (values[index - 1] + value) / 2.0
    upper = value if index == len(values) - 1 else (value + values[index + 1]) / 2.0
    return lower, upper


def _normalised_distance(
    midpoint: dict[str, float],
    seed_parameters: dict[str, float],
    full_ranges: dict[str, dict[str, float]],
    parameters: Iterable[str],
) -> float:
    distance = 0.0
    for parameter in parameters:
        span = full_ranges[parameter]["max"] - full_ranges[parameter]["min"]
        scale = span if span > 0 else 1.0
        seed_value = float(seed_parameters.get(parameter, midpoint[parameter]))
        distance += ((midpoint[parameter] - seed_value) / scale) ** 2
    return distance ** 0.5


def _metric_boundary_evidence(row: dict) -> float:
    distances = []
    ttc = row.get("min_ttc_s")
    gap = row.get("min_gap_m")
    if ttc is not None and not pd.isna(ttc):
        scale = EVALUATION_THRESHOLDS["high_risk_ttc_s"] - EVALUATION_THRESHOLDS["fail_ttc_s"]
        distances.extend(
            abs(float(ttc) - EVALUATION_THRESHOLDS[key]) / scale
            for key in ("fail_ttc_s", "high_risk_ttc_s")
        )
    if gap is not None and not pd.isna(gap):
        scale = EVALUATION_THRESHOLDS["high_risk_gap_m"] - EVALUATION_THRESHOLDS["fail_gap_m"]
        distances.extend(
            abs(float(gap) - EVALUATION_THRESHOLDS[key]) / scale
            for key in ("fail_gap_m", "high_risk_gap_m")
        )
    return 0.0 if not distances else 1.0 / (1.0 + min(distances))


def mine_boundary(
    results_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    seed_parameters: Optional[dict[str, float]] = None,
    scenario_type: Optional[str] = None,
) -> dict:
    """Find axis-aligned neighbouring samples whose evaluated state changes.

    Other focus parameters are held fixed while each parameter is sorted.  A
    transition is recorded only when two consecutive observed samples have
    different PASS/HIGH_RISK/FAIL states.  Pairs are ranked by transition value
    and then seed distance.  The returned region is the compact grid cell around
    the top-ranked transition, rather than a box around every risky sample.
    """
    required_result_columns = {"generated_case_id", "result"}
    if not required_result_columns.issubset(results_df.columns):
        missing = sorted(required_result_columns - set(results_df.columns))
        raise ValueError(f"Missing result columns: {missing}")

    metric_columns = [
        name for name in ("generated_case_id", "result", "min_ttc_s", "min_gap_m", "collision")
        if name in results_df.columns
    ]
    merged = cases_df.merge(
        results_df[metric_columns],
        on="generated_case_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "parameters": {},
            "transition_pairs": [],
            "boundary_evidence_score": 0.0,
            "reason": "No evaluated coarse cases could be joined to parameter data.",
        }

    if scenario_type is None and "scenario_type" in merged.columns:
        scenario_values = merged["scenario_type"].dropna().astype(str).unique().tolist()
        scenario_type = scenario_values[0] if len(scenario_values) == 1 else None
    candidate_parameters = FOCUS_PARAMETERS.get(scenario_type or "", [])
    parameters = [
        name
        for name in candidate_parameters
        if name in merged.columns and pd.api.types.is_numeric_dtype(merged[name])
    ]
    if not parameters:
        return {
            "status": "UNSUPPORTED_PARAMETER_SPACE",
            "parameters": {},
            "transition_pairs": [],
            "boundary_evidence_score": 0.0,
            "reason": "No supported numeric focus parameters were present.",
        }

    merged = merged[merged["result"].isin(RESULT_LEVEL)].copy()
    full_ranges = {
        parameter: {
            "min": float(merged[parameter].min()),
            "max": float(merged[parameter].max()),
        }
        for parameter in parameters
    }
    seed_parameters = seed_parameters or {}
    transitions: list[dict] = []

    for changed_parameter in parameters:
        fixed_parameters = [p for p in parameters if p != changed_parameter]
        grouped = merged.groupby(fixed_parameters, dropna=False, sort=True)
        for _, group in grouped:
            ordered = group.sort_values(
                [changed_parameter, "generated_case_id"], kind="stable"
            ).drop_duplicates(subset=[changed_parameter], keep="first")
            records = ordered.to_dict("records")
            for left, right in zip(records, records[1:]):
                if left["result"] == right["result"]:
                    continue
                midpoint = {
                    parameter: (float(left[parameter]) + float(right[parameter])) / 2.0
                    for parameter in parameters
                }
                transition = {
                    "case_a": left["generated_case_id"],
                    "result_a": left["result"],
                    "case_b": right["generated_case_id"],
                    "result_b": right["result"],
                    "changed_parameter": changed_parameter,
                    "interval": {
                        "min": float(left[changed_parameter]),
                        "max": float(right[changed_parameter]),
                    },
                    "fixed_parameters": {
                        parameter: float(left[parameter]) for parameter in fixed_parameters
                    },
                    "severity_delta": abs(
                        RESULT_LEVEL[left["result"]] - RESULT_LEVEL[right["result"]]
                    ),
                    "priority": _transition_priority(
                        left["result"], right["result"]
                    ),
                    "midpoint": midpoint,
                }
                transition["distance_to_seed"] = round(
                    _normalised_distance(
                        midpoint, seed_parameters, full_ranges, parameters
                    ),
                    6,
                )
                interval_width = abs(float(right[changed_parameter]) - float(left[changed_parameter]))
                axis_span = full_ranges[changed_parameter]["max"] - full_ranges[changed_parameter]["min"]
                normalised_interval = interval_width / axis_span if axis_span > 0 else 1.0
                metric_evidence = (
                    _metric_boundary_evidence(left) + _metric_boundary_evidence(right)
                ) / 2.0
                transition["score_components"] = {
                    "state_transition_value": transition["priority"],
                    "transition_type": "↔".join(sorted({left["result"], right["result"]}, key=RESULT_LEVEL.get)),
                    "normalized_parameter_interval": round(normalised_interval, 6),
                    "interval_width": round(interval_width, 6),
                    "seed_proximity": round(1.0 / (1.0 + transition["distance_to_seed"]), 6),
                    "metric_boundary_evidence": round(metric_evidence, 6),
                }
                transition["anchor_score"] = round(
                    transition["priority"] * 100.0
                    + (1.0 - normalised_interval) * 10.0
                    + transition["score_components"]["seed_proximity"] * 5.0
                    + metric_evidence * 5.0,
                    6,
                )
                transitions.append(transition)

    transitions.sort(
        key=lambda item: (
            -item["anchor_score"],
            item["distance_to_seed"],
            item["changed_parameter"],
            item["case_a"],
            item["case_b"],
        )
    )
    if not transitions:
        states = sorted(merged["result"].unique().tolist(), key=RESULT_LEVEL.get)
        return {
            "status": "NO_TRANSITION_DETECTED",
            "parameters": {},
            "transition_pairs": [],
            "boundary_evidence_score": 0.0,
            "reason": (
                "No adjacent coarse samples changed evaluation state; "
                "fine expansion was not generated."
            ),
            "observed_states": states,
            "evaluated_cases": int(len(merged)),
        }

    anchor = transitions[0]
    boundary_parameters = {}
    for parameter in parameters:
        values = sorted(float(value) for value in merged[parameter].dropna().unique())
        if parameter == anchor["changed_parameter"]:
            lower = anchor["interval"]["min"]
            upper = anchor["interval"]["max"]
            source = "anchor_transition_interval"
        else:
            anchor_value = float(anchor["midpoint"][parameter])
            lower, upper = _local_interval(values, anchor_value)
            source = "local_grid_cell"
        boundary_parameters[parameter] = {
            "min": round(float(lower), 6),
            "max": round(float(upper), 6),
            "coarse_min": full_ranges[parameter]["min"],
            "coarse_max": full_ranges[parameter]["max"],
            "seed_value": (
                float(seed_parameters[parameter])
                if parameter in seed_parameters
                else None
            ),
            "source": source,
        }

    observed_states = sorted(merged["result"].unique().tolist(), key=RESULT_LEVEL.get)
    changed_axes = len({item["changed_parameter"] for item in transitions})
    evidence_score = min(
        0.95,
        0.45
        + min(len(transitions), 5) * 0.06
        + min(changed_axes, len(parameters)) * 0.04
        + (0.08 if len(observed_states) >= 3 else 0.04),
    )

    all_public_pairs = []
    for item in transitions:
        all_public_pairs.append({
            key: value for key, value in item.items() if key != "midpoint"
        })
    public_pairs = all_public_pairs[:20]

    # Select the best pair on each available axis first, then fill by score.
    selected_pairs = []
    selected_ids = set()
    for item in all_public_pairs:
        axis = item["changed_parameter"]
        if axis in {pair["changed_parameter"] for pair in selected_pairs}:
            continue
        chosen = dict(item)
        chosen["diversity_reason"] = "该 Anchor 是此参数轴上评分最高的候选，用于保证 Parameter Axis Diversity。"
        selected_pairs.append(chosen)
        selected_ids.add((item["case_a"], item["case_b"], axis))
        if len(selected_pairs) == 3:
            break
    for item in all_public_pairs:
        identity = (item["case_a"], item["case_b"], item["changed_parameter"])
        if len(selected_pairs) == 3:
            break
        if identity in selected_ids:
            continue
        chosen = dict(item)
        chosen["diversity_reason"] = "不同参数轴已优先覆盖；该 Anchor 按综合证据评分补充。"
        selected_pairs.append(chosen)
        selected_ids.add(identity)

    return BoundaryResult({
        "status": "DETECTED",
        "method": "axis_aligned_adjacent_state_change",
        "parameters": boundary_parameters,
        "transition_pairs": public_pairs,
        "transition_pair_count": len(transitions),
        "selected_transition_pairs": selected_pairs,
        "anchor_pair": selected_pairs[0],
        "boundary_evidence_score": round(evidence_score, 3),
        "observed_states": observed_states,
        "evaluated_cases": int(len(merged)),
        "reason": (
            "Transition pairs use state span, normalized interval, seed proximity and "
            "metric-threshold evidence; selected anchors prioritize parameter-axis diversity."
        ),
    })


def assess_refinement(
    boundary: dict,
    fine_cases_df: pd.DataFrame,
    fine_results_df: pd.DataFrame,
) -> dict:
    """Measure observed refinement quality using real fine evaluation states."""
    selected_pairs = boundary.get("selected_transition_pairs", [])
    if not selected_pairs or fine_cases_df.empty or fine_results_df.empty:
        return {
            "status": "INSUFFICIENT_FINE_DATA",
            "fine_states": [],
            "has_multiple_states": False,
            "pairs": [],
        }

    merged = fine_cases_df.merge(
        fine_results_df[["generated_case_id", "result"]],
        on="generated_case_id",
        how="inner",
        validate="one_to_one",
    )
    pair_evidence = []
    for pair_index, pair in enumerate(selected_pairs, start=1):
        axis = pair["changed_parameter"]
        original_min = float(pair["interval"]["min"])
        original_max = float(pair["interval"]["max"])
        original_width = original_max - original_min
        mask = pd.Series(True, index=merged.index)
        for parameter, value in pair["fixed_parameters"].items():
            if parameter not in merged:
                mask &= False
            else:
                mask &= (merged[parameter].astype(float) - float(value)).abs() < 1e-8
        mask &= merged[axis].astype(float).between(
            original_min, original_max, inclusive="neither"
        )
        samples = merged.loc[mask].sort_values(axis).copy()
        anchor = (original_min + original_max) / 2.0
        distances = (samples[axis].astype(float) - anchor).abs()

        points = [
            {"value": original_min, "result": pair["result_a"], "source": "coarse"}
        ]
        points.extend(
            {
                "value": float(row[axis]),
                "result": row["result"],
                "source": "fine",
            }
            for _, row in samples.iterrows()
        )
        points.append(
            {"value": original_max, "result": pair["result_b"], "source": "coarse"}
        )
        points.sort(key=lambda item: item["value"])

        refined_transitions = []
        for left, right in zip(points, points[1:]):
            if left["result"] == right["result"]:
                continue
            refined_transitions.append({
                "min": left["value"],
                "max": right["value"],
                "width": right["value"] - left["value"],
                "result_a": left["result"],
                "result_b": right["result"],
                "priority": _transition_priority(left["result"], right["result"]),
            })
        refined_transitions.sort(
            key=lambda item: (-item["priority"], item["width"], item["min"])
        )
        refined = refined_transitions[0] if refined_transitions else None
        refined_width = refined["width"] if refined else None
        pair_evidence.append({
            "pair_index": pair_index,
            "transition_anchor": {
                "case_a": pair["case_a"],
                "result_a": pair["result_a"],
                "case_b": pair["case_b"],
                "result_b": pair["result_b"],
                "value": anchor,
            },
            "refined_axis": axis,
            "original_interval": {"min": original_min, "max": original_max},
            "original_width": original_width,
            "refined_interval": (
                {"min": refined["min"], "max": refined["max"]}
                if refined
                else None
            ),
            "refined_width": refined_width,
            "width_reduction_ratio": (
                round(1.0 - refined_width / original_width, 6)
                if refined_width is not None and original_width > 0
                else None
            ),
            "boundary_width_shrunk": (
                refined_width < original_width if refined_width is not None else False
            ),
            "refinement_samples": [round(float(value), 6) for value in samples[axis]],
            "fine_states": sorted(
                samples["result"].unique().tolist(), key=RESULT_LEVEL.get
            ),
            "sample_distance_to_anchor": {
                "mean": round(float(distances.mean()), 6) if not distances.empty else None,
                "max": round(float(distances.max()), 6) if not distances.empty else None,
                "unit": axis,
            },
        })

    fine_states = sorted(merged["result"].unique().tolist(), key=RESULT_LEVEL.get)
    return {
        "status": "ASSESSED",
        "fine_states": fine_states,
        "has_multiple_states": len(fine_states) >= 2,
        "all_detected_widths_shrunk": bool(pair_evidence) and all(
            item["boundary_width_shrunk"] for item in pair_evidence
        ),
        "pairs": pair_evidence,
    }
