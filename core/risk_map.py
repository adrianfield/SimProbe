"""Honest two-dimensional risk-map slicing without hidden aggregation."""

from __future__ import annotations

from itertools import product

import pandas as pd


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["result"].value_counts() if not frame.empty else {}
    return {state: int(counts.get(state, 0)) for state in ("PASS", "HIGH_RISK", "FAIL")}


def nearest_observed_value(values, seed_value: float) -> float:
    candidates = sorted({float(value) for value in values})
    return min(candidates, key=lambda value: (abs(value - float(seed_value)), value))


def build_risk_map_slice(
    cases_df: pd.DataFrame,
    results_df: pd.DataFrame,
    seed_parameters: dict[str, float],
    fixed_values: dict[str, float] | None = None,
    x_parameter: str = "rear_speed_kph",
    y_parameter: str = "rear_distance_m",
    fixed_parameters: tuple[str, ...] | list[str] = ("ego_speed_kph", "merge_time_s"),
) -> dict:
    merged = cases_df.merge(
        results_df[["generated_case_id", "result"]],
        on="generated_case_id",
        how="inner",
        validate="one_to_one",
    )
    fixed_parameters = [parameter for parameter in fixed_parameters if parameter in merged.columns]
    chosen = dict(fixed_values or {})
    for parameter in fixed_parameters:
        if parameter not in chosen:
            chosen[parameter] = nearest_observed_value(
                merged[parameter], seed_parameters[parameter]
            )
    mask = pd.Series(True, index=merged.index)
    for parameter, value in chosen.items():
        mask &= (merged[parameter].astype(float) - float(value)).abs() < 1e-9
    sliced = merged.loc[mask].copy()
    duplicates = sliced.duplicated([x_parameter, y_parameter], keep=False)
    if duplicates.any():
        raise ValueError("Risk-map slice contains multiple real cases for one cell")

    x_values = sorted({float(value) for value in merged[x_parameter]})
    y_values = sorted({float(value) for value in merged[y_parameter]})
    lookup = {
        (float(row[x_parameter]), float(row[y_parameter])): row
        for _, row in sliced.iterrows()
    }
    cells = []
    for x_value, y_value in product(x_values, y_values):
        row = lookup.get((x_value, y_value))
        cells.append({
            x_parameter: x_value,
            y_parameter: y_value,
            "generated_case_id": row["generated_case_id"] if row is not None else None,
            "result": row["result"] if row is not None else "N/A",
        })
    return {
        "cells": pd.DataFrame(cells),
        "fixed_values": chosen,
        "overall_distribution": _distribution(merged),
        "slice_distribution": _distribution(sliced),
        "real_case_count": int(len(sliced)),
    }
