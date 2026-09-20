"""Deterministic Scenario IR generation and coverage-aware coarse sampling."""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Dict

from .models import GeneratedCase, SeedScenario
from .scenario_registry import SCENARIO_REGISTRY, scenario_ir_fields


MAX_COARSE_CASES = 128
FULL_CARTESIAN = "FULL_CARTESIAN"
DETERMINISTIC_MAXIMIN = "DETERMINISTIC_MAXIMIN"

# Compatibility export used by v0.3 callers. Numeric policy authority now lives
# in Scenario Registry and build_parameter_space().
DEFAULT_COARSE = {
    scenario_type: {
        name: [] for name in sorted(entry["parameters"])
    }
    for scenario_type, entry in SCENARIO_REGISTRY.items()
}


def _canonical_ranges(ranges: Dict[str, list[float]]) -> tuple[list[str], list[list[float]]]:
    keys = sorted(ranges)
    values = [sorted({float(value) for value in ranges[key]}) for key in keys]
    if any(not level for level in values):
        raise ValueError("Every coarse parameter must contain at least one value")
    return keys, values


def _normalise(
    candidate: tuple[float, ...],
    keys: list[str],
    scenario_type: str,
) -> tuple[float, ...]:
    specs = SCENARIO_REGISTRY[scenario_type]["parameters"]
    return tuple(
        (value - specs[key].minimum) / (specs[key].maximum - specs[key].minimum)
        for key, value in zip(keys, candidate)
    )


def _distance_squared(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def select_coarse_parameters(
    seed: SeedScenario,
    parameter_ranges: Dict[str, list[float]],
    limit: int = MAX_COARSE_CASES,
) -> tuple[list[dict[str, float]], dict]:
    """Select reproducible cases from the complete Cartesian candidate space."""
    keys, levels = _canonical_ranges(parameter_ranges)
    candidates = [tuple(values) for values in product(*levels)]
    candidate_space_size = prod(len(values) for values in levels)
    if candidate_space_size <= limit:
        selected = candidates
        strategy = FULL_CARTESIAN
    else:
        normalised = {
            candidate: _normalise(candidate, keys, seed.scenario_type)
            for candidate in candidates
        }
        seed_vector = tuple(float(seed.parameters[key]) for key in keys)
        seed_nearest = min(
            candidates,
            key=lambda item: (_distance_squared(normalised[item], _normalise(seed_vector, keys, seed.scenario_type)), item),
        )
        selected = [seed_nearest]
        selected_set = {seed_nearest}

        # Add real Cartesian corners in canonical order before maximin filling.
        corners = sorted(set(product(*[(values[0], values[-1]) for values in levels])))
        for corner in corners:
            if corner not in selected_set and len(selected) < limit:
                selected.append(corner)
                selected_set.add(corner)

        min_distances = {
            candidate: min(
                _distance_squared(normalised[candidate], normalised[chosen])
                for chosen in selected
            )
            for candidate in candidates
            if candidate not in selected_set
        }
        while len(selected) < limit and min_distances:
            farthest = min(
                min_distances,
                key=lambda candidate: (-min_distances[candidate], candidate),
            )
            selected.append(farthest)
            selected_set.add(farthest)
            chosen_vector = normalised[farthest]
            del min_distances[farthest]
            for candidate in min_distances:
                min_distances[candidate] = min(
                    min_distances[candidate],
                    _distance_squared(normalised[candidate], chosen_vector),
                )
        strategy = DETERMINISTIC_MAXIMIN

    selected_rows = [dict(zip(keys, candidate)) for candidate in selected]
    coverage_axes = {}
    for key in keys:
        selected_values = sorted({row[key] for row in selected_rows})
        candidate_values = sorted({float(value) for value in parameter_ranges[key]})
        coverage_axes[key] = {
            "candidate_level_count": len(candidate_values),
            "selected_level_count": len(selected_values),
            "minimum_selected": selected_values[0],
            "maximum_selected": selected_values[-1],
            "covers_extrema": selected_values[0] == candidate_values[0]
            and selected_values[-1] == candidate_values[-1],
        }
    metadata = {
        "candidate_space_size": candidate_space_size,
        "sampling_strategy": strategy,
        "selected_case_count": len(selected_rows),
        "coverage_summary": {
            "seed_nearest_included": any(
                all(row[key] == min(parameter_ranges[key], key=lambda value: (abs(float(value) - float(seed.parameters[key])), float(value))) for key in keys)
                for row in selected_rows
            ),
            "axes": coverage_axes,
        },
    }
    return selected_rows, metadata


def coarse_generalize(
    seed: SeedScenario,
    limit: int = MAX_COARSE_CASES,
    parameter_ranges: Dict[str, list[float]] | None = None,
    return_metadata: bool = False,
):
    from .scenario_registry import build_parameter_space

    ranges = parameter_ranges or build_parameter_space(seed).parameter_ranges
    selected, metadata = select_coarse_parameters(seed, ranges, limit=limit)
    ir = scenario_ir_fields(seed)
    cases = [
        GeneratedCase(
            generated_case_id=f"{seed.case_id}_C{i:03d}",
            seed_case_id=seed.case_id,
            scenario_type=seed.scenario_type,
            round_name="coarse",
            parameters=params,
            **ir,
        )
        for i, params in enumerate(selected, start=1)
    ]
    return (cases, metadata) if return_metadata else cases


def fine_generalize(
    seed: SeedScenario,
    boundary: dict,
    limit: int = 30,
    samples_per_pair: int = 7,
    max_pairs: int = 3,
) -> list[GeneratedCase]:
    """Interpolate only along deterministic transition axes."""
    if boundary.get("status") and boundary.get("status") != "DETECTED":
        return []
    if not 5 <= samples_per_pair <= 9:
        raise ValueError("samples_per_pair must be between 5 and 9")
    selected_pairs = boundary.get("selected_transition_pairs")
    if selected_pairs is None:
        selected_pairs = boundary.get("transition_pairs", [])[:max_pairs]
    selected_pairs = selected_pairs[:max_pairs]
    if not selected_pairs:
        return []

    cases: list[GeneratedCase] = []
    seen = set()
    ir = scenario_ir_fields(seed, source="boundary_refinement")
    for pair in selected_pairs:
        axis = pair["changed_parameter"]
        lower = float(pair["interval"]["min"])
        upper = float(pair["interval"]["max"])
        for sample_index in range(1, samples_per_pair + 1):
            axis_value = lower + (upper - lower) * sample_index / (samples_per_pair + 1)
            parameters = dict(seed.parameters)
            parameters.update({key: float(value) for key, value in pair["fixed_parameters"].items()})
            parameters[axis] = float(axis_value)
            signature = tuple((key, round(float(value), 9)) for key, value in sorted(parameters.items()))
            if signature in seen:
                continue
            seen.add(signature)
            cases.append(GeneratedCase(
                generated_case_id=f"{seed.case_id}_F{len(cases) + 1:03d}",
                seed_case_id=seed.case_id,
                scenario_type=seed.scenario_type,
                round_name="fine",
                parameters=parameters,
                **ir,
            ))
            if len(cases) >= limit:
                return cases
    return cases
