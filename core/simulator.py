from __future__ import annotations
import math
import pandas as pd
from .models import GeneratedCase

VEHICLE_LENGTH_M = 4.6

class LocalKinematicSimulator:
    def __init__(self, dt: float = 0.1, horizon_s: float = 8.0):
        self.dt = dt
        self.horizon_s = horizon_s

    def run(self, case: GeneratedCase) -> pd.DataFrame:
        if case.scenario_type == "lead_brake":
            return self._lead_brake(case)
        if case.scenario_type == "highway_merge":
            return self._highway_merge(case)
        raise ValueError(case.scenario_type)

    def _lead_brake(self, case: GeneratedCase) -> pd.DataFrame:
        p = case.parameters
        ego_v = p["ego_speed_kph"] / 3.6
        lead_v = p["lead_speed_kph"] / 3.6
        lead_x = p["initial_gap_m"] + VEHICLE_LENGTH_M
        ego_x = 0.0
        lead_decel = p["lead_decel_mps2"]
        ego_reaction_s = 1.0
        ego_decel = -6.0
        brake_start = 0.5
        rows = []

        n = int(self.horizon_s / self.dt) + 1
        for i in range(n):
            t = i * self.dt
            gap = lead_x - ego_x - VEHICLE_LENGTH_M
            rel = ego_v - lead_v
            ttc = gap / rel if rel > 1e-6 and gap > 0 else math.inf
            rows.append({
                "time_s": t,
                "ego_x_m": ego_x,
                "ego_speed_mps": ego_v,
                "other_x_m": lead_x,
                "other_speed_mps": lead_v,
                "gap_m": gap,
                "ttc_s": ttc,
                "overlap_active": True,
            })
            if gap <= 0:
                break

            lead_a = lead_decel if t >= brake_start and lead_v > 0 else 0.0
            ego_a = ego_decel if t >= brake_start + ego_reaction_s and ego_v > 0 else 0.0

            ego_x += ego_v * self.dt + 0.5 * ego_a * self.dt**2
            lead_x += lead_v * self.dt + 0.5 * lead_a * self.dt**2
            ego_v = max(0.0, ego_v + ego_a * self.dt)
            lead_v = max(0.0, lead_v + lead_a * self.dt)

        return pd.DataFrame(rows)

    def _highway_merge(self, case: GeneratedCase) -> pd.DataFrame:
        p = case.parameters
        ego_v = p["ego_speed_kph"] / 3.6
        rear_v = p["rear_speed_kph"] / 3.6
        ego_x = 0.0
        rear_x = -p["rear_distance_m"] - VEHICLE_LENGTH_M
        merge_time = p["merge_time_s"]
        rows = []

        n = int(self.horizon_s / self.dt) + 1
        for i in range(n):
            t = i * self.dt
            gap = ego_x - rear_x - VEHICLE_LENGTH_M
            closing = rear_v - ego_v
            ttc = gap / closing if closing > 1e-6 and gap > 0 else math.inf
            overlap_active = t >= merge_time * 0.70
            rows.append({
                "time_s": t,
                "ego_x_m": ego_x,
                "ego_speed_mps": ego_v,
                "other_x_m": rear_x,
                "other_speed_mps": rear_v,
                "gap_m": gap,
                "ttc_s": ttc,
                "overlap_active": overlap_active,
            })
            if overlap_active and gap <= 0:
                break
            ego_x += ego_v * self.dt
            rear_x += rear_v * self.dt

        return pd.DataFrame(rows)
