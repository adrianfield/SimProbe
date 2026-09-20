import json
from pathlib import Path
from .models import SeedScenario
from .scenario_registry import scenario_ir_fields

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "seed_scenarios.json"

def load_seed_scenarios(path: Path = DEFAULT_PATH) -> dict[str, SeedScenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seeds = {}
    for item in raw:
        seed = SeedScenario(**item)
        defaults = scenario_ir_fields(seed, source="seed_asset")
        update = {
            key: getattr(seed, key) or defaults[key]
            for key in ("actors", "road", "constraints", "provenance")
        }
        seed = seed.model_copy(update=update)
        seeds[seed.case_id] = seed
    return seeds
