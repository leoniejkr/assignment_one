from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass
class ExperimentResult:
    """Single experiment result."""
    algorithm: str
    parameters: Dict[str, Any]
    run_number: int
    final_score: float
    runtime_seconds: float
    iterations: int
    convergence_history: List[tuple]
    best_found_at_iteration: int
    best_found_at_time: float

    def to_dict(self):
        return asdict(self)
