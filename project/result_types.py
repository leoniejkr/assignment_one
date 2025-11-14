from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

@dataclass
class ExperimentResult:
    """Single experiment result with comprehensive metrics."""
    
    # Basic identification
    algorithm: str
    parameters: Dict[str, Any]
    run_number: int
    
    # Performance metrics
    final_score: float
    runtime_seconds: float
    iterations: int
    
    # Evaluation tracking
    num_evaluations: int = 0
    evals_per_second: float = 0.0
    
    # Convergence tracking
    convergence_history: List[tuple] = field(default_factory=list)
    best_found_at_iteration: int = 0
    best_found_at_time: float = 0.0
    
    # Optional solution quality metrics
    orders_completed: Optional[int] = None
    completion_rate: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def compute_efficiency_metrics(self):
        """Compute derived efficiency metrics."""
        if self.runtime_seconds > 0:
            self.evals_per_second = self.num_evaluations / self.runtime_seconds
        else:
            self.evals_per_second = 0.0
    
    def __post_init__(self):
        """Automatically compute efficiency metrics after initialization."""
        self.compute_efficiency_metrics()
    
    def summary_string(self):
        """Get a concise summary string."""
        return (
            f"{self.algorithm} (run {self.run_number}): "
            f"score={self.final_score:.0f}, "
            f"time={self.runtime_seconds:.2f}s, "
            f"evals={self.num_evaluations}"
        )


@dataclass
class ComparisonResult:
    """Result of comparing multiple algorithms."""
    
    algorithm_name: str
    mean_score: float
    std_score: float
    mean_runtime: float
    mean_evaluations: float
    num_runs: int
    
    # Statistical measures
    best_score: float
    worst_score: float
    median_score: float
    
    # Efficiency
    mean_score_per_second: float
    mean_score_per_evaluation: float
    
    def __str__(self):
        return (
            f"{self.algorithm_name}:\n"
            f"  Score: {self.mean_score:.1f} ± {self.std_score:.1f} "
            f"(best={self.best_score:.0f}, worst={self.worst_score:.0f})\n"
            f"  Runtime: {self.mean_runtime:.2f}s\n"
            f"  Evaluations: {self.mean_evaluations:.0f}\n"
            f"  Efficiency: {self.mean_score_per_evaluation:.3f} score/eval"
        )


def create_comparison_from_results(results: List[ExperimentResult]) -> ComparisonResult:
    """Create a comparison result from multiple experiment results."""
    if not results:
        raise ValueError("Cannot create comparison from empty results list")
    
    scores = [r.final_score for r in results]
    runtimes = [r.runtime_seconds for r in results]
    evaluations = [r.num_evaluations for r in results]
    
    import numpy as np
    
    mean_runtime = np.mean(runtimes)
    mean_evals = np.mean(evaluations)
    mean_score = np.mean(scores)
    
    return ComparisonResult(
        algorithm_name=results[0].algorithm,
        mean_score=mean_score,
        std_score=np.std(scores),
        mean_runtime=mean_runtime,
        mean_evaluations=mean_evals,
        num_runs=len(results),
        best_score=max(scores),
        worst_score=min(scores),
        median_score=np.median(scores),
        mean_score_per_second=mean_score / mean_runtime if mean_runtime > 0 else 0,
        mean_score_per_evaluation=mean_score / mean_evals if mean_evals > 0 else 0,
    )