import time, json
import pandas as pd
from pathlib import Path
from result_types import ExperimentResult
from configs import define_experiments
from greedy import build_greedy_trips, find_warehouse_plan_for_order
from file_handling import write_submission

class ExperimentRunner:
    """Run experiments and store results."""

    def __init__(self, data, output_dir="project/results"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def run_experiment(self, algo_class, params, run_number):
        print(f"\nRunning {algo_class.__name__} | Run {run_number} | Params: {params}")
        start = time.time()
        algo = algo_class(self.data, **params)
        solution, score, history = algo.optimize()
        runtime = time.time() - start

        best_iter = history[-1][0] if history else 0
        best_time = history[-1][2] if history else runtime

        result = ExperimentResult(
            algorithm=algo_class.__name__,
            parameters=params,
            run_number=run_number,
            final_score=score,
            runtime_seconds=runtime,
            iterations=len(history) if history else 0,
            convergence_history=history,
            best_found_at_iteration=best_iter,
            best_found_at_time=best_time,
        )
        self.results.append(result)

        if solution:
            filename = f"{algo_class.__name__}_run{run_number}_score{score}.out"
            filepath = self.output_dir / filename
            assignments = []
            for oid, drone_id in enumerate(solution):
                order = self.data["orders"][oid]
                plan = find_warehouse_plan_for_order(order, self.data["warehouses"])
                assignments.append({"plan": plan, "drone": drone_id})
            trips = build_greedy_trips(assignments, self.data)
            write_submission(trips, str(filepath))
        return result

    def run_all(self, num_runs=5):
        print("\n=== Starting Full Experiment Suite ===")
        configs = define_experiments()
        total = sum(len(c) for c in configs.values()) * num_runs
        progress = 0
        for algo_name, setups in configs.items():
            for config in setups:
                for run in range(1, num_runs + 1):
                    progress += 1
                    print(f"\nProgress: {progress}/{total}")
                    self.run_experiment(config["class"], config["params"], run)
        self.save_results()

    def save_results(self):
        json_path = self.output_dir / "detailed_results.json"
        csv_path = self.output_dir / "summary_results.csv"
        json.dump([r.to_dict() for r in self.results], open(json_path, "w"), indent=2)
        pd.DataFrame([r.to_dict() for r in self.results]).to_csv(csv_path, index=False)
        print(f"✓ Results saved to {json_path} and {csv_path}")
