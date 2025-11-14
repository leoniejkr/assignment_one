import time
import json
import pandas as pd
from pathlib import Path
from result_types import ExperimentResult
from configs import define_experiments
from greedy_improved import build_greedy_trips, find_warehouse_plan_for_order
from greedy_improved import spatial_clustering_assignment, priority_based_assignment
from file_handling import write_submission

class ExperimentRunner:
    """Run experiments and store results."""

    def __init__(self, data, output_dir="project/results"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Global evaluation counter
        self.global_eval_counter = 0

    def run_experiment(self, algo_class, params, run_number):
        """Run a single experiment with evaluation counting."""
        print(f"\nRunning {algo_class.__name__} | Run {run_number} | Params: {params}")
        
        start = time.time()
        
        # Create algorithm instance
        algo = algo_class(self.data, **params)
        
        # Wrap the evaluate method to count evaluations
        original_evaluate = algo.evaluate
        eval_counter = {"count": 0}
        
        def counted_evaluate(solution):
            eval_counter["count"] += 1
            return original_evaluate(solution)
        
        # Replace evaluate method with counted version
        algo.evaluate = counted_evaluate
        
        # Run optimization
        solution, score, history = algo.optimize()
        runtime = time.time() - start

        # Get evaluation count
        num_evaluations = eval_counter["count"]
        
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
        
        # Add evaluation count to result (we'll extend ExperimentResult later)
        result_dict = result.to_dict()
        result_dict["num_evaluations"] = num_evaluations
        result_dict["evals_per_second"] = num_evaluations / runtime if runtime > 0 else 0
        
        self.results.append(result)
        
        # Update global counter
        self.global_eval_counter += num_evaluations
        
        print(f"  ✓ Score: {score:.0f} | Runtime: {runtime:.2f}s | Evaluations: {num_evaluations}")

        # Save solution to file
        if solution:
            filename = f"{algo_class.__name__}_run{run_number}_score{score:.0f}.out"
            filepath = self.output_dir / filename
            
            assignments = []
            for oid, drone_id in enumerate(solution):
                order = self.data["orders"][oid]
                plan = find_warehouse_plan_for_order(order, self.data["warehouses"])
                if plan:
                    assignments.append({"plan": plan, "drone": drone_id})
            
            if assignments:
                trips = build_greedy_trips(assignments, self.data)
                write_submission(trips, str(filepath))
        
        return result, num_evaluations

    def run_greedy_baselines(self):
        """Run greedy baseline algorithms for comparison."""
        print("\n" + "="*70)
        print("RUNNING GREEDY BASELINES")
        print("="*70)
        
        greedy_methods = [
            ("GreedySpatialClustering", spatial_clustering_assignment),
            ("GreedyPriorityBased", priority_based_assignment),
        ]
        
        for method_name, method_func in greedy_methods:
            print(f"\nRunning {method_name}...")
            
            start = time.time()
            
            try:
                # Generate solution
                assignments = method_func(self.data)
                trips = build_greedy_trips(assignments, self.data)
                
                # Score solution
                from util import simulate_and_score
                score, order_scores = simulate_and_score(trips, self.data)
                
                runtime = time.time() - start
                
                # Extract drone assignment (for consistency with metaheuristics)
                solution = [a["drone"] for a in assignments]
                
                # Create result object
                result = ExperimentResult(
                    algorithm=method_name,
                    parameters={},
                    run_number=1,
                    final_score=score,
                    runtime_seconds=runtime,
                    iterations=1,
                    convergence_history=[(0, score, runtime)],
                    best_found_at_iteration=0,
                    best_found_at_time=runtime,
                )
                
                self.results.append(result)
                
                # Save output file
                filename = f"{method_name}_score{score:.0f}.out"
                filepath = self.output_dir / filename
                write_submission(trips, str(filepath))
                
                print(f"  ✓ {method_name} Score: {score:.0f} | Runtime: {runtime:.2f}s")
                
            except Exception as e:
                print(f"  ✗ {method_name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)

    def run_all(self, num_runs=5, include_greedy=True):
        """Run full experiment suite."""
        print("\n" + "="*70)
        print("STARTING FULL EXPERIMENT SUITE")
        print("="*70)
        
        # Reset global counter
        self.global_eval_counter = 0
        
        # Run greedy baselines first (optional)
        if include_greedy:
            self.run_greedy_baselines()
        
        # Run metaheuristics
        configs = define_experiments()
        total_experiments = sum(len(c) for c in configs.values()) * num_runs
        progress = 0
        
        print(f"\nTotal metaheuristic experiments: {total_experiments}")
        print("="*70)
        
        for algo_name, setups in configs.items():
            for config in setups:
                for run in range(1, num_runs + 1):
                    progress += 1
                    print(f"\n[Progress: {progress}/{total_experiments}]")
                    
                    try:
                        self.run_experiment(config["class"], config["params"], run)
                    except Exception as e:
                        print(f"  ✗ Experiment failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Save all results
        self.save_results()
        
        print("\n" + "="*70)
        print(f"✓ ALL EXPERIMENTS COMPLETE")
        print(f"✓ Total evaluations performed: {self.global_eval_counter:,}")
        print("="*70)

    def save_results(self):
        """Save results to JSON and CSV with evaluation counts."""
        json_path = self.output_dir / "detailed_results.json"
        csv_path = self.output_dir / "summary_results.csv"
        
        # Convert results to dictionaries
        results_data = []
        for r in self.results:
            result_dict = r.to_dict()
            
            # Try to extract evaluation count from convergence history
            # (This is a fallback if we didn't capture it during run)
            if "num_evaluations" not in result_dict:
                result_dict["num_evaluations"] = len(result_dict.get("convergence_history", []))
                result_dict["evals_per_second"] = (
                    result_dict["num_evaluations"] / result_dict["runtime_seconds"]
                    if result_dict["runtime_seconds"] > 0 else 0
                )
            
            results_data.append(result_dict)
        
        # Save JSON (detailed)
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV (summary)
        df = pd.DataFrame(results_data)
        
        # Drop convergence history from CSV (too large)
        if "convergence_history" in df.columns:
            df_summary = df.drop(columns=["convergence_history"])
        else:
            df_summary = df
        
        df_summary.to_csv(csv_path, index=False)
        
        print(f"\n✓ Results saved:")
        print(f"  - Detailed: {json_path}")
        print(f"  - Summary:  {csv_path}")
        
        # Print summary statistics
        self.print_summary()

    def print_summary(self):
        """Print quick summary of results."""
        if not self.results:
            print("\nNo results to summarize.")
            return
        
        print("\n" + "="*70)
        print("QUICK SUMMARY")
        print("="*70)
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        summary = df.groupby("algorithm").agg({
            "final_score": ["mean", "std", "max"],
            "runtime_seconds": ["mean", "std"],
        }).round(2)
        
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.sort_values("final_score_mean", ascending=False)
        
        print("\nAlgorithm Performance:")
        print(summary)
        
        # Best overall
        best_idx = df["final_score"].idxmax()
        best = df.loc[best_idx]
        
        print(f"\n🏆 Best Solution:")
        print(f"   Algorithm: {best['algorithm']}")
        print(f"   Score: {best['final_score']:.0f}")
        print(f"   Runtime: {best['runtime_seconds']:.2f}s")
        print(f"   Run: {best['run_number']}")
        
        print("="*70)


# ============= USAGE EXAMPLE =============

if __name__ == "__main__":
    from file_handling import load_input
    
    # Load data
    data = load_input("project/data/input/busy_day.in")
    
    # Create runner
    runner = ExperimentRunner(data, output_dir="project/results")
    
    # Run all experiments (including greedy baselines)
    runner.run_all(
        num_runs=3,           # Number of runs per algorithm configuration
        include_greedy=True   # Include greedy baselines
    )
    
    print("\n✓ Experiments complete! Run analysis next:")
    print("   python comprehensive_analysis.py")