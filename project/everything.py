"""
Complete Experimental Framework for Systematic Comparison
Run multiple algorithms with different parameters and analyze results
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from file_handling import load_input, write_submission
from greedy import build_greedy_trips

from optimizers.genetic_algorithm import GeneticAlgorithm
from optimizers.hill_climing import HillClimbing
from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.tabu_search import TabuSearch

@dataclass
class ExperimentResult:
    """Store single experiment result"""
    algorithm: str
    parameters: Dict[str, Any]
    run_number: int
    final_score: int
    runtime_seconds: float
    iterations: int
    convergence_history: List[tuple]
    best_found_at_iteration: int
    best_found_at_time: float


class ExperimentRunner:
    """Run systematic experiments comparing algorithms"""

    def __init__(self, data, output_dir="project/results"):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ExperimentResult] = []
    
    def run_experiment(self, algorithm_class, params, run_number, num_runs):
        """Run single experiment"""
        print(f"\n{'─'*60}")
        print(f"Run {run_number}/{num_runs}: {algorithm_class.__name__}")
        print(f"Parameters: {params}")
        print(f"{'─'*60}")
        
        start = time.time()
        
        # Create algorithm instance
        algo = algorithm_class(self.data, **params)
        
        # Run optimization
        solution, score, history = algo.optimize()
        
        runtime = time.time() - start
        
        # Extract metrics
        iterations = len(history) if history else 0
        best_iter = history[-1][0] if history else 0
        best_time = history[-1][2] if history else runtime
        
        # Store result
        result = ExperimentResult(
            algorithm=algorithm_class.__name__,
            parameters=params,
            run_number=run_number,
            final_score=score,
            runtime_seconds=runtime,
            iterations=iterations,
            convergence_history=history,
            best_found_at_iteration=best_iter,
            best_found_at_time=best_time
        )
        
        self.results.append(result)
        
        # Save submission file
        if solution:
            filename = (f"{algorithm_class.__name__}_"
                       f"run{run_number}_score{score}.out")
            filepath = self.output_dir / filename
            
            assignments = []
            from greedy import find_warehouse_plan_for_order
            for oid, drone_id in enumerate(solution):
                order = self.data["orders"][oid]
                plan = find_warehouse_plan_for_order(order, self.data["warehouses"])
                assignments.append({"plan": plan, "drone": drone_id})
            
            trips = build_greedy_trips(assignments, self.data)
            write_submission(trips, str(filepath))
        
        return result
    
    def run_full_experiments(self, num_runs=2):
        """Run complete experimental suite"""
        print("="*60)
        print("STARTING FULL EXPERIMENTAL SUITE")
        print("="*60)
        
        # Define parameter grids for each algorithm
        experiments = self.define_experiments()
        
        total_experiments = sum(len(configs) for configs in experiments.values()) * num_runs
        current = 0
        
        for algo_name, configs in experiments.items():
            print(f"\n{'='*60}")
            print(f"ALGORITHM: {algo_name}")
            print(f"{'='*60}")
            
            for config_idx, config in enumerate(configs, 1):
                print(f"\nConfiguration {config_idx}/{len(configs)}")
                
                for run in range(1, num_runs + 1):
                    current += 1
                    print(f"\nProgress: {current}/{total_experiments} "
                          f"({current/total_experiments*100:.1f}%)")
                    
                    self.run_experiment(
                        config["class"],
                        config["params"],
                        run,
                        num_runs
                    )
        
        # Save all results
        self.save_results()
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*60)
    
    def define_experiments(self) -> Dict[str, List[Dict]]:
        """Define all experiment configurations"""
        
        experiments = {
            "HillClimbing": [
                {
                    "class": HillClimbing,
                    "params": {
                        "max_iterations": 1000,
                        "restarts": 3,
                        "time_limit": 60
                    }
                },
                {
                    "class": HillClimbing,
                    "params": {
                        "max_iterations": 2000,
                        "restarts": 5,
                        "time_limit": 60
                    }
                },
                {
                    "class": HillClimbing,
                    "params": {
                        "max_iterations": 500,
                        "restarts": 10,
                        "time_limit": 60
                    }
                }
            ],
            
            "SimulatedAnnealing": [
                {
                    "class": SimulatedAnnealing,
                    "params": {
                        "initial_temp": 10000,
                        "cooling_rate": 0.95,
                        "iterations_per_temp": 100,
                        "min_temp": 1,
                        "time_limit": 60
                    }
                },
                {
                    "class": SimulatedAnnealing,
                    "params": {
                        "initial_temp": 10000,
                        "cooling_rate": 0.99,
                        "iterations_per_temp": 100,
                        "min_temp": 1,
                        "time_limit": 60
                    }
                },
                {
                    "class": SimulatedAnnealing,
                    "params": {
                        "initial_temp": 5000,
                        "cooling_rate": 0.95,
                        "iterations_per_temp": 500,
                        "min_temp": 0.1,
                        "time_limit": 60
                    }
                },
                {
                    "class": SimulatedAnnealing,
                    "params": {
                        "initial_temp": 20000,
                        "cooling_rate": 0.99,
                        "iterations_per_temp": 200,
                        "min_temp": 1,
                        "time_limit": 60
                    }
                }
            ],
            
            "TabuSearch": [
                {
                    "class": TabuSearch,
                    "params": {
                        "tabu_tenure": 10,
                        "max_iterations": 1000,
                        "time_limit": 60
                    }
                },
                {
                    "class": TabuSearch,
                    "params": {
                        "tabu_tenure": 20,
                        "max_iterations": 1000,
                        "time_limit": 60
                    }
                },
                {
                    "class": TabuSearch,
                    "params": {
                        "tabu_tenure": 50,
                        "max_iterations": 1000,
                        "time_limit": 60
                    }
                },
                {
                    "class": TabuSearch,
                    "params": {
                        "tabu_tenure": 20,
                        "max_iterations": 2000,
                        "time_limit": 60
                    }
                }
            ],
            
            "GeneticAlgorithm": [
                {
                    "class": GeneticAlgorithm,
                    "params": {
                        "population_size": 50,
                        "generations": 50,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.8,
                        "time_limit": 60
                    }
                },
                {
                    "class": GeneticAlgorithm,
                    "params": {
                        "population_size": 100,
                        "generations": 50,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.8,
                        "time_limit": 60
                    }
                },
                {
                    "class": GeneticAlgorithm,
                    "params": {
                        "population_size": 50,
                        "generations": 100,
                        "mutation_rate": 0.3,
                        "crossover_rate": 0.8,
                        "time_limit": 60
                    }
                },
                {
                    "class": GeneticAlgorithm,
                    "params": {
                        "population_size": 100,
                        "generations": 100,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.9,
                        "time_limit": 60
                    }
                }
            ]
        }
        
        return experiments
    
    def save_results(self):
        """Save results to JSON and CSV"""
        # Save detailed JSON
        json_data = [asdict(r) for r in self.results]
        json_path = self.output_dir / "detailed_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Saved detailed results to {json_path}")
        
        # Save summary CSV
        summary_data = []
        for r in self.results:
            summary_data.append({
                "algorithm": r.algorithm,
                "params": str(r.parameters),
                "run": r.run_number,
                "score": r.final_score,
                "runtime": r.runtime_seconds,
                "iterations": r.iterations
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / "summary_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved summary to {csv_path}")
    
    def analyze_results(self):
        """Perform statistical analysis"""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        df = pd.DataFrame([
            {
                "algorithm": r.algorithm,
                "config": str(r.parameters),
                "score": r.final_score,
                "runtime": r.runtime_seconds
            }
            for r in self.results
        ])
        
        # Group by algorithm
        print("\n--- Performance by Algorithm ---")
        grouped = df.groupby("algorithm")["score"].agg(['mean', 'std', 'min', 'max', 'count'])
        print(grouped)
        
        # Best configuration per algorithm
        print("\n--- Best Configuration per Algorithm ---")
        for algo in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algo]
            best_config = algo_df.groupby("config")["score"].mean().idxmax()
            best_score = algo_df.groupby("config")["score"].mean().max()
            print(f"\n{algo}:")
            print(f"  Config: {best_config}")
            print(f"  Avg Score: {best_score:.2f}")
        
        # Overall winner
        print("\n--- Overall Best ---")
        best_run = df.loc[df["score"].idxmax()]
        print(f"Algorithm: {best_run['algorithm']}")
        print(f"Score: {best_run['score']}")
        print(f"Config: {best_run['config']}")
        
        return df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        df = pd.DataFrame([
            {
                "algorithm": r.algorithm,
                "score": r.final_score,
                "runtime": r.runtime_seconds,
                "config": str(r.parameters)[:30]
            }
            for r in self.results
        ])
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Box plot comparing algorithms
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="algorithm", y="score")
        plt.title("Score Distribution by Algorithm", fontsize=16)
        plt.xlabel("Algorithm", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "boxplot_comparison.png", dpi=300)
        print("✓ Saved boxplot_comparison.png")
        plt.close()
        
        # 2. Runtime vs Score scatter
        plt.figure(figsize=(10, 6))
        for algo in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algo]
            plt.scatter(algo_df["runtime"], algo_df["score"], 
                       label=algo, alpha=0.6, s=100)
        plt.xlabel("Runtime (seconds)", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Score vs Runtime", fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_vs_score.png", dpi=300)
        print("✓ Saved runtime_vs_score.png")
        plt.close()
        
        # 3. Convergence curves (for each algorithm, plot average)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        algorithms = df["algorithm"].unique()
        for idx, algo in enumerate(algorithms):
            algo_results = [r for r in self.results if r.algorithm == algo]
            
            ax = axes[idx]
            for r in algo_results[:5]:  # Plot first 5 runs
                if r.convergence_history:
                    times = [h[2] for h in r.convergence_history]
                    scores = [h[1] for h in r.convergence_history]
                    ax.plot(times, scores, alpha=0.3, color='blue')
            
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Best Score Found")
            ax.set_title(f"{algo} - Convergence")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "convergence_curves.png", dpi=300)
        print("✓ Saved convergence_curves.png")
        plt.close()
        
        # 4. Bar chart - mean scores
        plt.figure(figsize=(10, 6))
        means = df.groupby("algorithm")["score"].mean().sort_values(ascending=False)
        stds = df.groupby("algorithm")["score"].std()
        
        plt.bar(range(len(means)), means.values, yerr=stds.values, 
               capsize=5, alpha=0.7)
        plt.xticks(range(len(means)), means.index, rotation=45)
        plt.ylabel("Average Score", fontsize=12)
        plt.title("Average Performance by Algorithm", fontsize=16)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "average_performance.png", dpi=300)
        print("✓ Saved average_performance.png")
        plt.close()
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load data
    data = load_input("project/data/input/busy_day.in")
    
    # Create experiment runner
    runner = ExperimentRunner(data, output_dir="project/results")
    
    # Run full experimental suite
    print("Starting experiments...")
    print("This will take some time (estimated: 30-60 minutes)")
    print()
    
    runner.run_full_experiments(num_runs=2)
    
    # Analyze results
    runner.analyze_results()
    
    # Create visualizations
    runner.create_visualizations()
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"Results saved in: {runner.output_dir}/")
    print("Files created:")
    print("  - detailed_results.json")
    print("  - summary_results.csv")
    print("  - boxplot_comparison.png")
    print("  - runtime_vs_score.png")
    print("  - convergence_curves.png")
    print("  - average_performance.png")
    print("  - *.out (submission files)")