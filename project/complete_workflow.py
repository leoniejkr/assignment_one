"""
Complete workflow for drone delivery optimization experiments.

This script demonstrates the full pipeline:
1. Load data
2. Run greedy baselines
3. Run metaheuristic algorithms
4. Analyze results
5. Generate visualizations
"""

import sys
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("DRONE DELIVERY OPTIMIZATION - COMPLETE WORKFLOW")
    print("="*80)
    
    # ============= STEP 1: Load Data =============
    print("\n[STEP 1] Loading input data...")
    
    from file_handling import load_input
    
    # Change this to your input file
    input_file = "project/data/input/busy_day.in"
    
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please update the input_file path in this script.")
        sys.exit(1)
    
    data = load_input(input_file)
    
    print(f"✓ Loaded: {len(data['orders'])} orders, {len(data['warehouses'])} warehouses")
    print(f"  Grid: {data['rows']}x{data['cols']}, Drones: {data['num_drones']}, Deadline: {data['deadline']}T")
    
    # ============= STEP 2: Quick Test (Optional) =============
    
    print("\n" + "="*80)
    choice = input("\nRun quick test first? (recommended) [y/n]: ").strip().lower()
    
    if choice == 'y':
        print("\n[STEP 2] Running QUICK TEST (3 runs per algorithm, ~2-3 minutes)...")
        
        from runner import ExperimentRunner
        from configs import define_experiments_quick_test
        
        # Temporarily replace define_experiments
        import configs
        original_experiments = configs.define_experiments
        configs.define_experiments = define_experiments_quick_test
        
        runner = ExperimentRunner(data, output_dir="project/results/quick_test")
        runner.run_all(num_runs=3, include_greedy=True)
        
        # Restore original
        configs.define_experiments = original_experiments
        
        print("\n✓ Quick test complete!")
        print(f"  Results in: project/results/quick_test")
        
        proceed = input("\nProceed to full experiments? [y/n]: ").strip().lower()
        if proceed != 'y':
            print("\nExiting. Run this script again when ready for full experiments.")
            sys.exit(0)
    
    # ============= STEP 3: Full Experiments =============
    
    print("\n" + "="*80)
    print("[STEP 3] Running FULL EXPERIMENTS")
    print("="*80)
    
    num_runs = int(input("\nHow many runs per algorithm configuration? [default=5]: ").strip() or "5")
    
    print(f"\nStarting {num_runs} runs per configuration...")
    print("This may take 10-30 minutes depending on your hardware.")
    print("-"*80)
    
    from runner import ExperimentRunner
    
    runner = ExperimentRunner(data, output_dir="project/results")
    runner.run_all(num_runs=num_runs, include_greedy=True)
    
    print("\n✓ All experiments complete!")
    
    # ============= STEP 4: Analysis =============
    
    print("\n" + "="*80)
    print("[STEP 4] Running COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    from comprehensive_analysis import ComprehensiveAnalyzer
    
    analyzer = ComprehensiveAnalyzer("project/results")
    analyzer.run_full_analysis()
    
    # ============= STEP 5: Summary =============
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    
    print("\n📁 Output Files Generated:")
    print("  Results:")
    print("    - project/results/summary_results.csv")
    print("    - project/results/detailed_results.json")
    print("    - project/results/algorithm_ranking.csv")
    print("    - project/results/summary_statistics.csv")
    print("    - project/results/efficiency_metrics.csv")
    print("    - project/results/statistical_tests.csv")
    print("\n  Plots:")
    print("    - project/results/plots/score_distribution.png")
    print("    - project/results/plots/performance_comparison.png")
    print("    - project/results/plots/runtime_analysis.png")
    print("    - project/results/plots/convergence_curves.png")
    print("    - project/results/plots/algorithm_ranking.png")
    print("\n  Solutions:")
    print("    - project/results/*.out (submission files)")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review plots in project/results/plots/")
    print("  2. Examine summary_results.csv for detailed metrics")
    print("  3. Check statistical_tests.csv for significance")
    print("  4. Use best .out file for submission")
    print("="*80 + "\n")


def print_quick_stats(results_dir="project/results"):
    """Print quick statistics from existing results."""
    import pandas as pd
    from pathlib import Path
    
    csv_path = Path(results_dir) / "summary_results.csv"
    
    if not csv_path.exists():
        print(f"No results found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("QUICK RESULTS SUMMARY")
    print("="*80)
    
    # Use correct column names
    score_col = "score" if "score" in df.columns else "final_score"
    runtime_col = "runtime" if "runtime" in df.columns else "runtime_seconds"
    
    summary = df.groupby("algorithm").agg({
        score_col: ["mean", "std", "max"],
        runtime_col: "mean",
        "num_evaluations": "mean" if "num_evaluations" in df.columns else "count"
    }).round(2)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.sort_values(f"{score_col}_mean", ascending=False)
    
    print(summary)
    
    # Best solution
    best_idx = df[score_col].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n🏆 Best Overall Solution:")
    print(f"   Algorithm: {best['algorithm']}")
    print(f"   Score: {best[score_col]:.0f}")
    print(f"   Runtime: {best[runtime_col]:.2f}s")
    if "num_evaluations" in df.columns:
        print(f"   Evaluations: {best['num_evaluations']:.0f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Delivery Optimization Workflow")
    parser.add_argument("--stats-only", action="store_true", 
                       help="Only print statistics from existing results")
    parser.add_argument("--results-dir", default="project/results",
                       help="Results directory (default: project/results)")
    
    args = parser.parse_args()
    
    if args.stats_only:
        print_quick_stats(args.results_dir)
    else:
        main()