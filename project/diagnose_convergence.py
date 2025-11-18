"""
Diagnostic script to understand why greedy baselines beat some metaheuristics.
This will help determine if the issue is:
1. Lack of convergence (need more time/iterations)
2. Poor initial solutions (start from random)
3. Problem-specific (tiny problem favors greedy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast

def analyze_convergence_issue(results_dir="project/results"):
    """Comprehensive diagnosis of convergence issues."""
    
    print("\n" + "="*80)
    print("CONVERGENCE DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Load data
    csv_path = Path(results_dir) / "summary_results.csv"
    if not csv_path.exists():
        print(f"❌ No results found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    score_col = "score" if "score" in df.columns else "final_score"
    
    # Identify greedy vs meta
    greedy_algos = [a for a in df["algorithm"].unique() if "Greedy" in a or "greedy" in a]
    meta_algos = [a for a in df["algorithm"].unique() if a not in greedy_algos]
    
    if not greedy_algos:
        print("⚠️ No greedy baselines found")
        return
    
    # Get best greedy score
    best_greedy_score = df[df["algorithm"].isin(greedy_algos)][score_col].max()
    best_greedy_name = df[df["algorithm"].isin(greedy_algos)].groupby("algorithm")[score_col].mean().idxmax()
    
    print(f"\n🎯 Best Greedy Baseline: {best_greedy_name}")
    print(f"   Score: {best_greedy_score:.0f}")
    
    # Check each metaheuristic
    print("\n" + "-"*80)
    print("METAHEURISTIC ANALYSIS")
    print("-"*80)
    
    for algo in meta_algos:
        algo_df = df[df["algorithm"] == algo]
        
        mean_score = algo_df[score_col].mean()
        max_score = algo_df[score_col].max()
        min_score = algo_df[score_col].min()
        
        print(f"\n{algo}:")
        print(f"  Mean: {mean_score:.0f} | Max: {max_score:.0f} | Min: {min_score:.0f}")
        
        # Check if beats greedy
        if mean_score < best_greedy_score:
            deficit = best_greedy_score - mean_score
            print(f"  ❌ WORSE than greedy by {deficit:.0f} points ({(deficit/best_greedy_score)*100:.1f}%)")
            
            # Diagnose why
            print(f"\n  🔍 DIAGNOSIS:")
            
            # Check convergence history
            if "convergence_history" in algo_df.columns:
                for idx, row in algo_df.iterrows():
                    history = row["convergence_history"]
                    if isinstance(history, str):
                        try:
                            history = ast.literal_eval(history)
                        except:
                            continue
                    
                    if history and len(history) > 1:
                        # Check initial vs final score
                        initial_score = history[0][1]
                        final_score = history[-1][1]
                        improvement = final_score - initial_score
                        
                        # Check if still improving at end
                        last_5_improvements = []
                        if len(history) >= 5:
                            for i in range(-5, -1):
                                last_5_improvements.append(history[i+1][1] - history[i][1])
                        
                        still_improving = any(imp > 0 for imp in last_5_improvements)
                        
                        print(f"     Run {row['run_number']}:")
                        print(f"       Initial score: {initial_score:.0f}")
                        print(f"       Final score: {final_score:.0f}")
                        print(f"       Total improvement: {improvement:.0f}")
                        
                        if initial_score < best_greedy_score - 1000:
                            print(f"       ⚠️ Poor initial solution (< greedy by {best_greedy_score - initial_score:.0f})")
                            print(f"       → Consider using greedy initialization")
                        
                        if still_improving:
                            print(f"       ⚠️ Still improving at timeout")
                            print(f"       → Increase time_limit or iterations")
                        else:
                            print(f"       ✓ Converged (not improving in last 5 steps)")
                        
                        # Check convergence speed
                        time_to_90 = None
                        target_score = initial_score + 0.9 * improvement
                        for iter_num, score, time_sec in history:
                            if score >= target_score:
                                time_to_90 = time_sec
                                break
                        
                        if time_to_90:
                            print(f"       Time to 90% improvement: {time_to_90:.2f}s")
                            if time_to_90 < 1.0:
                                print(f"       → Very fast convergence, might be premature")
                        
                        break  # Only analyze first run
        else:
            improvement = mean_score - best_greedy_score
            print(f"  ✓ Beats greedy by {improvement:.0f} points ({(improvement/best_greedy_score)*100:.1f}%)")
    
    # Overall recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    underperforming = [a for a in meta_algos 
                      if df[df["algorithm"]==a][score_col].mean() < best_greedy_score]
    
    if underperforming:
        print(f"\n❌ {len(underperforming)} metaheuristic(s) underperform greedy:")
        for algo in underperforming:
            print(f"   - {algo}")
        
        print("\n💡 Suggested fixes:")
        print("   1. Use greedy initialization instead of random")
        print("      → Set use_greedy_init=True in base_optimizer.py")
        print("   2. Increase computational budget:")
        print("      → Hill Climbing: max_iterations = 500-1000")
        print("      → Simulated Annealing: increase initial_temp or iterations_per_temp")
        print("      → Tabu Search: max_iterations = 10000")
        print("   3. Increase time_limit to 120 seconds")
        print("   4. Test on larger problem instances (not just busy_day.in)")
    else:
        print("\n✅ All metaheuristics beat greedy baseline!")
        print("   Your implementations are working correctly.")
    
    # Problem size check
    print("\n" + "-"*80)
    print("PROBLEM SIZE CHECK")
    print("-"*80)
    
    from file_handling import load_input
    try:
        data = load_input("project/data/input/busy_day.in")
        print(f"\nCurrent problem (busy_day.in):")
        print(f"  Orders: {len(data['orders'])}")
        print(f"  Drones: {data['num_drones']}")
        print(f"  Warehouses: {len(data['warehouses'])}")
        print(f"  Deadline: {data['deadline']} turns")
        
        if len(data['orders']) < 10:
            print(f"\n⚠️ VERY SMALL PROBLEM!")
            print(f"   With only {len(data['orders'])} orders, the solution space is tiny.")
            print(f"   Greedy methods naturally perform well on small problems.")
            print(f"   Recommendation: Test on larger instances to see true algorithm performance.")
    except:
        print("⚠️ Could not load problem data")


def plot_initial_vs_final_scores(results_dir="project/results"):
    """Visualize initial vs final scores to show improvement."""
    import matplotlib.pyplot as plt
    
    csv_path = Path(results_dir) / "summary_results.csv"
    if not csv_path.exists():
        return
    
    df = pd.read_csv(csv_path)
    
    greedy_algos = [a for a in df["algorithm"].unique() if "Greedy" in a or "greedy" in a]
    meta_algos = [a for a in df["algorithm"].unique() if a not in greedy_algos]
    
    if "convergence_history" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_for_plot = []
    
    for algo in meta_algos:
        algo_df = df[df["algorithm"] == algo]
        
        for idx, row in algo_df.iterrows():
            history = row["convergence_history"]
            if isinstance(history, str):
                try:
                    history = ast.literal_eval(history)
                except:
                    continue
            
            if history and len(history) > 0:
                initial = history[0][1]
                final = history[-1][1]
                data_for_plot.append({
                    "Algorithm": algo,
                    "Initial": initial,
                    "Final": final,
                    "Improvement": final - initial
                })
    
    if not data_for_plot:
        return
    
    plot_df = pd.DataFrame(data_for_plot)
    
    x = np.arange(len(meta_algos))
    width = 0.35
    
    initial_means = [plot_df[plot_df["Algorithm"]==a]["Initial"].mean() for a in meta_algos]
    final_means = [plot_df[plot_df["Algorithm"]==a]["Final"].mean() for a in meta_algos]
    
    ax.bar(x - width/2, initial_means, width, label='Initial Score', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, final_means, width, label='Final Score', alpha=0.7, color='lightgreen')
    
    # Add greedy reference line
    if greedy_algos:
        score_col = "score" if "score" in df.columns else "final_score"
        best_greedy = df[df["algorithm"].isin(greedy_algos)][score_col].max()
        ax.axhline(y=best_greedy, color='red', linestyle='--', linewidth=2, 
                  label=f'Best Greedy: {best_greedy:.0f}')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title('Initial vs Final Scores (shows improvement potential)')
    ax.set_xticks(x)
    ax.set_xticklabels(meta_algos, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "plots" / "initial_vs_final_scores.png", dpi=300)
    plt.close()
    print("\n✓ Saved initial_vs_final_scores.png")


if __name__ == "__main__":
    analyze_convergence_issue()
    plot_initial_vs_final_scores()
    
    print("\n" + "="*80)
    print("Run complete! Check the diagnostic output above.")
    print("="*80)