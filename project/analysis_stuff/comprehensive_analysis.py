import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
from pathlib import Path
import ast

class ComprehensiveAnalyzer:
    """Complete analysis for drone delivery optimization experiments."""
    
    def __init__(self, results_dir="project/results"):
        self.dir = Path(results_dir)
        self.plots_dir = self.dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        csv_path = self.dir / "summary_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Identify greedy vs metaheuristic algorithms
        self.greedy_algos = [a for a in self.df["algorithm"].unique() 
                            if "Greedy" in a or "greedy" in a]
        self.meta_algos = [a for a in self.df["algorithm"].unique() 
                          if a not in self.greedy_algos]
        
        print(f"📊 Loaded {len(self.df)} experiment results")
        print(f"   Metaheuristics: {', '.join(self.meta_algos)}")
        if self.greedy_algos:
            print(f"   Greedy baselines: {', '.join(self.greedy_algos)}")
        
        # Parse parameters
        param_col = self._find_param_column()
        if param_col:
            try:
                self.df["params_dict"] = self.df[param_col].apply(
                    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.strip() else {}
                )
            except Exception as e:
                print(f"⚠️ Could not parse {param_col}: {e}")
                self.df["params_dict"] = [{} for _ in range(len(self.df))]
        else:
            self.df["params_dict"] = [{} for _ in range(len(self.df))]
        
        # Standardize column names
        self._standardize_columns()
    
    def _find_param_column(self):
        """Auto-detect parameter column."""
        candidates = ["params", "parameters", "config", "hyperparams"]
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate
        return None
    
    def _standardize_columns(self):
        """Ensure consistent column names."""
        if "final_score" in self.df.columns and "score" not in self.df.columns:
            self.df["score"] = self.df["final_score"]
        elif "score" in self.df.columns and "final_score" not in self.df.columns:
            self.df["final_score"] = self.df["score"]
        
        if "runtime_seconds" in self.df.columns and "runtime" not in self.df.columns:
            self.df["runtime"] = self.df["runtime_seconds"]
        elif "runtime" in self.df.columns and "runtime_seconds" not in self.df.columns:
            self.df["runtime_seconds"] = self.df["runtime"]
    
    # ============= DESCRIPTIVE STATISTICS =============
    
    def summary_statistics(self):
        """Comprehensive summary statistics by algorithm."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS BY ALGORITHM")
        print("="*70)
        
        grouped = self.df.groupby("algorithm")["score"].agg([
            ("Mean", "mean"),
            ("Std", "std"),
            ("Median", "median"),
            ("Min", "min"),
            ("Max", "max"),
            ("Q1", lambda x: x.quantile(0.25)),
            ("Q3", lambda x: x.quantile(0.75)),
            ("Count", "count"),
            ("CV", lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
        ])
        
        grouped["IQR"] = grouped["Q3"] - grouped["Q1"]
        grouped = grouped.sort_values("Mean", ascending=False)
        
        print(grouped.round(2))
        
        # Separate analysis for greedy vs metaheuristics
        if self.greedy_algos:
            print("\n" + "-"*70)
            print("GREEDY BASELINES vs METAHEURISTICS")
            print("-"*70)
            
            greedy_scores = grouped.loc[[a for a in grouped.index if a in self.greedy_algos]]
            meta_scores = grouped.loc[[a for a in grouped.index if a in self.meta_algos]]
            
            if not greedy_scores.empty:
                best_greedy = greedy_scores["Mean"].max()
                best_greedy_name = greedy_scores["Mean"].idxmax()
                print(f"\nBest Greedy: {best_greedy_name} = {best_greedy:.2f}")
                
                if not meta_scores.empty:
                    best_meta = meta_scores["Mean"].max()
                    best_meta_name = meta_scores["Mean"].idxmax()
                    worst_meta = meta_scores["Mean"].min()
                    worst_meta_name = meta_scores["Mean"].idxmin()
                    
                    print(f"Best Metaheuristic: {best_meta_name} = {best_meta:.2f}")
                    print(f"Worst Metaheuristic: {worst_meta_name} = {worst_meta:.2f}")
                    
                    improvement_best = ((best_meta - best_greedy) / best_greedy) * 100
                    improvement_worst = ((worst_meta - best_greedy) / best_greedy) * 100
                    
                    print(f"\nImprovement over best greedy:")
                    print(f"  Best metaheuristic: {improvement_best:+.2f}%")
                    print(f"  Worst metaheuristic: {improvement_worst:+.2f}%")
                    
                    if improvement_worst < 0:
                        print(f"\n  ⚠️ WARNING: {worst_meta_name} performs WORSE than greedy!")
                        print(f"     Likely causes:")
                        print(f"     1. Algorithm hasn't converged (increase iterations/time)")
                        print(f"     2. Poor parameter tuning")
                        print(f"     3. Weak initial solution")
                        print(f"     → Check convergence curves to diagnose")
                    elif improvement_best < 5:
                        print(f"  ⚠️ Note: Modest improvement (<5%) - complexity may not be justified")
                    else:
                        print(f"  ✓ Metaheuristics provide meaningful improvement")
        
        # Save to CSV
        grouped.to_csv(self.dir / "summary_statistics.csv")
        print(f"\n✓ Saved to summary_statistics.csv")
        
        return grouped
    
    def computational_efficiency(self):
        """Analyze computational efficiency metrics."""
        print("\n" + "="*70)
        print("COMPUTATIONAL EFFICIENCY ANALYSIS")
        print("="*70)
        
        efficiency_df = self.df.groupby("algorithm").agg({
            "runtime": ["mean", "std"],
            "score": ["mean", "std"],
            "iterations": ["mean", "std"] if "iterations" in self.df.columns else "count"
        }).round(2)
        
        self.df["score_per_sec"] = self.df["score"] / self.df["runtime"]
        
        if "iterations" in self.df.columns:
            self.df["score_per_iter"] = self.df["score"] / self.df["iterations"]
        
        efficiency_summary = self.df.groupby("algorithm").agg({
            "score_per_sec": ["mean", "std"],
            "score_per_iter": ["mean", "std"] if "iterations" in self.df.columns else "count"
        }).round(2)
        
        print("\nRuntime & Score Summary:")
        print(efficiency_df)
        
        print("\nEfficiency Metrics:")
        print(efficiency_summary)
        
        efficiency_summary.to_csv(self.dir / "efficiency_metrics.csv")
        print(f"\n✓ Saved to efficiency_metrics.csv")
        
        return efficiency_summary
    
    # ============= STATISTICAL TESTS =============
    
    def statistical_tests(self):
        """Comprehensive statistical significance testing."""
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*70)
        
        algos = self.df["algorithm"].unique()
        results = []
        
        # Pairwise t-tests
        print("\nPairwise t-tests (Welch's t-test):")
        print("-" * 70)
        
        for i, a1 in enumerate(algos):
            for a2 in algos[i+1:]:
                s1 = self.df[self.df["algorithm"] == a1]["score"]
                s2 = self.df[self.df["algorithm"] == a2]["score"]
                
                # Skip if sample size too small
                if len(s1) < 2 or len(s2) < 2:
                    continue
                
                # Welch's t-test
                t_stat, p_val = ttest_ind(s1, s2, equal_var=False)
                
                # Mann-Whitney U test
                u_stat, p_val_mw = mannwhitneyu(s1, s2, alternative='two-sided')
                
                mean_diff = s1.mean() - s2.mean()
                
                results.append({
                    "Algorithm 1": a1,
                    "Algorithm 2": a2,
                    "Mean Diff": round(mean_diff, 2),
                    "t-statistic": round(t_stat, 4),
                    "p-value (t)": round(p_val, 4),
                    "p-value (MW)": round(p_val_mw, 4),
                    "Significant (α=0.05)": "Yes" if p_val < 0.05 else "No"
                })
                
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"{a1:25s} vs {a2:25s} | diff={mean_diff:8.1f} | p={p_val:.4f} {sig_marker}")
        
        # ANOVA test (only for metaheuristics with multiple runs)
        if len(self.meta_algos) > 1:
            print("\n" + "-" * 70)
            print("One-way ANOVA for Metaheuristics:")
            meta_groups = [self.df[self.df["algorithm"] == a]["score"] 
                          for a in self.meta_algos 
                          if len(self.df[self.df["algorithm"] == a]) > 1]
            
            if len(meta_groups) > 1:
                f_stat, p_val_anova = f_oneway(*meta_groups)
                print(f"F-statistic: {f_stat:.3f}")
                print(f"p-value: {p_val_anova:.6f}")
                
                if p_val_anova < 0.05:
                    print("→ At least one metaheuristic differs significantly (α=0.05)")
                else:
                    print("→ No significant differences among metaheuristics")
        
        # Special comparison: Best meta vs best greedy
        if self.greedy_algos and self.meta_algos:
            print("\n" + "-" * 70)
            print("BEST METAHEURISTIC vs BEST GREEDY BASELINE:")
            
            # Find best of each category
            greedy_means = {a: self.df[self.df["algorithm"]==a]["score"].mean() 
                           for a in self.greedy_algos}
            meta_means = {a: self.df[self.df["algorithm"]==a]["score"].mean() 
                         for a in self.meta_algos}
            
            best_greedy_name = max(greedy_means, key=greedy_means.get)
            best_meta_name = max(meta_means, key=meta_means.get)
            
            s_greedy = self.df[self.df["algorithm"] == best_greedy_name]["score"]
            s_meta = self.df[self.df["algorithm"] == best_meta_name]["score"]
            
            if len(s_meta) >= 2:  # Need multiple runs for stats
                t_stat, p_val = ttest_ind(s_meta, s_greedy, equal_var=False)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                print(f"{best_meta_name} vs {best_greedy_name}")
                print(f"  Mean difference: {s_meta.mean() - s_greedy.mean():.2f}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_val:.4f} {sig}")
                
                if p_val < 0.05:
                    print(f"  ✓ Metaheuristic significantly better than greedy")
                else:
                    print(f"  ⚠️ No significant improvement over greedy baseline")
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.dir / "statistical_tests.csv", index=False)
            print(f"\n✓ Saved detailed results to statistical_tests.csv")
        
        return results
    
    # ============= VISUALIZATIONS =============
    
    def plot_score_distribution(self):
        """Box plot with ALL algorithms including greedy."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Sort: greedy first, then meta by mean score
        greedy_sorted = sorted([a for a in self.df["algorithm"].unique() if a in self.greedy_algos])
        meta_means = self.df[self.df["algorithm"].isin(self.meta_algos)].groupby("algorithm")["score"].mean()
        meta_sorted = meta_means.sort_values(ascending=False).index.tolist()
        algo_order = greedy_sorted + meta_sorted
        
        # Colors
        palette = ['#FF6B6B' if a in self.greedy_algos else '#4ECDC4' for a in algo_order]
        
        sns.boxplot(data=self.df, x="algorithm", y="score", ax=ax, 
                   palette=palette, order=algo_order)
        
        sns.stripplot(data=self.df, x="algorithm", y="score", ax=ax, 
                     color="black", alpha=0.3, size=4, order=algo_order)
        
        ax.set_title("Score Distribution: All Algorithms (Red=Greedy, Teal=Metaheuristics)", 
                    fontsize=14, fontweight="bold")
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Greedy Baselines'),
            Patch(facecolor='#4ECDC4', label='Metaheuristics')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "score_distribution_all.png", dpi=300)
        plt.close()
        print("✓ Saved score_distribution_all.png")
        
        # Also create metaheuristics-only version (original style)
        if self.meta_algos:
            fig, ax = plt.subplots(figsize=(12, 6))
            meta_df = self.df[self.df["algorithm"].isin(self.meta_algos)]
            
            sns.boxplot(data=meta_df, x="algorithm", y="score", ax=ax, palette="Set2")
            sns.stripplot(data=meta_df, x="algorithm", y="score", ax=ax, 
                         color="black", alpha=0.3, size=4)
            
            ax.set_title("Score Distribution by Metaheuristic Algorithm", fontsize=14, fontweight="bold")
            ax.set_xlabel("Algorithm", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "boxplot_comparison.png", dpi=300)
            plt.close()
            print("✓ Saved boxplot_comparison.png (metaheuristics only)")
    
    def plot_performance_comparison(self):
        """Bar chart with ALL algorithms."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        summary = self.df.groupby("algorithm")["score"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        
        colors = ['#FF6B6B' if algo in self.greedy_algos else '#4ECDC4' 
                 for algo in summary.index]
        
        bars = ax.bar(range(len(summary)), summary["mean"], 
                     yerr=summary["std"], capsize=5, alpha=0.7, color=colors)
        
        for i, (bar, mean_val) in enumerate(zip(bars, summary["mean"])):
            height = bar.get_height() + summary["std"].iloc[i] + (summary["mean"].max() * 0.01)
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{mean_val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary.index, rotation=45, ha='right')
        ax.set_ylabel("Mean Score", fontsize=12)
        ax.set_title("Algorithm Performance Comparison (Mean ± Std)", fontsize=14, fontweight="bold")
        ax.grid(axis='y', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', alpha=0.7, label='Metaheuristics'),
            Patch(facecolor='#FF6B6B', alpha=0.7, label='Greedy Baselines')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if self.greedy_algos:
            best_greedy = summary.loc[[a for a in summary.index if a in self.greedy_algos]]["mean"].max()
            ax.axhline(y=best_greedy, color='red', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Best Greedy: {best_greedy:.0f}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300)
        plt.close()
        print("✓ Saved performance_comparison.png")
        
        # Also create average performance (metaheuristics only, original style)
        if self.meta_algos:
            fig, ax = plt.subplots(figsize=(10, 6))
            meta_summary = summary.loc[[a for a in summary.index if a in self.meta_algos]]
            
            plt.bar(range(len(meta_summary)), meta_summary["mean"], 
                   yerr=meta_summary["std"], capsize=5, alpha=0.7)
            plt.xticks(range(len(meta_summary)), meta_summary.index, rotation=45)
            plt.ylabel("Average Score")
            plt.title("Average Performance by Algorithm")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "average_performance.png", dpi=300)
            plt.close()
            print("✓ Saved average_performance.png (metaheuristics only)")
    
    def plot_runtime_vs_score(self):
        """Scatter plot including greedy."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in self.df["algorithm"].unique():
            algo_df = self.df[self.df["algorithm"] == algo]
            color = '#FF6B6B' if algo in self.greedy_algos else None
            ax.scatter(algo_df["runtime"], algo_df["score"], label=algo, alpha=0.6, s=100, color=color)
        
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Score")
        ax.set_title("Score vs Runtime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "runtime_vs_score.png", dpi=300)
        plt.close()
        print("✓ Saved runtime_vs_score.png")
    
    def plot_efficiency(self):
        """Efficiency bar plot with ALL algorithms."""
        if "score_per_sec" not in self.df.columns:
            self.df["score_per_sec"] = self.df["score"] / self.df["runtime"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort order
        algo_order = sorted(self.greedy_algos) + sorted(self.meta_algos)
        colors = ['#FF6B6B' if a in self.greedy_algos else '#4ECDC4' for a in algo_order]
        
        plot_df = self.df[self.df["algorithm"].isin(algo_order)]
        
        sns.barplot(data=plot_df, x="algorithm", y="score_per_sec", 
                   order=algo_order, palette=colors, ax=ax)
        ax.set_title("Efficiency (Score per Second)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "efficiency_barplot.png", dpi=300)
        plt.close()
        print("✓ Saved efficiency_barplot.png")
    
    def plot_runtime_analysis(self):
        """4-panel runtime analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Runtime distribution
        sns.boxplot(data=self.df, x="algorithm", y="runtime", ax=axes[0, 0], palette="Set3")
        axes[0, 0].set_title("Runtime Distribution", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel("Runtime (seconds)")
        
        # 2. Score vs Runtime scatter
        colors_map = {algo: '#FF6B6B' if algo in self.greedy_algos else None
                     for algo in self.df["algorithm"].unique()}
        
        for algo in self.df["algorithm"].unique():
            algo_df = self.df[self.df["algorithm"] == algo]
            axes[0, 1].scatter(algo_df["runtime"], algo_df["score"], 
                             label=algo, alpha=0.6, s=80, color=colors_map[algo])
        axes[0, 1].set_xlabel("Runtime (seconds)")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Score vs Runtime", fontweight="bold")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Efficiency
        if "score_per_sec" not in self.df.columns:
            self.df["score_per_sec"] = self.df["score"] / self.df["runtime"]
        
        sns.barplot(data=self.df, x="algorithm", y="score_per_sec", ax=axes[1, 0], 
                   palette="Set1", errorbar="sd")
        axes[1, 0].set_title("Efficiency (Score per Second)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylabel("Score / Second")
        
        # 4. Runtime-Score trade-off
        summary = self.df.groupby("algorithm").agg({"runtime": "mean", "score": "mean"})
        
        for algo, row in summary.iterrows():
            color = '#FF6B6B' if algo in self.greedy_algos else '#4ECDC4'
            marker = 's' if algo in self.greedy_algos else 'o'
            axes[1, 1].scatter(row["runtime"], row["score"], s=200, alpha=0.6, 
                             color=color, marker=marker)
            axes[1, 1].annotate(algo, (row["runtime"], row["score"]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel("Mean Runtime (seconds)")
        axes[1, 1].set_ylabel("Mean Score")
        axes[1, 1].set_title("Runtime-Score Trade-off (□=Greedy, ●=Meta)", fontweight="bold")
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "runtime_analysis.png", dpi=300)
        plt.close()
        print("✓ Saved runtime_analysis.png")
    
    def plot_convergence_curves(self, max_runs=5):
        """Convergence curves (metaheuristics only)."""
        if "convergence_history" not in self.df.columns:
            print("⚠️  No convergence history available")
            return
        
        plot_algos = self.meta_algos if self.meta_algos else self.df["algorithm"].unique()
        
        n_algos = len(plot_algos)
        n_cols = 2
        n_rows = (n_algos + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, algo in enumerate(plot_algos):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            algo_df = self.df[self.df["algorithm"] == algo]
            
            for i, row in algo_df.head(max_runs).iterrows():
                history = row["convergence_history"]
                if isinstance(history, str):
                    try:
                        history = ast.literal_eval(history)
                    except:
                        continue
                
                if history and len(history) > 0:
                    times = [h[2] for h in history]
                    scores = [h[1] for h in history]
                    ax.plot(times, scores, alpha=0.4, linewidth=2)
            
            ax.set_title(f"{algo} Convergence", fontweight="bold")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Best Score Found")
            ax.grid(True, alpha=0.3)
            
            # Add greedy baseline reference if available
            if self.greedy_algos:
                best_greedy = self.df[self.df["algorithm"].isin(self.greedy_algos)]["score"].max()
                ax.axhline(y=best_greedy, color='red', linestyle='--', alpha=0.5, linewidth=1,
                          label=f'Best Greedy: {best_greedy:.0f}')
                ax.legend(fontsize=8)
        
        for idx in range(len(plot_algos), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "convergence_curves.png", dpi=300)
        plt.close()
        print("✓ Saved convergence_curves.png")
    
    def plot_parameter_sensitivity(self):
        """Parameter sensitivity heatmap."""
        param_df = self.df.join(pd.json_normalize(self.df["params_dict"]))
        if len(param_df.columns) > 6:
            corr = param_df.corr(numeric_only=True)[["score"]].sort_values("score", ascending=False)
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="viridis")
            plt.title("Parameter Sensitivity")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "parameter_sensitivity_heatmap.png", dpi=300)
            plt.close()
            print("✓ Saved parameter_sensitivity_heatmap.png")
    
    def plot_algorithm_ranking(self):
        """Ranking with confidence intervals, ALL algorithms."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        summary = self.df.groupby("algorithm")["score"].agg(["mean", "std", "count"])
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci"] = 1.96 * summary["se"]
        summary = summary.sort_values("mean", ascending=True)
        
        colors = ['#FF6B6B' if algo in self.greedy_algos else '#4ECDC4' 
                 for algo in summary.index]
        
        y_pos = range(len(summary))
        ax.barh(y_pos, summary["mean"], xerr=summary["ci"], 
               capsize=5, alpha=0.7, color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index)
        ax.set_xlabel("Mean Score (with 95% CI)", fontsize=12)
        ax.set_title("Algorithm Ranking (Red=Greedy, Teal=Metaheuristics)", fontsize=14, fontweight="bold")
        ax.grid(axis='x', alpha=0.3)
        
        for i, (algo, row) in enumerate(summary[::-1].iterrows()):
            x_pos = min(summary["mean"]) * 0.95
            ax.text(x_pos, len(summary)-i-1, f"#{i+1}", 
                   fontsize=11, fontweight='bold', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "algorithm_ranking.png", dpi=300)
        plt.close()
        print("✓ Saved algorithm_ranking.png")
        
        # Save ranking CSV
        ranking = summary.reset_index()
        ranking["rank"] = range(len(ranking), 0, -1)
        ranking["category"] = ranking["algorithm"].apply(
            lambda x: "Greedy" if x in self.greedy_algos else "Metaheuristic"
        )
        ranking = ranking.sort_values("rank")
        ranking.to_csv(self.dir / "algorithm_ranking.csv", index=False)
        print("✓ Saved algorithm_ranking.csv")
    
    def save_algorithm_ranking(self):
        """Save ranking CSV (for backward compatibility)."""
        summary = self.df.groupby("algorithm")["score"].agg(["mean"]).sort_values("mean", ascending=False)
        ranking = summary.reset_index()
        ranking["rank"] = range(1, len(ranking) + 1)
        ranking.to_csv(self.dir / "algorithm_ranking_simple.csv", index=False)
        print("✓ Saved algorithm_ranking_simple.csv")
    
    def statistical_significance_tests(self):
        """Alias for backward compatibility."""
        return self.statistical_tests()
    
    # ============= MAIN RUNNER =============
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        print("\n" + "="*70)
        print("COMPREHENSIVE DRONE DELIVERY OPTIMIZATION ANALYSIS")
        print("="*70)
        
        # Statistics
        self.summary_statistics()
        self.computational_efficiency()
        self.statistical_tests()
        
        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_score_distribution()
        self.plot_performance_comparison()
        self.plot_runtime_vs_score()
        self.plot_efficiency()
        self.plot_runtime_analysis()
        self.plot_convergence_curves()
        self.plot_parameter_sensitivity()
        self.plot_algorithm_ranking()
        self.save_algorithm_ranking()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print(f"✓ Results saved to: {self.dir}")
        print(f"✓ Plots saved to: {self.plots_dir}")
        print("="*70)


if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer("project/results")
    analyzer.run_full_analysis()