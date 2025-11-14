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
            ("CV", lambda x: x.std() / x.mean() if x.mean() != 0 else 0)  # Coefficient of variation
        ])
        
        # Add IQR
        grouped["IQR"] = grouped["Q3"] - grouped["Q1"]
        
        # Sort by mean score
        grouped = grouped.sort_values("Mean", ascending=False)
        
        print(grouped.round(2))
        
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
        
        # Calculate score per second
        self.df["score_per_sec"] = self.df["score"] / self.df["runtime"]
        
        # Calculate score per iteration
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
                
                # Welch's t-test (doesn't assume equal variance)
                t_stat, p_val = ttest_ind(s1, s2, equal_var=False)
                
                # Mann-Whitney U test (non-parametric alternative)
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
                print(f"{a1:20s} vs {a2:20s} | diff={mean_diff:7.1f} | p={p_val:.4f} {sig_marker}")
        
        # ANOVA test
        print("\n" + "-" * 70)
        print("One-way ANOVA (tests if ANY algorithm differs):")
        groups = [self.df[self.df["algorithm"] == a]["score"] for a in algos]
        f_stat, p_val_anova = f_oneway(*groups)
        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_val_anova:.6f}")
        
        if p_val_anova < 0.05:
            print("→ At least one algorithm performs significantly differently (α=0.05)")
        else:
            print("→ No significant differences detected between algorithms")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.dir / "statistical_tests.csv", index=False)
        print(f"\n✓ Saved detailed results to statistical_tests.csv")
        
        return results_df
    
    # ============= VISUALIZATIONS =============
    
    def plot_score_distribution(self):
        """Box plot with individual points overlay."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot
        sns.boxplot(data=self.df, x="algorithm", y="score", ax=ax, palette="Set2")
        
        # Overlay individual points
        sns.stripplot(data=self.df, x="algorithm", y="score", ax=ax, 
                     color="black", alpha=0.3, size=4)
        
        ax.set_title("Score Distribution by Algorithm", fontsize=14, fontweight="bold")
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "score_distribution.png", dpi=300)
        plt.close()
        print("✓ Saved score_distribution.png")
    
    def plot_performance_comparison(self):
        """Bar chart with error bars and annotations."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        summary = self.df.groupby("algorithm")["score"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        
        bars = ax.bar(range(len(summary)), summary["mean"], 
                     yerr=summary["std"], capsize=5, alpha=0.7, 
                     color=sns.color_palette("Set2", len(summary)))
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, summary["mean"])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + summary["std"].iloc[i] + 5,
                   f'{mean_val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary.index, rotation=45, ha='right')
        ax.set_ylabel("Mean Score", fontsize=12)
        ax.set_title("Algorithm Performance Comparison (Mean ± Std)", fontsize=14, fontweight="bold")
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300)
        plt.close()
        print("✓ Saved performance_comparison.png")
    
    def plot_runtime_analysis(self):
        """Multiple subplots for runtime analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Runtime distribution
        sns.boxplot(data=self.df, x="algorithm", y="runtime", ax=axes[0, 0], palette="Set3")
        axes[0, 0].set_title("Runtime Distribution", fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel("Runtime (seconds)")
        
        # 2. Score vs Runtime scatter
        for algo in self.df["algorithm"].unique():
            algo_df = self.df[self.df["algorithm"] == algo]
            axes[0, 1].scatter(algo_df["runtime"], algo_df["score"], 
                             label=algo, alpha=0.6, s=80)
        axes[0, 1].set_xlabel("Runtime (seconds)")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Score vs Runtime", fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Efficiency (score per second)
        if "score_per_sec" not in self.df.columns:
            self.df["score_per_sec"] = self.df["score"] / self.df["runtime"]
        
        sns.barplot(data=self.df, x="algorithm", y="score_per_sec", ax=axes[1, 0], 
                   palette="Set1", errorbar="sd")
        axes[1, 0].set_title("Efficiency (Score per Second)", fontweight="bold")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylabel("Score / Second")
        
        # 4. Runtime vs Score trade-off
        summary = self.df.groupby("algorithm").agg({"runtime": "mean", "score": "mean"})
        axes[1, 1].scatter(summary["runtime"], summary["score"], s=200, alpha=0.6)
        
        for algo, row in summary.iterrows():
            axes[1, 1].annotate(algo, (row["runtime"], row["score"]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel("Mean Runtime (seconds)")
        axes[1, 1].set_ylabel("Mean Score")
        axes[1, 1].set_title("Runtime-Score Trade-off", fontweight="bold")
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "runtime_analysis.png", dpi=300)
        plt.close()
        print("✓ Saved runtime_analysis.png")
    
    def plot_convergence_curves(self, max_runs=5):
        """Convergence curves from convergence history."""
        if "convergence_history" not in self.df.columns:
            print("⚠️  No convergence history available")
            return
        
        algorithms = self.df["algorithm"].unique()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, algo in enumerate(algorithms):
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
        
        # Hide unused subplots
        for idx in range(len(algorithms), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "convergence_curves.png", dpi=300)
        plt.close()
        print("✓ Saved convergence_curves.png")
    
    def plot_algorithm_ranking(self):
        """Visual ranking with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        summary = self.df.groupby("algorithm")["score"].agg(["mean", "std", "count"])
        summary["se"] = summary["std"] / np.sqrt(summary["count"])  # Standard error
        summary["ci"] = 1.96 * summary["se"]  # 95% confidence interval
        summary = summary.sort_values("mean", ascending=True)  # Low to high for horizontal
        
        y_pos = range(len(summary))
        ax.barh(y_pos, summary["mean"], xerr=summary["ci"], 
               capsize=5, alpha=0.7, color=sns.color_palette("viridis", len(summary)))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index)
        ax.set_xlabel("Mean Score (with 95% CI)", fontsize=12)
        ax.set_title("Algorithm Ranking", fontsize=14, fontweight="bold")
        ax.grid(axis='x', alpha=0.3)
        
        # Add rank numbers
        for i, (algo, row) in enumerate(summary[::-1].iterrows()):
            ax.text(5, len(summary)-i-1, f"#{i+1}", 
                   fontsize=10, fontweight='bold', va='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "algorithm_ranking.png", dpi=300)
        plt.close()
        print("✓ Saved algorithm_ranking.png")
        
        # Save ranking CSV
        ranking = summary.reset_index()
        ranking["rank"] = range(len(ranking), 0, -1)
        ranking = ranking.sort_values("rank")
        ranking.to_csv(self.dir / "algorithm_ranking.csv", index=False)
        print("✓ Saved algorithm_ranking.csv")
    
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
        self.plot_runtime_analysis()
        self.plot_convergence_curves()
        self.plot_algorithm_ranking()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print(f"✓ Results saved to: {self.dir}")
        print(f"✓ Plots saved to: {self.plots_dir}")
        print("="*70)


if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer("project/results")
    analyzer.run_full_analysis()