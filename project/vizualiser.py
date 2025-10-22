import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import ast
from pathlib import Path

class ExperimentVisualizer:
    """Individual visualization and analysis methods for experiments."""

    def __init__(self, results, output_dir="project/results"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.df = pd.DataFrame([
            {
                "algorithm": r.algorithm,
                "score": r.final_score,
                "runtime": r.runtime_seconds,
                "config": str(r.parameters),
                "convergence": r.convergence_history
            }
            for r in self.results
        ])
        # Parse parameters
        try:
            self.df["params_dict"] = self.df["config"].apply(lambda x: ast.literal_eval(str(x)) if x else {})
        except:
            self.df["params_dict"] = [{} for _ in range(len(self.df))]

    def plot_score_distribution(self):
        """Boxplot comparing algorithm scores"""
        plt.figure(figsize=(12,6))
        sns.boxplot(data=self.df, x="algorithm", y="score")
        plt.title("Score Distribution by Algorithm")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "boxplot_comparison.png", dpi=300)
        plt.close()
        print("✓ Saved boxplot_comparison.png")

    def plot_runtime_vs_score(self):
        """Scatter plot of runtime vs score"""
        plt.figure(figsize=(10,6))
        for algo in self.df["algorithm"].unique():
            algo_df = self.df[self.df["algorithm"] == algo]
            plt.scatter(algo_df["runtime"], algo_df["score"], label=algo, alpha=0.6, s=100)
        plt.xlabel("Runtime (s)")
        plt.ylabel("Score")
        plt.title("Score vs Runtime")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_vs_score.png", dpi=300)
        plt.close()
        print("✓ Saved runtime_vs_score.png")

    def plot_convergence_curves(self, max_runs=5):
        """Plot convergence curves per algorithm"""
        algorithms = self.df["algorithm"].unique()
        fig, axes = plt.subplots(2, 2, figsize=(15,12))
        axes = axes.flatten()
        for idx, algo in enumerate(algorithms):
            ax = axes[idx]
            algo_results = self.df[self.df["algorithm"] == algo]
            for conv in algo_results["convergence"][:max_runs]:
                if conv:
                    times = [h[2] for h in conv]
                    scores = [h[1] for h in conv]
                    ax.plot(times, scores, alpha=0.3)
            ax.set_title(f"{algo} Convergence")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "convergence_curves.png", dpi=300)
        plt.close()
        print("✓ Saved convergence_curves.png")

    def plot_average_performance(self):
        """Bar chart of mean scores with std"""
        means = self.df.groupby("algorithm")["score"].mean().sort_values(ascending=False)
        stds = self.df.groupby("algorithm")["score"].std()
        plt.figure(figsize=(10,6))
        plt.bar(range(len(means)), means.values, yerr=stds.values, capsize=5, alpha=0.7)
        plt.xticks(range(len(means)), means.index, rotation=45)
        plt.ylabel("Average Score")
        plt.title("Average Performance by Algorithm")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "average_performance.png", dpi=300)
        plt.close()
        print("✓ Saved average_performance.png")

    def plot_efficiency(self):
        """Bar chart of score per second (efficiency)"""
        self.df["score_per_second"] = self.df["score"] / self.df["runtime"]
        plt.figure(figsize=(8,6))
        sns.barplot(data=self.df, x="algorithm", y="score_per_second")
        plt.title("Efficiency (Score per Second)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_barplot.png", dpi=300)
        plt.close()
        print("✓ Saved efficiency_barplot.png")

    def plot_parameter_sensitivity(self):
        """Heatmap of correlation between parameters and score"""
        param_df = self.df.join(pd.json_normalize(self.df["params_dict"]))
        if len(param_df.columns) > 6:  # avoid empty heatmap
            corr = param_df.corr(numeric_only=True)[["score"]].sort_values("score", ascending=False)
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="viridis")
            plt.title("Parameter Sensitivity")
            plt.tight_layout()
            plt.savefig(self.output_dir / "parameter_sensitivity_heatmap.png", dpi=300)
            plt.close()
            print("✓ Saved parameter_sensitivity_heatmap.png")

    def save_algorithm_ranking(self):
        """Save CSV with algorithm ranking by mean score"""
        rank_df = self.df.groupby("algorithm")["score"].mean().sort_values(ascending=False).reset_index()
        rank_df["rank"] = range(1, len(rank_df)+1)
        rank_df.to_csv(self.output_dir / "algorithm_ranking.csv", index=False)
        print("✓ Saved algorithm_ranking.csv")
        print(rank_df)

    def statistical_significance_tests(self):
        """Perform t-tests and ANOVA between algorithms"""
        algos = self.df["algorithm"].unique()
        print("\n--- Statistical Significance Tests ---")
        for i, a1 in enumerate(algos):
            for a2 in algos[i+1:]:
                s1 = self.df[self.df["algorithm"] == a1]["score"]
                s2 = self.df[self.df["algorithm"] == a2]["score"]
                t, p = ttest_ind(s1, s2, equal_var=False)
                print(f"{a1} vs {a2} → p={p:.4f}")
        groups = [self.df[self.df["algorithm"]==a]["score"] for a in algos]
        F, p = f_oneway(*groups)
        print(f"ANOVA → F={F:.3f}, p={p:.4f}")
