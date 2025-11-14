import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import ast
from pathlib import Path

class ExperimentVisualizerAnalyzer:
    """Unified visualizer and analyzer for experiment results."""

    def __init__(self, results=None, results_dir="project/results"):
        """
        Initialize with either:
        - results: list of ExperimentResult objects (for convergence curves)
        - results_dir: directory containing summary_results.csv
        """
        self.results_dir = Path(results_dir)
        self.results = results

        # Load CSV if it exists
        csv_path = self.results_dir / "summary_results.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
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

        # Parse parameter dictionaries
        param_col = None
        for candidate in ["params", "parameters", "config", "hyperparams"]:
            if candidate in self.df.columns:
                param_col = candidate
                break
        if param_col:
            try:
                self.df["params_dict"] = self.df[param_col].apply(
                    lambda x: ast.literal_eval(str(x)) if isinstance(x, str) and x.strip() else {}
                )
            except:
                self.df["params_dict"] = [{} for _ in range(len(self.df))]
        else:
            self.df["params_dict"] = [{} for _ in range(len(self.df))]

    # ----------------- Visualization Methods -----------------

    def plot_score_distribution(self):
        plt.figure(figsize=(12,6))
        sns.boxplot(data=self.df, x="algorithm", y="score" if "score" in self.df.columns else "final_score")
        plt.title("Score Distribution by Algorithm")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "boxplot_comparison.png", dpi=300)
        plt.close()
        print("✓ Saved boxplot_comparison.png")

    def plot_runtime_vs_score(self):
        plt.figure(figsize=(10,6))
        for algo in self.df["algorithm"].unique():
            algo_df = self.df[self.df["algorithm"] == algo]
            plt.scatter(algo_df["runtime"] if "runtime" in algo_df.columns else algo_df["runtime_seconds"],
                        algo_df["score"] if "score" in algo_df.columns else algo_df["final_score"],
                        label=algo, alpha=0.6, s=100)
        plt.xlabel("Runtime (s)")
        plt.ylabel("Score")
        plt.title("Score vs Runtime")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "runtime_vs_score.png", dpi=300)
        plt.close()
        print("✓ Saved runtime_vs_score.png")

    def plot_convergence_curves(self, max_runs=5):
        if self.results is None:
            print("⚠️ No ExperimentResult objects provided for convergence curves.")
            return
        algorithms = list({r.algorithm for r in self.results})
        fig, axes = plt.subplots(2, 2, figsize=(15,12))
        axes = axes.flatten()
        for idx, algo in enumerate(algorithms):
            ax = axes[idx]
            algo_results = [r for r in self.results if r.algorithm == algo]
            for r in algo_results[:max_runs]:
                if r.convergence_history:
                    times = [h[2] for h in r.convergence_history]
                    scores = [h[1] for h in r.convergence_history]
                    ax.plot(times, scores, alpha=0.3)
            ax.set_title(f"{algo} Convergence")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "convergence_curves.png", dpi=300)
        plt.close()
        print("✓ Saved convergence_curves.png")

    def plot_average_performance(self):
        means = self.df.groupby("algorithm")["score" if "score" in self.df.columns else "final_score"].mean().sort_values(ascending=False)
        stds = self.df.groupby("algorithm")["score" if "score" in self.df.columns else "final_score"].std()
        plt.figure(figsize=(10,6))
        plt.bar(range(len(means)), means.values, yerr=stds.values, capsize=5, alpha=0.7)
        plt.xticks(range(len(means)), means.index, rotation=45)
        plt.ylabel("Average Score")
        plt.title("Average Performance by Algorithm")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "average_performance.png", dpi=300)
        plt.close()
        print("✓ Saved average_performance.png")

    def plot_efficiency(self):
        score_col = "score" if "score" in self.df.columns else "final_score"
        runtime_col = "runtime" if "runtime" in self.df.columns else "runtime_seconds"
        self.df["score_per_second"] = self.df[score_col] / self.df[runtime_col]
        plt.figure(figsize=(8,6))
        sns.barplot(data=self.df, x="algorithm", y="score_per_second")
        plt.title("Efficiency (Score per Second)")
        plt.tight_layout()
        plt.savefig(self.results_dir / "efficiency_barplot.png", dpi=300)
        plt.close()
        print("✓ Saved efficiency_barplot.png")

    def plot_parameter_sensitivity(self):
        param_df = self.df.join(pd.json_normalize(self.df["params_dict"]))
        if len(param_df.columns) > 6:  # avoid empty heatmap
            score_col = "score" if "score" in param_df.columns else "final_score"
            corr = param_df.corr(numeric_only=True)[[score_col]].sort_values(score_col, ascending=False)
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="viridis")
            plt.title("Parameter Sensitivity")
            plt.tight_layout()
            plt.savefig(self.results_dir / "parameter_sensitivity_heatmap.png", dpi=300)
            plt.close()
            print("✓ Saved parameter_sensitivity_heatmap.png")

    # ----------------- Statistical Methods -----------------

    def save_algorithm_ranking(self):
        score_col = "score" if "score" in self.df.columns else "final_score"
        rank_df = self.df.groupby("algorithm")[score_col].mean().sort_values(ascending=False).reset_index()
        rank_df["rank"] = range(1, len(rank_df)+1)
        rank_df.to_csv(self.results_dir / "algorithm_ranking.csv", index=False)
        print("✓ Saved algorithm_ranking.csv")
        print(rank_df)

    def statistical_significance_tests(self):
        score_col = "score" if "score" in self.df.columns else "final_score"
        algos = self.df["algorithm"].unique()
        print("\n--- Statistical Significance Tests ---")
        for i, a1 in enumerate(algos):
            for a2 in algos[i+1:]:
                s1 = self.df[self.df["algorithm"] == a1][score_col]
                s2 = self.df[self.df["algorithm"] == a2][score_col]
                t, p = ttest_ind(s1, s2, equal_var=False)
                print(f"{a1} vs {a2} → p={p:.4f}")
        groups = [self.df[self.df["algorithm"]==a][score_col] for a in algos]
        F, p = f_oneway(*groups)
        print(f"ANOVA → F={F:.3f}, p={p:.4f}")

    def summary_stats(self):
        score_col = "score" if "score" in self.df.columns else "final_score"
        print("\n--- Summary Stats ---")
        grouped = self.df.groupby("algorithm")[score_col].agg(["mean","std","median","min","max","count"])
        grouped["cv"] = grouped["std"] / grouped["mean"]
        print(grouped.sort_values("mean", ascending=False))
        return grouped

    def full_analysis(self):
        # stats:
        self.summary_stats()
        self.statistical_significance_tests()
        self.plot_efficiency()
        self.save_algorithm_ranking()

        # plots:
        self.plot_average_performance()
        self.plot_score_distribution()
        self.plot_runtime_vs_score()
        self.plot_parameter_sensitivity()
        self.plot_convergence_curves()

        print("\n✓ Full analysis completed.")
