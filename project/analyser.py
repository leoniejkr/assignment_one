import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import ast

class ExperimentAnalyzer:
    """Advanced statistical analysis of experiment results."""

    def __init__(self, results_dir="project/results"):
        import ast

        self.dir = results_dir
        self.df = pd.read_csv(f"{results_dir}/summary_results.csv")

        # auto-detect parameter column
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
            except Exception as e:
                print(f"⚠️ Could not parse {param_col}: {e}")
                self.df["params_dict"] = [{} for _ in range(len(self.df))]
        else:
            print("⚠️ No parameter column found — using empty dicts.")
            self.df["params_dict"] = [{} for _ in range(len(self.df))]

    def summary_stats(self):
        print("\n--- Summary Stats ---")
        grouped = self.df.groupby("algorithm")["final_score"].agg(["mean", "std", "median", "min", "max", "count"])
        grouped["cv"] = grouped["std"] / grouped["mean"]
        print(grouped.sort_values("mean", ascending=False))
        return grouped

    def significance_tests(self):
        print("\n--- Significance Tests ---")
        algos = self.df["algorithm"].unique()
        for i, a1 in enumerate(algos):
            for a2 in algos[i+1:]:
                s1, s2 = self.df[self.df["algorithm"] == a1]["final_score"], self.df[self.df["algorithm"] == a2]["final_score"]
                t, p = ttest_ind(s1, s2, equal_var=False)
                print(f"{a1} vs {a2} → p={p:.4f}")
        groups = [self.df[self.df["algorithm"] == a]["final_score"] for a in algos]
        F, p = f_oneway(*groups)
        print(f"\nANOVA → F={F:.3f}, p={p:.4f}")

    def parameter_sensitivity(self):
        print("\n--- Parameter Sensitivity ---")
        param_df = self.df.join(pd.json_normalize(self.df["params_dict"]))
        corr = param_df.corr(numeric_only=True)[["final_score"]].sort_values("final_score", ascending=False)
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="viridis")
        plt.title("Parameter Sensitivity")
        plt.tight_layout()
        plt.savefig(f"{self.dir}/parameter_sensitivity_heatmap.png")
        print("✓ Saved parameter_sensitivity_heatmap.png")

    def efficiency_plot(self):
        print("\n--- Efficiency Plot ---")
        self.df["final_score_per_second"] = self.df["final_score"] / self.df["runtime_seconds"]
        plt.figure(figsize=(8,6))
        sns.barplot(data=self.df, x="algorithm", y="final_score_per_second")
        plt.title("Efficiency (final_Score per Second)")
        plt.tight_layout()
        plt.savefig(f"{self.dir}/efficiency_barplot.png")
        print("✓ Saved efficiency_barplot.png")

    def rank_algorithms(self):
        print("\n--- Algorithm Ranking ---")
        rank_df = (self.df.groupby("algorithm")["final_score"]
                   .mean()
                   .sort_values(ascending=False)
                   .reset_index())
        rank_df["rank"] = range(1, len(rank_df) + 1)
        print(rank_df)
        rank_df.to_csv(f"{self.dir}/algorithm_ranking.csv", index=False)
        print("✓ Saved algorithm_ranking.csv")

    def full_analysis(self):
        self.summary_stats()
        self.significance_tests()
        self.parameter_sensitivity()
        self.efficiency_plot()
        self.rank_algorithms()
        print("\n✓ Full analysis completed.")
