"""
Step 4: Agreement Evaluation Module.
Computes Human-Human Inter-Rater Reliability (Ceiling) and System-Human Agreement
across baselines (B1, B2, B3) and proposed methods (S1, S2).
Includes Quadratic Weighted Kappa (QWK), ICC(2,1), Bland-Altman statistics,
bootstrap 95% confidence intervals, non-inferiority testing, CSV tables, and plots.
"""

from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from eval.common import (
    human_disagreement, load_config, load_dataset, out_dir, save_json, write_run_meta,
)


# --- Core Statistical Primitives ---

def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, min_val: int = 1, max_val: int = 5) -> float:
    """Quadratic Weighted Kappa (QWK) on discrete ordinal rubric scale [min_val, max_val]."""
    y_t = np.clip(np.round(np.asarray(y_true)), min_val, max_val).astype(int)
    y_p = np.clip(np.round(np.asarray(y_pred)), min_val, max_val).astype(int)
    if len(np.unique(y_t)) <= 1 and len(np.unique(y_p)) <= 1:
        return 1.0 if y_t[0] == y_p[0] else 0.0
    labels = list(range(min_val, max_val + 1))
    return float(cohen_kappa_score(y_t, y_p, labels=labels, weights="quadratic"))


def compute_icc_2_1(matrix: np.ndarray) -> float:
    """
    Two-way random effects, single rater, absolute agreement ICC(2,1).
    matrix: (n_targets, k_raters)
    """
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n < 2 or k < 2:
        return 0.0

    # Grand mean
    grand_mean = np.mean(matrix)
    
    # Target (row) means and rater (col) means
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    # Sum of squares
    ssr = k * np.sum((row_means - grand_mean) ** 2)
    ssc = n * np.sum((col_means - grand_mean) ** 2)
    sst = np.sum((matrix - grand_mean) ** 2)
    sse = sst - ssr - ssc
    if sse < 0:
        sse = 0.0

    # Mean squares
    df_r = n - 1
    df_c = k - 1
    df_e = (n - 1) * (k - 1)

    msr = ssr / df_r if df_r > 0 else 0.0
    msc = ssc / df_c if df_c > 0 else 0.0
    mse = sse / df_e if df_e > 0 else 0.0

    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom == 0.0:
        return 0.0
    icc = (msr - mse) / denom
    return float(np.clip(icc, -1.0, 1.0))


def compute_bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate Bland-Altman mean bias and 95% limits of agreement (bias +/- 1.96 * SD)."""
    diffs = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_lower = mean_bias - 1.96 * sd_diff
    loa_upper = mean_bias + 1.96 * sd_diff
    return {
        "mean_bias": round(mean_bias, 3),
        "sd_bias": round(sd_diff, 3),
        "loa_lower": round(loa_lower, 3),
        "loa_upper": round(loa_upper, 3),
    }


def compute_agreement_metrics(y_human: np.ndarray, y_system: np.ndarray) -> Dict[str, float]:
    """Comprehensive system-human agreement metrics."""
    y_h = np.asarray(y_human, dtype=float)
    y_s = np.asarray(y_system, dtype=float)
    n = len(y_h)
    if n == 0:
        return {}

    mae = float(mean_absolute_error(y_h, y_s))
    exact_acc = float(np.mean(np.round(y_h) == np.round(y_s)))
    within_1pt = float(np.mean(np.abs(np.round(y_h) - np.round(y_s)) <= 1.0))
    qwk = compute_qwk(y_h, y_s)

    # ICC between human and system treated as 2 raters
    pair_matrix = np.column_stack([y_h, y_s])
    icc = compute_icc_2_1(pair_matrix)

    # Correlations
    pearson_r = float(stats.pearsonr(y_h, y_s)[0]) if np.std(y_h) > 0 and np.std(y_s) > 0 else 0.0
    spearman_rho = float(stats.spearmanr(y_h, y_s)[0]) if np.std(y_h) > 0 and np.std(y_s) > 0 else 0.0

    ba = compute_bland_altman(y_h, y_s)

    return {
        "N": n,
        "MAE": round(mae, 3),
        "Exact_Agreement": round(exact_acc, 3),
        "Within_1Pt": round(within_1pt, 3),
        "QWK": round(qwk, 3),
        "ICC": round(icc, 3),
        "Pearson_r": round(pearson_r, 3),
        "Spearman_rho": round(spearman_rho, 3),
        "Mean_Bias": ba["mean_bias"],
        "LoA_Lower": ba["loa_lower"],
        "LoA_Upper": ba["loa_upper"],
    }


def bootstrap_ci(
    data_func,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap percentile confidence interval [lower, upper]."""
    rng = np.random.RandomState(random_seed)
    estimates = []
    for _ in range(n_resamples):
        val = data_func(rng)
        if not np.isnan(val):
            estimates.append(val)
    if not estimates:
        return (0.0, 0.0)
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(estimates, 100.0 * alpha))
    upper = float(np.percentile(estimates, 100.0 * (1.0 - alpha)))
    return (round(lower, 3), round(upper, 3))


# --- Evaluation Engine ---

class AgreementEvaluator:
    def __init__(self, output_dir: str | Path = "eval/outputs", non_inferiority_margin: float = 0.10, bootstrap_resamples: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.margin = non_inferiority_margin
        self.resamples = bootstrap_resamples

    def evaluate_human_ceiling(self, ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate human-human inter-rater reliability across unique rater pairs.
        Returns:
            Dict containing per-dimension and overall ICC(2,1) and QWK, plus pairwise tables.
        """
        if ratings_df.empty or "rater_id" not in ratings_df.columns:
            return {"error": "No human ratings provided."}

        raters = ratings_df["rater_id"].unique()
        if len(raters) < 2:
            return {"error": f"Need at least 2 distinct raters for human-human reliability; found {len(raters)}."}

        # Pivot to target x rater matrix for overall score
        pivot_overall = ratings_df.pivot(index="answer_id", columns="rater_id", values="overall_score").dropna()
        n_completed = len(pivot_overall)
        if n_completed < 2:
            return {"error": f"Insufficient shared ratings across raters (shared answers: {n_completed})."}

        # Ceiling = mean PAIRWISE single-rater ICC(2,1). This is the like-for-like reference for
        # system-vs-single-rater agreement. (The all-rater ICC is reported too, for information.)
        human_icc_all = compute_icc_2_1(pivot_overall.values)
        pair_iccs = [compute_icc_2_1(pivot_overall[[a, b]].values)
                     for a, b in itertools.combinations(pivot_overall.columns, 2)]
        human_icc = float(np.mean(pair_iccs))

        # Pairwise QWK average
        pairs = list(itertools.combinations(raters, 2))
        qwk_list = []
        for r1, r2 in pairs:
            shared = ratings_df[ratings_df["rater_id"].isin([r1, r2])].pivot(
                index="answer_id", columns="rater_id", values="overall_score"
            ).dropna()
            if len(shared) >= 2:
                qwk_list.append(compute_qwk(shared[r1].values, shared[r2].values))

        avg_qwk = float(np.mean(qwk_list)) if qwk_list else 0.0

        # Dimension breakdown
        dims = ["relevance_correctness", "depth_completeness", "structure_clarity", "language_use"]
        dim_results = {}
        for d in dims:
            if d in ratings_df.columns:
                p_dim = ratings_df.pivot(index="answer_id", columns="rater_id", values=d).dropna()
                dim_icc = compute_icc_2_1(p_dim.values) if len(p_dim) >= 2 else 0.0
                dim_results[d] = round(dim_icc, 3)

        return {
            "num_raters": len(raters),
            "shared_answers": n_completed,
            "human_ceiling_icc": round(human_icc, 3),
            "human_icc_all_raters": round(human_icc_all, 3),
            "typical_abs_disagreement": human_disagreement(ratings_df),
            "human_ceiling_qwk": round(avg_qwk, 3),
            "dimension_icc": dim_results,
        }

    def evaluate_methods_against_human(
        self,
        merged_df: pd.DataFrame,
        human_col: str = "human_overall_mean",
        methods_cols: Optional[Dict[str, str]] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compare each system method (B1, B2, B3, S1, S2) against the human mean score.
        Computes 95% CIs via empirical bootstrap.
        """
        if methods_cols is None:
            methods_cols = {
                "B1 (TF-IDF)": "score_B1",
                "B2 (Embedding)": "score_B2",
                "B3 (Naive LLM)": "score_B3",
                "S1 (Rubric LLM)": "score_S1",
                "S2 (Ensemble)": "score_S2",
            }

        valid_df = merged_df.dropna(subset=[human_col]).copy()
        if valid_df.empty:
            return pd.DataFrame()

        records = []
        y_human = valid_df[human_col].values

        for method_label, col_name in methods_cols.items():
            if col_name not in valid_df.columns:
                continue
            method_sub = valid_df.dropna(subset=[col_name])
            if method_sub.empty:
                continue

            y_h = method_sub[human_col].values
            y_s = method_sub[col_name].values

            base_metrics = compute_agreement_metrics(y_h, y_s)
            n_samples = len(y_h)

            # Bootstrap CIs for ICC and QWK
            def sample_icc(rng):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                pair = np.column_stack([y_h[idx], y_s[idx]])
                return compute_icc_2_1(pair)

            def sample_qwk(rng):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                return compute_qwk(y_h[idx], y_s[idx])

            icc_ci = bootstrap_ci(sample_icc, n_resamples=self.resamples)
            qwk_ci = bootstrap_ci(sample_qwk, n_resamples=self.resamples)

            rec = {
                "Method": method_label,
                "N": base_metrics["N"],
                "ICC": base_metrics["ICC"],
                "ICC_95_CI": f"[{icc_ci[0]:.2f}, {icc_ci[1]:.2f}]",
                "ICC_vs_Single_Rater": (
                    round(self.system_vs_single_raters_icc(ratings_df, method_sub.set_index("answer_id")[col_name]), 3)
                    if ratings_df is not None else None
                ),
                "QWK": base_metrics["QWK"],
                "QWK_95_CI": f"[{qwk_ci[0]:.2f}, {qwk_ci[1]:.2f}]",
                "MAE": base_metrics["MAE"],
                "Exact": base_metrics["Exact_Agreement"],
                "Within_1Pt": base_metrics["Within_1Pt"],
                "Pearson_r": base_metrics["Pearson_r"],
                "Spearman_rho": base_metrics["Spearman_rho"],
                "Mean_Bias": base_metrics["Mean_Bias"],
                "LoA": f"[{base_metrics['LoA_Lower']:.2f}, {base_metrics['LoA_Upper']:.2f}]",
            }
            records.append(rec)

        res_df = pd.DataFrame(records)
        csv_path = self.output_dir / "system_human_agreement_table.csv"
        res_df.to_csv(csv_path, index=False)
        return res_df

    @staticmethod
    def _rater_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
        return ratings_df.pivot(index="answer_id", columns="rater_id", values="overall_score").dropna()

    @staticmethod
    def _mean_pairwise_icc(mat: np.ndarray) -> float:
        k = mat.shape[1]
        return float(np.mean([compute_icc_2_1(mat[:, [i, j]]) for i, j in itertools.combinations(range(k), 2)]))

    def system_vs_single_raters_icc(self, ratings_df: pd.DataFrame, system_scores: pd.Series) -> float:
        """Mean ICC(2,1) between the system and EACH individual rater (like-for-like with human-human)."""
        piv = self._rater_matrix(ratings_df)
        sys = system_scores.dropna()
        common = piv.index.intersection(sys.index)
        if len(common) < 3:
            return float("nan")
        vals = [compute_icc_2_1(np.column_stack([piv.loc[common, r].values, sys.loc[common].values]))
                for r in piv.columns]
        return float(np.mean(vals))

    def test_non_inferiority(self, ratings_df: pd.DataFrame, system_scores: pd.Series,
                             min_answers: int = 10, method_label: str = "S1") -> Dict[str, Any]:
        """
        Is the system as consistent with a single human rater as human raters are with each other?

        ceiling = mean pairwise ICC(2,1) between human raters
        system  = mean ICC(2,1) between the system and each individual rater
        Verdict uses the bootstrap 95% CI of (ceiling - system), resampling answers:
          non-inferior : upper CI bound <= margin
          inconclusive : point estimate <= margin but upper bound > margin (not enough data to conclude)
          inferior     : point estimate > margin
        """
        piv = self._rater_matrix(ratings_df)
        if piv.shape[1] < 2:
            return {"error": "Need >= 2 raters who rated the same answers."}
        sys = system_scores.dropna()
        common = piv.index.intersection(sys.index)
        n = len(common)
        if n < min_answers:
            return {"error": f"Insufficient data: {n} answers rated by all raters AND scored by {method_label} "
                             f"(need >= {min_answers})."}
        H, s = piv.loc[common].values, sys.loc[common].values.astype(float)

        def stat(idx):
            Hh, ss = H[idx], s[idx]
            return self._mean_pairwise_icc(Hh), float(np.mean([compute_icc_2_1(np.column_stack([Hh[:, j], ss]))
                                                               for j in range(Hh.shape[1])]))

        ceiling, system = stat(np.arange(n))
        rng = np.random.RandomState(42)
        diffs = []
        for _ in range(self.resamples):
            c, sv = stat(rng.choice(n, size=n, replace=True))
            diffs.append(c - sv)
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        diff = ceiling - system
        if hi <= self.margin:
            verdict, text = "non_inferior", f"{method_label} IS non-inferior (upper 95% CI bound {hi:.3f} <= margin {self.margin:.2f})"
        elif diff <= self.margin:
            verdict, text = "inconclusive", (f"INCONCLUSIVE: point estimate {diff:.3f} <= margin {self.margin:.2f}, "
                                             f"but upper 95% CI bound {hi:.3f} exceeds it (more data needed)")
        else:
            verdict, text = "inferior", f"{method_label} is INFERIOR to the human ceiling (difference {diff:.3f} > margin {self.margin:.2f})"
        return {
            "method": method_label, "n_answers": n, "n_raters": int(piv.shape[1]),
            "human_ceiling_icc": round(ceiling, 3), "system_icc_single_rater": round(system, 3),
            "margin_delta": self.margin, "icc_difference": round(diff, 3),
            "diff_95_ci": (round(lo, 3), round(hi, 3)), "verdict": verdict,
            "is_non_inferior": verdict == "non_inferior", "conclusion": text,
        }

    # --- Plotting Utilities ---

    def plot_bland_altman(self, y_human: np.ndarray, y_s1: np.ndarray, method_name: str = "S1 (Proposed)") -> Path:
        """Generate publication-ready Bland-Altman agreement plot."""
        means = (y_human + y_s1) / 2.0
        diffs = y_s1 - y_human
        ba = compute_bland_altman(y_human, y_s1)

        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.scatter(means, diffs, alpha=0.65, color="#2563eb", edgecolors="none", s=40)
        ax.axhline(ba["mean_bias"], color="#dc2626", linestyle="-", linewidth=1.5, label=f"Mean Bias: {ba['mean_bias']:.2f}")
        ax.axhline(ba["loa_upper"], color="#16a34a", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {ba['loa_upper']:.2f}")
        ax.axhline(ba["loa_lower"], color="#16a34a", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {ba['loa_lower']:.2f}")

        ax.set_title(f"Bland-Altman Plot: {method_name} vs Human Raters", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean of Human Raters and System Score", fontsize=10)
        ax.set_ylabel("Difference (System - Human)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        plt.tight_layout()

        out_path = self.output_dir / "bland_altman_s1.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_agreement_comparison(self, agreement_df: pd.DataFrame, human_ceiling_icc: Optional[float] = None) -> Path:
        """Generate bar chart comparing ICC and QWK across methods alongside human ceiling."""
        if agreement_df.empty or "Method" not in agreement_df.columns:
            return self.output_dir / "agreement_comparison.png"

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        x = np.arange(len(agreement_df))
        width = 0.35

        ax.bar(x - width/2, agreement_df["ICC"], width, label="ICC (Absolute)", color="#3b82f6")
        ax.bar(x + width/2, agreement_df["QWK"], width, label="QWK (Quadratic Kappa)", color="#8b5cf6")

        if human_ceiling_icc is not None:
            ax.axhline(human_ceiling_icc, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Human Ceiling (ICC {human_ceiling_icc:.2f})")
            ax.axhspan(human_ceiling_icc - self.margin, human_ceiling_icc, color="#fca5a5", alpha=0.25, label=f"Non-inferiority Margin ({self.margin:.2f})")

        ax.set_ylabel("Reliability Coefficient", fontsize=10)
        ax.set_title("Scoring Agreement Across Methods vs Human Benchmarks", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(agreement_df["Method"], rotation=15, ha="right", fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)
        plt.tight_layout()

        out_path = self.output_dir / "agreement_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


# --- CLI: python -m eval.agreement [--demo] [--config config.yaml] ---

ARTIFACTS = ["human_ceiling.json", "non_inferiority.json", "system_human_agreement_table.csv",
             "bland_altman_s1.png", "agreement_comparison.png"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Human-human ceiling and system-human agreement")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data + MOCK LLM (pipeline test only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = out_dir(cfg, args.demo)
    for name in ARTIFACTS:  # never leave stale outputs from an earlier run
        (out / name).unlink(missing_ok=True)

    merged_df, ratings_df, _ = load_dataset(cfg, args.demo)
    write_run_meta(cfg, args.demo, n_answers=len(merged_df))
    if args.demo:
        print("*** DEMO MODE: synthetic data and MOCK LLM. These numbers test the pipeline only; they are NOT evidence. ***")

    ev_cfg = cfg["eval"]
    evaluator = AgreementEvaluator(output_dir=out, non_inferiority_margin=float(ev_cfg["non_inferiority_margin"]),
                                   bootstrap_resamples=int(ev_cfg["bootstrap_resamples"]))

    ceiling = evaluator.evaluate_human_ceiling(ratings_df)
    save_json(out / "human_ceiling.json", ceiling)
    print("\n--- HUMAN CEILING ---")
    print(json.dumps(ceiling, indent=2))

    eligible = merged_df
    if "human_count" in merged_df.columns:
        eligible = merged_df[merged_df["human_count"] >= int(ev_cfg.get("min_raters_per_answer", 2))]
    print(f"\nAnswers with >= {ev_cfg.get('min_raters_per_answer', 2)} human ratings: {len(eligible)} of {len(merged_df)}")

    agreement_df = evaluator.evaluate_methods_against_human(eligible, ratings_df=ratings_df)
    if agreement_df.empty:
        print("No system scores available to compare (insufficient data).")
    else:
        print("\n--- SYSTEM vs HUMAN (mean of raters; see ICC_vs_Single_Rater for the like-for-like measure) ---")
        print(agreement_df.to_string(index=False))

    if "score_S1" in eligible.columns:
        ni = evaluator.test_non_inferiority(ratings_df, eligible.set_index("answer_id")["score_S1"],
                                            min_answers=int(ev_cfg.get("min_answers_for_inference", 10)))
        save_json(out / "non_inferiority.json", ni)
        print("\n--- NON-INFERIORITY (S1) ---")
        print(json.dumps(ni, indent=2))

        pair = eligible[["human_overall_mean", "score_S1"]].dropna()
        if len(pair) >= 2:
            evaluator.plot_bland_altman(pair["human_overall_mean"].values, pair["score_S1"].values)
    if not agreement_df.empty:
        evaluator.plot_agreement_comparison(agreement_df, ceiling.get("human_ceiling_icc"))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
