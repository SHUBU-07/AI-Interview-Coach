"""
Step 5: Reliability Evaluation Module.
Measures:
1. Test-retest ICC (score consistency across repeated stochastic or temperature runs on identical answers).
2. Prompt sensitivity (score drift across 2-3 paraphrased prompt variations).
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from coach.scorer_llm import S1Scorer
from eval.agreement import compute_icc_2_1, compute_qwk

logger = logging.getLogger(__name__)


@dataclass
class TestRetestResult:
    k_runs: int
    n_answers: int
    test_retest_icc: float
    mean_score_variance: float
    max_score_discrepancy: float
    per_dimension_icc: Dict[str, float]


@dataclass
class PromptSensitivityResult:
    variants_evaluated: List[str]
    n_answers: int
    mean_absolute_difference: float
    max_pairwise_difference: float
    inter_prompt_icc: float
    correlation_matrix: Dict[str, Dict[str, float]]


class ReliabilityEvaluator:
    def __init__(self, s1_scorer: S1Scorer, output_dir: str | Path = "eval/outputs"):
        self.scorer = s1_scorer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_test_retest(self, answers_df: pd.DataFrame, questions_dict: Dict[str, str], k_runs: int = 3) -> TestRetestResult:
        """
        Rerun S1 k times on the exact same candidate answers.
        Computes intra-system test-retest ICC(2,1) and dimension-level consistency.
        """
        if answers_df.empty:
            return TestRetestResult(
                k_runs=k_runs, n_answers=0, test_retest_icc=0.0,
                mean_score_variance=0.0, max_score_discrepancy=0.0, per_dimension_icc={}
            )

        run_matrix = []  # shape: (n_answers, k_runs)
        dim_matrices = {d: [] for d in ["relevance_correctness", "depth_completeness", "structure_clarity", "language_use"]}

        for _, row in answers_df.iterrows():
            aid = row["answer_id"]
            qid = row["question_id"]
            transcript = row["transcript"]
            q_text = questions_dict.get(qid, "Interview Question")

            scores_this_answer = []
            dim_scores_this_answer = {d: [] for d in dim_matrices.keys()}
            failed = False

            for r_idx in range(k_runs):
                res = self.scorer.score_answer(
                    answer_id=f"{aid}_run_{r_idx}",
                    question=q_text,
                    transcript=transcript,
                    prompt_variant="default",
                )
                if not res.validation_passed:
                    failed = True
                scores_this_answer.append(res.overall_score)
                for d in dim_matrices.keys():
                    val = res.dimension_scores.get(d)
                    dim_scores_this_answer[d].append(val.score if val else 3)

            if failed:
                continue  # an LLM run failed validation: exclude this answer
            run_matrix.append(scores_this_answer)
            for d in dim_matrices.keys():
                dim_matrices[d].append(dim_scores_this_answer[d])

        mat = np.array(run_matrix)  # (N, k)
        test_retest_icc = compute_icc_2_1(mat)
        variances = np.var(mat, axis=1, ddof=1) if k_runs > 1 else np.zeros(len(mat))
        mean_var = float(np.mean(variances))
        max_disc = float(np.max(np.ptp(mat, axis=1))) if len(mat) > 0 else 0.0

        per_dim_icc = {}
        for d, d_mat in dim_matrices.items():
            per_dim_icc[d] = round(compute_icc_2_1(np.array(d_mat)), 3)

        result = TestRetestResult(
            k_runs=k_runs,
            n_answers=len(mat),
            test_retest_icc=round(test_retest_icc, 3),
            mean_score_variance=round(mean_var, 4),
            max_score_discrepancy=round(max_disc, 2),
            per_dimension_icc=per_dim_icc,
        )

        # Export summary table
        df_summary = pd.DataFrame([{
            "N_Answers": result.n_answers,
            "K_Runs": result.k_runs,
            "Test_Retest_ICC": result.test_retest_icc,
            "Mean_Score_Variance": result.mean_score_variance,
            "Max_Score_Discrepancy": result.max_score_discrepancy,
        }])
        df_summary.to_csv(self.output_dir / "reliability_test_retest.csv", index=False)
        return result

    def evaluate_prompt_sensitivity(
        self,
        answers_df: pd.DataFrame,
        questions_dict: Dict[str, str],
        variants: Optional[List[str]] = None,
    ) -> PromptSensitivityResult:
        """
        Evaluate score drift across paraphrased prompt variations.
        """
        if variants is None:
            variants = ["default", "paraphrase_1", "paraphrase_2"]

        if answers_df.empty:
            return PromptSensitivityResult(
                variants_evaluated=variants, n_answers=0, mean_absolute_difference=0.0,
                max_pairwise_difference=0.0, inter_prompt_icc=0.0, correlation_matrix={}
            )

        variant_matrix = []  # shape: (n_answers, len(variants))
        for _, row in answers_df.iterrows():
            aid = row["answer_id"]
            qid = row["question_id"]
            transcript = row["transcript"]
            q_text = questions_dict.get(qid, "Interview Question")

            scores_for_answer = []
            failed = False
            for var in variants:
                res = self.scorer.score_answer(
                    answer_id=f"{aid}_{var}",
                    question=q_text,
                    transcript=transcript,
                    prompt_variant=var,
                )
                if not res.validation_passed:
                    failed = True
                scores_for_answer.append(res.overall_score)
            if failed:
                continue
            variant_matrix.append(scores_for_answer)

        mat = np.array(variant_matrix)  # (N, V)
        inter_prompt_icc = compute_icc_2_1(mat)

        # Pairwise differences
        diffs = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                diffs.extend(np.abs(mat[:, i] - mat[:, j]))

        mad = float(np.mean(diffs)) if diffs else 0.0
        max_diff = float(np.max(diffs)) if diffs else 0.0

        # Correlation matrix
        corr_dict = {}
        for i, v1 in enumerate(variants):
            corr_dict[v1] = {}
            for j, v2 in enumerate(variants):
                r = stats.pearsonr(mat[:, i], mat[:, j])[0] if np.std(mat[:, i]) > 0 and np.std(mat[:, j]) > 0 else 1.0
                corr_dict[v1][v2] = round(float(r), 3)

        result = PromptSensitivityResult(
            variants_evaluated=variants,
            n_answers=len(mat),
            mean_absolute_difference=round(mad, 3),
            max_pairwise_difference=round(max_diff, 2),
            inter_prompt_icc=round(inter_prompt_icc, 3),
            correlation_matrix=corr_dict,
        )

        df_summary = pd.DataFrame([{
            "N_Answers": result.n_answers,
            "Variants": ", ".join(variants),
            "Inter_Prompt_ICC": result.inter_prompt_icc,
            "Mean_Absolute_Drift": result.mean_absolute_difference,
            "Max_Pairwise_Drift": result.max_pairwise_difference,
        }])
        df_summary.to_csv(self.output_dir / "reliability_prompt_sensitivity.csv", index=False)
        return result


if __name__ == "__main__":
    import argparse
    from eval.common import build_scorers, load_config, load_dataset, out_dir, write_run_meta

    parser = argparse.ArgumentParser(description="Reliability: test-retest and prompt sensitivity")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N answers (saves LLM time)")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = out_dir(cfg, args.demo)
    for f in ["reliability_test_retest.csv", "reliability_prompt_sensitivity.csv"]:
        (out / f).unlink(missing_ok=True)

    merged, _, questions_df = load_dataset(cfg, args.demo, require_scores=False)
    write_run_meta(cfg, args.demo, n_answers=len(merged))
    answers_df = merged.head(args.limit) if args.limit else merged
    q_dict = dict(zip(questions_df["question_id"], questions_df["question_text"]))

    scorers = build_scorers(cfg, demo=args.demo)
    evaluator = ReliabilityEvaluator(s1_scorer=scorers["s1"], output_dir=out)

    if args.demo:
        print("*** DEMO MODE: MOCK LLM is deterministic, so test-retest/prompt ICC are trivially ~1. Pipeline test only. ***")
    elif float(cfg["llm"]["temperature"]) == 0.0:
        print("[note] temperature=0: repeated runs are near-deterministic, so high test-retest ICC is expected. "
              "Prompt sensitivity is the more informative check here.")

    print(f"Test-retest ({args.runs} runs x {len(answers_df)} answers)...")
    tr = evaluator.evaluate_test_retest(answers_df, q_dict, k_runs=args.runs)
    print(f"  Test-retest ICC: {tr.test_retest_icc} (n={tr.n_answers}, mean variance {tr.mean_score_variance})")
    print("Prompt sensitivity (3 paraphrased prompts)...")
    ps = evaluator.evaluate_prompt_sensitivity(answers_df, q_dict)
    print(f"  Inter-prompt ICC: {ps.inter_prompt_icc} (n={ps.n_answers}, mean abs drift {ps.mean_absolute_difference})")
    print(f"Saved to: {out}")
