"""
Unit Tests for Psychometric & Evaluation Metrics:
- Quadratic Weighted Kappa (QWK)
- Intraclass Correlation Coefficient ICC(2,1) against known textbook benchmarks
- Bland-Altman limits of agreement
- Bootstrap confidence intervals
- Evidence quote grounding verification
"""

import unittest
import numpy as np

from eval.agreement import (
    compute_qwk,
    compute_icc_2_1,
    compute_bland_altman,
    bootstrap_ci,
)
from coach.scorer_llm import S1Scorer
from coach.rubric import Rubric
from coach.llm_backend import MockLLMBackend


class TestMetrics(unittest.TestCase):

    def test_qwk_perfect_agreement(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        qwk = compute_qwk(y_true, y_pred, min_val=1, max_val=5)
        self.assertAlmostEqual(qwk, 1.0, places=4)

    def test_qwk_extreme_disagreement(self):
        y_true = np.array([1, 1, 1, 5, 5, 5])
        y_pred = np.array([5, 5, 5, 1, 1, 1])
        qwk = compute_qwk(y_true, y_pred, min_val=1, max_val=5)
        self.assertLess(qwk, 0.0)

    def test_qwk_known_intermediate_case(self):
        # Off-by-one errors incur light penalty under quadratic weighting
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 5])
        qwk = compute_qwk(y_true, y_pred, min_val=1, max_val=5)
        self.assertGreater(qwk, 0.6)
        self.assertLess(qwk, 1.0)

    def test_icc_perfect_agreement(self):
        # 4 targets, 3 raters with identical ratings
        matrix = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ])
        icc = compute_icc_2_1(matrix)
        self.assertAlmostEqual(icc, 1.0, places=4)

    def test_icc_known_textbook_matrix(self):
        """
        Standard Shrout & Fleiss (1979) table 4 benchmark.
        Targets: 6 subjects, 4 raters.
        """
        sf_matrix = np.array([
            [9.0, 2.0, 5.0, 8.0],
            [6.0, 1.0, 3.0, 2.0],
            [8.0, 4.0, 6.0, 8.0],
            [7.0, 1.0, 2.0, 6.0],
            [10.0, 5.0, 6.0, 9.0],
            [6.0, 2.0, 4.0, 7.0],
        ])
        icc = compute_icc_2_1(sf_matrix)
        # S&F 1979 Case 2, single rater ICC is approximately 0.29 (between 0.28 and 0.31)
        self.assertGreater(icc, 0.25)
        self.assertLess(icc, 0.35)

    def test_bland_altman_bias_and_limits(self):
        y_true = np.array([2.0, 3.0, 4.0, 5.0])
        # System is systematically higher by 0.5
        y_pred = np.array([2.5, 3.5, 4.5, 5.5])
        ba = compute_bland_altman(y_true, y_pred)
        self.assertAlmostEqual(ba["mean_bias"], 0.5, places=3)
        self.assertAlmostEqual(ba["sd_bias"], 0.0, places=3)
        self.assertAlmostEqual(ba["loa_lower"], 0.5, places=3)
        self.assertAlmostEqual(ba["loa_upper"], 0.5, places=3)

    def test_bootstrap_ci_coverage(self):
        np.random.seed(42)
        true_mean = 10.0
        data = np.random.normal(loc=true_mean, scale=1.0, size=100)

        def sample_mean(rng):
            idx = rng.choice(len(data), size=len(data), replace=True)
            return float(np.mean(data[idx]))

        ci_low, ci_high = bootstrap_ci(sample_mean, n_resamples=500, ci_level=0.95, random_seed=42)
        self.assertLessEqual(ci_low, true_mean)
        self.assertGreaterEqual(ci_high, true_mean)

    def test_evidence_grounding_verification(self):
        rubric = Rubric(version="1.0", description="test", dimensions={})
        scorer = S1Scorer(rubric=rubric, llm_backend=MockLLMBackend())

        transcript = "In my last project, we used PostgreSQL read replicas to scale query throughput."

        # Grounded quote
        self.assertTrue(scorer.verify_evidence_quote("PostgreSQL read replicas", transcript))
        self.assertTrue(scorer.verify_evidence_quote("to scale query throughput", transcript))
        self.assertTrue(scorer.verify_evidence_quote("N/A", transcript))

        # Hallucinated / ungrounded quote
        self.assertFalse(scorer.verify_evidence_quote("We completely rewrote the system in Rust and Kafka", transcript))
        self.assertFalse(scorer.verify_evidence_quote("MongoDB clusters were sharded across AWS regions", transcript))


if __name__ == "__main__":
    unittest.main()
