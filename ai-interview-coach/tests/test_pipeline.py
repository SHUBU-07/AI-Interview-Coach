"""Regression tests for bugs found in the original upload (each test names the bug it guards against)."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from coach.baselines import B1TfidfScorer, B3UnconstrainedLLMScorer
from coach.db import CoachDB, SystemScoreRecord
from coach.ensemble import S2EnsembleScorer
from coach.feedback import FeedbackGenerator
from coach.llm_backend import BaseLLMBackend, get_llm_backend
from coach.rubric import Rubric
from coach.scorer_llm import DimensionScore, S1Scorer, S1ScoringResult
from eval.agreement import AgreementEvaluator
from eval.common import human_disagreement

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = Rubric.load_from_yaml(ROOT / "rubric.yaml")
GOOD = json.dumps([{"dimension": d, "score": 4, "evidence_quote": "locks a record", "reason": "ok"}
                   for d in ["relevance_correctness", "depth_completeness", "structure_clarity", "language_use"]])
TRANSCRIPT = "Pessimistic control locks a record on read and optimistic control checks a version at commit."


class ScriptedLLM(BaseLLMBackend):
    """Returns scripted outputs in order (cycling)."""
    def __init__(self, outputs):
        super().__init__("scripted", 0.0)
        self.outputs, self.i = outputs, 0

    def generate(self, prompt, system_prompt=None):
        out = self.outputs[self.i % len(self.outputs)]
        self.i += 1
        return out, {"latency_ms": 1.0}


class TestHonestFailureHandling(unittest.TestCase):
    def test_unknown_backend_raises_instead_of_silent_mock(self):
        with self.assertRaises(ValueError):
            get_llm_backend({"llm": {"backend": "gemini"}})

    def test_b3_unparsable_output_is_nan_not_midpoint(self):
        res = B3UnconstrainedLLMScorer(ScriptedLLM(["I would rather not rate this."])).score_answer("a", "q", "t")
        self.assertTrue(math.isnan(res.overall_score))

    def test_b3_parses_score(self):
        res = B3UnconstrainedLLMScorer(ScriptedLLM(["Score: 8/10"])).score_answer("a", "q", "t")
        self.assertAlmostEqual(res.overall_score, 1 + 7 * 4 / 9, places=2)

    def test_b1_without_reference_is_nan(self):
        self.assertTrue(math.isnan(B1TfidfScorer().score_answer("a", "some answer", "").overall_score))

    def test_s1_invalid_llm_output_is_flagged(self):
        res = S1Scorer(RUBRIC, ScriptedLLM(["not json at all"]), max_retries=1).score_answer("a", "q", TRANSCRIPT)
        self.assertFalse(res.validation_passed)

    def test_s1_ignores_dimension_that_was_not_requested(self):
        extra = json.loads(GOOD) + [{"dimension": "delivery", "score": 5, "evidence_quote": "N/A", "reason": "x"}]
        res = S1Scorer(RUBRIC, ScriptedLLM([json.dumps(extra)])).score_answer("a", "q", TRANSCRIPT, has_audio=False)
        self.assertTrue(res.validation_passed)
        self.assertNotIn("delivery", res.dimension_scores)

    def test_ensemble_excludes_failed_runs(self):
        s1 = S1Scorer(RUBRIC, ScriptedLLM(["garbage", GOOD, GOOD, GOOD, GOOD, GOOD]), max_retries=0)
        res = S2EnsembleScorer([s1]).score_answer("a", "q", TRANSCRIPT, num_runs=3)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.valid_runs, 2)
        self.assertAlmostEqual(res.overall_score, 4.0)  # the failed run's fallback 3.0 must not drag the mean down


class TestFeedbackBug(unittest.TestCase):
    def test_priority_action_quotes_the_weakest_dimension_not_the_last_loop_item(self):
        dims = {
            "relevance_correctness": DimensionScore("relevance_correctness", 1, "alpha beta gamma", "weak", True),
            "language_use": DimensionScore("language_use", 5, "zzz final strength", "strong", True),
        }
        res = S1ScoringResult("a", 3.0, dims, "", "m", "1", 0.0, "v", "h", 1.0, True)
        fb = FeedbackGenerator(RUBRIC).generate_feedback(res, "q", TRANSCRIPT)
        self.assertIn("alpha beta gamma", fb.prioritized_action)
        self.assertNotIn("zzz final strength", fb.prioritized_action)


class TestNonInferiority(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.truth = rng.uniform(1, 5, 40)
        rows = []
        for rater, noise in (("A", 0.3), ("B", 0.3), ("C", 0.3)):
            for i, t in enumerate(self.truth):
                rows.append({"rater_id": rater, "answer_id": f"a{i}", "overall_score": float(np.clip(t + rng.normal(0, noise), 1, 5))})
        self.ratings = pd.DataFrame(rows)
        self.ev = AgreementEvaluator(output_dir=tempfile.mkdtemp(), non_inferiority_margin=0.10, bootstrap_resamples=200)
        self.ids = [f"a{i}" for i in range(40)]

    def test_system_as_good_as_a_rater_is_non_inferior(self):
        rng = np.random.RandomState(1)
        system = pd.Series(np.clip(self.truth + rng.normal(0, 0.3, 40), 1, 5), index=self.ids)
        self.assertEqual(self.ev.test_non_inferiority(self.ratings, system)["verdict"], "non_inferior")

    def test_noisy_system_is_inferior(self):
        rng = np.random.RandomState(2)
        system = pd.Series(np.clip(self.truth + rng.normal(0, 1.5, 40), 1, 5), index=self.ids)
        self.assertEqual(self.ev.test_non_inferiority(self.ratings, system)["verdict"], "inferior")

    def test_too_few_answers_reports_insufficient_data(self):
        system = pd.Series(self.truth[:5], index=self.ids[:5])
        self.assertIn("error", self.ev.test_non_inferiority(self.ratings, system, min_answers=10))

    def test_human_disagreement_value(self):
        df = pd.DataFrame({"rater_id": ["A", "B", "A", "B"], "answer_id": ["x", "x", "y", "y"],
                           "overall_score": [4.0, 3.0, 2.0, 2.5]})
        self.assertAlmostEqual(human_disagreement(df), 0.75)


class TestDatabaseExport(unittest.TestCase):
    def test_invalid_scores_never_reach_the_evaluation_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "t.db")
            pid = db.create_participant(True, "Hindi", "South India", "Unspecified")
            ok = db.save_answer(pid, "Q1", "answer one", asr_transcript=None)
            bad = db.save_answer(pid, "Q1", "answer two")
            for aid, valid in ((ok, True), (bad, False)):
                db.save_system_score(SystemScoreRecord(
                    f"SC_{aid}", aid, "S1", "m", "1", 0.0, "v", "h",
                    json.dumps({"overall": 3.0, "validation_passed": valid}), "raw"))
            merged = db.export_merged_dataset(Path(tmp) / "m.csv").set_index("answer_id")
            self.assertEqual(merged.loc[ok, "score_S1"], 3.0)
            self.assertTrue(pd.isna(merged.loc[bad, "score_S1"]))

    def test_consent_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                CoachDB(Path(tmp) / "t.db").create_participant(False, "x", "y", "z")

    def test_delete_participant_removes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "t.db")
            pid = db.create_participant(True, "Hindi", "South India", "Unspecified")
            aid = db.save_answer(pid, "Q1", "x")
            db.save_sus_response(pid, [3] * 10, 50.0)
            db.delete_participant_data(pid)
            self.assertEqual(db.get_all_answers(), [])
            self.assertEqual(db.get_all_sus_responses(), [])


class TestCommandLine(unittest.TestCase):
    def test_python_m_eval_agreement_actually_runs(self):
        """Original bug: `python -m eval.agreement` printed nothing because the CLI lived in __main__.py."""
        r = subprocess.run([sys.executable, "-m", "eval.agreement", "--demo"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("HUMAN CEILING", r.stdout)
        self.assertTrue((ROOT / "eval" / "outputs" / "demo" / "human_ceiling.json").exists())

    def test_demo_report_is_labelled_demo_and_has_no_hardcoded_numbers(self):
        subprocess.run([sys.executable, "run_pipeline.py", "--demo"], cwd=ROOT, capture_output=True, text=True, check=True)
        text = (ROOT / "results_demo.md").read_text(encoding="utf-8")
        self.assertIn("DEMO ONLY", text)
        ceiling = json.loads((ROOT / "eval" / "outputs" / "demo" / "human_ceiling.json").read_text())
        self.assertIn(str(ceiling["human_ceiling_icc"]), text)  # report value must come from the computed artifact


if __name__ == "__main__":
    unittest.main()
