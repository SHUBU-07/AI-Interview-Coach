"""
S2 Ensemble Scorer:
Runs S1 several times (or across several model backends) and aggregates by mean/median.

Notes
- With ONE deterministic scorer at temperature 0, repeated runs return identical outputs,
  so S2 == S1. The ensemble only adds information with temperature > 0 or >= 2 different
  models/backends in `s1_scorers`.
- Runs whose JSON failed validation (fallback scores) are excluded from aggregation.
"""

from dataclasses import dataclass
import json
import logging
import statistics
import time
from typing import Dict, List, Optional
import uuid

from coach.scorer_llm import S1Scorer, S1ScoringResult, DimensionScore
from coach.db import CoachDB, SystemScoreRecord

logger = logging.getLogger(__name__)


@dataclass
class S2EnsembleResult:
    answer_id: str
    overall_score: float
    dimension_scores: Dict[str, DimensionScore]
    run_results: List[S1ScoringResult]
    aggregation: str
    score_variance: float
    latency_ms: float
    validation_passed: bool = True
    valid_runs: int = 0


class S2EnsembleScorer:
    def __init__(self, s1_scorers: List[S1Scorer], aggregation: str = "mean", db: Optional[CoachDB] = None):
        self.s1_scorers = s1_scorers
        self.aggregation = aggregation.lower()
        self.db = db

    def _agg(self, values):
        return statistics.median(values) if self.aggregation == "median" else statistics.mean(values)

    def score_answer(self, answer_id: str, question: str, transcript: str,
                     has_audio: bool = False, num_runs: int = 3) -> S2EnsembleResult:
        t0 = time.time()
        results: List[S1ScoringResult] = []
        for i in range(num_runs):
            scorer = self.s1_scorers[i % len(self.s1_scorers)]
            results.append(scorer.score_answer(
                answer_id=answer_id, question=question, transcript=transcript,
                has_audio=has_audio, prompt_variant="default",
            ))
        total_latency = round((time.time() - t0) * 1000, 2)

        valid = [r for r in results if r.validation_passed]
        ensemble_ok = len(valid) > 0
        used = valid if ensemble_ok else results  # if everything failed, keep fallbacks but flag as invalid

        overalls = [r.overall_score for r in used]
        var = statistics.variance(overalls) if len(overalls) > 1 else 0.0
        final_overall = round(self._agg(overalls), 2)

        final_dimensions: Dict[str, DimensionScore] = {}
        for d in used[0].dimension_scores.keys():
            d_scores = [r.dimension_scores[d].score for r in used if d in r.dimension_scores]
            if not d_scores:
                continue
            agg_score = int(round(self._agg(d_scores)))
            best_run = min(used, key=lambda r: abs(r.dimension_scores[d].score - agg_score)
                           if d in r.dimension_scores else 99)
            best = best_run.dimension_scores[d]
            final_dimensions[d] = DimensionScore(
                dimension=d, score=agg_score, evidence_quote=best.evidence_quote,
                reason=f"[Ensemble {self.aggregation.upper()} of {len(used)} valid run(s)] {best.reason}",
                quote_grounded=best.quote_grounded,
            )

        res = S2EnsembleResult(
            answer_id=answer_id, overall_score=final_overall, dimension_scores=final_dimensions,
            run_results=results, aggregation=self.aggregation, score_variance=round(var, 4),
            latency_ms=total_latency, validation_passed=ensemble_ok, valid_runs=len(valid),
        )

        if self.db:
            payload = {
                "overall": final_overall, "variance": round(var, 4), "validation_passed": ensemble_ok,
                "valid_runs": len(valid), "runs": [r.overall_score for r in results],
                "dimensions": {k: v.score for k, v in final_dimensions.items()},
            }
            self.db.save_system_score(SystemScoreRecord(
                score_id=f"SC_{uuid.uuid4().hex[:8]}", answer_id=answer_id, method="S2",
                model_name=f"ensemble-{len(results)}-runs", model_version="1.0",
                temperature=results[0].temperature, prompt_version=f"S2_ensemble_{self.aggregation}",
                prompt_hash="s2_ensemble", scores_json=json.dumps(payload),
                raw_output=f"Ensemble runs: {[r.overall_score for r in results]} -> final: {final_overall}",
                latency_ms=total_latency,
            ))
        return res
