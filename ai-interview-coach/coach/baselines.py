"""
Baseline scorers for AI Interview Coach:
- B1: TF-IDF cosine similarity to a reference answer, mapped to 1-5
- B2: sentence-embedding cosine similarity to a reference answer, mapped to 1-5
- B3: unconstrained LLM prompt ("score out of 10"), rescaled to 1-5

Honesty rule: if a baseline cannot produce a score (model unavailable, unparsable LLM
output) it returns NaN. It never substitutes a made-up value, and NaN rows are excluded
from the agreement evaluation instead of being counted as real scores.
"""

from dataclasses import dataclass
import json
import logging
import math
import re
import time
from typing import Dict, Optional
import uuid

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coach.llm_backend import BaseLLMBackend
from coach.db import CoachDB, SystemScoreRecord

logger = logging.getLogger(__name__)
NAN = float("nan")
DIMS = ["relevance_correctness", "depth_completeness", "structure_clarity", "language_use"]


@dataclass
class BaselineScoreResult:
    answer_id: str
    method: str
    overall_score: float
    raw_similarity: float
    dimension_scores: Dict[str, float]
    raw_output: str
    latency_ms: float


def _log(db: Optional[CoachDB], answer_id: str, method: str, model_name: str, model_version: str,
         temperature: float, prompt_version: str, prompt_hash: str, payload: dict,
         raw_output: str, latency: float) -> None:
    """Persist an audit record, but only for real (non-NaN) scores."""
    if db is None or payload.get("overall") is None or math.isnan(payload["overall"]):
        return
    db.save_system_score(SystemScoreRecord(
        score_id=f"SC_{uuid.uuid4().hex[:8]}", answer_id=answer_id, method=method,
        model_name=model_name, model_version=model_version, temperature=temperature,
        prompt_version=prompt_version, prompt_hash=prompt_hash,
        scores_json=json.dumps(payload), raw_output=raw_output, latency_ms=latency,
    ))


def _similarity_to_score(sim: float) -> float:
    return round(1.0 + 4.0 * max(0.0, min(1.0, sim)), 2)


class B1TfidfScorer:
    """B1: TF-IDF similarity between candidate answer and reference answer, mapped to 1-5."""

    def __init__(self, db: Optional[CoachDB] = None, max_features: int = 1000):
        self.db = db
        self.max_features = max_features

    def score_answer(self, answer_id: str, transcript: str, reference_answer: str) -> BaselineScoreResult:
        t0 = time.time()
        if not transcript.strip() or not reference_answer.strip() or reference_answer.strip().lower() == "nan":
            sim, score = NAN, NAN  # no reference answer -> cannot score
        else:
            try:
                vec = TfidfVectorizer(stop_words="english", max_features=self.max_features)
                tfidf = vec.fit_transform([transcript, reference_answer])
                sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
                score = _similarity_to_score(sim)
            except Exception as e:  # e.g. empty vocabulary after stop-word removal
                logger.error(f"B1 scoring error: {e}")
                sim, score = NAN, NAN
        latency = round((time.time() - t0) * 1000, 2)
        raw = f"TF-IDF cosine similarity: {sim:.4f} -> score: {score:.2f}"
        _log(self.db, answer_id, "B1", "tfidf-sklearn", "1.0", 0.0, "b1_tfidf_reference_cos", "tfidf",
             {"overall": score, "similarity": sim}, raw, latency)
        return BaselineScoreResult(answer_id, "B1", score, sim, {d: score for d in DIMS}, raw, latency)


class B2EmbeddingScorer:
    """B2: sentence-transformers embedding similarity mapped to 1-5."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", db: Optional[CoachDB] = None):
        self.model_name = model_name
        self.db = db
        self._model = None
        self._unavailable = False  # set once if the model cannot be loaded (avoids repeated slow failures)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score_answer(self, answer_id: str, transcript: str, reference_answer: str) -> BaselineScoreResult:
        t0 = time.time()
        if self._unavailable or not transcript.strip() or not reference_answer.strip() \
                or reference_answer.strip().lower() == "nan":
            sim, score = NAN, NAN
        else:
            try:
                emb = self._get_model().encode([transcript, reference_answer], normalize_embeddings=True)
                sim = float(np.dot(emb[0], emb[1]))
                score = _similarity_to_score(sim)
            except Exception as e:
                logger.error(f"B2 unavailable ({e}). B2 scores will be NaN and excluded from evaluation.")
                self._unavailable = True
                sim, score = NAN, NAN
        latency = round((time.time() - t0) * 1000, 2)
        raw = f"Embedding cosine similarity ({self.model_name}): {sim:.4f} -> score: {score:.2f}"
        _log(self.db, answer_id, "B2", self.model_name, "1.0", 0.0, "b2_embedding_reference_cos",
             "minilm_l6", {"overall": score, "similarity": sim}, raw, latency)
        return BaselineScoreResult(answer_id, "B2", score, sim, {d: score for d in DIMS}, raw, latency)


class B3UnconstrainedLLMScorer:
    """B3: unconstrained naive LLM prompt asking for a score out of 10, rescaled to 1-5."""

    def __init__(self, llm_backend: BaseLLMBackend, db: Optional[CoachDB] = None):
        self.llm = llm_backend
        self.db = db

    def score_answer(self, answer_id: str, question: str, transcript: str) -> BaselineScoreResult:
        t0 = time.time()
        prompt = (
            "You are evaluating a job candidate's interview response.\n"
            f"Question: {question}\n\n"
            f"Candidate Answer: {transcript}\n\n"
            "Please rate the overall quality of this answer on a scale from 1 to 10. "
            "State your score clearly as 'Score: X/10' where X is an integer from 1 to 10."
        )
        prompt_hash = self.llm.compute_prompt_hash(prompt)
        try:
            raw_output, meta = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"B3 LLM call failed: {e}")
            return BaselineScoreResult(answer_id, "B3", NAN, NAN, {d: NAN for d in DIMS}, f"ERROR: {e}", 0.0)
        latency = meta.get("latency_ms", (time.time() - t0) * 1000)

        match = re.search(r"(\b(?:10|[1-9])(?:\.\d+)?)\s*(?:/|\s*out of\s*)\s*10", raw_output, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:score|rate|rating|give it|give this answer)\s*(?:is|:)?\s*(\b(?:10|[1-9])(?:\.\d+)?)",
                              raw_output, re.IGNORECASE)
        if match:
            score_10 = float(match.group(1))
            score_5 = max(1.0, min(5.0, round(1.0 + (score_10 - 1.0) * (4.0 / 9.0), 2)))
        else:
            logger.warning("B3: could not parse a score from LLM output; recording NaN (no fallback value).")
            score_10, score_5 = NAN, NAN

        _log(self.db, answer_id, "B3", self.llm.model_name,
             getattr(self.llm, "extra_kwargs", {}).get("version", "1.0"), self.llm.temperature,
             "b3_unconstrained_10pt", prompt_hash, {"overall": score_5, "score_10": score_10},
             raw_output, latency)
        return BaselineScoreResult(answer_id, "B3", score_5, score_10, {d: score_5 for d in DIMS}, raw_output, latency)
