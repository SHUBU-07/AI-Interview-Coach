"""
S1 Proposed Scorer:
Rubric-anchored LLM evaluation with exact construct definitions, structured JSON output,
evidence grounding verification, and full inference logging.
"""

from dataclasses import dataclass, asdict
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from coach.rubric import Rubric, RubricDimension
from coach.llm_backend import BaseLLMBackend
from coach.db import CoachDB, SystemScoreRecord

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    dimension: str
    score: int
    evidence_quote: str
    reason: str
    quote_grounded: bool = True  # True if evidence_quote is verified in candidate transcript


@dataclass
class S1ScoringResult:
    answer_id: str
    overall_score: float
    dimension_scores: Dict[str, DimensionScore]
    raw_output: str
    model_name: str
    model_version: str
    temperature: float
    prompt_version: str
    prompt_hash: str
    latency_ms: float
    validation_passed: bool
    retry_count: int = 0


class S1Scorer:
    PROMPT_VERSION = "S1_rubric_anchored_v1.0"

    def __init__(self, rubric: Rubric, llm_backend: BaseLLMBackend, db: Optional[CoachDB] = None, max_retries: int = 2):
        self.rubric = rubric
        self.llm = llm_backend
        self.db = db
        self.max_retries = max_retries

    def build_scoring_prompt(self, question: str, transcript: str, has_audio: bool = False, prompt_variant: str = "default") -> str:
        rubric_text = self.rubric.format_all_for_prompt(has_audio=has_audio)
        dim_ids = [d.id for d in self.rubric.get_eval_dimensions(has_audio=has_audio)]

        if prompt_variant == "paraphrase_1":
            instruction_header = (
                "You are an expert, objective interview evaluator assessing candidate competence against strict rubric anchors.\n"
                "Review the interview question and the verbatim candidate answer below."
            )
        elif prompt_variant == "paraphrase_2":
            instruction_header = (
                "You are conducting a formal assessment of an interview response using a validated behavioral rubric.\n"
                "Evaluate the candidate's transcript strictly against the specified level descriptors."
            )
        else:
            instruction_header = (
                "You are an expert psychometric evaluator assessing an interview response against an anchored scoring rubric.\n"
                "Evaluate the answer strictly based on evidence provided in the transcript."
            )

        prompt = f"""{instruction_header}

{rubric_text}

---
Question Asked:
{question}

Candidate Answer:
{transcript}

---
Instructions:
1. Score each dimension on an integer scale from 1 to 5 strictly aligned with the anchor descriptors above.
2. For EVERY dimension, extract an exact verbatim quote from the candidate answer as `evidence_quote`. Do NOT invent or alter the quote. If the candidate failed to address the dimension, quote the closest excerpt or write 'N/A'.
3. Provide a concise analytical `reason` justifying why the evidence warrants that specific level.
4. Output MUST be a strictly valid JSON array of objects, one per dimension, matching this schema:
[
  {{
    "dimension": "<dimension_id>",
    "score": <integer 1 to 5>,
    "evidence_quote": "<verbatim quote from transcript>",
    "reason": "<specific justification citing the anchor>"
  }}
]

Required dimension IDs to score: {json.dumps(dim_ids)}
Respond ONLY with the JSON array. Do not include markdown codeblocks or conversational text.
"""
        return prompt

    def verify_evidence_quote(self, quote: str, transcript: str) -> bool:
        """Verify that the evidence quote is genuinely grounded in the candidate transcript."""
        if not quote or quote.strip() in ["N/A", "none", "None"]:
            return True
        clean_quote = re.sub(r"[^\w\s]", "", quote.lower()).strip()
        clean_transcript = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
        if not clean_quote:
            return True
        # Exact substring or fuzzy token overlap
        if clean_quote in clean_transcript:
            return True
        # Check token sequence overlap (at least 75% consecutive words)
        q_words = clean_quote.split()
        if len(q_words) >= 4:
            chunk = " ".join(q_words[:min(6, len(q_words))])
            if chunk in clean_transcript:
                return True
        return False

    def parse_and_validate(self, raw_output: str, transcript: str, expected_dims: List[str]) -> Tuple[Dict[str, DimensionScore], bool]:
        clean_text = raw_output.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # Find outer array bounds if extra text is present
        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx : end_idx + 1]

        data = json.loads(clean_text)
        if not isinstance(data, list):
            raise ValueError("Expected a top-level JSON array")

        dimension_scores = {}
        for item in data:
            dim_id = item.get("dimension")
            if dim_id not in expected_dims:
                continue  # ignore dimensions that were not requested (e.g. delivery without audio)
            score = int(item.get("score"))
            quote = str(item.get("evidence_quote", "")).strip()
            reason = str(item.get("reason", "")).strip()

            if score < 1 or score > 5:
                raise ValueError(f"Score {score} out of bounds [1, 5] for {dim_id}")

            grounded = self.verify_evidence_quote(quote, transcript)
            dimension_scores[dim_id] = DimensionScore(
                dimension=dim_id,
                score=score,
                evidence_quote=quote,
                reason=reason,
                quote_grounded=grounded,
            )

        # Check that all expected dimensions are present
        missing = [d for d in expected_dims if d not in dimension_scores]
        if missing:
            raise ValueError(f"Missing required dimensions in LLM output: {missing}")

        return dimension_scores, True

    def score_answer(
        self,
        answer_id: str,
        question: str,
        transcript: str,
        has_audio: bool = False,
        prompt_variant: str = "default",
        max_retries: Optional[int] = None,
    ) -> S1ScoringResult:
        if max_retries is None:
            max_retries = self.max_retries
        eval_dims = [d.id for d in self.rubric.get_eval_dimensions(has_audio=has_audio)]
        prompt = self.build_scoring_prompt(question, transcript, has_audio=has_audio, prompt_variant=prompt_variant)
        prompt_hash = self.llm.compute_prompt_hash(prompt)

        raw_output = ""
        last_error = ""
        dimension_scores = {}
        validation_passed = False
        total_latency = 0.0

        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                raw_output, meta = self.llm.generate(prompt)
                latency = meta.get("latency_ms", (time.time() - t0) * 1000)
                total_latency += latency

                dimension_scores, validation_passed = self.parse_and_validate(raw_output, transcript, eval_dims)
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"S1 scoring attempt {attempt+1} failed: {e}")
                time.sleep(0.5)

        # Fallback if all attempts failed
        if not validation_passed:
            logger.error(f"S1 validation permanently failed for answer {answer_id}: {last_error}")
            for d in eval_dims:
                dimension_scores[d] = DimensionScore(
                    dimension=d,
                    score=3,  # Neutral midpoint fallback
                    evidence_quote="N/A",
                    reason=f"LLM parsing failed: {last_error}",
                    quote_grounded=False,
                )

        scores_list = [ds.score for ds in dimension_scores.values()]
        overall_score = round(sum(scores_list) / len(scores_list), 2) if scores_list else 0.0

        result = S1ScoringResult(
            answer_id=answer_id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            raw_output=raw_output,
            model_name=self.llm.model_name,
            model_version=getattr(self.llm, "extra_kwargs", {}).get("version", "1.0"),
            temperature=self.llm.temperature,
            prompt_version=f"{self.PROMPT_VERSION}:{prompt_variant}",
            prompt_hash=prompt_hash,
            latency_ms=round(total_latency, 2),
            validation_passed=validation_passed,
            retry_count=attempt,
        )

        # Save audited log record to database if DB instance attached
        if self.db:
            scores_payload = {
                "overall": overall_score,
                "validation_passed": validation_passed,
                "dimensions": {
                    k: {
                        "score": v.score,
                        "evidence_quote": v.evidence_quote,
                        "reason": v.reason,
                        "quote_grounded": v.quote_grounded,
                    }
                    for k, v in dimension_scores.items()
                },
            }
            record = SystemScoreRecord(
                score_id=f"SC_{uuid.uuid4().hex[:8]}",
                answer_id=answer_id,
                method="S1",
                model_name=result.model_name,
                model_version=result.model_version,
                temperature=result.temperature,
                prompt_version=result.prompt_version,
                prompt_hash=result.prompt_hash,
                scores_json=json.dumps(scores_payload),
                raw_output=raw_output,
                latency_ms=result.latency_ms,
            )
            self.db.save_system_score(record)

        return result
